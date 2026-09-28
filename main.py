"""
main.py — Лабораторна робота №1: AI-агент з довготривалою RAG-пам'яттю.

Курс «Прикладний штучний інтелект та MLOps процесу розробки програмного
забезпечення», Лекція 4 (smolagents + Ollama).

ЩО ТУТ Є
    1. MemoryManager — довготривала пам'ять агента: факти зберігаються як
       вектори у faiss-індексі, пошук — за косинусною близькістю.
    2. Ембединги рахує ЛОКАЛЬНИЙ Ollama (bge-m3, 1024 виміри) через
       POST /api/embed. Ніякого sentence-transformers і ніякого torch:
       модель уже стоїть поруч з LLM, вона багатомовна й розуміє українську.
    3. Чотири інструменти @tool, два з яких керують самою пам'яттю —
       агент САМ вирішує, що запам'ятати і коли відповісти з пам'яті.
    4. Сценарний демо-прогін за замовчуванням (не входить в input()-цикл),
       інтерактивний режим — за прапорцем --chat.

ЯК ЗАПУСТИТИ
    python main.py           # сценарій із 4 кроків, доводить що пам'ять працює
    python main.py --chat    # інтерактивний діалог
    python main.py --help    # усі прапорці

    Потрібен запущений Ollama з моделями `qwen3:8b` і `bge-m3`.
    Якщо чогось бракує — скрипт друкує зрозуміле пояснення і виходить
    без трейсбеку.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path


# ══════════════════════════════════════════════════════════════════
# 0. КОНСОЛЬ.
#    Windows-консоль стартує у cp1251 і падає на кирилиці/емодзі ще до
#    першого змістовного рядка — особливо при `python main.py > log.txt`.
#    Викликаємо ПЕРЕД будь-яким print().
# ══════════════════════════════════════════════════════════════════
def setup_console() -> None:
    """Перемикає stdout/stderr на UTF-8. Безпечно викликати багато разів."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass  # не TextIOWrapper (наприклад, під pytest) — нічого страшного


setup_console()

# ── .env (не обов'язковий) ────────────────────────────────────────
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

# ══════════════════════════════════════════════════════════════════
# 1. НАЛАШТУВАННЯ. Усе перекривається через .env — див. .env.example.
# ══════════════════════════════════════════════════════════════════
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
AGENT_MODEL = os.getenv("AGENT_MODEL", "qwen3:8b")
EMBED_MODEL = os.getenv("EMBED_MODEL", "bge-m3:latest")
NUM_CTX = int(os.getenv("NUM_CTX", "8192"))
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "120"))
# Поріг косинусної близькості: нижче — спогад вважаємо нерелевантним і
# просто не показуємо агентові. Без порогу faiss завжди поверне k «найкращих»
# записів, навіть якщо всі вони про інше, і модель почне вигадувати з них.
MEMORY_MIN_SCORE = float(os.getenv("MEMORY_MIN_SCORE", "0.40"))
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", str(Path(__file__).resolve().parent / "output")))

# ── М'яка перевірка залежностей ───────────────────────────────────
# Правило лекції: скрипт ніколи не падає трейсбеком — він пояснює.
MISSING: list[str] = []
try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None
    MISSING.append("numpy")
try:
    import requests
except ImportError:  # pragma: no cover
    requests = None
    MISSING.append("requests")
try:
    import faiss
except ImportError:  # pragma: no cover
    faiss = None
    MISSING.append("faiss-cpu")
try:
    from smolagents import CodeAgent, LiteLLMModel, tool
except ImportError:  # pragma: no cover
    CodeAgent = LiteLLMModel = None
    MISSING.append("smolagents[litellm]")

    def tool(func):  # заглушка, щоб модуль хоча б імпортувався
        return func


# ══════════════════════════════════════════════════════════════════
# 2. БЕЗПЕЧНИЙ ШЛЯХ ДЛЯ ЗАПИСУ ФАЙЛІВ.
#
#    Ім'я файлу, яке склала LLM, — це НЕДОВІРЕНИЙ ввід. Варіанти 6 і 20
#    записують файли на диск; без цієї функції агент цілком здатен
#    попросити записати у "../../.env" або "C:/Windows/system32/x.dll".
#    Правило: усе, що агент пише, лягає лише в OUTPUT_DIR.
# ══════════════════════════════════════════════════════════════════
_RESERVED_WINDOWS_NAMES = {
    "con", "prn", "aux", "nul",
    *(f"com{i}" for i in range(1, 10)),
    *(f"lpt{i}" for i in range(1, 10)),
}


def _basename(raw: str) -> str:
    """Останній компонент шляху, незалежно від типу слеша та літери диска."""
    normalised = raw.replace("\\", "/").rstrip("/")
    return normalised.rsplit("/", 1)[-1].strip()


def safe_output_path(filename: str, base_dir: Path | None = None) -> Path:
    """
    Перетворює довільне ім'я файлу на шлях усередині робочої теки.

    Кидає ValueError, якщо ім'я порожнє, містить \\x00 або (після
    нормалізації) намагається вийти за межі base_dir.
    """
    base = Path(base_dir) if base_dir is not None else OUTPUT_DIR
    base = base.resolve()

    raw = str(filename).strip()
    if not raw or "\x00" in raw:
        raise ValueError("Порожнє або некоректне ім'я файлу")

    # Прибираємо будь-які теки та літери диска: лишається тільки базове ім'я.
    # replace("\\", "/") потрібен, бо на Linux зворотний слеш — звичайний символ.
    name = _basename(raw)
    if not name or name in {".", ".."}:
        raise ValueError(f"Некоректне ім'я файлу: {filename!r}")
    if name.split(".")[0].lower() in _RESERVED_WINDOWS_NAMES:
        raise ValueError(f"Зарезервоване ім'я файлу Windows: {name!r}")

    base.mkdir(parents=True, exist_ok=True)
    candidate = (base / name).resolve()
    if not candidate.is_relative_to(base):  # страхувальний пояс
        raise ValueError(f"Шлях виходить за межі робочої теки: {filename!r}")
    return candidate


# ══════════════════════════════════════════════════════════════════
# 3. ЕМБЕДИНГИ ЧЕРЕЗ OLLAMA.
#
#    Форма виклику (перевірено живим запитом на Ollama 0.32):
#        POST http://localhost:11434/api/embed
#        {"model": "bge-m3:latest", "input": ["текст 1", "текст 2"]}
#      → {"embeddings": [[float x1024], [float x1024]], ...}
#    Ключ саме "embeddings" (множина, список списків), не "embedding".
#
#    Чому не sentence-transformers:
#      • тягне torch — ~2 ГБ завантаження на кожного студента;
#      • all-MiniLM-L6-v2 — АНГЛОМОВНА модель, а ми зберігаємо в пам'ять
#        українські факти («Перевернуте слово "Кіт"»). Вона їх не розрізняє.
#    bge-m3 багатомовна, 1024 виміри, і вже стоїть поруч з LLM.
# ══════════════════════════════════════════════════════════════════
class OllamaEmbedder:
    """Рахує L2-нормовані вектори текстів локальним Ollama."""

    def __init__(
        self,
        model: str = EMBED_MODEL,
        base_url: str = OLLAMA_BASE_URL,
        timeout: int = REQUEST_TIMEOUT,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._dimension: int | None = None

    @property
    def dimension(self) -> int:
        """Розмір вектора. Визначається одним пробним запитом і кешується."""
        if self._dimension is None:
            self._dimension = len(self.embed(["dimension probe"])[0])
        return self._dimension

    def embed(self, texts: list[str]) -> "np.ndarray":
        """Один POST на весь список — на порядок швидше, ніж запит на текст."""
        if not texts:
            raise ValueError("embed() отримав порожній список")
        resp = requests.post(
            f"{self.base_url}/api/embed",
            json={"model": self.model, "input": texts},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        vectors = np.asarray(resp.json()["embeddings"], dtype="float32")
        return l2_normalize(vectors)


def l2_normalize(vectors: "np.ndarray") -> "np.ndarray":
    """
    Нормуємо вектори до довжини 1.

    Навіщо: після нормування скалярний добуток ДОРІВНЮЄ косинусній
    близькості. Тому далі беремо faiss.IndexFlatIP (inner product) —
    і score виходить у зрозумілому діапазоні [-1, 1], який можна
    порівняти з порогом. З «сирим» IndexFlatL2 відстані ні з чим
    не порівняєш: їхній масштаб залежить від моделі.
    """
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (vectors / norms).astype("float32")


# ══════════════════════════════════════════════════════════════════
# 4. ДОВГОТРИВАЛА ПАМ'ЯТЬ (RAG).
#
#    Це і є головна ідея лабораторної: вікно контексту LLM скінченне,
#    тому «пам'ять» живе поза моделлю — у векторному індексі, а в промпт
#    підмішуються лише релевантні шматки.
# ══════════════════════════════════════════════════════════════════
class MemoryManager:
    """Векторне сховище фактів на faiss + ембедер (будь-який, з методом embed)."""

    def __init__(self, embedder, min_score: float = MEMORY_MIN_SCORE, verbose: bool = True) -> None:
        self.embedder = embedder
        self.min_score = min_score
        self.verbose = verbose
        self.dimension = embedder.dimension
        # IndexFlatIP + нормовані вектори = точний косинусний пошук.
        # Flat означає повний перебір: для сотень фактів це мікросекунди,
        # а результат точний (на відміну від наближеного HNSW/IVF).
        self.index = faiss.IndexFlatIP(self.dimension)
        self.memories: list[str] = []
        self._log(f"Пам'ять готова: ембедер {getattr(embedder, 'model', '?')}, "
                  f"{self.dimension} вимірів")

    # ── службове ──────────────────────────────────────────────────
    def _log(self, message: str) -> None:
        if self.verbose:
            print(f"   [memory] {message}")

    def __len__(self) -> int:
        return len(self.memories)

    def reset(self) -> None:
        """Забути все. Індекс створюємо новий — faiss не має reset() на IP."""
        self.index = faiss.IndexFlatIP(self.dimension)
        self.memories = []
        self._log("очищено")

    # ── запис ─────────────────────────────────────────────────────
    def add_memory(self, text: str) -> None:
        text = (text or "").strip()
        if not text:
            raise ValueError("Порожній спогад зберігати немає сенсу")
        vector = self.embedder.embed([text])
        self.index.add(vector)
        self.memories.append(text)
        self._log(f"збережено #{len(self.memories)}: {text[:70]}")

    # ── читання ───────────────────────────────────────────────────
    def search(self, query: str, k: int = 3) -> list[tuple[str, float]]:
        """Повертає [(текст, косинусна близькість)] — уже відсіяне за порогом."""
        if not self.memories or k <= 0:
            return []
        k = min(k, self.index.ntotal)  # faiss не любить k > ntotal
        scores, indices = self.index.search(self.embedder.embed([query]), k)
        hits: list[tuple[str, float]] = []
        for idx, score in zip(indices[0], scores[0]):
            if idx < 0:  # faiss ставить -1, якщо результатів менше за k
                continue
            if float(score) < self.min_score:
                continue
            hits.append((self.memories[int(idx)], float(score)))
        return hits

    def retrieve_relevant_memory(self, query: str, k: int = 3) -> str:
        """Готовий текстовий блок для промпта. Порожній рядок = нічого не знайшли."""
        hits = self.search(query, k)
        if not hits:
            self._log("релевантних спогадів немає")
            return ""
        for text, score in hits:
            self._log(f"знайдено (cos={score:.2f}): {text[:70]}")
        return "\n".join(text for text, _ in hits)


# ══════════════════════════════════════════════════════════════════
# 5. ІНСТРУМЕНТИ АГЕНТА.
#
#    Docstring інструмента читає МОДЕЛЬ — тому він англійською і описує
#    рівно те, коли інструмент застосовувати. Коментарі — для людини.
#
#    Пам'ять живе в модульній змінній: інструмент отримує її через
#    get_memory(). Так @tool-функції лишаються звичайними функціями,
#    які легко тестувати, підставивши фейковий ембедер.
# ══════════════════════════════════════════════════════════════════
_MEMORY: MemoryManager | None = None


def attach_memory(memory: MemoryManager | None) -> None:
    """Прив'язує сховище пам'яті до інструментів (викликається з main/тестів)."""
    global _MEMORY
    _MEMORY = memory


def get_memory() -> MemoryManager:
    if _MEMORY is None:
        raise RuntimeError("Пам'ять не ініціалізована: викличте attach_memory() першим")
    return _MEMORY


@tool
def count_words(text: str) -> int:
    """
    Counts the number of words in a text.

    Args:
        text: The text to analyze.
    """
    print(f"   [tool] count_words({text!r})")
    return len(text.split())


@tool
def reverse_text(text: str) -> str:
    """
    Reverses a text character by character.

    Args:
        text: The text to reverse.
    """
    print(f"   [tool] reverse_text({text!r})")
    return text[::-1]


@tool
def save_fact_to_memory(fact_description: str, fact_content: str) -> str:
    """
    Saves a fact to long-term memory so it can be recalled in later turns.
    Call this whenever the user asks you to remember something, or after you
    compute a result the user may ask about again.

    Args:
        fact_description: Short description of the fact, e.g. "user's favorite color".
        fact_content: The value of the fact, e.g. "blue".
    """
    record = f"Fact about '{fact_description}': '{fact_content}'"
    get_memory().add_memory(record)
    return "Fact saved to long-term memory."


@tool
def answer_from_context(answer_text: str) -> str:
    """
    Use this when the answer is already present in the Memory block and no
    computation is needed. Returns the answer unchanged.

    Args:
        answer_text: The final answer for the user, in Ukrainian.
    """
    print(f"   [tool] answer_from_context({answer_text!r})")
    return answer_text


TOOLS = [count_words, reverse_text, save_fact_to_memory, answer_from_context]

# ══════════════════════════════════════════════════════════════════
# 6. ПРОМПТ.
#
#    Цей текст не написаний «на око» — його підбирали вимірюванням.
#    Демо-сценарій (4 ходи) прогнали багато разів на кожному варіанті
#    промпта і рахували два останні ходи: чи агент дістав правильний факт
#    з пам'яті (2 перевірки на прогін). qwen3:8b, вересень 2026:
#
#      A  самі правила, без прикладів                11/16  (8 прогонів)
#      B  A + «ти ЗОБОВ'ЯЗАНИЙ кликати answer_...»    4/6   (3 прогони)
#      C  минулорічний XML-промпт з 5 прикладами       8/8   (4 прогони)
#      D  A + 2 приклади українською                 15/20  (10 прогонів)
#      E  D + тег <thought>                          20/20  (10 прогонів) ←
#
#    Що з цього випливає:
#      • Приклади мусять бути УКРАЇНСЬКОЮ. Саме правило «відповідай
#        українською» не працює: у варіанті A лише 9 з 32 відповідей були
#        українською, у D і E — 80 з 80.
#      • Тег <thought> — не просто спадщина llama3. Примусовий короткий
#        план перед кодом тримає qwen3:8b у сценарії: варіант D (без нього)
#        5 разів із 20 повернув текст самого запитання замість відповіді.
#      • Наказ «ОБОВ'ЯЗКОВО клич answer_from_context» (B) робить гірше:
#        модель виконує букву наказу і пхає в інструмент текст питання.
#      • Чого НЕ лишилося з минулорічної версії: п'яти дубльованих
#        прикладів (варіант C відповідає їхніми ж фразами — «I counted 4
#        words and saved this fact to memory» — замість своїх) і явного
#        code_block_tags: ("<code>", "</code>") і так типове значення
#        у smolagents 1.26.
#
#    Відтворити вимірювання можна, підмінивши main.INSTRUCTIONS і
#    прогнавши DEMO_STEPS у циклі — рівно це й робилося.
# ══════════════════════════════════════════════════════════════════
INSTRUCTIONS = """You are a Python assistant with a long-term vector memory.

Always start your answer with a plan inside <thought> and </thought> tags.

Rules:
1. A "## Memory" block may be prepended to the task. It contains facts you
   saved in earlier turns. If it already answers the question, call
   answer_from_context(answer_text=...) and pass its result to final_answer().
   Do not recompute what you already remember.
2. If the Memory block is missing or does not answer the question, solve the
   task with the other tools. Never compute in your head what a tool can do.
3. Whenever the user asks you to remember something, or you produce a result
   worth recalling later, call
   save_fact_to_memory(fact_description=..., fact_content=...)
   BEFORE final_answer(). Save the input as well as the result.
4. Always call tools with keyword arguments: count_words(text="...").
5. The last line of your code must always be final_answer(...).
6. Write the final answer to the user in Ukrainian.

Example A - nothing in memory yet, so compute and save:

## Current task
Count the words in 'this is a test' and remember the count.

<thought>
I will count the words and save the result.
</thought>
<code>
n = count_words(text="this is a test")
save_fact_to_memory(fact_description="word count of 'this is a test'", fact_content=str(n))
final_answer(f"У фразі 'this is a test' {n} слова. Я це запам'ятав.")
</code>

Example B - the Memory block already holds the answer:

## Memory (facts you saved in earlier turns)
Fact about 'user's favorite color': 'blue'

## Current task
What is my favorite color?

<thought>
The memory block already answers this.
</thought>
<code>
answer = answer_from_context(answer_text="Ваш улюблений колір - синій.")
final_answer(answer)
</code>
"""


def build_agent(model, verbosity_level: int = 0) -> "CodeAgent":
    """Створює CodeAgent з нашими інструментами.

    code_block_tags НЕ передаємо: ("<code>", "</code>") — і так типове
    значення у smolagents 1.26.
    """
    return CodeAgent(
        tools=TOOLS,
        model=model,
        instructions=INSTRUCTIONS,
        max_steps=4,
        verbosity_level=verbosity_level,
    )


def build_model(model_name: str = AGENT_MODEL, num_ctx: int = NUM_CTX):
    """
    Модель точно так само, як у Лекції 4.

    1. ``ollama_chat/``, а не ``ollama/``: другий веде на застарілий
       /api/generate, перший — на /api/chat з chat-шаблоном і tool calling.
    2. ``num_ctx=8192`` — про VRAM, а не про обрізання промпту: без нього
       Ollama бере повне вікно моделі (qwen3:8b → 40960 токенів,
       ~11 ГБ замість ~6.3 ГБ).
    3. ``reasoning_effort="none"`` → LiteLLM надсилає {"think": false}.
       qwen3 — гібридна reasoning-модель; без цього прапорця вона пише
       довгу <think>-трасу перед кожним кроком (≈47 c замість ≈17 c).
    """
    return LiteLLMModel(
        model_id=f"ollama_chat/{model_name}",
        api_base=OLLAMA_BASE_URL,
        num_ctx=num_ctx,
        reasoning_effort="none",
    )


# ══════════════════════════════════════════════════════════════════
# 7. ПЕРЕВІРКА СЕРЕДОВИЩА — щоб студент бачив причину, а не трейсбек.
# ══════════════════════════════════════════════════════════════════
def check_environment(model_name: str, embed_model: str) -> list[str]:
    """Повертає список проблем. Порожній список = все гаразд."""
    if MISSING:
        return [
            "Не встановлені пакети: " + ", ".join(MISSING),
            "   Виправити:  pip install -r requirements.txt",
        ]
    try:
        resp = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        resp.raise_for_status()
        available = {m["name"] for m in resp.json().get("models", [])}
    except Exception as exc:
        return [
            f"Ollama не відповідає на {OLLAMA_BASE_URL} ({type(exc).__name__}).",
            "   Windows/macOS — запустіть застосунок Ollama;  Linux — `ollama serve`",
        ]

    problems: list[str] = []
    for name, hint in ((model_name, model_name), (embed_model, embed_model.split(":")[0])):
        # Ollama показує моделі як "qwen3:8b" / "bge-m3:latest" — звіряємо обидві форми.
        if name not in available and f"{name}:latest" not in available:
            problems.append(f"Модель '{name}' не знайдена в Ollama.")
            problems.append(f"   Виправити:  ollama pull {hint}")
    return problems


# ══════════════════════════════════════════════════════════════════
# 8. ОДИН ХІД ДІАЛОГУ: RAG-пошук → промпт з контекстом → агент.
# ══════════════════════════════════════════════════════════════════
def build_task_with_memory(task: str, context: str) -> str:
    """Підмішує знайдені спогади у промпт окремим, явно названим блоком."""
    if not context:
        return task
    return (
        "## Memory (facts you saved in earlier turns)\n"
        f"{context}\n\n"
        "## Current task\n"
        f"{task}"
    )


def run_turn(agent, memory: MemoryManager, task: str, k: int = 3) -> str:
    """Виконує один хід і повертає відповідь агента (або текст помилки)."""
    context = memory.retrieve_relevant_memory(task, k=k)
    started = time.time()
    try:
        # reset=True: історію кроків не тягнемо між ходами — весь «спогад»
        # має приходити тільки з векторної пам'яті. Саме це ми й перевіряємо.
        result = agent.run(build_task_with_memory(task, context), reset=True)
    except Exception as exc:
        return f"[!] Помилка виконання: {type(exc).__name__}: {exc}"
    print(f"   [time] {time.time() - started:.1f} c")
    return str(result)


# ══════════════════════════════════════════════════════════════════
# 9. РЕЖИМИ ЗАПУСКУ.
# ══════════════════════════════════════════════════════════════════
DEMO_STEPS = [
    "Порахуй, скільки слів у фразі 'the quick brown fox', і запам'ятай результат.",
    "Переверни слово 'Кіт' і запам'ятай, що вийшло.",
    "Яке слово я просив тебе перевернути і що з нього вийшло?",
    "Скільки слів було у фразі, яку я просив порахувати?",
]


def run_demo(agent, memory: MemoryManager) -> int:
    """
    Сценарій із 4 ходів: два перші щось рахують і зберігають,
    два останні перевіряють, чи агент дістане це з пам'яті.
    Перші два ходи мають викликати save_fact_to_memory,
    останні два — answer_from_context.
    """
    print("\n─── Демо-сценарій: перевіряємо довготривалу пам'ять ───")
    for number, task in enumerate(DEMO_STEPS, 1):
        print(f"\n[{number}/{len(DEMO_STEPS)}] > {task}")
        print(f"    ← {run_turn(agent, memory, task)}")
    print(f"\n─── Готово. У пам'яті {len(memory)} фактів. ───")
    print("Інтерактивний режим:  python main.py --chat")
    return 0


def run_chat(agent, memory: MemoryManager) -> int:
    """Інтерактивний діалог. Коректно завершується на Ctrl+C, Ctrl+D та EOF."""
    print("\n─── Інтерактивний режим з RAG-пам'яттю ───")
    print("'/new' — очистити пам'ять,  'exit' або Ctrl+D — вихід.")
    while True:
        try:
            task = input("\n> Ваше завдання: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nЗавершено.")
            return 0
        if not task:
            continue
        if task.lower() in {"exit", "quit", "вихід", "/exit"}:
            print("Бувай.")
            return 0
        if task.lower() == "/new":
            memory.reset()
            print("Пам'ять очищено — починаємо новий чат.")
            continue
        try:
            print(f"    ← {run_turn(agent, memory, task)}")
        except KeyboardInterrupt:
            print("\n[!] Перервано користувачем.")


# ══════════════════════════════════════════════════════════════════
# 10. ТОЧКА ВХОДУ.
# ══════════════════════════════════════════════════════════════════
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="ЛР №1: AI-агент на smolagents з довготривалою RAG-пам'яттю (Ollama + faiss).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Приклади:\n"
               "  python main.py                 сценарний демо-прогін (за замовчуванням)\n"
               "  python main.py --chat          інтерактивний діалог\n"
               "  python main.py --model qwen3:4b   слабкий GPU\n",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--demo", action="store_true",
                      help="сценарний прогін із 4 кроків (режим за замовчуванням)")
    mode.add_argument("--chat", action="store_true",
                      help="інтерактивний діалог замість сценарію")
    parser.add_argument("--model", default=AGENT_MODEL,
                        help=f"модель Ollama для агента (типово {AGENT_MODEL})")
    parser.add_argument("--embed-model", default=EMBED_MODEL,
                        help=f"модель ембедингів (типово {EMBED_MODEL})")
    parser.add_argument("--num-ctx", type=int, default=NUM_CTX,
                        help=f"вікно контексту Ollama (типово {NUM_CTX})")
    parser.add_argument("--verbose", action="store_true",
                        help="показувати повні кроки агента (verbosity_level=2)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    setup_console()
    args = parse_args(argv)

    print("═" * 62)
    print("  ЛР №1 — AI-агент з довготривалою RAG-пам'яттю")
    print("═" * 62)

    problems = check_environment(args.model, args.embed_model)
    if problems:
        print("\nЗапуск неможливий:")
        for line in problems:
            print(f"[!] {line}" if not line.startswith("   ") else line)
        return 0  # не трейсбек і не «червоний» CI — просто зрозуміла причина

    embedder = OllamaEmbedder(model=args.embed_model)
    try:
        memory = MemoryManager(embedder)
    except Exception as exc:
        print(f"\n[!] Не вдалося порахувати ембединги: {type(exc).__name__}: {exc}")
        print(f"    Перевірте:  ollama pull {args.embed_model.split(':')[0]}")
        return 0
    attach_memory(memory)

    model = build_model(args.model, args.num_ctx)
    agent = build_agent(model, verbosity_level=2 if args.verbose else 0)
    print(f"   [agent] ollama_chat/{args.model}, num_ctx={args.num_ctx}, think=off")

    return run_chat(agent, memory) if args.chat else run_demo(agent, memory)


if __name__ == "__main__":
    sys.exit(main())
