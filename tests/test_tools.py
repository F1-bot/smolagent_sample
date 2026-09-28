"""
Регресійні тести до ЛР №1 — БЕЗ жодного виклику моделі та без мережі.

Інструмент агента — це звичайний детермінований код, і покривати його треба
як звичайний код. Саму LLM у тестах не ганяють: вона повільна й недетермінована.
Ембедер тут теж підмінений на фейковий — тому pytest не потребує ані Ollama,
ані завантажених моделей і минає за секунди.

    pytest -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import main  # noqa: E402


# ══════════════════════════════════════════════════════════════════
# Фейкові ембедери: той самий інтерфейс, що й у OllamaEmbedder
# (.model, .dimension, .embed(list[str]) -> np.ndarray), але без мережі.
# ══════════════════════════════════════════════════════════════════
class HashingEmbedder:
    """Детерміновані вектори: кожне слово потрапляє у свій «кошик»."""

    model = "fake-hashing"

    def __init__(self, dim: int = 64) -> None:
        self.dim = dim

    @property
    def dimension(self) -> int:
        return self.dim

    def embed(self, texts: list[str]) -> np.ndarray:
        rows = []
        for text in texts:
            vector = np.zeros(self.dim, dtype="float32")
            for word in text.lower().split():
                vector[hash(word) % self.dim] += 1.0
            rows.append(vector)
        return main.l2_normalize(np.vstack(rows))


class ScriptedEmbedder:
    """Повертає наперед задані вектори — щоб перевіряти поріг точно."""

    model = "fake-scripted"
    dimension = 2

    def __init__(self, table: dict[str, list[float]]) -> None:
        self.table = table

    def embed(self, texts: list[str]) -> np.ndarray:
        rows = [self.table[text] for text in texts]
        return main.l2_normalize(np.asarray(rows, dtype="float32"))


@pytest.fixture
def memory():
    """Чиста пам'ять, прив'язана до інструментів; після тесту — від'єднана."""
    manager = main.MemoryManager(HashingEmbedder(), min_score=0.0, verbose=False)
    main.attach_memory(manager)
    yield manager
    main.attach_memory(None)


# ══════════════════════════════════════════════════════════════════
# 1. Прості інструменти
# ══════════════════════════════════════════════════════════════════
@pytest.mark.parametrize(
    "text, expected",
    [
        ("the quick brown fox", 4),
        ("one", 1),
        ("", 0),
        ("   ", 0),
        ("  подвійні   пробіли  всередині ", 3),
        ("рядок\nз\tрізними\rроздільниками", 4),  # split() ріже будь-який пробіл
    ],
)
def test_count_words(text, expected):
    assert main.count_words(text=text) == expected


@pytest.mark.parametrize(
    "text, expected",
    [
        ("desserts", "stressed"),
        ("Кіт", "тіК"),        # кирилиця не має ламатися
        ("", ""),
        ("ab", "ba"),
    ],
)
def test_reverse_text(text, expected):
    assert main.reverse_text(text=text) == expected


def test_reverse_text_is_an_involution():
    original = "Прикладний ШІ та MLOps"
    assert main.reverse_text(text=main.reverse_text(text=original)) == original


def test_answer_from_context_returns_text_unchanged():
    answer = "Ваш улюблений колір — синій."
    assert main.answer_from_context(answer_text=answer) == answer


# ── Захист від «ехо запиту» ───────────────────────────────────────
# Приблизно в одному прогоні з двадцяти qwen3:8b передає в цей інструмент
# текст самого запитання, і користувач отримує своє ж питання як відповідь.
# Інструмент мусить упізнати це сам і повернути помилку — вона стає
# Observation, за яким агент виправляється наступним кроком.
@pytest.fixture
def task_is_reverse_question(monkeypatch):
    monkeypatch.setattr(
        main, "_CURRENT_TASK",
        "Яке слово я просив тебе перевернути і що з нього вийшло?")


def test_answer_rejects_verbatim_echo_of_the_question(task_is_reverse_question):
    echo = "Яке слово я просив тебе перевернути і що з нього вийшло?"
    assert main.answer_from_context(answer_text=echo).startswith("ERROR")


def test_answer_rejects_any_question_without_a_number(task_is_reverse_question):
    assert main.answer_from_context(
        answer_text="Скільки слів було у фразі, яку я просив порахувати?").startswith("ERROR")


def test_answer_rejects_empty_string(task_is_reverse_question):
    assert main.answer_from_context(answer_text="   ").startswith("ERROR")


@pytest.mark.parametrize("good", [
    "Ви просили перевернути слово 'Кіт', вийшло 'тіК'.",
    "У фразі 'the quick brown fox' було 4 слова.",
    "Чи пам'ятаю я? Так: збережено 4 слова.",  # питальний знак, але є цифра
])
def test_answer_lets_real_answers_through(task_is_reverse_question, good):
    assert main.answer_from_context(answer_text=good) == good


# ══════════════════════════════════════════════════════════════════
# 2. MemoryManager
# ══════════════════════════════════════════════════════════════════
def test_empty_memory_returns_empty_string(memory):
    assert len(memory) == 0
    assert memory.retrieve_relevant_memory("будь-що") == ""
    assert memory.search("будь-що") == []


def test_add_and_retrieve(memory):
    memory.add_memory("Fact about 'favorite color': 'blue'")
    memory.add_memory("Fact about 'reversed word Кіт': 'тіК'")
    assert len(memory) == 2
    assert memory.index.ntotal == 2

    context = memory.retrieve_relevant_memory("favorite color", k=1)
    assert "blue" in context


def test_k_larger_than_number_of_memories(memory):
    """k > ntotal не має падати: faiss отримує вже обрізане k."""
    memory.add_memory("єдиний факт")
    hits = memory.search("єдиний факт", k=10)
    assert len(hits) == 1
    assert memory.retrieve_relevant_memory("єдиний факт", k=10) == "єдиний факт"


def test_non_positive_k_returns_nothing(memory):
    memory.add_memory("факт")
    assert memory.search("факт", k=0) == []
    assert memory.retrieve_relevant_memory("факт", k=-1) == ""


def test_retrieved_block_keeps_every_hit_on_its_own_line(memory):
    memory.add_memory("перший факт")
    memory.add_memory("другий факт")
    block = memory.retrieve_relevant_memory("факт", k=2)
    assert len(block.splitlines()) == 2


def test_empty_fact_is_rejected(memory):
    with pytest.raises(ValueError):
        memory.add_memory("   ")


def test_reset_clears_both_texts_and_index(memory):
    memory.add_memory("щось важливе")
    memory.reset()
    assert len(memory) == 0
    assert memory.index.ntotal == 0
    assert memory.retrieve_relevant_memory("щось важливе") == ""


def test_min_score_filters_out_irrelevant_memories():
    """Головна відмінність від минулорічної версії: поріг релевантності."""
    table = {
        "збережений факт": [1.0, 0.0],
        "схоже питання": [0.9, 0.1],   # cos ≈ 0.994 — проходить
        "зовсім про інше": [0.0, 1.0],  # cos = 0.0 — відсіюється
    }
    manager = main.MemoryManager(ScriptedEmbedder(table), min_score=0.4, verbose=False)
    manager.add_memory("збережений факт")

    assert manager.retrieve_relevant_memory("схоже питання") == "збережений факт"
    assert manager.retrieve_relevant_memory("зовсім про інше") == ""


def test_scores_are_cosine_similarities():
    table = {"a": [1.0, 0.0], "b": [0.0, 1.0]}
    manager = main.MemoryManager(ScriptedEmbedder(table), min_score=-1.0, verbose=False)
    manager.add_memory("a")
    (_, score), = manager.search("a")
    assert score == pytest.approx(1.0, abs=1e-5)
    (_, score), = manager.search("b")
    assert score == pytest.approx(0.0, abs=1e-5)


def test_l2_normalize_handles_zero_vector():
    normalized = main.l2_normalize(np.array([[0.0, 0.0], [3.0, 4.0]], dtype="float32"))
    assert np.allclose(normalized[0], [0.0, 0.0])          # без ділення на нуль
    assert np.allclose(normalized[1], [0.6, 0.8])
    assert normalized.dtype == np.float32                   # faiss приймає лише float32


# ══════════════════════════════════════════════════════════════════
# 3. Інструменти, що працюють з пам'яттю
# ══════════════════════════════════════════════════════════════════
def test_save_fact_to_memory_writes_into_the_store(memory):
    result = main.save_fact_to_memory(
        fact_description="улюблений колір", fact_content="синій"
    )
    assert "saved" in result.lower()
    assert len(memory) == 1
    assert "улюблений колір" in memory.memories[0]
    assert "синій" in memory.memories[0]


def test_tools_fail_loudly_without_attached_memory():
    main.attach_memory(None)
    with pytest.raises(RuntimeError):
        main.save_fact_to_memory(fact_description="x", fact_content="y")


def test_saved_fact_is_findable_in_a_later_turn(memory):
    """Мініатюра всього сценарію ЛР: зберегли на першому ході — знайшли на другому."""
    main.save_fact_to_memory(fact_description="reversed word Кіт", fact_content="тіК")
    context = memory.retrieve_relevant_memory("reversed word Кіт")
    assert "тіК" in context


# ══════════════════════════════════════════════════════════════════
# 4. Складання промпта
# ══════════════════════════════════════════════════════════════════
def test_task_without_context_is_passed_through():
    assert main.build_task_with_memory("завдання", "") == "завдання"


def test_task_with_context_has_both_blocks():
    prompt = main.build_task_with_memory("завдання", "факт із пам'яті")
    assert "## Memory" in prompt
    assert "## Current task" in prompt
    assert prompt.index("факт із пам'яті") < prompt.index("завдання")


def test_instructions_mention_every_tool():
    for name in ("answer_from_context", "save_fact_to_memory", "count_words", "final_answer"):
        assert name in main.INSTRUCTIONS


# ══════════════════════════════════════════════════════════════════
# 5. Безпека шляхів — для варіантів, що пишуть файли (6, 14, 20).
#
#    Ім'я файлу приходить від LLM, тобто це недовірений ввід.
# ══════════════════════════════════════════════════════════════════
def test_safe_path_keeps_a_plain_name(tmp_path):
    path = main.safe_output_path("report.txt", base_dir=tmp_path)
    assert path == (tmp_path / "report.txt").resolve()
    assert path.parent.exists()


@pytest.mark.parametrize(
    "attack",
    [
        "../../.env",
        "..\\..\\Windows\\system32\\evil.dll",
        "/etc/passwd",
        "C:/Windows/system32/evil.dll",
        "subdir/../../../escape.txt",
    ],
)
def test_safe_path_never_escapes_the_base_dir(tmp_path, attack):
    path = main.safe_output_path(attack, base_dir=tmp_path)
    assert path.is_relative_to(tmp_path.resolve())
    assert path.parent == tmp_path.resolve()  # теки теж не створюємо


@pytest.mark.parametrize("bad", ["", "   ", ".", "..", "with\x00null", "/", "../"])
def test_safe_path_rejects_nonsense(tmp_path, bad):
    with pytest.raises(ValueError):
        main.safe_output_path(bad, base_dir=tmp_path)


@pytest.mark.parametrize("bad", ["CON", "nul.txt", "LPT1", "com3.log"])
def test_safe_path_rejects_reserved_windows_names(tmp_path, bad):
    with pytest.raises(ValueError):
        main.safe_output_path(bad, base_dir=tmp_path)


def test_safe_path_is_actually_writable(tmp_path):
    path = main.safe_output_path("../out.txt", base_dir=tmp_path)
    path.write_text("дані", encoding="utf-8")
    assert path.read_text(encoding="utf-8") == "дані"


# ══════════════════════════════════════════════════════════════════
# 6. CLI — розбір аргументів працює без моделі
# ══════════════════════════════════════════════════════════════════
def test_demo_is_the_default_mode():
    args = main.parse_args([])
    assert args.chat is False


def test_chat_flag():
    assert main.parse_args(["--chat"]).chat is True


def test_demo_and_chat_are_mutually_exclusive():
    with pytest.raises(SystemExit):
        main.parse_args(["--demo", "--chat"])


def test_model_override():
    args = main.parse_args(["--model", "qwen3:4b", "--num-ctx", "4096"])
    assert args.model == "qwen3:4b"
    assert args.num_ctx == 4096


def test_demo_scenario_saves_then_recalls():
    """Сценарій має саме таку форму: спочатку створюємо факти, потім питаємо про них."""
    assert len(main.DEMO_STEPS) >= 4
    assert "запам'ятай" in main.DEMO_STEPS[0]
    assert "запам'ятай" not in main.DEMO_STEPS[-1]
