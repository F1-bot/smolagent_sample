import faiss
from sentence_transformers import SentenceTransformer
import numpy as np
from smolagents import tool, CodeAgent, LiteLLMModel
from typing import Union
import os

class MemoryManager:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        print("🧠 Ініціалізація Менеджера Пам'яті...")
        self.encoder = SentenceTransformer(model_name)
        self.dimension = self.encoder.get_sentence_embedding_dimension()
        self.index = faiss.IndexFlatL2(self.dimension)
        self.memories = []
        print("✅ Менеджер Пам'яті готовий.")

    def add_memory(self, text: str):
        print(f"📝 Збереження спогаду: '{text[:70]}...'")
        vector = self.encoder.encode([text])
        self.index.add(vector)
        self.memories.append(text)

    def retrieve_relevant_memory(self, query: str, k: int = 3) -> str:
        if not self.memories:
            return ""
        if self.index.ntotal < k:
            k = self.index.ntotal

        if k == 0: return ""

        print(f"🔍 Пошук {k} релевантних спогадів для запиту: '{query}'")
        query_vector = self.encoder.encode([query])
        distances, indices = self.index.search(query_vector, k)
        relevant_memories = [self.memories[i] for i in indices[0]]
        context = "\n".join(relevant_memories)
        print(f"💡 Знайдено релевантний контекст:\n--- КОНТЕКСТ ---\n{context}\n--- КІНЕЦЬ КОНТЕКСТУ ---")
        return context

memory_manager = MemoryManager()

@tool
def count_words(text: str) -> int:
    """
    Counts the number of words in a given text string.
    Args:
        text: The text string to analyze.
    """
    print(f"Інструмент 'count_words' отримав текст: '{text}'")
    words = text.split()
    return len(words)


@tool
def reverse_text(text: str) -> str:
    """
    Reverses a given text string.
    Args:
        text: The text string to reverse.
    """
    print(f"Інструмент 'reverse_text' отримав текст: '{text}'")
    return text[::-1]


@tool
def save_fact_to_memory(fact_description: str, fact_content: str) -> str:
    """
    Saves an important fact or a result from another tool to your long-term memory.
    Args:
        fact_description: A short description of what the fact is (e.g., 'user's favorite color').
        fact_content: The actual content or value of the fact (e.g., 'blue').
    """
    memory_to_add = f"Fact about '{fact_description}': '{fact_content}'"
    memory_manager.add_memory(memory_to_add)
    return "Fact saved successfully."


@tool
def answer_from_context(answer_text: str) -> str:
    """
    Use this tool when the answer is already known from the provided context or memory.
    Args:
        answer_text: The final, human-readable answer string.
    """
    return answer_text

ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")

model = LiteLLMModel(
    model_id="ollama/llama3:8b",
    api_base=ollama_host
)

instructions_for_llama3 = """
You are an expert Python assistant with a long-term memory. You must respond using an XML format with <thought> and <code> tags.

Follow these rules strictly:
1.  **Think in XML:** Always start by explaining your plan inside `<thought>` and `</thought>` tags.
2.  **Use Memory Critically:** Analyze the provided context. If it DIRECTLY answers the question, use `answer_from_context`. If the context is irrelevant or doesn't help, state that you cannot find the answer.
3.  **Save Facts:** If you calculate a new result, save it using `save_fact_to_memory`.
4.  **Write Code in XML:** Provide your Python script inside `<code>` and `</code>` tags.
5.  **Call `final_answer()`:** The last line of your code MUST be a call to `final_answer()`.

--- EXAMPLES ---

--- Example 1: Calculating and saving ---
Task: "Count the words in 'this is a test' and remember the count."
<thought>
I will use `count_words`, then `save_fact_to_memory`, and finally call `final_answer`.
</thought>
<code>
text_to_process = 'this is a test'
word_count_result = count_words(text_to_process)
save_fact_to_memory(fact_description='word count for "this is a test"', fact_content=str(word_count_result))
final_answer(f"I counted {word_count_result} words and saved this fact to memory.")
</code>

--- Example 2: Answering directly and correctly from memory ---
Context: "Fact about 'Reversed word "Кіт"': 'тіК'\nFact about 'user's favorite color': 'blue'"
Current Task: "What is my favorite color?"
<thought>
The context contains several facts. I need to find the one relevant to 'favorite color'. The fact "Fact about 'user's favorite color': 'blue'" is the correct one. I will use `answer_from_context` to state this fact.
</thought>
<code>
known_fact = answer_from_context(answer_text="Based on my memory, your favorite color is blue.")
final_answer(known_fact)
</code>

--- Example 3: Using only the `count_words` tool ---
Task: "How many words are in the phrase 'the quick brown fox'?"

Your generated code should look EXACTLY like this:
<code>
text_to_process = 'the quick brown fox'
word_count = count_words(text_to_process)
final_answer(f"There are {word_count} words in the phrase.")
</code>

--- Example 4: Using only the reverse_text tool ---
Task: "Reverse the word 'desserts'"
Your generated code should look EXACTLY like this:
<code>
original_text = 'desserts'
reversed_word = reverse_text(original_text)
final_answer(f"The reverse of 'desserts' is '{reversed_word}'.")
</code>

--- Example 5: Using BOTH tools ---
Task: "Analyze 'hello world': count its words and reverse it."
Your generated code should look EXACTLY like this:
<code>
text_to_process = 'hello world'
word_count_result = count_words(text_to_process)
reversed_text_result = reverse_text(text_to_process)
final_answer(f"Analysis complete. Word count: {word_count_result}. Reversed text: '{reversed_text_result}'.")
</code>

Now, solve the user's current task following this XML format.
"""

agent = CodeAgent(
    tools=[count_words, reverse_text, save_fact_to_memory, answer_from_context],
    model=model,
    instructions=instructions_for_llama3,
    code_block_tags=("<code>", "</code>")
)

print("\n--- Інтерактивний режим з Активною RAG-пам'яттю ---")
print("Щоб почати новий чат, введіть '/new'. Щоб вийти, введіть 'exit'.")

while True:
    task = input("\n> Введіть ваше завдання: ")
    if task.lower() in ['exit', 'вихід']: break
    if task.lower() == '/new':
        memory_manager = MemoryManager()
        print("\n🧹 Пам'ять очищено. Починаємо новий чат.")
        continue
    retrieved_context = memory_manager.retrieve_relevant_memory(task)
    contextual_task = task
    if retrieved_context:
        contextual_task = f"Here is relevant context from our past conversation:\n---\n{retrieved_context}\n---\n\nNow, using this context, solve the following task: {task}"
    print(f"\n🚀 Запускаємо агента із завданням...\n")
    try:
        result = agent.run(contextual_task, reset=True)
        print(f"\n✅ Фінальний результат від агента: {result}")
    except Exception as e:
        print(f"\n❌ Під час виконання сталася помилка: {e}")
