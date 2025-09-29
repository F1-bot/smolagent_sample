# Autonomous AI Agent with Long-Term Memory

**[UA]** Цей репозиторій містить код для лабораторної роботи з розробки інтерактивного AI-асистента з довготривалою пам'яттю на базі архітектури RAG (Retrieval-Augmented Generation). Агент використовує `smol-agents`, векторну базу даних `faiss` та локальну LLM через `Ollama`.

**[EN]** This repository contains the code for a lab project on developing an interactive AI assistant with long-term memory based on the RAG (Retrieval-Augmented Generation) architecture. The agent utilizes `smol-agents`, the `faiss` vector database, and a local LLM via `Ollama`.

## 🚀 Key Features

**Long-Term Memory**

Implements RAG to overcome the limited context window of LLMs, enabling the agent to recall information from previous conversations.
**Dynamic Tool Use**

The agent can dynamically choose from a set of predefined tools to perform tasks like text analysis, calculations, or data manipulation.
**Active Memory Management**

The agent can autonomously decide which facts are important to save for future reference.
**Prompt Engineering**

Behavior is controlled through a detailed system prompt using few-shot examples and XML-like tags for robust interaction.
**Local LLM Integration**

Runs with local language models via Ollama, ensuring privacy and offline capabilities.

## 🛠️ Technical Architecture

The project is built on the following core components:

- **`smol-agents`** – A lightweight library for creating and managing AI agents.
- **`Ollama`** – For running large language models (like Llama 3) locally.
- **`SentenceTransformers`** – Used to generate semantic embeddings for text.
- **`faiss`** – A library from Facebook AI for efficient similarity search on dense vectors, used here as the vector database.
- **`Python`** – The core programming language.

## ⚙️ Setup & Installation

Follow these steps to set up and run the project.

### 1. Install Ollama
Visit the [official Ollama website](https://ollama.com/) and download the installer for your operating system. Follow the installation instructions. Ollama will run as a background service.

### 2. Download a Language Model
Open your terminal and run the following command to download the Llama 3 8B model.

```bash
ollama pull llama3:8b
```

*Note: You can use any other model available on Ollama.*

### 3. Set Up Python Environment and Install Dependencies
It is highly recommended to use a virtual environment.

```bash
# Create a virtual environment
python -m venv .venv

# Activate the environment
# Windows
# .venv\Scripts\activate
# macOS/Linux
# source .venv/bin/activate

# Install the required libraries
pip install "smolagents[litellm]" sentence-transformers faiss-cpu numpy
```

## ▶️ Usage Guide

1. Save the provided code as `main.py`.
2. Ensure your virtual environment is activated.
3. Run the script from your terminal:

```bash
python main.py
```
On the first run, `sentence-transformers` will download the `all-MiniLM-L6-v2` model. After initialization, you will see a prompt to enter your task.

- Type your task and press Enter.
- To start a new chat and clear the agent's memory, type `/new`.
- To exit the program, type `exit`.

### Example Interaction:
```
--- Інтерактивний режим з Активною RAG-пам'яттю ---
Щоб почати новий чат, введіть '/new'. Щоб вийти, введіть 'exit'.

> Введіть ваше завдання: Count the words in 'the quick brown fox' and remember it.

... (Agent processing) ...

✅ Фінальний результат від агента: I counted 4 words and saved this fact to memory.

> Введіть ваше завдання: what was the sentence I asked you to analyze?

... (Agent processing with RAG) ...

✅ Фінальний результат від агента: Based on my memory, the sentence you asked me to analyze was 'the quick brown fox'.
```

## 📚 Individual Assignment Variants

This project can be extended with various specialized tools. Below are the different agent configurations that can be built, each with a unique set of tools and a key requirement for its behavior.

<details>
<summary>Click to expand the list of assignment variants</summary>

| Варіант | Найменування | Інструменти | Ключова вимога |
|:---:|---|---|---|
| 1 | Текстовий Аналітик | `count_sentences(text)`, `find_longest_word(text)` | Агент повинен завжди надавати відповідь у форматі Markdown-таблиці. |
| 2 | Математичний Помічник | `calculate_square_root(number)`, `calculate_factorial(number)` | Агент повинен завжди перевіряти вхідні дані і повідомляти про помилку. |
| 3 | Генератор Паролів | `generate_password(length)`, `check_password_strength(password)` | Після генерації пароля агент повинен автоматично перевіряти його надійність. |
| 4 | Перекладач Морзе | `text_to_morse(text)`, `morse_to_text(morse_code)` | Агент повинен чітко розрізняти, який тип конвертації потрібен. |
| 5 | Конвертер Валют | `convert_usd_to_uah(amount)`, `convert_uah_to_usd(amount)` | Агент повинен завжди вказувати курс, за яким була проведена конвертація. |
| 6 | Файловий Інспектор | `create_file_with_content(filename, content)`, `get_file_size(filename)` | Агент повинен вміти обробляти помилки (наприклад, файл не знайдено). |
| 7 | Текстовий Шифратор | `encrypt_caesar(text, shift)`, `decrypt_caesar(text, shift)` | Агент повинен зберігати в пам’ять ключ (зсув) для дешифрування. |
| 8 | Погодний Інформатор | `get_temperature(city)`, `get_weather_condition(city)` | Агент повинен завжди об’єднувати інформацію в одне зв’язне речення. |
| 9 | Аналізатор URL | `extract_domain(url)`, `check_if_https(url)` | Агент повинен надавати відповідь у вигляді JSON-рядка. |
| 10 | Генератор Імен | `generate_random_name(gender)`, `get_name_initials(full_name)` | Агент повинен генерувати ім’я та одразу ж повертати його ініціали. |
| 11 | Калькулятор Знижок | `calculate_discount_price(...)`, `calculate_tax(...)` | Агент повинен вміти послідовно застосовувати знижку, а потім податок. |
| 12 | Лічильник Калорій | `get_calories_per_100g(product)`, `calculate_total_calories(...)` | Агент повинен питати у користувача вагу продукту, якщо вона не вказана. |
| 13 | Конвертер Одиниць | `convert_km_to_miles(km)`, `convert_miles_to_km(miles)` | Агент повинен завжди вказувати коефіцієнт конвертації у відповіді. |
| 14 | Генератор Кольорів | `generate_random_hex_color()`, `hex_to_rgb(hex_code)` | Агент повинен зберігати в пам’ять згенерований HEX-код для конвертації. |
| 15 | Лічильник Символів | `count_characters(...)`, `count_vowels(text)` | Агент повинен надавати звіт за кількома критеріями одночасно. |
| 16 | Таймер | `set_timer(seconds)`, `check_timer_status()` | Агент повинен використовувати `time.sleep()` і повідомляти про завершення. |
| 17 | Генератор Цитат | `get_random_quote_by_category(category)`, `get_quote_author(quote)` | Агент повинен завжди повертати цитату разом з її автором. |
| 18 | Географічний Довідник | `get_country_capital(country)`, `get_country_population(country)` | Агент повинен вміти порівнювати населення двох країн. |
| 19 | Текстовий Нормалізатор | `text_to_lowercase(text)`, `remove_punctuation(text)` | Агент повинен застосовувати обидва інструменти послідовно. |
| 20 | Генератор QR-кодів | `generate_qr_code(data, filename)`, `read_text_from_file(filename)` | Агент повинен згенерувати QR-код, зберегти і зчитати його для перевірки. |

</details>
