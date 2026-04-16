# 🤖 Smart Memory Chatbot — RAG + Web Search + Router

An intelligent chatbot that **thinks before answering** — it decides whether to answer from a knowledge base, search the web, or use its own intelligence. Combined with short-term and long-term memory, it gets smarter with every conversation.

---

## 🚀 What It Does

This chatbot has a **router** at its core. Before answering, it asks:

> "Does my knowledge base have the answer? If not, should I search the web or answer from my own knowledge?"

- Factual question in knowledge base → answers from RAG
- Real-time question (news, prices, current events) → searches the web via DuckDuckGo
- General question → answers from LLM directly
- Meanwhile, it remembers you across all conversations

**Example:**
> "What is the capital of France?" → LLM answers directly  
> "What are the latest AI news today?" → Web search  
> "Explain transformers from the paper" → RAG from knowledge base  
> "My name is Ali" → saved to long-term memory, remembered forever

---

## ✨ Features

- 🔀 **Smart Router** — decides between RAG, Web Search, or LLM per query
- 🌐 **Web Search** — real-time answers via DuckDuckGo
- 📚 **Knowledge RAG** — semantic search over your custom knowledge base
- 💬 **Short-Term Memory** — sliding window of last 5 messages, older ones summarized
- 🧠 **Long-Term Memory** — extracts and stores user facts in ChromaDB per user
- 👤 **Per-User Memory** — each user has their own isolated memory store
- 🛡️ **Safe & Robust** — input validation, error handling on web search, fallback logic

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| LLM | Groq (llama-3.1-8b-instant) |
| Embeddings | HuggingFace (all-MiniLM-L6-v2) |
| Vector Store | ChromaDB (x2 — user memory + knowledge) |
| Web Search | DuckDuckGo (via LangChain) |
| Short-Term Memory | Custom Summarizer |
| Long-Term Memory | LLM Extraction + ChromaDB |
| Language | Python 3.11+ |

---

## ⚙️ How It Works

```
User Query
      ↓
┌──────────────────────┐
│  Short-Term Memory   │  → Summarize old messages, keep last 5
└──────────┬───────────┘
           ↓
┌──────────────────────┐
│   Retrieve User      │  → Fetch relevant user memories from ChromaDB
│   Long-Term Memory   │
└──────────┬───────────┘
           ↓
┌──────────────────────┐
│   Knowledge RAG      │  → Search knowledge base for relevant docs
└──────────┬───────────┘
           ↓
┌──────────────────────┐
│      Router          │  → RAG docs found? → use RAG
│                      │    Real-time query? → Web Search
│                      │    Otherwise?       → LLM directly
└──────────┬───────────┘
           ↓
┌──────────────────────┐
│   LLM Generation     │  → Answer using chosen context + memory
└──────────┬───────────┘
           ↓
┌──────────────────────┐
│  Memory Extraction   │  → Extract key facts → save to user ChromaDB
└──────────┬───────────┘
           ↓
        Response
```

---

## 📦 Installation

```bash
# 1. Clone the repository
git clone https://github.com/alirazaaihub/smart-memory-chatbot
cd smart-memory-chatbot

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate       # Mac/Linux
venv\Scripts\activate          # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

---

## 🔑 Environment Setup

Open `chatbot.py` and replace the API key:

```python
GROQ_API_KEY = "your_groq_api_key_here"
```

Get your free Groq API key at: https://console.groq.com

---

## ▶️ Usage

**Optional — Add documents to knowledge base first:**

Place your text/PDF content into ChromaDB under `db/knowledge/` using a separate ingestion script, or add documents programmatically using `knowledge_db.add_documents(...)`.

**Run the chatbot:**

```bash
python chatbot.py
```

**Example conversation:**

```
You: My name is Ali and I study AI
AI: Nice to meet you Ali! I'll remember that.

You: What are the latest developments in AI today?
AI: [searches web] According to recent news...

You: Explain attention mechanism
AI: [retrieves from knowledge base] The attention mechanism...

You: exit
```

---

## 📁 Project Structure

```
smart-memory-chatbot/
│
├── chatbot.py              # Main chatbot with router, memory, RAG, web search
├── requirements.txt        # Project dependencies
├── db/
│   ├── user_memory/        # ChromaDB — per-user long-term memory (auto-created)
│   └── knowledge/          # ChromaDB — knowledge base for RAG (add your docs here)
└── README.md
```

---

## 📋 Dependencies

```
langchain
langchain-groq
langchain-huggingface
langchain-chroma
langchain-core
langchain-community
sentence-transformers
chromadb
duckduckgo-search
```

Install all:

```bash
pip install langchain langchain-groq langchain-huggingface langchain-chroma langchain-core langchain-community sentence-transformers chromadb duckduckgo-search
```

---

## 🧩 Key Concepts Used

| Concept | How It's Used |
|--------|----------------|
| **Smart Router** | LLM decides source: RAG / Web / LLM based on query type |
| **Web Search** | DuckDuckGo fetches real-time info when needed |
| **Knowledge RAG** | ChromaDB searched for relevant context from knowledge base |
| **Short-Term Memory** | Last 5 messages kept; older ones compressed by LLM |
| **Long-Term Memory** | LLM extracts user facts → saved to per-user ChromaDB |
| **Per-User Isolation** | Each user's memory filtered by `user_id` metadata |
| **Input Validation** | Empty queries handled gracefully, web search errors caught |

---

## ⚙️ Configuration

Tweak these in `chatbot.py`:

```python
MAX_SHORT_TERM = 5      # Messages to keep before summarizing
TOP_K = 5               # Number of docs to retrieve from vector stores
```

---

## 🙋 About

Built by **Ali raza** — an 18-year-old self-taught AI developer from Pakistan.  
This is part of my Agentic AI portfolio built using LangChain and Groq.

📌 [LinkedIn](www.linkedin.com/in/alirazaaihub) • [GitHub](https://github.com/alirazaaihub)

---

## 📄 License

MIT License — feel free to use and modify.
