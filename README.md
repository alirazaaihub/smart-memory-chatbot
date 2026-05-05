# 🤖 Smart Routing Agent

An agentic AI system that intelligently routes user queries to the most appropriate knowledge source — **RAG pipeline, web search, or direct LLM generation** — using structured output-based decision making, dual-layer memory, and a fully async LangGraph execution engine.

---

## 🚀 What Makes This Different

Basic chatbots just pass every query to an LLM. This system:

- Classifies every query using structured LLM output — no string parsing, no hallucinated routing  
- Retrieves from a persistent vector knowledge base for domain-specific questions  
- Uses live web search for real-time and current-event queries  
- Automatically extracts and stores permanent user facts in long-term memory  
- Maintains thread-isolated sessions (no cross-user contamination)  
- Compresses long conversations automatically to stay within context limits  

**Example:**
> User: "Latest trends in AI?" → Web search  
> User: "Explain my uploaded docs" → RAG  
> User: "Explain recursion" → Direct LLM  

---

## ✨ Features

- 🔀 **Structured Output Routing** — strict schema-based decision (`rag | web | llm`)  
- 🧠 **Dual-Layer Memory** — short-term (MemorySaver) + long-term (ChromaDB)  
- ⚡ **Fully Async Execution** — async nodes + `asyncio.to_thread()` for blocking calls  
- 🔒 **Thread-Aware Sessions** — each `user_id` has isolated state  
- 💬 **Auto Summarization** — compresses conversations beyond 10 messages  
- 🌐 **REST API** — clean FastAPI backend for integration  

---

## 🛠️ Tech Stack

| Component | Technology |
|----------|------------|
| Agent Framework | LangGraph |
| LLM | LLaMA 3.3 70B (Groq) |
| Embeddings | Gemini Embedding 001 |
| Vector Store | ChromaDB |
| Web Search | DuckDuckGo |
| Backend API | FastAPI + Uvicorn |
| Memory | MemorySaver + ChromaDB |

---

## ⚙️ System Architecture

### Routing & Generation Flow

```
User Query
    ↓
[ Router Node (LLM) ]
    ↓
 ┌───────────────┬───────────────┬───────────────┐
 │               │               │               │
rag            web             llm
 │               │               │
 ↓               ↓               ↓
[RAG]        [Web Search]    [LLM Direct]
 │               │               │
 └───────────────┴───────────────┘
                ↓
        [Generate Node]
                ↓
        [Summarize Node]
                ↓
             Response
```

---

## 🧠 Memory Architecture

| Layer | Backend | Scope | Lifespan |
|------|--------|------|----------|
| Short-term | MemorySaver | Per thread | Runtime |
| Long-term | ChromaDB | Per user | Persistent |

> Long-term memory extraction runs asynchronously using `asyncio.create_task()`

---

## 📦 Installation

```bash
# 1. Clone the repository
git clone https://github.com/alirazaaihub/smart-memory-chatbot.git
cd smart-routing-agent

# 2. Create virtual environment
python -m venv venv

# Linux / Mac
source venv/bin/activate

# Windows
venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## 🔑 Environment Setup

Create a `.env` file:

```env
GROQ_API_KEY=your_groq_api_key_here
GOOGLE_API_KEY=your_google_api_key_here
```

---

## ▶️ Usage

### Start Server

```bash
uvicorn main:app --reload
```

### API Endpoint

`POST /chat`

### Example Request

```json
{
  "user_id": "ali_123",
  "query": "What are the latest trends in agentic AI?"
}
```

### Example Response

```json
{
  "user_id": "ali_123",
  "response": "Based on recent developments..."
}
```

---

## 📁 Project Structure

```
smart-routing-agent/
│
├── agent.py
├── main.py
├── requirements.txt
├── .env.example
├── .gitignore
├── memory_db/
├── vectore_db/
├── agent.log
└── README.md
```

---

## 📋 Dependencies

```
langchain
langchain-groq
langchain-google-genai
langchain-community
langchain-chroma
langgraph
fastapi
uvicorn
pydantic
chromadb
```

Install all:

```bash
pip install langchain langchain-groq langchain-google-genai langchain-community langchain-chroma langgraph fastapi uvicorn pydantic chromadb
```

---

## 🧩 Key Concepts Used

| Concept | How It's Used |
|--------|----------------|
| Structured Routing | LLM outputs strict schema → no hallucinated decisions |
| RAG Retrieval | ChromaDB used for domain-specific knowledge |
| Web Search Routing | Real-time queries handled via external search |
| Async Execution | All nodes async for scalability |
| Short-Term Memory | Maintains conversation context per thread |
| Long-Term Memory | Stores user facts persistently |
| Auto Summarization | Prevents context overflow |

---
📌 [LinkedIn](www.linkedin.com/in/alirazaaihub)

## 🙋 About

Built by **Ali Raza** — AI/ML Engineering Student, Pakistan.  
Focused on Agentic AI, RAG systems, and production LLM applications.

---

## 📄 License

MIT License — free to use and modify.
