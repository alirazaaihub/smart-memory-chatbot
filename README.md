🤖 Smart Routing Agent
An agentic AI system that intelligently routes user queries to the most appropriate knowledge source — RAG pipeline, web search, or direct LLM generation — using structured output-based decision making, dual-layer memory, and a fully async LangGraph execution engine.

🚀 What Makes This Different
Basic chatbots just pass every query to an LLM. This system:

Classifies every query using structured LLM output — no string parsing, no hallucinated routing
Retrieves from a persistent vector knowledge base for domain-specific questions
Hits live web search for real-time and current-event queries
Automatically extracts and stores permanent user facts in a separate long-term memory layer
Maintains thread-isolated conversation history — no cross-user state contamination
Compresses long conversations automatically to stay within context limits


✨ Features

🔀 Structured Output Routing — RouteDecision schema enforces strict rag / web / llm classification
🧠 Dual-Layer Memory — Short-term via LangGraph MemorySaver + Long-term via persistent ChromaDB
⚡ Fully Async Execution — All nodes are async; sync DB calls offloaded via asyncio.to_thread()
🔒 Thread-Aware Sessions — Each user_id gets an isolated conversation thread
💬 Auto Summarization — Conversation history beyond 10 messages is auto-compressed
🌐 REST API — Clean FastAPI interface for easy frontend or service integration


🛠️ Tech Stack
ComponentTechnologyLLMLLaMA 3.3 70B via GroqEmbeddingsGemini Embedding 001 (Google)Vector StoreChromaDB (persistent)Agent FrameworkLangGraph (StateGraph)Web SearchDuckDuckGo API WrapperAPI LayerFastAPI + UvicornMemoryLangGraph MemorySaver + ChromaDB

⚙️ Architecture
Routing & Generation Flow
User Query
    │
    ▼
┌─────────────┐    RouteDecision (structured output)
│ Router Node │ ──────────────────────────────────────┐
│ (LLaMA 70B) │                                       │
└─────────────┘                                       │
       │                                              │
       ├──── rag ──► ┌──────────────┐                 │
       │             │ RAG Retrieve │ ─► Chroma        │
       │             └──────┬───────┘   Vector DB      │
       │                    │                          │
       ├──── web ──► ┌──────────────┐                  │
       │             │  Web Search  │ ─► DuckDuckGo    │
       │             └──────┬───────┘                  │
       │                    │                          │
       └──── llm ──► ┌──────────────┐                  │
                     │  LLM Direct  │ ◄────────────────┘
                     └──────┬───────┘
                            │
                            ▼
                   ┌─────────────────┐
                   │  Generate Node  │ ◄── Long-Term Memory
                   │  (+ User Facts) │     (Chroma user_db)
                   └────────┬────────┘
                            │
                            ▼
                   ┌─────────────────┐
                   │ Summarize Node  │  (auto-compresses >10 msgs)
                   └────────┬────────┘
                            │
                            ▼
                         Response
Memory Architecture
LayerBackendScopeLifespanShort-termMemorySaver (LangGraph)Per thread_id / sessionProcess lifetimeLong-termChromaDB (user_db)Per user_idPersistent across sessions

Long-term memory is extracted asynchronously after each turn via asyncio.create_task() — it does not block response generation.


📁 Project Structure
smart-routing-agent/
├── agent.py            # LangGraph graph definition, nodes, memory logic
├── main.py             # FastAPI server, /chat endpoint
├── requirements.txt    # Pinned dependencies
├── .env.example        # Environment variable template
├── .gitignore          # Excludes .env, chroma dirs, __pycache__
├── memory_db/          # ChromaDB — user long-term memory (auto-created)
├── vectore_db/         # ChromaDB — RAG knowledge base (manually populated)
├── agent.log           # Rotating log file (daily, 7-day retention)
└── README.md

📦 Installation
1. Clone the repository
bashgit clone https://github.com/alirazaaihub/smart.git-memory-chatbot.git
cd smart-routing-agent
2. Create and activate a virtual environment
bashpython -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows
3. Install dependencies
bashpip install -r requirements.txt
4. Configure environment variables
bashcp .env.example .env
Edit .env and fill in your keys:
envGROQ_API_KEY=your_groq_api_key_here
GOOGLE_API_KEY=your_google_api_key_here
5. (Optional) Populate the RAG knowledge base
Add your domain-specific documents to the vectore_db Chroma collection before running. The user_db collection is created automatically on first use.
6. Run the server
bashuvicorn main:app --reload
API will be available at http://localhost:8000

📡 API Reference
POST /chat
Send a query to the agent.
Request Body:
json{
  "user_id": "ali_123",
  "query": "What are the latest trends in agentic AI?"
}
Response:
json{
  "user_id": "ali_123",
  "response": "Based on recent developments..."
}
GET /
Health check endpoint.
json{
  "status": "running",
  "memory": "thread-aware"
}

🔀 Routing Logic
The router classifies every incoming query into one of three paths:
RouteTrigger ConditionExample QueriesragTechnical / internal / domain-specific knowledge"What's on our menu?", "How does X work in our system?"webCurrent events, news, real-time information"Latest AI news", "Today's weather"llmGeneral knowledge, conversation, reasoning"Explain recursion", "Write me a poem"

🔑 Environment Variables
VariableDescriptionRequiredGROQ_API_KEYGroq API key for LLaMA inference✅ YesGOOGLE_API_KEYGoogle API key for Gemini embeddings✅ Yes

📋 .env.example
envGROQ_API_KEY=your_groq_api_key_here
GOOGLE_API_KEY=your_google_api_key_here

🚫 .gitignore
gitignore.env
__pycache__/
*.pyc
*.pyo
memory_db/
vectore_db/
*.log
venv/
.venv/

📦 requirements.txt
langchain>=0.3.0
langchain-groq>=0.2.0
langchain-google-genai>=2.0.0
langchain-community>=0.3.0
langchain-chroma>=0.1.4
langgraph>=0.2.0
fastapi>=0.115.0
uvicorn>=0.30.0
pydantic>=2.0.0
chromadb>=0.5.0

⚠️ Known Limitations

MemorySaver is in-memory only — short-term conversation history resets on server restart. For production, replace with PostgresSaver or RedisSaver.
DuckDuckGo is rate-limited — not suitable for high-traffic production use. Consider replacing with Tavily or Serper API.
No authentication on the API layer — add JWT middleware before any public deployment.


🔮 Future Improvements

 Replace MemorySaver with PostgresSaver for persistent short-term memory
 Add JWT-based authentication to the FastAPI layer
 Integrate Tavily API for reliable production web search
 Add DELETE /memory/{user_id} endpoint to clear user long-term memory
 Containerize with Docker


🙋 About
Built by Ali Raza — AI/ML Engineering Student, Punjab, Pakistan.
Part of a self-directed agentic AI learning curriculum covering LangChain, LangGraph, RAG pipelines, fine-tuning, and MCP server development.
📌 LinkedIn • GitHub

📄 License
MIT License — free to use and modify.
