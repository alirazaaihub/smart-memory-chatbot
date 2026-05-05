Smart Routing Agent
An agentic AI system that intelligently routes user queries to the most appropriate knowledge source — RAG pipeline, web search, or direct LLM generation — using structured output-based decision making, dual-layer memory, and a fully async LangGraph execution engine.

Architecture Overview
User Query
    │
    ▼
┌─────────────┐     structured output      ┌──────────────┐
│  Router Node │ ─── RouteDecision ──────► │  RAG Retrieve │ ─► Chroma Vector DB
│  (LLaMA 70B) │                           └──────────────┘
│              │                           ┌──────────────┐
│              │ ──────────────────────► │  Web Search   │ ─► DuckDuckGo API
│              │                           └──────────────┘
│              │                           ┌──────────────┐
│              │ ──────────────────────► │  LLM Direct   │
└─────────────┘                           └──────────────┘
                                                  │
                                                  ▼
                                         ┌─────────────────┐
                                         │  Generate Node   │◄── Long-Term Memory
                                         │  (+ User Memory) │    (Chroma user_db)
                                         └─────────────────┘
                                                  │
                                                  ▼
                                         ┌─────────────────┐
                                         │  Summarize Node  │ (auto-compresses >10 msgs)
                                         └─────────────────┘
                                                  │
                                                  ▼
                                              Response

Key Design Decisions
1. Structured Output Routing (not prompt parsing)
The router uses llm.with_structured_output(RouteDecision) to enforce a strict Literal["rag", "web", "llm"] response, eliminating string parsing errors and hallucinated routing decisions.
2. Dual-Layer Memory
LayerBackendScopeLifespanShort-termMemorySaver (LangGraph)Per thread/sessionProcess lifetimeLong-termChroma (user_db)Per user_idPersistent across sessions
Long-term memory is extracted asynchronously after each turn — it does not block response generation.
3. Fully Async Execution
All graph nodes are async functions. Synchronous operations (Chroma similarity search) are offloaded via asyncio.to_thread() to prevent event loop blocking.
4. Thread-Aware Sessions
Each user gets an isolated conversation thread via LangGraph's thread_id config, preventing cross-user state contamination.

Tech Stack
ComponentTechnologyLLMLLaMA 3.3 70B via GroqEmbeddingsGemini Embedding 001 (Google)Vector StoreChroma (persistent)Agent FrameworkLangGraph (StateGraph)Web SearchDuckDuckGo API WrapperAPI LayerFastAPI + UvicornMemoryLangGraph MemorySaver + Chroma

Project Structure
smart-memory-chtbot/
├── agent.py          # LangGraph graph definition, nodes, memory logic
├── main.py           # FastAPI server, /chat endpoint
├── requirements.txt  # Pinned dependencies
├── .env.example      # Environment variable template
├── .gitignore        # Excludes .env, chroma dirs, __pycache__
└── README.md

Setup & Installation
1. Clone the repository
bashgit clone https://github.com/alirazaaihub/smart-memory-chatbot.git
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
5. (Optional) Populate your vector database
Add your domain-specific documents to the vectore_db Chroma collection before running. The user_db collection is created automatically on first use.
6. Run the server
bashuvicorn main:app --reload
API will be available at http://localhost:8000

API Reference
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
Response:
json{
  "status": "running",
  "memory": "thread-aware"
}

Routing Logic
The router classifies every incoming query into one of three paths:
RouteTrigger ConditionExample QueriesragTechnical/internal/domain-specific knowledge"What's on our menu?", "How does X work in our system?"webCurrent events, news, real-time information"Latest AI news", "Today's weather"llmGeneral knowledge, conversation, reasoning"Explain recursion", "Write me a poem"

Environment Variables
VariableDescriptionRequiredGROQ_API_KEYGroq API key for LLaMA inferenceYesGOOGLE_API_KEYGoogle API key for Gemini embeddingsYes

.env.example
envGROQ_API_KEY=your_groq_api_key_here
GOOGLE_API_KEY=your_google_api_key_here

.gitignore
gitignore.env
__pycache__/
*.pyc
*.pyo
memory_db/
vectore_db/
*.log
venv/
.venv/

requirements.txt (recommended pins)
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

Known Limitations

MemorySaver is in-memory only — short-term conversation history resets on server restart. For production persistence, replace with PostgresSaver or RedisSaver.
DuckDuckGo search wrapper is rate-limited and not suitable for high-traffic production use. Consider replacing with Tavily or Serper API.
No authentication or rate limiting on the FastAPI layer — add middleware before any public deployment.


Future Improvements

 Replace MemorySaver with PostgresSaver for persistent short-term memory
 Add JWT-based authentication to the FastAPI layer
 Integrate Tavily API for reliable production web search
 Add a DELETE /memory/{user_id} endpoint to clear user long-term memory
 Containerize with Docker


Author
Built by Ali — AI/ML Engineering Student, Punjab, Pakistan.
Part of a self-directed agentic AI learning curriculum covering LangChain, LangGraph, RAG pipelines, fine-tuning, and MCP server development.
