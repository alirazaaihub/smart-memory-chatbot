import os
import logging
import inspect
import asyncio
from typing import List, Annotated, TypedDict, Literal
from pydantic import BaseModel, Field # Structured output ke liye

from logging.handlers import TimedRotatingFileHandler
from langchain_groq import ChatGroq
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, SystemMessage
from langchain_core.documents import Document
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_chroma import Chroma
from langgraph.checkpoint.memory import MemorySaver

# --- 1. Structured Output Schema ---
class RouteDecision(BaseModel):
    """Decide the best tool based on user query."""
    tool: Literal["rag", "web", "llm"] = Field(description="The tool to use for the query.")

# --- 2. Custom Logger ---
class TimeSmartLogger:
    def get_logger(self, logLevel=logging.INFO):
        logger_name = inspect.stack()[1][3]
        logger = logging.getLogger(logger_name)
        logger.setLevel(logLevel)
        if not logger.handlers:
            log_file = "agent.log"
            handler = TimedRotatingFileHandler(log_file, when="midnight", interval=1, backupCount=7)
            handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))
            logger.addHandler(handler)
            logger.addHandler(logging.StreamHandler())
        return logger

ts_logger = TimeSmartLogger()

# --- 3. Configuration ---
GROQ_API_KEY = "here enter your groq api key"
GOOGLE_API_KEY = "here enter your google api"

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0, groq_api_key=GROQ_API_KEY)
# LLM ko structure output ke liye bind karna
structured_llm = llm.with_structured_output(RouteDecision)

embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", google_api_key=GOOGLE_API_KEY)

user_db = Chroma(persist_directory="memory_db", embedding_function=embeddings)
vector_db = Chroma(persist_directory="vectore_db", embedding_function=embeddings)
search = DuckDuckGoSearchAPIWrapper(max_results=5)

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    query: str
    context: str
    decision: str
    user_id: str

# --- 4. Async Helper Functions ---

async def get_user_memory(query, user_id):
    log = ts_logger.get_logger()
    log.info(f"Retrieving memory for User: {user_id}")
    docs = await asyncio.to_thread(user_db.similarity_search, query, k=3, filter={"user_id": user_id})
    return "\n".join([d.page_content for d in docs]) if docs else ""

async def extract_and_save_memory(text, user_id):
    log = ts_logger.get_logger()
    prompt = f"Extract only permanent facts about the user from this text in one-line bullet points. If no facts, return 'None'.\nText: {text}"
    res = await llm.ainvoke([SystemMessage(content=prompt)])
    content = res.content.strip()
    if content.lower() != "none":
        doc = Document(page_content=content, metadata={"user_id": user_id})
        await asyncio.to_thread(user_db.add_documents, [doc])
        log.info(f"Memory saved for {user_id}")

# --- 5. Async Graph Nodes ---

async def router_node(state: AgentState):
    log = ts_logger.get_logger()
    query = state['messages'][-1].content
    
    # Structured call
    decision_obj = await structured_llm.ainvoke([
        SystemMessage(content="Analyze the query and route it. 'rag' for technical/internal/about user resturant  data, 'web' for news/current events, 'llm' for general talk."),
        HumanMessage(content=query)
    ])
    
    decision = decision_obj.tool
    context_data = ""

    if decision == "rag":
        log.info("--- ROUTING TO RAG ---")
        rag_docs = await asyncio.to_thread(vector_db.similarity_search, query, k=3)
        context_data = "\n".join([d.page_content for d in rag_docs])
    elif decision == "web":
        log.info("--- ROUTING TO WEB ---")
    else:
        log.info("--- ROUTING TO LLM ---")

    return {"decision": decision, "query": query, "context": context_data}

async def web_node(state: AgentState):
    log = ts_logger.get_logger()
    results = await asyncio.to_thread(search.run, state['query'])
    return {"context": results}

async def rag_node(state: AgentState):
    # Context pehle hi router ne nikaal liya hai
    return {"context": state['context']}

async def generate_node(state: AgentState):
    query = state['messages'][-1].content
    log = ts_logger.get_logger()
    user_id = state.get("user_id", "default_user")
    
    long_term = await get_user_memory(query, user_id)
    
    system_prompt = f"""You are a helpful assistant.
    - Use long term memory silently.
    User long-term-memory: {long_term}
    Context: {state.get('context', 'None')}"""
    
    response = await llm.ainvoke([SystemMessage(content=system_prompt)] + state['messages'])
    asyncio.create_task(extract_and_save_memory(state['messages'][-1].content, user_id))
    
    return {"messages": [response]}

async def summarize_node(state: AgentState):
    msgs = state['messages']
    if len(msgs) > 10:
        summary = await llm.ainvoke([SystemMessage(content="Summarize chat briefly.")] + msgs[:-5])
        return {"messages": [SystemMessage(content=f"Summary: {summary.content}")] + msgs[-5:]}
    return {"messages": msgs}

# --- 6. Graph Construction ---

def build_graph():

    builder = StateGraph(AgentState)
    builder.add_node("router", router_node)
    builder.add_node("web_search", web_node)
    builder.add_node("rag_retrieve", rag_node)
    builder.add_node("generate", generate_node)
    builder.add_node("summarize", summarize_node)

    builder.add_edge(START, "router")

# Conditional edges based on structured decision
    builder.add_conditional_edges(
        "router", 
        lambda x: x["decision"], 
        {
            "web": "web_search",
            "rag": "rag_retrieve",
            "llm": "generate"
        }
    )

    builder.add_edge("web_search", "generate")
    builder.add_edge("rag_retrieve", "generate")
    builder.add_edge("generate", "summarize")
    builder.add_edge("summarize", END)

    memory = MemorySaver()
    return builder.compile(checkpointer=memory)

graph = build_graph()
