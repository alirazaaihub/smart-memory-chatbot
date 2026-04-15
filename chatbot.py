from typing import TypedDict, Annotated, List
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage
from langgraph.graph.message import add_messages

from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

from langchain_community.utilities import DuckDuckGoSearchAPIWrapper

import uuid
import os


# CONFIG

MAX_SHORT_TERM = 5
TOP_K = 5

USER_DB_DIR = "db/user_memory"
KNOWLEDGE_DB_DIR = "db/knowledge"

#  SAFE API KEY
GROQ_API_KEY = "Enter your api key"


# LLM

llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model_name="llama-3.1-8b-instant",
    temperature=0
)


# SEARCH

search = DuckDuckGoSearchAPIWrapper(max_results=5)


# EMBEDDINGS

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# VECTOR STORES

user_db = Chroma(
    persist_directory=USER_DB_DIR,
    embedding_function=embeddings
)

knowledge_db = Chroma(
    persist_directory=KNOWLEDGE_DB_DIR,
    embedding_function=embeddings
)


# STATE

class State(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]



#   SHORT TERM MEMORY

def summarize(state: State):
    msgs = state["messages"]

    if len(msgs) <= MAX_SHORT_TERM:
        return state

    old = msgs[:-MAX_SHORT_TERM]
    recent = msgs[-MAX_SHORT_TERM:]

    summary = llm.invoke(
        [SystemMessage(content="Summarize briefly")] + old
    )

    state["messages"] = [
        SystemMessage(content=f"Summary:\n{summary.content}")
    ] + recent

    return state



#   USER MEMORY

def save_user_memory(user_id, text):
    doc = Document(
        page_content=text,
        metadata={"user_id": user_id, "id": str(uuid.uuid4())}
    )
    user_db.add_documents([doc])


def get_user_memory(user_id, query):
    docs = user_db.similarity_search(
        query, k=TOP_K, filter={"user_id": user_id}
    )
    return "\n".join([d.page_content for d in docs]) if docs else ""



#   KNOWLEDGE RAG

def get_knowledge(query):
    return knowledge_db.similarity_search(query, k=TOP_K)



#   WEB SEARCH (SAFE)

def search_web(query):
    if not query.strip():
        return ""

    try:
        result = search.run(query)
        return result if result else ""
    except Exception:
        return ""



#   ROUTER (SAFE)

def decide_source(query, rag_docs):
    if not query.strip():
        return "llm"

    # simple deterministic rule first
    if rag_docs:
        return "rag"

    prompt = f"""
Decide best source:

Query: {query}

Options:
- web
- llm

Rules:
- If real-time info → web
- Else → llm

Answer ONLY one word.
"""
    decision = llm.invoke([SystemMessage(content=prompt)]).content.strip().lower()

    if decision not in ["web", "llm"]:
        return "llm"

    return decision


#   MEMORY EXTRACTION (IMPROVED)

def extract_memory(text):
    if not text.strip():
        return None

    prompt = f"""
Extract useful long-term memory.

Rules:
- Only stable info
- Ignore temporary info
- One line per fact

Text:
{text}
"""

    res = llm.invoke([SystemMessage(content=prompt)]).content.strip()

    if not res or res.lower() == "none":
        return None

    return res



#   MAIN CHATBOT

def chatbot(user_id, state, query):

    #  Input validation
    if not query.strip():
        return state, " Please enter a valid query."

    # 1 Add user msg
    state["messages"].append(HumanMessage(content=query))

    # 2 STM
    state = summarize(state)

    # 3 Memory
    user_memory = get_user_memory(user_id, query)

    # 4 RAG
    rag_docs = get_knowledge(query)

    # 5 Router
    source = decide_source(query, rag_docs)

    # 6 Context
    context = ""

    if source == "rag" and rag_docs:
        context = "\n".join([d.page_content for d in rag_docs])

    elif source == "web":
        context = search_web(query)

    # 7 Final Answer
    final_prompt = f"""
You are a smart AI assistant.

User Memory:
{user_memory}

Context:
{context}

Conversation:
{" ".join([m.content for m in state["messages"]])}

Question:
{query}

Answer naturally.
"""

    response = llm.invoke([SystemMessage(content=final_prompt)])
    answer = response.content.strip()

    # 8 Save STM
    state["messages"].append(AIMessage(content=answer))

    # 9 Save LTM
    memory = extract_memory(query)
    if memory:
        save_user_memory(user_id, memory)

    return state, answer



# RUN

if __name__ == "__main__":
    state = {"messages": []}
    user_id = "user_1"

    while True:
        q = input("You: ").strip()

        if q.lower() == "exit":
            break

        state, ans = chatbot(user_id, state, q)

        print("AI:", ans)
