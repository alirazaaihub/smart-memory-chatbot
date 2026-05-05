from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

from agent import graph
from langchain_core.messages import HumanMessage

app = FastAPI(title="Agentic AI API")

class ChatRequest(BaseModel):
    user_id: str
    query: str

class ChatResponse(BaseModel):
    user_id: str
    response: str

@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    try:
        #  Thread-based memory config (CRITICAL)
        config = {
            "configurable": {
                "thread_id": req.user_id
            }
        }

        result = await graph.ainvoke(
            {
                "messages": [HumanMessage(content=req.query)],
                "user_id": req.user_id
            },
            config=config
        )

        response = result["messages"][-1].content

        return ChatResponse(
            user_id=req.user_id,
            response=response
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
def health():
    return {"status": "running", "memory": "thread-aware"}

if __name__ == "__main__":
    uvicorn.run("main:app", reload=True)
