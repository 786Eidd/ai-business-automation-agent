"""
FastAPI service exposing the full agent workflow.

Run (two terminals):
    uvicorn src.external_service:app --port 8002       # the mock third-party system
    uvicorn src.app:app --reload --port 8001            # the agent itself

Then:
    curl -X POST http://localhost:8001/automate \
         -H "Content-Type: application/json" \
         -d '{"request_text": "Can I get a refund for ORD-1001? It arrived damaged."}'
"""
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional

from .graph import build_graph

app = FastAPI(
    title="AI Business Automation Agent",
    description=(
        "Takes a free-text business request, reasons about it, retrieves the "
        "relevant order/knowledge context, calls the right tool/API, takes the "
        "business action, and either resolves it directly or hands it to a "
        "human — orchestrated as an actual LangGraph state graph."
    ),
)

_graph = None


@app.on_event("startup")
def load_graph():
    global _graph
    _graph = build_graph()


class AutomateRequest(BaseModel):
    request_text: str


class AutomateResponse(BaseModel):
    request_text: str
    intent: str
    confidence: float
    order_id: Optional[str] = None
    action: str
    status: str  # "resolved" | "pending_human_review"
    reply: str
    reply_path: str
    detail: Optional[str] = None


@app.post("/automate", response_model=AutomateResponse)
def automate(req: AutomateRequest):
    result = _graph.invoke({"request_text": req.request_text})
    return AutomateResponse(
        request_text=req.request_text,
        intent=result["intent"],
        confidence=result["confidence"],
        order_id=result.get("order_id"),
        action=result["action"],
        status="pending_human_review" if result["human_review"] else "resolved",
        reply=result["reply"],
        reply_path=result["reply_path"],
        detail=result.get("detail"),
    )


@app.get("/health")
def health():
    return {"status": "ok", "graph_loaded": _graph is not None}
