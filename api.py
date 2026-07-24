"""
api.py
Module 12 - Production AI Systems
FastAPI wrapper around AgentX, exposing:
  POST /research        -> run full agent pipeline, return final report (JSON)
  POST /research/stream  -> stream the final writer response token-by-token
  GET  /health           -> simple health check

Run with:
    uvicorn api:app --reload --port 8000
"""
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from agent.orchestrator import run_research_agent
from agent.config import client, SMART_MODEL
from agent.guardrails import run_guardrails, GuardrailBlocked

app = FastAPI(title="AgentX - Research Agent API")


class ResearchRequest(BaseModel):
    topic: str
    auto_approve: bool = True  # APIs can't show a terminal approval prompt


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/research")
def research(req: ResearchRequest):
    report = run_research_agent(req.topic, auto_approve=req.auto_approve)
    return {"topic": req.topic, "report": report}


@app.post("/research/stream")
def research_stream(req: ResearchRequest):
    """Lightweight streaming demo: streams a direct answer (not the full
    multi-agent pipeline, which isn't natively streamable step-by-step)."""
    try:
        run_guardrails(req.topic)
    except GuardrailBlocked as e:
        return {"error": str(e)}

    def generate():
        stream = client.chat.completions.create(
            model=SMART_MODEL,
            messages=[{"role": "user", "content": f"Give a quick research summary on: {req.topic}"}],
            stream=True,
        )
        for chunk in stream:
            content = chunk.choices[0].delta.content
            if content:
                yield content

    return StreamingResponse(generate(), media_type="text/plain")
