# 🤖 AgentX — Autonomous Research & Report Agent

A full, runnable multi-agent system built on **Groq** (Llama models) that
demonstrates every module from the Agentic AI Roadmap in one working project.

Give it a topic → it plans sub-questions → researches each (with tools) →
critiques its own findings → writes a polished markdown report → asks for
human approval → saves the file — all while logging traces, costs, and metrics.

## 📁 Project Structure
```
agentx/
├── agent/
│   ├── config.py         # Module: Groq client + model routing (Optimization)
│   ├── memory.py          # Module 4  - Memory Systems (working/episodic/semantic)
│   ├── tools.py            # Module 5 & 11 - Tool Engineering + MCP-style server
│   ├── reasoning.py       # Module 1 & 3 - Planning + Chain-of-Thought reasoning
│   ├── agents.py           # Module 6 - Multi-Agent roles (Researcher/Critic/Writer)
│   ├── guardrails.py       # Module 9 - Safety & Guardrails
│   ├── hitl.py              # Module 7 - Human-in-the-Loop approval gate
│   ├── evaluation.py       # Module 8 - Agent Evaluation + benchmarking
│   ├── observability.py    # Module 13 - Tracing, token & cost monitoring
│   ├── optimization.py     # Module 14 - Caching + model routing
│   └── orchestrator.py     # Module 2 & 10 - FSM agent loop tying it all together
├── main.py                  # CLI entry point
├── api.py                    # Module 12 - FastAPI production layer (+streaming)
├── requirements.txt
└── .env.example
```

## 🧩 Module → File Mapping
| Roadmap Module | Where it lives |
|---|---|
| 1. AI Agent Fundamentals (planning) | `reasoning.py` (`make_plan`) |
| 2. Agent Communication & Execution (FSM, retries) | `orchestrator.py` |
| 3. Advanced Reasoning (CoT) | `reasoning.py` (`reason_step`) |
| 4. Agent Memory Systems | `memory.py` |
| 5. Tool Engineering | `tools.py` |
| 6. Multi-Agent Systems | `agents.py` |
| 7. Human-in-the-Loop | `hitl.py` |
| 8. Agent Evaluation | `evaluation.py` |
| 9. Safety & Guardrails | `guardrails.py` |
| 10. Agent Frameworks (hand-rolled StateGraph) | `orchestrator.py` |
| 11. MCP | `tools.py` (`MCPToolServer`) |
| 12. Production AI Systems | `api.py` |
| 13. AI Observability | `observability.py` |
| 14. AI Optimization | `optimization.py` |
| 15. Capstone (this whole project) | Autonomous Research Agent |

## 🚀 Setup
```bash
cd agentx
pip install -r requirements.txt
cp .env.example .env        # then edit .env and add your real GROQ_API_KEY
export GROQ_API_KEY=your_key_here    # or use `export $(cat .env | xargs)`
```
Get a free Groq API key at: https://console.groq.com

## ▶️ Run — CLI
```bash
python main.py "Impact of AI agents on software development"
```
You'll see: planning → research (with tool calls) → critique → final report →
a terminal approval prompt (type `yes`) → saved to `reports/`.

Run the offline benchmark (Module 8):
```bash
python main.py --benchmark
```

## ▶️ Run — Production API (Module 12)
```bash
uvicorn api:app --reload --port 8000
```
Then:
```bash
curl -X POST http://localhost:8000/research \
  -H "Content-Type: application/json" \
  -d '{"topic": "Benefits of MCP for AI agents", "auto_approve": true}'
```
Streaming endpoint:
```bash
curl -N -X POST http://localhost:8000/research/stream \
  -H "Content-Type: application/json" \
  -d '{"topic": "Vector databases explained"}'
```

## 📊 Observability
Every LLM call is logged to `traces.jsonl` (latency, tokens, estimated cost).
A summary prints automatically at the end of each CLI run.

## 🧠 Memory
- `episodic_memory.json` — persists a record of every topic researched.
- `SemanticMemory` (in-memory) — retrieves related past research to give
  the Researcher agent context. Swap `_embed()` in `memory.py` for a real
  embedding model (e.g. `sentence-transformers`) and the store for
  FAISS/Chroma/Qdrant in a real production deployment.

## 🔒 Notes & Production Upgrades
- `web_search` in `tools.py` is a mock KB — plug in a real search API
  (Tavily, Serper, Bing) for genuine web research.
- `MCPToolServer` mimics MCP's client/server tool-exposure pattern using
  plain Python — swap it for the official `mcp` SDK to get real
  STDIO/HTTP/SSE transport and multi-server support.
- `hitl.py`'s `input()` approval prompt would become a Slack message /
  dashboard button in a real product.
