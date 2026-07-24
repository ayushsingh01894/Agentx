"""
AgentX Streamlit App — single-file Autonomous Research & Report Agent UI
Run with:  streamlit run streamlit_app.py

Set your Groq API key either as an environment variable GROQ_API_KEY,
or paste it directly into the sidebar field when the app opens.
"""

import os
import re
import json
import time
import hashlib
import functools
from datetime import datetime

import streamlit as st
from groq import Groq

# =============================================================================
# PAGE CONFIG + STYLING
# =============================================================================
st.set_page_config(
    page_title="AgentX — Research Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main .block-container {padding-top: 2rem; max-width: 1100px;}
    h1 {font-weight: 800 !important;}
    .agentx-hero {
        background: linear-gradient(135deg, #6C5CE7 0%, #00B4D8 100%);
        padding: 1.6rem 2rem; border-radius: 16px; color: white; margin-bottom: 1.5rem;
    }
    .agentx-hero h1 {color: white; margin: 0; font-size: 1.9rem;}
    .agentx-hero p {color: #EAEAFF; margin: 0.3rem 0 0 0; font-size: 0.95rem;}
    .stage-card {
        border: 1px solid rgba(150,150,150,0.25); border-radius: 12px;
        padding: 1rem 1.2rem; margin-bottom: 0.8rem; background: rgba(120,120,255,0.03);
    }
    .metric-pill {
        display: inline-block; padding: 0.25rem 0.7rem; border-radius: 999px;
        background: rgba(0,180,216,0.12); color: #00B4D8; font-size: 0.8rem;
        font-weight: 600; margin-right: 0.4rem;
    }
    .status-blocked {color: #E63946; font-weight: 700;}
    .status-ok {color: #2A9D8F; font-weight: 700;}
    code {font-size: 0.85rem;}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="agentx-hero">
  <h1>🤖 AgentX — Autonomous Research &amp; Report Agent</h1>
  <p>Planning · Multi-Agent Reasoning · Tools · Memory · Guardrails · HITL · Observability — all in one run.</p>
</div>
""", unsafe_allow_html=True)

# =============================================================================
# SESSION STATE (in-memory: cache, episodic memory, semantic memory, traces)
# =============================================================================
if "episodic_memory" not in st.session_state:
    st.session_state.episodic_memory = []       # list of {topic, report}
if "semantic_memory" not in st.session_state:
    st.session_state.semantic_memory = []       # list of (text, embedding)
if "response_cache" not in st.session_state:
    st.session_state.response_cache = {}
if "traces" not in st.session_state:
    st.session_state.traces = []
if "last_report" not in st.session_state:
    st.session_state.last_report = None
if "pending_approval" not in st.session_state:
    st.session_state.pending_approval = None    # holds report waiting for HITL approval

FAST_MODEL = "llama-3.1-8b-instant"
SMART_MODEL = "llama-3.3-70b-versatile"
COST_PER_1K_TOKENS = 0.0006

# =============================================================================
# GUARDRAILS  (Safety module)
# =============================================================================
PII_PATTERNS = [r"\b\d{10}\b", r"[\w.+-]+@[\w-]+\.[\w.-]+", r"\b\d{4}\s?\d{4}\s?\d{4}\b"]
INJECTION_PHRASES = ["ignore previous instructions", "ignore all previous instructions",
                      "reveal your system prompt", "disregard your rules"]

def check_guardrails(text: str):
    if any(re.search(p, text) for p in PII_PATTERNS):
        return "PII detected in your topic — please remove personal data (emails/phone numbers)."
    if any(p in text.lower() for p in INJECTION_PHRASES):
        return "Possible prompt injection detected — request blocked."
    return None

# =============================================================================
# MEMORY  (simple in-session semantic retrieval)
# =============================================================================
def _embed(text: str, dim: int = 24):
    import random
    random.seed(abs(hash(text.lower())) % (2**32))
    return [random.random() for _ in range(dim)]

def _cosine(a, b):
    import math
    dot = sum(x*y for x, y in zip(a, b))
    na = math.sqrt(sum(x*x for x in a)); nb = math.sqrt(sum(y*y for y in b))
    return dot / (na*nb + 1e-8)

def retrieve_memory(query, top_k=2):
    store = st.session_state.semantic_memory
    if not store:
        return []
    q = _embed(query)
    scored = sorted(((t, _cosine(q, e)) for t, e in store), key=lambda x: x[1], reverse=True)
    return [t for t, _ in scored[:top_k]]

def add_memory(text):
    st.session_state.semantic_memory.append((text, _embed(text)))

# =============================================================================
# CACHE + MODEL ROUTING  (Optimization module)
# =============================================================================
def cache_key(text):
    return hashlib.sha256(" ".join(text.lower().split()).encode()).hexdigest()

def get_cached(topic):
    return st.session_state.response_cache.get(cache_key(topic))

def set_cached(topic, report):
    st.session_state.response_cache[cache_key(topic)] = report

# =============================================================================
# TOOLS  (Tool Engineering + mini MCP-style registry)
# =============================================================================
def tool_web_search(query: str) -> str:
    mock_kb = {
        "ai agents": "AI agents combine LLMs with planning, tools, and memory to complete multi-step goals autonomously.",
        "groq": "Groq provides ultra-fast LLM inference hardware (LPU) with an OpenAI-compatible API for open models like Llama.",
        "mcp": "Model Context Protocol (MCP) is an open standard letting AI apps connect uniformly to external tools and data.",
    }
    for k, v in mock_kb.items():
        if k in query.lower():
            return v
    return f"No strong match found for '{query}' (mock search — plug a real search API for production)."

def tool_calculator(expression: str) -> str:
    allowed = set("0123456789+-*/(). ")
    if not set(expression) <= allowed:
        return "Invalid expression."
    try:
        return str(eval(expression, {"__builtins__": {}}))
    except Exception as e:
        return f"Error: {e}"

TOOL_SCHEMAS = [
    {"type": "function", "function": {
        "name": "tool_web_search", "description": "Search for information on a topic.",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "tool_calculator", "description": "Evaluate a basic math expression.",
        "parameters": {"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]}}},
]
TOOL_REGISTRY = {"tool_web_search": tool_web_search, "tool_calculator": tool_calculator}

# =============================================================================
# OBSERVABILITY  (tracing wrapper)
# =============================================================================
def log_trace(step, response):
    usage = getattr(response, "usage", None)
    total = usage.total_tokens if usage else None
    st.session_state.traces.append({
        "step": step,
        "input_tokens": usage.prompt_tokens if usage else None,
        "output_tokens": usage.completion_tokens if usage else None,
        "total_tokens": total,
        "est_cost_usd": round((total/1000)*COST_PER_1K_TOKENS, 6) if total else None,
        "time": datetime.now().strftime("%H:%M:%S"),
    })

# =============================================================================
# AGENT PIPELINE  (Planner -> Researcher(s) -> Critic -> Writer)
# =============================================================================
def get_client(api_key):
    return Groq(api_key=api_key)

def make_plan(client, topic):
    resp = client.chat.completions.create(
        model=SMART_MODEL, temperature=0.3,
        messages=[
            {"role": "system", "content": "You are a research planner. Break the topic into exactly 3 concise, "
                                           "distinct sub-questions. Reply with ONLY a numbered list, no extra text."},
            {"role": "user", "content": topic},
        ],
    )
    log_trace("planner", resp)
    return resp.choices[0].message.content

def parse_plan(text):
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    cleaned = []
    for line in lines:
        parts = line.split(".", 1) if "." in line[:3] else line.split(")", 1)
        cleaned.append(parts[-1].strip() if len(parts) > 1 else line)
    return cleaned[:3] if cleaned else [text]

def researcher_agent(client, question, memory_context=""):
    messages = [
        {"role": "system", "content": f"You are a Researcher agent. Use tools when useful. Relevant memory: {memory_context}"},
        {"role": "user", "content": question},
    ]
    resp = client.chat.completions.create(model=SMART_MODEL, messages=messages, tools=TOOL_SCHEMAS, tool_choice="auto")
    log_trace("researcher", resp)
    msg = resp.choices[0].message
    if msg.tool_calls:
        for call in msg.tool_calls:
            args = json.loads(call.function.arguments)
            result = TOOL_REGISTRY[call.function.name](**args)
            messages.append({"role": "assistant", "content": None, "tool_calls": [call]})
            messages.append({"role": "tool", "tool_call_id": call.id, "content": result})
        follow_up = client.chat.completions.create(model=SMART_MODEL, messages=messages)
        log_trace("researcher_followup", follow_up)
        return follow_up.choices[0].message.content, True
    return msg.content, False

def critic_agent(client, findings):
    resp = client.chat.completions.create(
        model=SMART_MODEL, temperature=0.2,
        messages=[
            {"role": "system", "content": "You are a Critic agent. Review these research findings for gaps, vague "
                                           "claims, or missing detail. Give short, actionable feedback (max 3 bullet points)."},
            {"role": "user", "content": findings},
        ],
    )
    log_trace("critic", resp)
    return resp.choices[0].message.content

def writer_agent(client, topic, findings, critique):
    resp = client.chat.completions.create(
        model=SMART_MODEL, temperature=0.4,
        messages=[
            {"role": "system", "content": "You are a Writer agent. Produce a clear, well-structured markdown report "
                                           "with a title, short intro, 3 findings sections, and a conclusion. "
                                           "Address the reviewer feedback where relevant."},
            {"role": "user", "content": f"Topic: {topic}\n\nFindings:\n{findings}\n\nReviewer feedback:\n{critique}"},
        ],
    )
    log_trace("writer", resp)
    return resp.choices[0].message.content

# =============================================================================
# SIDEBAR — CONTROLS
# =============================================================================
with st.sidebar:
    st.header("⚙️ Settings")
    api_key = st.text_input("Groq API Key", value=os.environ.get("GROQ_API_KEY", ""), type="password",
                             help="Get a free key at console.groq.com")
    st.caption("Or set the GROQ_API_KEY environment variable before launching.")

    st.divider()
    st.header("🧠 Memory")
    past_topics = [e["topic"] for e in st.session_state.episodic_memory]
    if past_topics:
        for t in past_topics[-5:]:
            st.markdown(f"- {t}")
    else:
        st.caption("No past research yet this session.")

    st.divider()
    st.header("📊 Observability")
    if st.session_state.traces:
        total_cost = sum(t["est_cost_usd"] or 0 for t in st.session_state.traces)
        total_tokens = sum(t["total_tokens"] or 0 for t in st.session_state.traces)
        st.markdown(f"<span class='metric-pill'>{len(st.session_state.traces)} LLM calls</span>"
                    f"<span class='metric-pill'>{total_tokens} tokens</span>"
                    f"<span class='metric-pill'>${round(total_cost,5)}</span>", unsafe_allow_html=True)
        with st.expander("View trace log"):
            st.dataframe(st.session_state.traces, use_container_width=True)
    else:
        st.caption("No LLM calls yet.")

    st.divider()
    if st.button("🗑️ Clear session (memory + cache + traces)"):
        for k in ["episodic_memory", "semantic_memory", "response_cache", "traces", "last_report", "pending_approval"]:
            st.session_state[k] = [] if isinstance(st.session_state[k], list) else ({} if isinstance(st.session_state[k], dict) else None)
        st.rerun()

# =============================================================================
# MAIN — TOPIC INPUT + RUN
# =============================================================================
col1, col2 = st.columns([4, 1])
with col1:
    topic = st.text_input("🔎 Research topic", placeholder="e.g. Impact of AI agents on software development")
with col2:
    st.write("")
    st.write("")
    run_clicked = st.button("🚀 Run Agent", use_container_width=True, type="primary")

if run_clicked:
    if not api_key:
        st.error("Please enter your Groq API key in the sidebar first.")
    elif not topic.strip():
        st.warning("Please enter a research topic.")
    else:
        block_reason = check_guardrails(topic)
        if block_reason:
            st.markdown(f"<p class='status-blocked'>🚫 Blocked by guardrails: {block_reason}</p>", unsafe_allow_html=True)
        else:
            cached = get_cached(topic)
            if cached:
                st.success("⚡ Served instantly from cache — no new LLM calls needed.")
                st.session_state.last_report = cached
                st.session_state.pending_approval = None
            else:
                client = get_client(api_key)
                progress = st.progress(0, text="Planning research...")

                with st.container():
                    st.markdown("<div class='stage-card'>", unsafe_allow_html=True)
                    st.subheader("🧭 Plan")
                    plan_text = make_plan(client, topic)
                    sub_questions = parse_plan(plan_text)
                    for i, q in enumerate(sub_questions, 1):
                        st.markdown(f"**{i}.** {q}")
                    st.markdown("</div>", unsafe_allow_html=True)

                progress.progress(25, text="Researching sub-questions...")
                memory_context = "; ".join(retrieve_memory(topic))
                findings = []
                with st.container():
                    st.markdown("<div class='stage-card'>", unsafe_allow_html=True)
                    st.subheader("🔬 Research Findings")
                    for q in sub_questions:
                        try:
                            answer, used_tool = researcher_agent(client, q, memory_context)
                        except Exception as e:
                            answer, used_tool = f"[Error: {e}]", False
                        findings.append(answer)
                        tool_badge = " 🛠️ *(used a tool)*" if used_tool else ""
                        with st.expander(f"❓ {q}{tool_badge}"):
                            st.write(answer)
                    st.markdown("</div>", unsafe_allow_html=True)

                combined_findings = "\n\n".join(findings)
                add_memory(f"{topic}: {combined_findings[:200]}")

                progress.progress(55, text="Critic reviewing findings...")
                with st.container():
                    st.markdown("<div class='stage-card'>", unsafe_allow_html=True)
                    st.subheader("🧐 Critic Feedback")
                    critique = critic_agent(client, combined_findings)
                    st.write(critique)
                    st.markdown("</div>", unsafe_allow_html=True)

                progress.progress(80, text="Writer drafting final report...")
                report = writer_agent(client, topic, combined_findings, critique)
                progress.progress(100, text="Done!")
                time.sleep(0.3)
                progress.empty()

                st.session_state.pending_approval = {"topic": topic, "report": report}
                st.session_state.last_report = None

# =============================================================================
# HITL APPROVAL GATE
# =============================================================================
if st.session_state.pending_approval:
    pa = st.session_state.pending_approval
    st.markdown("<div class='stage-card'>", unsafe_allow_html=True)
    st.subheader("🔔 Human Approval Required")
    st.write(f"Approve saving the final report for **{pa['topic']}**?")
    with st.expander("Preview report", expanded=True):
        st.markdown(pa["report"])
    c1, c2 = st.columns(2)
    with c1:
        if st.button("✅ Approve & Save", use_container_width=True):
            st.session_state.episodic_memory.append({"topic": pa["topic"], "report": pa["report"]})
            set_cached(pa["topic"], pa["report"])
            st.session_state.last_report = pa["report"]
            st.session_state.pending_approval = None
            st.rerun()
    with c2:
        if st.button("❌ Reject", use_container_width=True):
            st.session_state.last_report = pa["report"]
            st.session_state.pending_approval = None
            st.warning("Rejected — report not saved to memory.")
            st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

# =============================================================================
# FINAL REPORT DISPLAY
# =============================================================================
if st.session_state.last_report:
    st.markdown("<div class='stage-card'>", unsafe_allow_html=True)
    st.subheader("📄 Final Report")
    st.markdown(st.session_state.last_report)
    st.download_button(
        "⬇️ Download report as Markdown",
        data=st.session_state.last_report,
        file_name=f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
        mime="text/markdown",
    )
    st.markdown("</div>", unsafe_allow_html=True)