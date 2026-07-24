"""
orchestrator.py
Module 2 - Agent Communication & Execution (FSM loop, retries, failure handling)
Module 10 - Agent Frameworks (a hand-rolled StateGraph-style FSM, LangGraph-inspired)

This ties together EVERY module into one runnable pipeline:

  guardrails -> memory -> planner -> [researcher -> critic] loop -> writer
      -> HITL approval -> save (tool) -> episodic memory log -> evaluation summary

State transitions:
  START -> PLAN -> RESEARCH -> CRITIQUE -> WRITE -> APPROVE -> SAVE -> END
                       ^___________retry on failure___________|
"""
from .guardrails import run_guardrails, GuardrailBlocked
from .memory import WorkingMemory, SemanticMemory, EpisodicMemory
from .reasoning import make_plan, parse_plan, reason_step
from .agents import researcher_agent, critic_agent, writer_agent
from .hitl import request_approval
from .tools import save_report
from .evaluation import RunMetrics
from .optimization import get_cached, set_cached
from .observability import print_trace_summary

semantic_memory = SemanticMemory()
episodic_memory = EpisodicMemory()

MAX_RETRIES = 2


def run_research_agent(topic: str, auto_approve: bool = False) -> str:
    metrics = RunMetrics()
    working_memory = WorkingMemory()
    working_memory.set("topic", topic)

    # ---- STATE: GUARDRAILS ----
    try:
        run_guardrails(topic)
    except GuardrailBlocked as e:
        print(f"🚫 Blocked by guardrails: {e}")
        metrics.finish(success=False)
        return ""

    # ---- Cache check (Module 14 - Optimization) ----
    cached = get_cached(topic)
    if cached:
        print("⚡ Served from cache (no LLM calls needed).")
        metrics.finish(success=True)
        return cached

    # ---- STATE: PLAN ----
    print(f"\n🧭 Planning research for: '{topic}'")
    plan_response = make_plan(topic)
    sub_questions = parse_plan(plan_response.choices[0].message.content)
    print("Sub-questions:", sub_questions)

    # ---- Retrieve relevant memory context (Module 4) ----
    memory_context = "; ".join(semantic_memory.retrieve(topic))

    # ---- STATE: RESEARCH (with retry/failure handling - Module 2) ----
    findings = []
    for q in sub_questions:
        attempt, success = 0, False
        while attempt <= MAX_RETRIES and not success:
            try:
                result = researcher_agent(q, memory_context)
                findings.append(result.choices[0].message.content)
                success = True
            except Exception as e:
                attempt += 1
                metrics.record_retry()
                print(f"⚠️  Research step failed ({e}), retry {attempt}/{MAX_RETRIES}...")
        if not success:
            findings.append(f"[Could not resolve: {q}]")

    combined_findings = "\n\n".join(findings)
    semantic_memory.add(f"{topic}: {combined_findings[:200]}")

    # ---- STATE: CRITIQUE (Reflection - Module 1 / Module 6 Reviewer) ----
    print("\n🔍 Critic reviewing findings...")
    critique_response = critic_agent(combined_findings)
    critique = critique_response.choices[0].message.content

    # ---- STATE: WRITE ----
    print("✍️  Writer synthesizing final report...")
    write_response = writer_agent(topic, combined_findings, critique)
    report = write_response.choices[0].message.content

    # ---- STATE: APPROVE (Module 7 - HITL) ----
    approved = request_approval(f"Save final report for '{topic}'", auto_approve=auto_approve)
    if not approved:
        metrics.finish(success=False)
        return report  # still return the draft, just not saved

    # ---- STATE: SAVE (tool call - Module 5) ----
    filename = topic.lower().replace(" ", "_")[:40] + ".md"
    save_result = save_report(filename, report)
    print(f"💾 {save_result}")

    # ---- Episodic memory log (Module 4) ----
    episodic_memory.log_episode(topic, report)

    # ---- Cache result (Module 14) ----
    set_cached(topic, report)

    metrics.finish(success=True)
    print(f"\n📈 Run metrics: {metrics.summary()}")
    print_trace_summary()

    return report
