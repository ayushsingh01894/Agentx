"""
agents.py
Module 6 - Multi-Agent Systems
Defines role-based agents that communicate through a shared "blackboard"
(the `state` dict passed between them) — Coordinator -> Researcher(s) -> Critic -> Writer.
"""
import json
from .config import client, SMART_MODEL, pick_model
from .observability import traced_llm_call
from .tools import mcp_server


@traced_llm_call("researcher_agent")
def researcher_agent(question: str, memory_context: str = ""):
    """Researcher role: may call tools (via the MCP tool server) to gather facts."""
    from .tools import TOOL_SCHEMAS
    messages = [
        {"role": "system", "content": f"You are a Researcher agent. Use tools when useful. "
                                       f"Relevant memory: {memory_context}"},
        {"role": "user", "content": question},
    ]
    response = client.chat.completions.create(
        model=SMART_MODEL, messages=messages, tools=TOOL_SCHEMAS, tool_choice="auto"
    )
    msg = response.choices[0].message

    if msg.tool_calls:
        for call in msg.tool_calls:
            args = json.loads(call.function.arguments)
            tool_result = mcp_server.call_tool(call.function.name, args)
            messages.append({"role": "assistant", "content": None, "tool_calls": [call]})
            messages.append({"role": "tool", "tool_call_id": call.id, "content": tool_result})
        follow_up = client.chat.completions.create(model=SMART_MODEL, messages=messages)
        return follow_up
    return response


@traced_llm_call("critic_agent")
def critic_agent(draft_findings: str):
    """Reviewer/Critic role: checks findings for gaps or unsupported claims (Reflection)."""
    return client.chat.completions.create(
        model=SMART_MODEL,
        messages=[
            {"role": "system", "content": "You are a Critic agent. Review these research findings for gaps, "
                                           "vague claims, or missing detail. Give short, actionable feedback "
                                           "(max 3 bullet points)."},
            {"role": "user", "content": draft_findings},
        ],
        temperature=0.2,
    )


@traced_llm_call("writer_agent")
def writer_agent(topic: str, findings: str, critique: str):
    """Writer role: synthesizes findings + critique into a final polished report."""
    return client.chat.completions.create(
        model=SMART_MODEL,
        messages=[
            {"role": "system", "content": "You are a Writer agent. Produce a clear, well-structured markdown "
                                           "report with a title, short intro, 3 findings sections, and a conclusion. "
                                           "Address the critic's feedback where relevant."},
            {"role": "user", "content": f"Topic: {topic}\n\nFindings:\n{findings}\n\nCritic feedback:\n{critique}"},
        ],
        temperature=0.4,
    )
