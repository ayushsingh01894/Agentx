"""
reasoning.py
Module 1 - Planning (task decomposition) + Module 3 - Advanced Reasoning (CoT)
"""
from .config import client, SMART_MODEL
from .observability import traced_llm_call


@traced_llm_call("planner")
def make_plan(topic: str):
    """Hierarchical/Sequential planning: break a research topic into sub-questions."""
    return client.chat.completions.create(
        model=SMART_MODEL,
        messages=[
            {"role": "system", "content": "You are a research planner. Break the topic into exactly 3 concise, "
                                           "distinct sub-questions. Reply with ONLY a numbered list, no extra text."},
            {"role": "user", "content": topic},
        ],
        temperature=0.3,
    )


def parse_plan(plan_text: str) -> list[str]:
    lines = [l.strip() for l in plan_text.splitlines() if l.strip()]
    cleaned = []
    for line in lines:
        # strip leading "1. " / "1) " numbering
        parts = line.split(".", 1) if "." in line[:3] else line.split(")", 1)
        cleaned.append(parts[-1].strip() if len(parts) > 1 else line)
    return cleaned[:3] if cleaned else [plan_text]


@traced_llm_call("chain_of_thought")
def reason_step(question: str):
    """Chain-of-Thought reasoning for a single sub-question."""
    return client.chat.completions.create(
        model=SMART_MODEL,
        messages=[
            {"role": "system", "content": "Think step by step internally, then give a concise final answer "
                                           "(2-3 sentences). Do not show your internal steps, only the answer."},
            {"role": "user", "content": question},
        ],
        temperature=0.3,
    )
