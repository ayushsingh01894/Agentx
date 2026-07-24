"""
observability.py
Module 13 - AI Observability: traces every LLM/tool call with latency, tokens, cost.
"""
import time
import json
import functools
from pathlib import Path

TRACE_FILE = Path(__file__).parent.parent / "traces.jsonl"

# Approx Groq pricing per 1K tokens (illustrative only, check console.groq.com for real rates)
COST_PER_1K_TOKENS = 0.0006


def _append_trace(record: dict):
    with open(TRACE_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def traced_llm_call(step_name: str):
    """Decorator that wraps a function returning a Groq completion response
    and logs latency, token usage, and estimated cost."""
    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            start = time.time()
            result = fn(*args, **kwargs)
            duration = round(time.time() - start, 3)

            usage = getattr(result, "usage", None)
            input_tokens = usage.prompt_tokens if usage else None
            output_tokens = usage.completion_tokens if usage else None
            total_tokens = usage.total_tokens if usage else None
            cost = round((total_tokens / 1000) * COST_PER_1K_TOKENS, 6) if total_tokens else None

            _append_trace({
                "step": step_name,
                "duration_sec": duration,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens,
                "estimated_cost_usd": cost,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            })
            return result
        return wrapper
    return decorator


def print_trace_summary():
    if not TRACE_FILE.exists():
        print("No traces recorded yet.")
        return
    total_cost = 0.0
    total_time = 0.0
    count = 0
    with open(TRACE_FILE, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            total_cost += rec.get("estimated_cost_usd") or 0
            total_time += rec.get("duration_sec") or 0
            count += 1
    print(f"\n📊 Observability Summary — {count} LLM calls | "
          f"Total time: {round(total_time, 2)}s | Estimated cost: ${round(total_cost, 5)}")
