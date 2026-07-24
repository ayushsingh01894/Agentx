"""
evaluation.py
Module 8 - Agent Evaluation
Tracks per-run metrics (success, retries, duration) and provides a tiny
benchmark harness to sanity-check the agent against known test cases.
"""
import time


class RunMetrics:
    def __init__(self):
        self.start_time = time.time()
        self.retries = 0
        self.tool_calls = 0
        self.success = False

    def record_retry(self):
        self.retries += 1

    def record_tool_call(self):
        self.tool_calls += 1

    def finish(self, success: bool):
        self.success = success
        self.duration = round(time.time() - self.start_time, 2)

    def summary(self):
        return {
            "success": self.success,
            "duration_sec": self.duration,
            "retries": self.retries,
            "tool_calls": self.tool_calls,
        }


def run_benchmark(orchestrator_fn, test_topics: list[str]):
    """Offline evaluation: runs the agent on a small fixed set of topics and
    reports pass/fail + timing for each (Module 8: Benchmarking)."""
    results = []
    for topic in test_topics:
        start = time.time()
        try:
            report = orchestrator_fn(topic, auto_approve=True)
            ok = bool(report) and len(report) > 50
        except Exception as e:
            ok = False
            report = str(e)
        results.append({
            "topic": topic,
            "passed": ok,
            "duration_sec": round(time.time() - start, 2),
        })
    return results
