"""
main.py — CLI entry point for AgentX (Autonomous Research & Report Agent)

Usage:
    python main.py "Impact of AI agents on software development"
    python main.py --benchmark
"""
import sys
from agent.orchestrator import run_research_agent
from agent.evaluation import run_benchmark


def main():
    if len(sys.argv) < 2:
        print('Usage: python main.py "your research topic"  OR  python main.py --benchmark')
        sys.exit(1)

    if sys.argv[1] == "--benchmark":
        print("Running offline benchmark (Module 8 - Evaluation)...\n")
        test_topics = [
            "Benefits of vector databases",
            "How does the Model Context Protocol work",
        ]
        results = run_benchmark(run_research_agent, test_topics)
        for r in results:
            status = "✅ PASS" if r["passed"] else "❌ FAIL"
            print(f"{status} | {r['topic']} | {r['duration_sec']}s")
        return

    topic = " ".join(sys.argv[1:])
    report = run_research_agent(topic, auto_approve=False)
    print("\n" + "=" * 60)
    print(report)
    print("=" * 60)


if __name__ == "__main__":
    main()
