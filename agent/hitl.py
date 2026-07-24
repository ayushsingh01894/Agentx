"""
hitl.py
Module 7 - Human-in-the-Loop
Provides an approval gate before any "sensitive" final action (here: saving/
publishing the report). In a real product this would be a UI button / Slack
approval instead of a terminal input() call.
"""

def request_approval(action_description: str, auto_approve: bool = False) -> bool:
    """Returns True if the action is approved.

    Set auto_approve=True for non-interactive environments (e.g. API/tests) —
    in production this would be replaced by an actual human click/approval event.
    """
    print(f"\n🔔 [HITL] Approval requested for: {action_description}")
    if auto_approve:
        print("✅ Auto-approved (non-interactive mode).")
        return True
    try:
        answer = input("Approve this action? (yes/no): ").strip().lower()
    except EOFError:
        # No interactive stdin available (e.g. running via script/API) -> default to approve
        print("⚠️  No interactive input available, defaulting to approve.")
        return True
    approved = answer in ("y", "yes")
    print("✅ Approved." if approved else "❌ Rejected — escalating to human review.")
    return approved
