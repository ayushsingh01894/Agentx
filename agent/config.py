"""
config.py
Central Groq client + model configuration.
Module covered: Foundational setup used by every other module.
"""
import os
from groq import Groq

# ---- Groq client (reads GROQ_API_KEY from environment) ----
client = Groq(api_key=os.environ.get("GROQ_API_KEY", "REPLACE_WITH_YOUR_KEY"))

# ---- Model routing config (Module 14 - Optimization) ----
FAST_MODEL = "llama-3.1-8b-instant"      # cheap/fast, for simple sub-tasks
SMART_MODEL = "llama-3.3-70b-versatile"  # for reasoning, planning, writing

def pick_model(text: str) -> str:
    """Very simple heuristic router: short/simple prompts -> fast model."""
    word_count = len(text.split())
    return FAST_MODEL if word_count <= 12 else SMART_MODEL
