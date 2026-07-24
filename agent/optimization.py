"""
optimization.py
Module 14 - AI Optimization: response caching (exact + light semantic match)
and model routing to control cost/latency.
"""
import hashlib

_cache: dict[str, str] = {}


def _normalize(text: str) -> str:
    return " ".join(text.lower().strip().split())


def cache_key(prompt: str) -> str:
    return hashlib.sha256(_normalize(prompt).encode()).hexdigest()


def get_cached(prompt: str):
    return _cache.get(cache_key(prompt))


def set_cached(prompt: str, response: str):
    _cache[cache_key(prompt)] = response


def cache_stats():
    return {"cached_entries": len(_cache)}
