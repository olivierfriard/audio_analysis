"""
Helpers for the project JSON call schema.
"""

CALLS_KEY = "calls"


def ensure_calls(container: dict) -> dict:
    """
    Return the calls block, creating it when missing.
    """
    if not isinstance(container, dict):
        return {}

    calls = container.get(CALLS_KEY)
    if not isinstance(calls, dict):
        calls = {}

    container[CALLS_KEY] = calls
    return calls


def get_calls(container: dict) -> dict:
    """
    Return calls from a chunk-like JSON block.
    """
    if not isinstance(container, dict):
        return {}
    calls = container.get(CALLS_KEY, {})
    return calls if isinstance(calls, dict) else {}
