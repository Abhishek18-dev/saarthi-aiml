"""Shared dependency helpers for FastAPI routes."""

from __future__ import annotations


def get_request_context() -> dict[str, str]:
    """Return lightweight request context metadata.

    In production this would include auth info, tracing IDs, etc.
    """
    return {"source": "api", "version": "1.0.0"}
