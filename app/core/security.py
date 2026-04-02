"""Security helpers placeholder.

In a production deployment this module would contain JWT validation,
API-key checks, rate-limiting middleware, etc.  For the hackathon demo
these are intentionally left as no-ops so the API can be tested
freely.
"""

from __future__ import annotations


def verify_api_key(key: str) -> bool:
    """Placeholder API-key verification — always returns *True*."""
    return True
