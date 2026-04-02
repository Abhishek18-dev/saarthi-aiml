"""Background tasks for heavy / async operations.

v2: ``precompute_embeddings()`` now populates the global
``EmbeddingCache`` so the startup warm-up actually benefits
subsequent requests (was a no-op before).
"""

from __future__ import annotations

import time
from typing import Any

from app.core.logger import logger
from app.services.matching.vectorizer import EmbeddingCache
from app.services.common.utils import load_users_frozen, invalidate_caches


def precompute_embeddings() -> dict[str, Any]:
    """Pre-compute and cache all user embeddings + similarity matrix.

    Called at startup or on-demand.  The result is stored in
    ``EmbeddingCache`` so all subsequent requests are instantaneous.
    """
    logger.info("Background: pre-computing embeddings …")
    users = load_users_frozen()
    stats = EmbeddingCache.warm(users)
    logger.info(
        "Background: embeddings ready — %d users, %.2fs elapsed",
        stats["users"],
        stats["elapsed_seconds"],
    )
    return stats


def refresh_data() -> None:
    """Invalidate caches and re-load data from disk.

    Useful after the mock JSON files are edited without restarting
    the server.
    """
    logger.info("Background: refreshing data caches …")
    invalidate_caches()
    # Re-warm.
    precompute_embeddings()
    logger.info("Background: data refresh complete")


def log_match_event(
    user_id: int,
    matches: list[dict[str, Any]],
) -> None:
    """Log a matching event for analytics (placeholder).

    In production this would write to a database or event stream.
    Handles both V1 (matchScore) and internal (similarity_score) keys.
    """
    top_score = 0.0
    if matches:
        first = matches[0]
        top_score = first.get(
            "similarity_score",
            first.get("matchScore", first.get("match_score", 0.0)),
        )
    logger.info(
        "Analytics: user %d matched with %d users — top score %.4f",
        user_id,
        len(matches),
        top_score,
    )
