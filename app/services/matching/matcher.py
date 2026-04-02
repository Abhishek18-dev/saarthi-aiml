"""Matcher — orchestrates vectorization + similarity to find top
matches for a given user.

v2 changes
----------
* Uses ``EmbeddingCache`` — zero redundant embedding computation.
* Uses ``build_user_index()`` — O(1) user ID → index look-up.
* Adds semantic skill-group bonus from ``constants``.
"""

from __future__ import annotations

from typing import Any

from app.config import settings
from app.core.logger import logger
from app.services.common.utils import load_users_frozen, build_user_index, get_user_by_id
from app.services.common.constants import compute_group_overlap
from app.services.matching.vectorizer import EmbeddingCache
from app.services.matching.similarity import get_top_similar


def get_matches(
    user_id: int,
    top_k: int | None = None,
) -> list[dict[str, Any]]:
    """Return the top-K most similar users for *user_id*.

    Parameters
    ----------
    user_id : int
        ID of the requesting user.
    top_k : int, optional
        Number of matches to return.  Falls back to
        ``settings.default_top_k``.

    Returns
    -------
    list[dict]
        Each dict contains ``user_id``, ``name``, ``skills``,
        ``level``, ``goal``, ``similarity_score``, and
        ``shared_skill_groups``.

    Raises
    ------
    ValueError
        If *user_id* does not exist in the database.
    """
    if top_k is None:
        top_k = settings.default_top_k

    # O(1) index look-up.
    index_map = build_user_index()
    user_index = index_map.get(user_id)
    if user_index is None:
        raise ValueError(f"User with id={user_id} not found")

    users = load_users_frozen()
    target_user = users[user_index]

    logger.info(
        "Computing matches for user %d (%s) — top_k=%d",
        user_id,
        target_user.get("name", "unknown"),
        top_k,
    )

    # Cached embeddings + similarity matrix — no re-computation.
    _, sim_matrix = EmbeddingCache.get()

    # Retrieve top-K (excluding self).
    top_indices = get_top_similar(sim_matrix, user_index, top_k=top_k)

    # Package results with skill-group bonus info.
    target_skills = target_user.get("skills", [])
    results: list[dict[str, Any]] = []
    for idx, score in top_indices:
        matched_user = users[idx]
        matched_skills = matched_user.get("skills", [])

        # Compute semantic group overlap for richer match explanation.
        group_overlap = compute_group_overlap(target_skills, matched_skills)

        results.append(
            {
                "user_id": matched_user["id"],
                "name": matched_user.get("name", f"User {matched_user['id']}"),
                "skills": matched_skills,
                "level": matched_user.get("level", "Unknown"),
                "goal": matched_user.get("goal", "General"),
                "similarity_score": score,
                "shared_skill_groups": round(group_overlap, 2),
            }
        )

    logger.info(
        "Returned %d matches for user %d", len(results), user_id
    )
    return results
