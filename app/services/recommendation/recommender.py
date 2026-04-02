"""Hybrid Recommender v2 — blends FIVE signals into a single score.

Scoring formula
---------------
    final = (skill_sim * 0.40)
          + (level_match * 0.15)
          + (goal_match * 0.10)
          + (activity * 0.15)
          + (interest_sim * 0.20)     ← NEW in v2

v2 improvements
---------------
1. **Interest similarity** — separate embedding comparing goals,
   interests, and bios semantically.  "AI researcher" and "data
   science enthusiast" now score high even when keywords differ.
2. **Pre-compute once, reuse** — all sub-scores are computed in the
   scoring loop and stored; the breakdown dict references the stored
   values instead of re-calling scoring functions.
3. **Cold-start** uses strict ``<`` instead of ``<=`` so boundary
   users aren't falsely classified as cold-start.
4. **Goal match** upgraded with partial credit for compatible goals.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from app.config import settings
from app.core.logger import logger
from app.services.common.constants import LEVEL_ORDER, MAX_LEVEL_DISTANCE
from app.services.common.utils import load_users_frozen, build_user_index, get_user_by_id
from app.services.matching.vectorizer import EmbeddingCache


# ── Goal compatibility map (partial credit) ────────────────────────
_COMPATIBLE_GOALS: dict[tuple[str, str], float] = {
    ("Project", "Project"): 1.0,
    ("Study", "Study"): 1.0,
    ("Job", "Job"): 1.0,
    ("Research", "Research"): 1.0,
    ("Mentor", "Mentor"): 1.0,
    # Cross-compatible goals get partial credit.
    ("Study", "Research"): 0.7,
    ("Research", "Study"): 0.7,
    ("Study", "Project"): 0.6,
    ("Project", "Study"): 0.6,
    ("Job", "Project"): 0.5,
    ("Project", "Job"): 0.5,
    ("Mentor", "Study"): 0.6,
    ("Study", "Mentor"): 0.6,
    ("Mentor", "Project"): 0.5,
    ("Project", "Mentor"): 0.5,
}


# ── Public API ──────────────────────────────────────────────────────

def get_recommendations(
    user_id: int,
    page: int = 1,
    limit: int = 10,
) -> list[dict[str, Any]]:
    """Return a paginated list of recommended people for *user_id*.

    For cold-start users the list is replaced by *trending* users.
    """
    limit = min(limit, settings.max_limit)
    target = get_user_by_id(user_id)
    if target is None:
        raise ValueError(f"User with id={user_id} not found")

    # Cold-start detection — strict less-than.
    if target.get("joined_days_ago", 0) < settings.cold_start_days:
        logger.info("Cold-start detected for user %d — returning trending", user_id)
        return _trending_users(user_id, page, limit)

    users = load_users_frozen()
    index_map = build_user_index()
    user_idx = index_map[user_id]

    # Cached matrices — no redundant computation.
    _, sim_matrix = EmbeddingCache.get()
    interest_sim_matrix = EmbeddingCache.get_interest_sim()

    scored: list[tuple[dict[str, Any], float, dict[str, float]]] = []

    for idx, other in enumerate(users):
        if other["id"] == user_id:
            continue

        # Compute all sub-scores ONCE.
        skill_sim = float(sim_matrix[user_idx][idx])
        level_score = _level_match_score(target, other)
        goal_score = _goal_match_score(target, other)
        activity_score = _activity_score(other)
        interest_sim = float(interest_sim_matrix[user_idx][idx])

        final = (
            skill_sim * settings.weight_skill_similarity
            + level_score * settings.weight_level_match
            + goal_score * settings.weight_goal_match
            + activity_score * settings.weight_activity
            + interest_sim * settings.weight_interest_similarity
        )

        # Store breakdown (re-use computed values, no re-calling).
        breakdown = {
            "skill_similarity": round(skill_sim * settings.weight_skill_similarity, 4),
            "level_match": round(level_score * settings.weight_level_match, 4),
            "goal_match": round(goal_score * settings.weight_goal_match, 4),
            "activity": round(activity_score * settings.weight_activity, 4),
            "interest_similarity": round(interest_sim * settings.weight_interest_similarity, 4),
        }

        scored.append((other, final, breakdown))

    # Sort descending by final score.
    scored.sort(key=lambda x: x[1], reverse=True)

    # Paginate.
    start = (page - 1) * limit
    end = start + limit
    page_slice = scored[start:end]

    results: list[dict[str, Any]] = []
    for user, score, breakdown in page_slice:
        results.append(
            {
                "user_id": user["id"],
                "name": user.get("name", f"User {user['id']}"),
                "skills": user.get("skills", []),
                "level": user.get("level", "Unknown"),
                "goal": user.get("goal", "General"),
                "recommendation_score": round(score, 4),
                "breakdown": breakdown,
            }
        )

    logger.info(
        "Recommendations for user %d — page %d, showing %d results",
        user_id,
        page,
        len(results),
    )
    return results


# ── Scoring helpers ─────────────────────────────────────────────────

def _level_match_score(target: dict, other: dict) -> float:
    """Score how well experience levels align (0..1).

    Identical levels → 1.0; maximum distance → 0.0.
    """
    t_level = LEVEL_ORDER.get(target.get("level", ""), 1)
    o_level = LEVEL_ORDER.get(other.get("level", ""), 1)
    if MAX_LEVEL_DISTANCE == 0:
        return 1.0
    distance = abs(t_level - o_level)
    return 1.0 - (distance / MAX_LEVEL_DISTANCE)


def _goal_match_score(target: dict, other: dict) -> float:
    """Score goal compatibility with partial credit.

    Uses a look-up table of compatible goal pairs.  Unknown combos
    get a baseline of 0.2 instead of 0.
    """
    t_goal = target.get("goal", "General")
    o_goal = other.get("goal", "General")
    return _COMPATIBLE_GOALS.get((t_goal, o_goal), 0.2)


def _activity_score(user: dict) -> float:
    """Mock activity signal normalised to [0, 1].

    Uses ``days_active``, ``contributions``, ``projects_joined``,
    and recency.
    """
    activity = user.get("activity", {})
    days_active = activity.get("days_active", 0)
    contributions = activity.get("contributions", 0)
    projects_joined = activity.get("projects_joined", 0)
    last_active = activity.get("last_active_days_ago", 30)

    # Recency boost: active today → 1.0, 30+ days ago → ~0.0
    recency = max(0.0, 1.0 - (last_active / 30.0))

    # Normalise raw counts with soft caps.
    act_raw = (
        min(days_active / 90.0, 1.0) * 0.30
        + min(contributions / 50.0, 1.0) * 0.30
        + min(projects_joined / 5.0, 1.0) * 0.15
        + recency * 0.25
    )

    return min(act_raw, 1.0)


def _trending_users(
    exclude_id: int,
    page: int,
    limit: int,
) -> list[dict[str, Any]]:
    """Return the most active users as a cold-start fallback.

    Trending = sorted by activity score (engagement signal).
    """
    users = load_users_frozen()
    others = [u for u in users if u["id"] != exclude_id]

    # Sort by activity score descending.
    others.sort(key=lambda u: _activity_score(u), reverse=True)

    start = (page - 1) * limit
    end = start + limit

    return [
        {
            "user_id": u["id"],
            "name": u.get("name", f"User {u['id']}"),
            "skills": u.get("skills", []),
            "level": u.get("level", "Unknown"),
            "goal": u.get("goal", "General"),
            "recommendation_score": round(_activity_score(u), 4),
            "trending": True,
        }
        for u in others[start:end]
    ]
