"""Trust Score Calculator.

Computes a 0–100 trust score based on two signals:
1. **Activity** — normalised from days active, contributions, and
   recency.
2. **Ratings** — average of community ratings (if any).

The two signals are blended 50/50.  Users with no ratings receive
a neutral 50 for the rating component so they aren't unfairly
penalised.
"""

from __future__ import annotations

from typing import Any

from app.core.logger import logger
from app.services.common.utils import get_user_by_id


def calculate_trust_score(user_id: int) -> dict[str, Any]:
    """Return the trust-score breakdown for *user_id*.

    Returns
    -------
    dict
        Contains ``trust_score`` (int, 0-100), ``activity_component``,
        ``rating_component``, and ``label`` (descriptive tier).
    """
    user = get_user_by_id(user_id)
    if user is None:
        raise ValueError(f"User with id={user_id} not found")

    activity = user.get("activity", {})
    ratings = user.get("ratings", [])

    # ── Activity component (0-100) ─────────────────────────────────
    days_active = activity.get("days_active", 0)
    contributions = activity.get("contributions", 0)
    projects_joined = activity.get("projects_joined", 0)
    last_active = activity.get("last_active_days_ago", 30)

    # Weighted sub-signals, soft-capped.
    act_raw = (
        min(days_active / 90.0, 1.0) * 0.3
        + min(contributions / 50.0, 1.0) * 0.35
        + min(projects_joined / 5.0, 1.0) * 0.2
        + max(0.0, 1.0 - last_active / 30.0) * 0.15
    )
    activity_component = round(act_raw * 100, 1)

    # ── Rating component (0-100) ───────────────────────────────────
    if ratings:
        avg_rating = sum(ratings) / len(ratings)
        rating_component = round((avg_rating / 5.0) * 100, 1)
    else:
        rating_component = 50.0  # neutral default

    # ── Blend ──────────────────────────────────────────────────────
    trust_score = round(0.5 * activity_component + 0.5 * rating_component)
    trust_score = max(0, min(100, trust_score))

    label = _score_label(trust_score)

    logger.info(
        "Trust score for user %d: %d (%s)", user_id, trust_score, label
    )

    return {
        "user_id": user_id,
        "trust_score": trust_score,
        "activity_component": activity_component,
        "rating_component": rating_component,
        "label": label,
    }


# ── Helper ──────────────────────────────────────────────────────────

def _score_label(score: int) -> str:
    """Map a numeric trust score to a human-friendly tier label."""
    if score >= 90:
        return "Excellent"
    if score >= 75:
        return "Very Good"
    if score >= 60:
        return "Good"
    if score >= 40:
        return "Fair"
    return "New / Low Activity"
