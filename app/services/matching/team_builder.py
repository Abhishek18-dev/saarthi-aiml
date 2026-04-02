"""Team Builder — form balanced teams using similarity clustering.

v2 changes
----------
* Uses ``EmbeddingCache`` — zero redundant embedding computation.
* Vectorised numpy scoring replaces the Python inner loop.
* Adds ``skill_coverage`` metric showing how many unique skill groups
  the team covers (diversity metric for demos).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from app.config import settings
from app.core.logger import logger
from app.services.common.utils import load_users_frozen, build_user_index, get_user_by_id
from app.services.common.constants import get_skill_group
from app.services.matching.vectorizer import EmbeddingCache


def build_team(
    user_id: int,
    team_size: int | None = None,
) -> list[dict[str, Any]]:
    """Build a single team around *user_id*.

    Parameters
    ----------
    user_id : int
        Seed user for the team.
    team_size : int, optional
        Desired team size (clamped between ``team_min_size`` and
        ``team_max_size`` from settings).

    Returns
    -------
    list[dict]
        Ordered list of team members starting with the seed user.
    """
    if team_size is None:
        team_size = settings.team_min_size
    team_size = max(settings.team_min_size, min(team_size, settings.team_max_size))

    index_map = build_user_index()
    seed_index = index_map.get(user_id)
    if seed_index is None:
        raise ValueError(f"User with id={user_id} not found")

    users = load_users_frozen()
    target = users[seed_index]

    logger.info(
        "Building team of %d around user %d (%s)",
        team_size,
        user_id,
        target.get("name", ""),
    )

    # Cached similarity matrix.
    _, sim_matrix = EmbeddingCache.get()

    # Greedy team construction — vectorised scoring.
    n_users = len(users)
    selected_mask = np.zeros(n_users, dtype=bool)
    selected_indices: list[int] = [seed_index]
    selected_mask[seed_index] = True

    while len(selected_indices) < team_size and not selected_mask.all():
        # Average similarity of each remaining user to all selected members.
        selected_cols = sim_matrix[:, selected_indices]  # shape (N, len(selected))
        avg_scores = selected_cols.mean(axis=1)          # shape (N,)
        avg_scores[selected_mask] = -1.0                 # exclude already selected

        best_idx = int(np.argmax(avg_scores))
        if avg_scores[best_idx] <= -1.0:
            break  # no remaining candidates

        selected_indices.append(best_idx)
        selected_mask[best_idx] = True

    # Build team with skill coverage analysis.
    team: list[dict[str, Any]] = []
    all_skill_groups: set[str] = set()

    for idx in selected_indices:
        member = users[idx]
        role = _suggest_role(member)
        team.append(
            {
                "user_id": member["id"],
                "name": member.get("name", f"User {member['id']}"),
                "skills": member.get("skills", []),
                "level": member.get("level", "Unknown"),
                "role": role,
            }
        )
        # Track unique skill groups this team covers.
        for skill in member.get("skills", []):
            group = get_skill_group(skill)
            if group:
                all_skill_groups.add(group)

    logger.info(
        "Team assembled: %s — covers %d skill groups",
        [m["name"] for m in team],
        len(all_skill_groups),
    )
    return team


def build_multiple_teams(
    team_size: int | None = None,
) -> list[list[dict[str, Any]]]:
    """Partition all users into balanced teams.

    Iterates through users that haven't been assigned yet, using each
    unassigned user as the next seed.
    """
    if team_size is None:
        team_size = settings.team_min_size
    team_size = max(settings.team_min_size, min(team_size, settings.team_max_size))

    users = load_users_frozen()
    _, sim_matrix = EmbeddingCache.get()

    n_users = len(users)
    assigned = np.zeros(n_users, dtype=bool)
    teams: list[list[dict[str, Any]]] = []

    for seed_idx in range(n_users):
        if assigned[seed_idx]:
            continue

        current_team_indices: list[int] = [seed_idx]
        assigned[seed_idx] = True

        while len(current_team_indices) < team_size and not assigned.all():
            selected_cols = sim_matrix[:, current_team_indices]
            avg_scores = selected_cols.mean(axis=1)
            avg_scores[assigned] = -1.0

            best_idx = int(np.argmax(avg_scores))
            if avg_scores[best_idx] <= -1.0:
                break

            current_team_indices.append(best_idx)
            assigned[best_idx] = True

        team: list[dict[str, Any]] = []
        for idx in current_team_indices:
            member = users[idx]
            team.append(
                {
                    "user_id": member["id"],
                    "name": member.get("name", f"User {member['id']}"),
                    "skills": member.get("skills", []),
                    "level": member.get("level", "Unknown"),
                    "role": _suggest_role(member),
                }
            )
        teams.append(team)

    logger.info("Built %d teams from %d users", len(teams), n_users)
    return teams


# ── Helpers ─────────────────────────────────────────────────────────

def _suggest_role(user: dict[str, Any]) -> str:
    """Heuristically assign a team role based on profile attributes."""
    level = user.get("level", "Beginner")
    goal = user.get("goal", "")
    skills_lower = {s.lower() for s in user.get("skills", [])}

    if level == "Advanced" or goal == "Mentor":
        return "Team Lead"
    if goal == "Research":
        return "Researcher"

    frontend_skills = {"react", "vue.js", "flutter", "css", "tailwind css",
                       "next.js", "angular", "html", "dart"}
    backend_skills = {"python", "django", "fastapi", "node.js", "java",
                      "spring boot", "express", "go"}
    ml_skills = {"machine learning", "tensorflow", "deep learning",
                 "computer vision", "nlp", "pytorch"}

    if skills_lower & ml_skills:
        return "ML / Data"
    if skills_lower & frontend_skills:
        return "Frontend / UI"
    if skills_lower & backend_skills:
        return "Backend / API"
    return "Contributor"
