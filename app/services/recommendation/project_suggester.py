"""Project Suggester v2 — hybrid Jaccard + semantic project matching.

v2 improvements
---------------
1. **Semantic scoring** — each project is embedded and compared to the
   user's profile embedding via cosine similarity.  Combined with
   Jaccard for a much smarter ranking.
2. **Difficulty match bonus** — projects matching the user's level get
   a small boost.
3. Uses cached embeddings via ``EmbeddingCache``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from app.config import settings
from app.core.logger import logger
from app.services.common.constants import SKILL_PROJECT_MAP
from app.services.common.utils import load_projects_frozen, get_user_by_id, build_user_index
from app.services.matching.vectorizer import (
    EmbeddingCache,
    embed_texts,
    project_to_text,
)


def suggest_projects(
    user_id: int,
    top_k: int = 5,
    min_results: int = 3,
) -> list[dict[str, Any]]:
    """Return project suggestions tailored to *user_id*.

    Scoring: ``final = jaccard_weight * jaccard + semantic_weight * cosine_sim``

    Parameters
    ----------
    user_id : int
        Target user.
    top_k : int
        Maximum suggestions to return.
    min_results : int
        Minimum results — if the DB produces fewer, the fallback map
        fills the gap.
    """
    index_map = build_user_index()
    user_idx = index_map.get(user_id)
    if user_idx is None:
        raise ValueError(f"User with id={user_id} not found")

    user = get_user_by_id(user_id)
    if user is None:
        raise ValueError(f"User with id={user_id} not found")

    user_skills = {s.lower() for s in user.get("skills", [])}
    user_level = user.get("level", "Beginner")
    results: list[dict[str, Any]] = []

    # ── Get user embedding from cache ──────────────────────────────
    user_embedding = EmbeddingCache.get_embeddings()[user_idx].reshape(1, -1)

    # ── Score all database projects ────────────────────────────────
    projects = load_projects_frozen()
    if projects:
        # Batch-embed all projects.
        project_texts = [project_to_text(p) for p in projects]
        project_embeddings = embed_texts(project_texts)

        # Cosine similarities: user vs all projects.
        semantic_scores = cosine_similarity(user_embedding, project_embeddings)[0]

        scored: list[tuple[dict[str, Any], float]] = []
        for i, proj in enumerate(projects):
            proj_skills = {s.lower() for s in proj.get("required_skills", [])}
            if not proj_skills:
                continue

            # Jaccard similarity.
            intersection = user_skills & proj_skills
            union = user_skills | proj_skills
            jaccard = len(intersection) / len(union) if union else 0.0

            # Semantic similarity.
            semantic = float(semantic_scores[i])

            # Difficulty bonus (+0.1 if levels match).
            difficulty_bonus = 0.1 if proj.get("difficulty") == user_level else 0.0

            combined = (
                settings.project_jaccard_weight * jaccard
                + settings.project_semantic_weight * semantic
                + difficulty_bonus
            )

            if combined > 0.05:  # threshold to avoid noise
                scored.append((proj, combined))

        scored.sort(key=lambda x: x[1], reverse=True)

        for proj, score in scored[:top_k]:
            results.append(
                {
                    "title": proj["title"],
                    "description": proj.get("description", ""),
                    "required_skills": proj.get("required_skills", []),
                    "difficulty": proj.get("difficulty", "Unknown"),
                    "category": proj.get("category", "General"),
                    "match_score": round(score, 4),
                    "source": "database",
                }
            )

    # ── Fallback: skill map ────────────────────────────────────────
    if len(results) < min_results:
        seen_titles = {r["title"].lower() for r in results}
        for skill in user.get("skills", []):
            if len(results) >= top_k:
                break
            ideas = SKILL_PROJECT_MAP.get(skill, [])
            for idea in ideas:
                if idea.lower() not in seen_titles and len(results) < top_k:
                    results.append(
                        {
                            "title": idea,
                            "description": f"Suggested project based on your {skill} skill",
                            "required_skills": [skill],
                            "difficulty": user_level,
                            "category": "Suggested",
                            "match_score": 0.5,
                            "source": "skill_map",
                        }
                    )
                    seen_titles.add(idea.lower())

    logger.info(
        "Suggested %d projects for user %d", len(results), user_id
    )
    return results


def suggest_projects_by_skills(
    skills: list[str],
    top_k: int = 5,
    min_results: int = 3,
) -> list[dict[str, Any]]:
    """Return project suggestions based on a raw skill list.

    This variant is used by the Spring Boot backend which sends
    ``{"skills": ["React", "Java"]}`` — no userId required.

    Scoring uses the same Jaccard + semantic hybrid as ``suggest_projects``.
    """
    from app.services.matching.vectorizer import embed_text

    user_skills = {s.lower() for s in skills}
    results: list[dict[str, Any]] = []

    # Build a synthetic text from skills for semantic matching.
    skills_text = f"Skills: {', '.join(skills)}."
    user_embedding = embed_text(skills_text).reshape(1, -1)

    # Score all database projects.
    projects = load_projects_frozen()
    if projects:
        project_texts = [project_to_text(p) for p in projects]
        project_embeddings = embed_texts(project_texts)
        semantic_scores = cosine_similarity(user_embedding, project_embeddings)[0]

        scored: list[tuple[dict[str, Any], float]] = []
        for i, proj in enumerate(projects):
            proj_skills = {s.lower() for s in proj.get("required_skills", [])}
            if not proj_skills:
                continue

            intersection = user_skills & proj_skills
            union = user_skills | proj_skills
            jaccard = len(intersection) / len(union) if union else 0.0

            semantic = float(semantic_scores[i])

            combined = (
                settings.project_jaccard_weight * jaccard
                + settings.project_semantic_weight * semantic
            )

            if combined > 0.05:
                scored.append((proj, combined))

        scored.sort(key=lambda x: x[1], reverse=True)

        for proj, score in scored[:top_k]:
            results.append(
                {
                    "title": proj["title"],
                    "description": proj.get("description", ""),
                    "required_skills": proj.get("required_skills", []),
                    "difficulty": proj.get("difficulty", "Unknown"),
                    "category": proj.get("category", "General"),
                    "match_score": round(score, 4),
                    "source": "database",
                }
            )

    # Fallback: skill map.
    if len(results) < min_results:
        seen_titles = {r["title"].lower() for r in results}
        for skill in skills:
            if len(results) >= top_k:
                break
            ideas = SKILL_PROJECT_MAP.get(skill, [])
            for idea in ideas:
                if idea.lower() not in seen_titles and len(results) < top_k:
                    results.append(
                        {
                            "title": idea,
                            "description": f"Suggested project based on {skill}",
                            "required_skills": [skill],
                            "difficulty": "Beginner",
                            "category": "Suggested",
                            "match_score": 0.5,
                            "source": "skill_map",
                        }
                    )
                    seen_titles.add(idea.lower())

    logger.info("Suggested %d projects for skills %s", len(results), skills)
    return results
