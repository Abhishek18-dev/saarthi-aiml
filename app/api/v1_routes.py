"""V1 API routes — Spring Boot integration layer.

These endpoints are the EXACT contract that Spring Boot's RestTemplate
calls internally.  The frontend NEVER calls these directly.

Architecture:
    React Frontend → Spring Boot (8080) → RestTemplate → FastAPI (8000)

All responses use camelCase JSON via Pydantic aliases, matching what
Java's Jackson ObjectMapper expects out of the box.

Endpoints
---------
POST /api/v1/match-users         — Find semantically similar users
POST /api/v1/trust-score         — Calculate trust score for a user
POST /api/v1/suggest-projects    — Suggest projects by skill list
POST /api/v1/build-team          — Build a team around a seed user
POST /api/v1/recommendations     — Full recommendations (people + projects)
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

from app.core.logger import logger
from app.schemas.v1_schema import (
    V1MatchRequest,
    V1MatchResult,
    V1TrustScoreRequest,
    V1TrustScoreResponse,
    V1SuggestProjectsRequest,
    V1ProjectSuggestion,
    V1BuildTeamRequest,
    V1TeamMember,
    V1RecommendRequest,
    V1RecommendationPerson,
    V1RecommendationsResponse,
)
from app.services.matching.matcher import get_matches
from app.services.matching.team_builder import build_team
from app.services.matching.vectorizer import EmbeddingCache
from app.services.common.utils import load_users, use_runtime_users
from app.services.recommendation.recommender import get_recommendations
from app.services.recommendation.project_suggester import (
    suggest_projects,
    suggest_projects_by_skills,
)
from app.services.recommendation.trust_score import calculate_trust_score


router = APIRouter(prefix="/api/v1", tags=["v1 — Spring Boot Integration"])


def _resolve_users_and_mode(
    runtime_users: list | None,
) -> tuple[list[dict], str]:
    """Auto-detect mode from request payload.

    REAL mode is activated only when request.users is provided.
    """
    if runtime_users:
        users = [u.model_dump(by_alias=False) for u in runtime_users]
        return users, "REAL"
    return load_users(), "DEMO"


# ────────────────────────────────────────────────────────────────────
# 1. MATCH USERS
# ────────────────────────────────────────────────────────────────────

@router.post(
    "/match-users",
    response_model=list[V1MatchResult],
    response_model_by_alias=True,
    summary="Find top skill-based matches (Spring Boot contract)",
    responses={
        404: {"description": "User not found"},
        422: {"description": "Invalid request body"},
    },
)
def v1_match_users(body: V1MatchRequest):
    """Return a flat JSON array of matched users.

    Spring Boot calls this with ``{"userId": 1}`` and expects::

        [
          { "userId": 2, "name": "...", "skills": [...], "matchScore": 85.5 }
        ]

    ``matchScore`` is a percentage (0–100), not a 0–1 float.
    """
    # Auto mode switch: request.users => REAL mode, otherwise DEMO mode.
    users, mode = _resolve_users_and_mode(body.users)
    logger.info("Running in %s mode", mode)

    with use_runtime_users(users):
        if mode == "REAL":
            EmbeddingCache.invalidate()
            EmbeddingCache.warm(users)

        try:
            raw_matches = get_matches(body.user_id, top_k=5)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    # Convert 0–1 similarity → 0–100 percentage and flatten.
    results = [
        V1MatchResult(
            user_id=m["user_id"],
            name=m["name"],
            skills=m["skills"],
            level=m.get("level", "Unknown"),
            goal=m.get("goal", "General"),
            match_score=round(m["similarity_score"] * 100, 1),
        )
        for m in raw_matches
    ]

    logger.info(
        "V1 match-users: user %d → %d matches", body.user_id, len(results)
    )
    return results


# ────────────────────────────────────────────────────────────────────
# 2. TRUST SCORE
# ────────────────────────────────────────────────────────────────────

@router.post(
    "/trust-score",
    response_model=V1TrustScoreResponse,
    response_model_by_alias=True,
    summary="Calculate trust score (Spring Boot contract)",
    responses={404: {"description": "User not found"}},
)
def v1_trust_score(body: V1TrustScoreRequest):
    """Return a clean trust-score JSON.

    Spring Boot calls with ``{"userId": 1}`` and expects::

        { "userId": 1, "trustScore": 78, "label": "Very Good" }
    """
    # Auto mode switch: request.users => REAL mode, otherwise DEMO mode.
    users, mode = _resolve_users_and_mode(body.users)
    logger.info("Running in %s mode", mode)

    with use_runtime_users(users):
        if mode == "REAL":
            EmbeddingCache.invalidate()
            EmbeddingCache.warm(users)

        try:
            result = calculate_trust_score(body.user_id)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    return V1TrustScoreResponse(
        user_id=result["user_id"],
        trust_score=result["trust_score"],
        label=result["label"],
    )


# ────────────────────────────────────────────────────────────────────
# 3. SUGGEST PROJECTS
# ────────────────────────────────────────────────────────────────────

@router.post(
    "/suggest-projects",
    response_model=list[V1ProjectSuggestion],
    response_model_by_alias=True,
    summary="Suggest projects by skill list (Spring Boot contract)",
)
def v1_suggest_projects(body: V1SuggestProjectsRequest):
    """Return project suggestions based on a raw skill list.

    Spring Boot calls with ``{"skills": ["React", "Java"]}`` and expects::

        [
          { "title": "Portfolio Website", "difficulty": "Beginner", ... }
        ]

    This does NOT require a userId — the Java backend extracts the
    user's skills from its own DB and sends them directly.
    """
    raw_projects = suggest_projects_by_skills(body.skills, top_k=5)

    results = [
        V1ProjectSuggestion(
            title=p["title"],
            description=p.get("description", ""),
            difficulty=p.get("difficulty", "Unknown"),
            required_skills=p.get("required_skills", []),
            match_score=round(p.get("match_score", 0.5) * 100, 1),
        )
        for p in raw_projects
    ]

    logger.info(
        "V1 suggest-projects: skills=%s → %d projects", body.skills, len(results)
    )
    return results


# ────────────────────────────────────────────────────────────────────
# 4. BUILD TEAM
# ────────────────────────────────────────────────────────────────────

@router.post(
    "/build-team",
    response_model=list[V1TeamMember],
    response_model_by_alias=True,
    summary="Build a balanced team (Spring Boot contract)",
    responses={404: {"description": "User not found"}},
)
def v1_build_team(body: V1BuildTeamRequest):
    """Return a flat array of team members.

    Spring Boot calls with ``{"userId": 1, "teamSize": 3}`` and expects::

        [
          { "userId": 1, "name": "...", "skills": [...], "role": "..." }
        ]
    """
    # Auto mode switch: request.users => REAL mode, otherwise DEMO mode.
    users, mode = _resolve_users_and_mode(body.users)
    logger.info("Running in %s mode", mode)

    with use_runtime_users(users):
        if mode == "REAL":
            EmbeddingCache.invalidate()
            EmbeddingCache.warm(users)

        try:
            raw_team = build_team(body.user_id, team_size=body.team_size)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    results = [
        V1TeamMember(
            user_id=m["user_id"],
            name=m["name"],
            skills=m["skills"],
            role=m["role"],
        )
        for m in raw_team
    ]

    logger.info(
        "V1 build-team: seed=%d, size=%d", body.user_id, len(results)
    )
    return results


# ────────────────────────────────────────────────────────────────────
# 5. RECOMMENDATIONS (people + projects combined)
# ────────────────────────────────────────────────────────────────────

@router.post(
    "/recommendations",
    response_model=V1RecommendationsResponse,
    response_model_by_alias=True,
    summary="Full hybrid recommendations (Spring Boot contract)",
    responses={404: {"description": "User not found"}},
)
def v1_recommendations(body: V1RecommendRequest):
    """Return people recommendations + project suggestions.

    Spring Boot calls with ``{"userId": 2}`` and expects::

        {
          "people": [ { "userId": ..., "matchScore": 85.2 } ],
          "projects": [ { "title": "...", "difficulty": "..." } ]
        }
    """
    # Auto mode switch: request.users => REAL mode, otherwise DEMO mode.
    users, mode = _resolve_users_and_mode(body.users)
    logger.info("Running in %s mode", mode)

    with use_runtime_users(users):
        if mode == "REAL":
            EmbeddingCache.invalidate()
            EmbeddingCache.warm(users)

        try:
            raw_people = get_recommendations(body.user_id, page=1, limit=10)
            raw_projects = suggest_projects(body.user_id, top_k=5)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    people = [
        V1RecommendationPerson(
            user_id=p["user_id"],
            name=p["name"],
            skills=p["skills"],
            match_score=round(p["recommendation_score"] * 100, 1),
            trending=p.get("trending", False),
        )
        for p in raw_people
    ]

    projects = [
        V1ProjectSuggestion(
            title=p["title"],
            description=p.get("description", ""),
            difficulty=p.get("difficulty", "Unknown"),
            required_skills=p.get("required_skills", []),
            match_score=round(p.get("match_score", 0.5) * 100, 1),
        )
        for p in raw_projects
    ]

    logger.info(
        "V1 recommendations: user %d → %d people, %d projects",
        body.user_id, len(people), len(projects),
    )
    return V1RecommendationsResponse(people=people, projects=projects)
