"""Recommendation API routes.

Endpoints
---------
POST /api/recommend       — Hybrid people + project recommendations.
GET  /api/trust-score     — Trust score for a given user.
POST /api/suggest-projects — Project suggestions for a user.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from app.schemas.recommend_schema import (
    RecommendRequest,
    RecommendResponse,
    TrustScoreResponse,
)
from app.services.recommendation.recommender import get_recommendations
from app.services.recommendation.project_suggester import suggest_projects
from app.services.recommendation.trust_score import calculate_trust_score

router = APIRouter(tags=["recommendation"])


@router.post(
    "/recommend",
    response_model=RecommendResponse,
    summary="Get hybrid people + project recommendations",
    responses={404: {"description": "User not found"}},
)
def recommend(body: RecommendRequest) -> RecommendResponse:
    """Return paginated people recommendations and project suggestions.

    The people list uses a 5-signal hybrid scoring formula blending
    skill similarity, level match, goal compatibility, activity, and
    interest similarity.  Projects are ranked by a Jaccard + semantic
    combined score.
    """
    try:
        people = get_recommendations(
            body.user_id, page=body.page, limit=body.limit
        )
        projects = suggest_projects(body.user_id, top_k=body.limit)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    return RecommendResponse(
        user_id=body.user_id,
        page=body.page,
        limit=body.limit,
        people=people,  # type: ignore[arg-type]
        projects=projects,  # type: ignore[arg-type]
    )


@router.get(
    "/trust-score",
    response_model=TrustScoreResponse,
    summary="Calculate trust score for a user",
    responses={404: {"description": "User not found"}},
)
def trust_score(
    user_id: int = Query(..., description="User ID to compute trust score for"),
) -> TrustScoreResponse:
    """Return a 0-100 trust score based on activity + ratings."""
    try:
        result = calculate_trust_score(user_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    return TrustScoreResponse(**result)


@router.post(
    "/suggest-projects",
    summary="Get project suggestions tailored to a user's skills",
    responses={404: {"description": "User not found"}},
)
def api_suggest_projects(
    user_id: int,
    top_k: int = Query(default=5, ge=1, le=20),
) -> dict:
    """Return project suggestions using Jaccard + semantic matching."""
    try:
        projects = suggest_projects(user_id, top_k=top_k)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    return {
        "user_id": user_id,
        "projects": projects,
        "total": len(projects),
    }
