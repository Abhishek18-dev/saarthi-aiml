"""Recommendation request / response schemas."""

from __future__ import annotations

from pydantic import BaseModel, Field


class RecommendRequest(BaseModel):
    """POST body for ``/recommend``."""

    user_id: int = Field(..., description="Target user ID")
    page: int = Field(default=1, ge=1, description="Page number (1-indexed)")
    limit: int = Field(default=10, ge=1, le=50, description="Results per page")

    model_config = {
        "json_schema_extra": {
            "examples": [{"user_id": 2, "page": 1, "limit": 5}]
        }
    }


class PersonRecommendation(BaseModel):
    """Single recommended person entry."""

    user_id: int
    name: str
    skills: list[str]
    level: str
    goal: str
    recommendation_score: float
    breakdown: dict[str, float] | None = None
    trending: bool = False


class ProjectSuggestion(BaseModel):
    """Single project suggestion entry."""

    title: str
    description: str
    required_skills: list[str]
    difficulty: str
    category: str
    match_score: float
    source: str


class RecommendResponse(BaseModel):
    """Response wrapper for ``/recommend``."""

    user_id: int
    page: int
    limit: int
    people: list[PersonRecommendation]
    projects: list[ProjectSuggestion]


class TrustScoreResponse(BaseModel):
    """Response for ``/trust-score``."""

    user_id: int
    trust_score: int
    activity_component: float
    rating_component: float
    label: str
