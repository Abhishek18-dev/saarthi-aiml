"""Matching request / response schemas."""

from __future__ import annotations

from pydantic import BaseModel, Field


class MatchRequest(BaseModel):
    """POST body for ``/match-users``."""

    user_id: int = Field(..., description="ID of the user to find matches for")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of results")

    model_config = {"json_schema_extra": {"examples": [{"user_id": 1, "top_k": 5}]}}


class MatchResult(BaseModel):
    """Single match entry in the response."""

    user_id: int
    name: str
    skills: list[str]
    level: str
    goal: str
    similarity_score: float
    shared_skill_groups: float = 0.0


class MatchResponse(BaseModel):
    """Response wrapper for ``/match-users``."""

    user_id: int
    matches: list[MatchResult]
    total: int


class TeamRequest(BaseModel):
    """POST body for ``/build-team``."""

    user_id: int = Field(..., description="Seed user for the team")
    team_size: int = Field(default=3, ge=3, le=5, description="Desired team size")

    model_config = {"json_schema_extra": {"examples": [{"user_id": 1, "team_size": 4}]}}


class TeamMember(BaseModel):
    """Single team member entry."""

    user_id: int
    name: str
    skills: list[str]
    level: str
    role: str


class TeamResponse(BaseModel):
    """Response wrapper for ``/build-team``."""

    seed_user_id: int
    team: list[TeamMember]
    team_size: int
