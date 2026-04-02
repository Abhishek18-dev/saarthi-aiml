"""V1 API schemas — camelCase JSON contracts for Spring Boot.

These schemas use Pydantic's ``alias`` feature so the Python code
uses snake_case internally, but the JSON serialised over HTTP uses
camelCase — exactly what Java's Jackson mapper expects.

Every response schema uses ``model_config = {"populate_by_name": True}``
so that both snake_case and camelCase are accepted on input, and
camelCase is always used on output (``by_alias=True`` in the route
``response_model_by_alias``).
"""

from __future__ import annotations

from pydantic import BaseModel, Field


# ── Requests ────────────────────────────────────────────────────────

class V1UserActivity(BaseModel):
    """Optional activity payload for runtime users."""

    days_active: int = Field(default=0, alias="daysActive")
    projects_joined: int = Field(default=0, alias="projectsJoined")
    contributions: int = 0
    last_active_days_ago: int = Field(default=30, alias="lastActiveDaysAgo")

    model_config = {"populate_by_name": True}


class V1User(BaseModel):
    """Runtime user payload accepted from Spring Boot in REAL mode."""

    id: int = Field(..., alias="userId")
    name: str = "Unknown"
    skills: list[str] = Field(default_factory=list)
    level: str = "Beginner"
    goal: str = "General"
    bio: str = ""
    interests: list[str] = Field(default_factory=list)
    activity: V1UserActivity = Field(default_factory=V1UserActivity)
    ratings: list[float] = Field(default_factory=list)
    joined_days_ago: int = Field(default=0, alias="joinedDaysAgo")

    model_config = {"populate_by_name": True}

class V1MatchRequest(BaseModel):
    """POST /api/v1/match-users — input from Spring Boot."""

    user_id: int = Field(..., alias="userId", description="User ID to match")
    users: list[V1User] | None = Field(
        default=None,
        description="Optional runtime users; when present, REAL mode is used",
    )

    model_config = {"populate_by_name": True}


class V1TrustScoreRequest(BaseModel):
    """POST /api/v1/trust-score — input from Spring Boot."""

    user_id: int = Field(..., alias="userId")
    users: list[V1User] | None = Field(
        default=None,
        description="Optional runtime users; when present, REAL mode is used",
    )

    model_config = {"populate_by_name": True}


class V1SuggestProjectsRequest(BaseModel):
    """POST /api/v1/suggest-projects — input from Spring Boot.

    Note: this takes a *skills list*, not a userId.
    """

    skills: list[str] = Field(..., description="List of skill strings")


class V1BuildTeamRequest(BaseModel):
    """POST /api/v1/build-team — input from Spring Boot."""

    user_id: int = Field(..., alias="userId")
    team_size: int = Field(default=3, ge=2, le=10, alias="teamSize")
    users: list[V1User] | None = Field(
        default=None,
        description="Optional runtime users; when present, REAL mode is used",
    )

    model_config = {"populate_by_name": True}


class V1RecommendRequest(BaseModel):
    """POST /api/v1/recommendations — input from Spring Boot."""

    user_id: int = Field(..., alias="userId")
    users: list[V1User] | None = Field(
        default=None,
        description="Optional runtime users; when present, REAL mode is used",
    )

    model_config = {"populate_by_name": True}


# ── Responses ───────────────────────────────────────────────────────

class V1MatchResult(BaseModel):
    """Single match entry — flat, clean JSON for Java/React."""

    user_id: int = Field(..., alias="userId")
    name: str
    skills: list[str]
    level: str
    goal: str
    match_score: float = Field(..., alias="matchScore")

    model_config = {"populate_by_name": True}


class V1TrustScoreResponse(BaseModel):
    """Trust score — minimal response for Spring Boot."""

    user_id: int = Field(..., alias="userId")
    trust_score: int = Field(..., alias="trustScore")
    label: str

    model_config = {"populate_by_name": True}


class V1ProjectSuggestion(BaseModel):
    """Single project suggestion — flat response."""

    title: str
    description: str
    difficulty: str
    required_skills: list[str] = Field(default_factory=list, alias="requiredSkills")
    match_score: float = Field(0.0, alias="matchScore")

    model_config = {"populate_by_name": True}


class V1TeamMember(BaseModel):
    """Single team member — minimal for Spring Boot."""

    user_id: int = Field(..., alias="userId")
    name: str
    skills: list[str]
    role: str

    model_config = {"populate_by_name": True}


class V1RecommendationPerson(BaseModel):
    """Single recommended person for the recommendations endpoint."""

    user_id: int = Field(..., alias="userId")
    name: str
    skills: list[str]
    match_score: float = Field(..., alias="matchScore")
    trending: bool = False

    model_config = {"populate_by_name": True}


class V1RecommendationsResponse(BaseModel):
    """Full recommendations response — people + projects."""

    people: list[V1RecommendationPerson]
    projects: list[V1ProjectSuggestion]

    model_config = {"populate_by_name": True}
