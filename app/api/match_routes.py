"""Matching API routes.

Endpoints
---------
POST /api/match-users   — Find semantically similar users.
POST /api/build-team    — Build a team around a seed user.
POST /api/build-teams   — Auto-partition all users into teams.
"""

from __future__ import annotations

import time

from fastapi import APIRouter, BackgroundTasks, HTTPException
from fastapi.responses import JSONResponse

from app.schemas.match_schema import (
    MatchRequest,
    MatchResponse,
    TeamRequest,
    TeamResponse,
)
from app.services.matching.matcher import get_matches
from app.services.matching.team_builder import build_team, build_multiple_teams
from app.background.tasks import log_match_event

router = APIRouter(tags=["matching"])


@router.post(
    "/match-users",
    response_model=MatchResponse,
    summary="Find top skill-based matches for a user",
    responses={404: {"description": "User not found"}},
)
def match_users(
    body: MatchRequest,
    bg: BackgroundTasks,
) -> MatchResponse:
    """Return the top-K semantically similar users.

    Uses SentenceTransformer embeddings + cosine similarity + semantic
    skill group overlap.
    """
    start = time.perf_counter()
    try:
        matches = get_matches(body.user_id, top_k=body.top_k)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    # Fire-and-forget analytics.
    bg.add_task(log_match_event, body.user_id, matches)

    response = MatchResponse(
        user_id=body.user_id,
        matches=matches,  # type: ignore[arg-type]
        total=len(matches),
    )

    return response


@router.post(
    "/build-team",
    response_model=TeamResponse,
    summary="Build a balanced team around a seed user",
    responses={404: {"description": "User not found"}},
)
def api_build_team(body: TeamRequest) -> TeamResponse:
    """Form a team using greedy similarity-based clustering."""
    try:
        team = build_team(body.user_id, team_size=body.team_size)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    return TeamResponse(
        seed_user_id=body.user_id,
        team=team,  # type: ignore[arg-type]
        team_size=len(team),
    )


@router.post(
    "/build-teams",
    summary="Auto-partition all users into balanced teams",
)
def api_build_teams(team_size: int = 3) -> dict:
    """Partition the entire user pool into teams."""
    teams = build_multiple_teams(team_size=team_size)
    return {
        "teams": teams,
        "total_teams": len(teams),
        "team_size_requested": team_size,
    }
