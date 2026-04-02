"""User profile schema."""

from __future__ import annotations

from pydantic import BaseModel, Field


class UserProfile(BaseModel):
    """Full user profile as stored in the mock database."""

    id: int
    name: str = "Unknown"
    skills: list[str] = Field(default_factory=list)
    level: str = "Beginner"
    goal: str = "General"
    bio: str = ""
    interests: list[str] = Field(default_factory=list)
