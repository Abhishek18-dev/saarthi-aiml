"""Prompt templates for future LLM integration.

When an LLM API (OpenAI, Gemini, local Ollama, etc.) is connected,
these templates turn raw user data into well-structured prompts. For
now they are pure string formatters — no API calls are made.
"""

from __future__ import annotations

from typing import Any


def match_explanation_prompt(
    user: dict[str, Any],
    match: dict[str, Any],
    score: float,
) -> str:
    """Generate a prompt that asks an LLM to explain *why* two users
    are a good match.

    Example output (the prompt text itself, not the LLM response):
        "Explain why User A (skills: React, JS) with goal 'Project'
         is a good match for User B (skills: React, Node.js) with
         goal 'Project'. Similarity score: 0.92."
    """
    return (
        f"Explain in 2-3 sentences why {user.get('name', 'User A')} "
        f"(skills: {', '.join(user.get('skills', []))}, "
        f"level: {user.get('level')}, goal: {user.get('goal')}) "
        f"is a good match for {match.get('name', 'User B')} "
        f"(skills: {', '.join(match.get('skills', []))}, "
        f"level: {match.get('level')}, goal: {match.get('goal')}). "
        f"Similarity score: {score:.2f}."
    )


def project_recommendation_prompt(
    user: dict[str, Any],
    projects: list[dict[str, Any]],
) -> str:
    """Generate a prompt that asks an LLM to rank / describe project
    suggestions for the user."""
    project_list = "\n".join(
        f"  - {p.get('title', 'Untitled')}: {p.get('description', '')}"
        for p in projects
    )
    return (
        f"Given a developer named {user.get('name', 'Unknown')} with "
        f"skills [{', '.join(user.get('skills', []))}] at "
        f"{user.get('level', 'unknown')} level aiming for "
        f"{user.get('goal', 'general growth')}, rank and briefly "
        f"describe why each of these projects would be a great fit:\n"
        f"{project_list}"
    )


def team_summary_prompt(team: list[dict[str, Any]]) -> str:
    """Generate a prompt for an LLM to produce a team summary."""
    members = "\n".join(
        f"  - {m.get('name', 'Member')}: "
        f"skills={', '.join(m.get('skills', []))}, role={m.get('role', 'TBD')}"
        for m in team
    )
    return (
        f"Write a short team profile for the following members:\n"
        f"{members}\n"
        f"Highlight complementary skills and suggest a team name."
    )
