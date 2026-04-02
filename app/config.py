"""Application configuration.

Centralizes all tuneable settings. Values can be overridden via
environment variables in production; defaults are safe for local
development on a standard laptop (CPU-only).
"""

from __future__ import annotations

import os
from pathlib import Path


# ── Paths ──────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

USERS_JSON = DATA_DIR / "users.json"
PROJECTS_JSON = DATA_DIR / "projects.json"


class Settings:
    """Application-wide settings container."""

    # ── General ────────────────────────────────────────────────────
    app_name: str = "SkillSync AI"
    app_version: str = "2.0.0"
    debug: bool = os.getenv("APP_ENV", "development") == "development"

    # ── Model ──────────────────────────────────────────────────────
    embedding_model: str = "all-MiniLM-L6-v2"

    # ── Matching ───────────────────────────────────────────────────
    default_top_k: int = 5
    team_min_size: int = 3
    team_max_size: int = 5

    # ── Recommendation weights (must sum to 1.0) ───────────────────
    weight_skill_similarity: float = 0.40
    weight_level_match: float = 0.15
    weight_goal_match: float = 0.10
    weight_activity: float = 0.15
    weight_interest_similarity: float = 0.20  # NEW: semantic interest match

    # ── Project suggestion scoring ─────────────────────────────────
    project_jaccard_weight: float = 0.60
    project_semantic_weight: float = 0.40

    # ── Cold-start threshold (days since signup) ──────────────────
    cold_start_days: int = 14  # users with joined_days_ago < this

    # ── Pagination defaults ────────────────────────────────────────
    default_page: int = 1
    default_limit: int = 10
    max_limit: int = 50


settings = Settings()
