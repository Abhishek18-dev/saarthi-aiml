"""Common utility functions used by multiple services.

Handles JSON-based data loading/caching so services never need to
worry about file I/O directly.

Safety: ``load_users`` and ``load_projects`` return *copies* of the
cached data so callers cannot accidentally mutate the shared cache.
"""

from __future__ import annotations

import copy
import json
from contextlib import contextmanager
from contextvars import ContextVar
from functools import lru_cache
from typing import Any

from app.config import USERS_JSON, PROJECTS_JSON
from app.core.logger import logger


# ── Data loaders ────────────────────────────────────────────────────

_runtime_users: ContextVar[tuple[dict[str, Any], ...] | None] = ContextVar(
    "runtime_users",
    default=None,
)


@contextmanager
def use_runtime_users(
    users: list[dict[str, Any]] | tuple[dict[str, Any], ...],
):
    """Temporarily override user data for the current request context."""
    frozen_users = tuple(copy.copy(u) for u in users)
    token = _runtime_users.set(frozen_users)
    try:
        yield
    finally:
        _runtime_users.reset(token)


def _get_runtime_users() -> tuple[dict[str, Any], ...] | None:
    """Return request-scoped runtime users when REAL mode is active."""
    return _runtime_users.get()

@lru_cache(maxsize=1)
def _load_users_raw() -> tuple[dict[str, Any], ...]:
    """Internal: load users from disk and freeze as a tuple.

    Using a tuple prevents accidental mutation of the cached object.
    """
    logger.info("Loading users from %s", USERS_JSON)
    with open(USERS_JSON, "r", encoding="utf-8") as fh:
        users: list[dict[str, Any]] = json.load(fh)
    logger.info("Loaded %d users", len(users))
    return tuple(users)


def load_users() -> list[dict[str, Any]]:
    """Return a fresh list copy of all users (safe to mutate)."""
    runtime_users = _get_runtime_users()
    if runtime_users is not None:
        return [copy.copy(u) for u in runtime_users]
    return [copy.copy(u) for u in _load_users_raw()]


def load_users_frozen() -> tuple[dict[str, Any], ...]:
    """Return the cached tuple directly (fast, read-only access)."""
    runtime_users = _get_runtime_users()
    if runtime_users is not None:
        return runtime_users
    return _load_users_raw()


@lru_cache(maxsize=1)
def _load_projects_raw() -> tuple[dict[str, Any], ...]:
    """Internal: load projects from disk and freeze as a tuple."""
    logger.info("Loading projects from %s", PROJECTS_JSON)
    with open(PROJECTS_JSON, "r", encoding="utf-8") as fh:
        projects: list[dict[str, Any]] = json.load(fh)
    logger.info("Loaded %d projects", len(projects))
    return tuple(projects)


def load_projects() -> list[dict[str, Any]]:
    """Return a fresh list copy of all projects."""
    return [copy.copy(p) for p in _load_projects_raw()]


def load_projects_frozen() -> tuple[dict[str, Any], ...]:
    """Return the cached tuple directly (fast, read-only access)."""
    return _load_projects_raw()


# ── Index helpers ──────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _build_user_index_cached() -> dict[int, int]:
    """Return a dict mapping user_id → index in the users list.

    Cached for the lifetime of the data.  O(1) look-up instead of
    O(N) linear scan every time.
    """
    return {u["id"]: idx for idx, u in enumerate(_load_users_raw())}


def build_user_index() -> dict[int, int]:
    """Return user_id → index map for current mode data source."""
    runtime_users = _get_runtime_users()
    if runtime_users is not None:
        return {u["id"]: idx for idx, u in enumerate(runtime_users)}
    return _build_user_index_cached()


def get_user_by_id(user_id: int) -> dict[str, Any] | None:
    """Retrieve a single user by their integer ID.

    Uses the index map for O(1) look-up.
    """
    users = load_users_frozen()
    idx = build_user_index().get(user_id)
    if idx is None:
        return None
    return dict(users[idx])  # shallow copy


def invalidate_caches() -> None:
    """Clear all cached data — useful after a data refresh."""
    _load_users_raw.cache_clear()
    _load_projects_raw.cache_clear()
    _build_user_index_cached.cache_clear()

    # Also invalidate the embedding cache.
    from app.services.matching.vectorizer import EmbeddingCache
    EmbeddingCache.invalidate()

    logger.info("All data & embedding caches invalidated")
