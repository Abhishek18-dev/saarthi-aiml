"""Vectorizer — converts user profiles to dense semantic embeddings.

Key optimisations (v2)
----------------------
1. **EmbeddingCache** — singleton that stores pre-computed embeddings
   and the similarity matrix.  Re-used across matcher, recommender,
   and team builder so the SentenceTransformer forward pass runs
   *once* (or after an explicit invalidation), not per request.

2. **Lazy model loading** — ``_get_model()`` loads the model once at
   first use.  The ``all-MiniLM-L6-v2`` model is ~80 MB, runs in
   ~5-15 ms per user on CPU.

3. **Interest embedding** — NEW: separate embedding for user interests
   so the recommender can compute interest-based similarity.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from app.config import settings
from app.core.logger import logger


# ── Global model singleton ──────────────────────────────────────────
_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    """Lazy-load the SentenceTransformer model (once per process)."""
    global _model
    if _model is None:
        logger.info("Loading embedding model '%s' …", settings.embedding_model)
        start = time.perf_counter()
        _model = SentenceTransformer(settings.embedding_model, device="cpu")
        elapsed = time.perf_counter() - start
        logger.info("Embedding model loaded in %.2fs", elapsed)
    return _model


# ── Profile → text ──────────────────────────────────────────────────

def user_to_text(user: dict[str, Any]) -> str:
    """Convert a user profile dict into a rich descriptive string.

    The string deliberately interleaves skills, level, goal, bio, and
    interests so the language model captures both *keyword* and
    *contextual* meaning.
    """
    skills = ", ".join(user.get("skills", []))
    level = user.get("level", "Unknown")
    goal = user.get("goal", "General")
    bio = user.get("bio", "")
    interests = ", ".join(user.get("interests", []))

    parts: list[str] = [
        f"Skills: {skills}",
        f"Level: {level}",
        f"Goal: {goal}",
    ]
    if bio:
        parts.append(f"Bio: {bio}")
    if interests:
        parts.append(f"Interests: {interests}")

    return ". ".join(parts) + "."


def user_interests_text(user: dict[str, Any]) -> str:
    """Extract just the interest/goal signal for interest-based matching."""
    interests = ", ".join(user.get("interests", []))
    goal = user.get("goal", "General")
    bio = user.get("bio", "")
    return f"Goal: {goal}. Interests: {interests}. {bio}".strip()


def project_to_text(project: dict[str, Any]) -> str:
    """Convert a project dict into a descriptive string for embedding."""
    skills = ", ".join(project.get("required_skills", []))
    title = project.get("title", "")
    desc = project.get("description", "")
    category = project.get("category", "")
    return f"{title}. {desc}. Skills: {skills}. Category: {category}."


# ── Embedding functions ─────────────────────────────────────────────

def embed_text(text: str) -> np.ndarray:
    """Return the embedding vector for a single text string.

    Returns a 1-D numpy array of shape ``(384,)`` for MiniLM-L6-v2.
    """
    model = _get_model()
    return model.encode(text, convert_to_numpy=True)  # type: ignore[return-value]


def embed_texts(texts: list[str]) -> np.ndarray:
    """Batch-embed multiple texts.

    Returns a 2-D numpy array of shape ``(N, 384)``.
    """
    model = _get_model()
    return model.encode(texts, convert_to_numpy=True, show_progress_bar=False)  # type: ignore[return-value]


def embed_user(user: dict[str, Any]) -> np.ndarray:
    """Convenience: profile dict → embedding vector."""
    return embed_text(user_to_text(user))


def embed_users(users: list[dict[str, Any]] | tuple[dict[str, Any], ...]) -> np.ndarray:
    """Batch-embed a list of user profiles."""
    texts = [user_to_text(u) for u in users]
    return embed_texts(texts)


# ── Embedding Cache (the big performance win) ──────────────────────

class EmbeddingCache:
    """Process-global cache for user embeddings and similarity matrix.

    Instead of re-computing embeddings on every request, all services
    call ``EmbeddingCache.get()`` which returns the cached embeddings
    and similarity matrix.  The cache is populated at startup by
    ``warm()`` and can be invalidated when data changes.
    """

    _embeddings: np.ndarray | None = None
    _interest_embeddings: np.ndarray | None = None
    _sim_matrix: np.ndarray | None = None
    _interest_sim_matrix: np.ndarray | None = None
    _user_count: int = 0
    _warm_time: float = 0.0

    @classmethod
    def warm(cls, users: list[dict[str, Any]] | tuple[dict[str, Any], ...]) -> dict[str, Any]:
        """Pre-compute and cache all embeddings + similarity matrices.

        Returns a stats dict for logging / health endpoints.
        """
        start = time.perf_counter()

        # Profile embeddings (full text).
        cls._embeddings = embed_users(users)
        cls._sim_matrix = cosine_similarity(cls._embeddings)

        # Interest embeddings (goals + interests + bio only).
        interest_texts = [user_interests_text(u) for u in users]
        cls._interest_embeddings = embed_texts(interest_texts)
        cls._interest_sim_matrix = cosine_similarity(cls._interest_embeddings)

        cls._user_count = len(users)
        cls._warm_time = time.perf_counter() - start

        logger.info(
            "EmbeddingCache warmed: %d users, shape=%s, %.2fs",
            cls._user_count,
            cls._embeddings.shape,
            cls._warm_time,
        )

        return {
            "users": cls._user_count,
            "embedding_shape": list(cls._embeddings.shape),
            "interest_embedding_shape": list(cls._interest_embeddings.shape),
            "sim_matrix_shape": list(cls._sim_matrix.shape),
            "elapsed_seconds": round(cls._warm_time, 3),
        }

    @classmethod
    def get(cls) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(embeddings, sim_matrix)``.

        Auto-warms if the cache is cold (first request).
        """
        from app.services.common.utils import load_users_frozen

        users = load_users_frozen()
        if (
            cls._embeddings is None
            or cls._sim_matrix is None
            or cls._user_count != len(users)
        ):
            cls.warm(users)
        return cls._embeddings, cls._sim_matrix  # type: ignore[return-value]

    @classmethod
    def get_interest_sim(cls) -> np.ndarray:
        """Return the interest-based similarity matrix.

        Auto-warms if the cache is cold.
        """
        from app.services.common.utils import load_users_frozen

        users = load_users_frozen()
        if cls._interest_sim_matrix is None or cls._user_count != len(users):
            cls.warm(users)
        return cls._interest_sim_matrix  # type: ignore[return-value]

    @classmethod
    def get_embeddings(cls) -> np.ndarray:
        """Return the raw user embeddings (for project matching etc.)."""
        embeddings, _ = cls.get()
        return embeddings

    @classmethod
    def invalidate(cls) -> None:
        """Clear the cache so next ``get()`` call recomputes."""
        cls._embeddings = None
        cls._interest_embeddings = None
        cls._sim_matrix = None
        cls._interest_sim_matrix = None
        cls._user_count = 0
        logger.info("EmbeddingCache invalidated")

    @classmethod
    def stats(cls) -> dict[str, Any]:
        """Return cache statistics for health/debug endpoints."""
        return {
            "cached": cls._embeddings is not None,
            "user_count": cls._user_count,
            "warm_time_seconds": round(cls._warm_time, 3),
        }
