"""SkillSync AI — FastAPI application entry point.

Run with:
    uvicorn app.main:app --reload

Interactive docs:
    http://127.0.0.1:8000/docs     (Swagger UI)
    http://127.0.0.1:8000/redoc    (ReDoc)
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import Response

from app.api.match_routes import router as match_router
from app.api.recommend_routes import router as recommend_router
from app.api.v1_routes import router as v1_router
from app.background.tasks import precompute_embeddings
from app.config import settings
from app.core.logger import logger
from app.services.matching.vectorizer import EmbeddingCache


# ── Lifespan ────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Startup / shutdown lifecycle hook.

    On startup we pre-compute embeddings so the first request is fast.
    """
    logger.info("🚀 Starting %s v%s", settings.app_name, settings.app_version)
    try:
        stats = precompute_embeddings()
        logger.info(
            "✅ Embedding cache warmed — %d users in %.2fs",
            stats["users"],
            stats["elapsed_seconds"],
        )
    except Exception as exc:
        logger.warning("⚠️  Embedding warm-up failed: %s — will compute on first request", exc)
    yield
    logger.info("👋 Shutting down %s", settings.app_name)


# ── App factory ─────────────────────────────────────────────────────

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description=(
        "AI-powered skill matching and recommendation platform.\n\n"
        "**Architecture:** React → Spring Boot (8080) → FastAPI AI Engine (8000)\n\n"
        "**V1 API** (`/api/v1/*`): Spring Boot integration layer (camelCase JSON)\n"
        "**Internal API** (`/api/*`): Direct endpoints (snake_case JSON)\n\n"
        "**Features:**\n"
        "- Semantic user matching via SentenceTransformers\n"
        "- 5-signal hybrid recommendation engine\n"
        "- Interest-based similarity scoring\n"
        "- Greedy team clustering with role assignment\n"
        "- Semantic project suggestions\n"
        "- Cold-start handling for new users\n"
        "- Trust score calculation\n\n"
        "All models run on CPU — no GPU required."
    ),
    lifespan=lifespan,
)


# ── Middleware ──────────────────────────────────────────────────────

# CORS (allow everything for hackathon / dev).
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_process_time_header(request: Request, call_next) -> Response:
    """Add ``X-Process-Time`` header (milliseconds) to every response.

    Impressive for demos — shows real latency in the response headers.
    """
    start = time.perf_counter()
    response: Response = await call_next(request)
    elapsed_ms = round((time.perf_counter() - start) * 1000, 2)
    response.headers["X-Process-Time"] = f"{elapsed_ms}ms"
    return response


# ── Mount routers ──────────────────────────────────────────────────
# V1 routes — called by Spring Boot via RestTemplate (camelCase JSON).
app.include_router(v1_router)
# Internal routes — direct access / backward compatibility.
app.include_router(match_router, prefix="/api")
app.include_router(recommend_router, prefix="/api")

# Compatibility aliases without the /api prefix.
app.include_router(match_router, include_in_schema=False)
app.include_router(recommend_router, include_in_schema=False)


# ── Root endpoints ─────────────────────────────────────────────────

@app.get("/", tags=["meta"])
def root() -> dict:
    """Landing endpoint — confirms the API is live."""
    return {
        "app": settings.app_name,
        "version": settings.app_version,
        "docs": "/docs",
        "status": "running",
    }


@app.get("/health", tags=["meta"])
def health_check() -> dict:
    """Health-check endpoint for uptime monitors."""
    cache_stats = EmbeddingCache.stats()
    return {
        "status": "healthy",
        "embedding_cache": cache_stats,
    }


@app.get("/stats", tags=["meta"])
def system_stats() -> dict:
    """System statistics — embedding cache state and config summary."""
    cache_stats = EmbeddingCache.stats()
    if not cache_stats["cached"]:
        # Keep /stats deterministic in environments where lifespan may not run.
        EmbeddingCache.get()
        cache_stats = EmbeddingCache.stats()

    return {
        "app": settings.app_name,
        "version": settings.app_version,
        "embedding_model": settings.embedding_model,
        "cache": cache_stats,
        "recommendation_weights": {
            "skill_similarity": settings.weight_skill_similarity,
            "level_match": settings.weight_level_match,
            "goal_match": settings.weight_goal_match,
            "activity": settings.weight_activity,
            "interest_similarity": settings.weight_interest_similarity,
        },
    }
