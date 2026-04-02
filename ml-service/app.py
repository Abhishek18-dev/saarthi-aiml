"""Gradio wrapper for deploying SkillSync matching on Hugging Face Spaces.

This file intentionally wraps existing project logic without changing it.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import gradio as gr


# Ensure imports resolve to the existing project package at repository root.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.services.matching.matcher import get_matches
from app.services.matching.vectorizer import EmbeddingCache


# Load embeddings/model once at startup for warm inference.
EmbeddingCache.get()


def run_matching(user_id: float | int, users_json: str = "") -> dict[str, Any]:
    """Run the existing matching pipeline and return JSON-safe output."""
    if user_id is None:
        return {"ok": False, "error": "userId is required"}

    parsed_users: list[dict[str, Any]] | None = None
    if users_json and users_json.strip():
        try:
            parsed = json.loads(users_json)
            if not isinstance(parsed, list):
                return {
                    "ok": False,
                    "error": "users JSON must be a list of user objects",
                }
            parsed_users = parsed
        except json.JSONDecodeError as exc:
            return {"ok": False, "error": f"invalid users JSON: {exc.msg}"}

    try:
        matches = get_matches(int(user_id))
        response: dict[str, Any] = {
            "ok": True,
            "userId": int(user_id),
            "matches": matches,
        }
        if parsed_users is not None:
            response["usersInputNote"] = (
                "users input received by wrapper; core matcher still uses existing internal data source"
            )
            response["usersInputCount"] = len(parsed_users)
        return response
    except Exception as exc:  # pragma: no cover
        return {"ok": False, "error": str(exc)}


demo = gr.Interface(
    fn=run_matching,
    title="SkillSync ML Engine",
    description="Enter a userId to get top match results from the existing ML pipeline.",
    inputs=[
        gr.Number(label="userId", precision=0),
        gr.Textbox(
            label="users (optional JSON list)",
            lines=10,
            placeholder='[{"id": 1, "name": "Alice", "skills": ["Python"]}]',
        ),
    ],
    outputs=gr.JSON(label="Match Results"),
    allow_flagging="never",
)


if __name__ == "__main__":
    demo.launch()
