"""Integration tests for SkillSync AI v2.

Run with:
    pytest tests/ -v
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


# ── Health / root ──────────────────────────────────────────────────

class TestRootEndpoints:
    def test_root(self) -> None:
        resp = client.get("/")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "running"
        assert data["version"] == "2.0.0"

    def test_health(self) -> None:
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        # Embedding cache should be warmed by lifespan.
        assert "embedding_cache" in data

    def test_stats(self) -> None:
        resp = client.get("/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert "cache" in data
        assert "recommendation_weights" in data
        assert data["cache"]["cached"] is True

    def test_response_has_timing_header(self) -> None:
        resp = client.get("/health")
        assert "X-Process-Time" in resp.headers


# ── Match endpoint ─────────────────────────────────────────────────

class TestMatchEndpoint:
    def test_match_valid_user(self) -> None:
        resp = client.post("/api/match-users", json={"user_id": 1})
        assert resp.status_code == 200
        data = resp.json()
        assert data["user_id"] == 1
        assert len(data["matches"]) > 0
        # Self should not appear in results.
        for m in data["matches"]:
            assert m["user_id"] != 1
            # v2: shared_skill_groups is present.
            assert "shared_skill_groups" in m

    def test_match_custom_top_k(self) -> None:
        resp = client.post(
            "/api/match-users", json={"user_id": 2, "top_k": 3}
        )
        assert resp.status_code == 200
        assert len(resp.json()["matches"]) <= 3

    def test_match_invalid_user(self) -> None:
        resp = client.post("/api/match-users", json={"user_id": 9999})
        assert resp.status_code == 404

    def test_match_scores_are_sorted(self) -> None:
        resp = client.post("/api/match-users", json={"user_id": 1, "top_k": 5})
        matches = resp.json()["matches"]
        scores = [m["similarity_score"] for m in matches]
        assert scores == sorted(scores, reverse=True)


# ── Team endpoint ──────────────────────────────────────────────────

class TestTeamEndpoint:
    def test_build_team_default(self) -> None:
        resp = client.post("/api/build-team", json={"user_id": 1})
        assert resp.status_code == 200
        data = resp.json()
        assert data["seed_user_id"] == 1
        assert len(data["team"]) >= 3

    def test_build_team_custom_size(self) -> None:
        resp = client.post(
            "/api/build-team", json={"user_id": 2, "team_size": 5}
        )
        assert resp.status_code == 200
        assert len(resp.json()["team"]) <= 5

    def test_build_team_seed_is_first_member(self) -> None:
        resp = client.post("/api/build-team", json={"user_id": 1})
        team = resp.json()["team"]
        assert team[0]["user_id"] == 1

    def test_build_teams_all(self) -> None:
        resp = client.post("/api/build-teams?team_size=3")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_teams"] >= 1

    def test_build_team_invalid_user(self) -> None:
        resp = client.post("/api/build-team", json={"user_id": 9999})
        assert resp.status_code == 404


# ── Recommend endpoint ─────────────────────────────────────────────

class TestRecommendEndpoint:
    def test_recommend_valid(self) -> None:
        resp = client.post(
            "/api/recommend", json={"user_id": 2, "page": 1, "limit": 5}
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["user_id"] == 2
        assert len(data["people"]) > 0
        assert len(data["projects"]) > 0
        # v2: breakdown includes interest_similarity.
        for person in data["people"]:
            if person.get("breakdown"):
                assert "interest_similarity" in person["breakdown"]

    def test_recommend_cold_start(self) -> None:
        """User 8 joined 10 days ago (< cold_start_days=14) — trending."""
        resp = client.post(
            "/api/recommend", json={"user_id": 8, "page": 1, "limit": 5}
        )
        assert resp.status_code == 200
        data = resp.json()
        if data["people"]:
            assert data["people"][0].get("trending") is True

    def test_recommend_boundary_user_not_cold_start(self) -> None:
        """User 10 joined exactly 14 days ago — should NOT be cold-start
        (cold-start is strictly < 14)."""
        resp = client.post(
            "/api/recommend", json={"user_id": 10, "page": 1, "limit": 5}
        )
        assert resp.status_code == 200
        data = resp.json()
        # User 10 gets real recommendations, not trending.
        for person in data["people"]:
            assert person.get("trending") is not True

    def test_recommend_scores_are_sorted(self) -> None:
        resp = client.post(
            "/api/recommend", json={"user_id": 2, "page": 1, "limit": 9}
        )
        people = resp.json()["people"]
        scores = [p["recommendation_score"] for p in people]
        assert scores == sorted(scores, reverse=True)

    def test_recommend_invalid_user(self) -> None:
        resp = client.post(
            "/api/recommend", json={"user_id": 9999}
        )
        assert resp.status_code == 404

    def test_recommend_pagination(self) -> None:
        page1 = client.post(
            "/api/recommend", json={"user_id": 2, "page": 1, "limit": 3}
        ).json()
        page2 = client.post(
            "/api/recommend", json={"user_id": 2, "page": 2, "limit": 3}
        ).json()
        # Pages should return different users (if enough users exist).
        ids_1 = {p["user_id"] for p in page1["people"]}
        ids_2 = {p["user_id"] for p in page2["people"]}
        assert ids_1.isdisjoint(ids_2) or len(page2["people"]) == 0


# ── Trust score endpoint ──────────────────────────────────────────

class TestTrustScore:
    def test_trust_score_valid(self) -> None:
        resp = client.get("/api/trust-score?user_id=4")
        assert resp.status_code == 200
        data = resp.json()
        assert 0 <= data["trust_score"] <= 100
        assert data["label"] in [
            "Excellent", "Very Good", "Good", "Fair", "New / Low Activity"
        ]

    def test_trust_score_high_activity_user(self) -> None:
        """User 4 (Sneha) is very active — should score high."""
        resp = client.get("/api/trust-score?user_id=4")
        data = resp.json()
        assert data["trust_score"] >= 75

    def test_trust_score_invalid(self) -> None:
        resp = client.get("/api/trust-score?user_id=9999")
        assert resp.status_code == 404


# ── Project suggestion endpoint ───────────────────────────────────

class TestProjectSuggestion:
    def test_suggest_projects(self) -> None:
        resp = client.post("/api/suggest-projects?user_id=1&top_k=5")
        assert resp.status_code == 200
        data = resp.json()
        assert data["user_id"] == 1
        assert len(data["projects"]) > 0

    def test_suggest_projects_have_score(self) -> None:
        resp = client.post("/api/suggest-projects?user_id=2&top_k=5")
        for proj in resp.json()["projects"]:
            assert "match_score" in proj
            assert proj["match_score"] > 0

    def test_suggest_projects_sorted_by_score(self) -> None:
        resp = client.post("/api/suggest-projects?user_id=1&top_k=10")
        projects = resp.json()["projects"]
        db_projects = [p for p in projects if p["source"] == "database"]
        scores = [p["match_score"] for p in db_projects]
        assert scores == sorted(scores, reverse=True)
