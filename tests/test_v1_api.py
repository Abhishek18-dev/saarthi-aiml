"""Integration tests for V1 API — Spring Boot contract validation.

These tests verify the EXACT JSON shape that Spring Boot's RestTemplate
expects.  If any of these fail, the Java backend will break.

Run with:
    pytest tests/test_v1_api.py -v
"""

from __future__ import annotations

import json

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


REAL_MODE_USERS = [
    {
        "id": 101,
        "name": "Real User 101",
        "skills": ["Python", "FastAPI"],
        "level": "Intermediate",
        "goal": "Project",
        "bio": "Backend engineer",
        "interests": ["api", "ml"],
        "activity": {
            "days_active": 40,
            "projects_joined": 3,
            "contributions": 20,
            "last_active_days_ago": 1,
        },
        "joined_days_ago": 100,
    },
    {
        "id": 102,
        "name": "Real User 102",
        "skills": ["Python", "Machine Learning"],
        "level": "Intermediate",
        "goal": "Study",
        "bio": "ML engineer",
        "interests": ["research"],
        "activity": {
            "days_active": 30,
            "projects_joined": 2,
            "contributions": 15,
            "last_active_days_ago": 2,
        },
        "joined_days_ago": 80,
    },
    {
        "id": 103,
        "name": "Real User 103",
        "skills": ["React", "TypeScript"],
        "level": "Beginner",
        "goal": "Project",
        "bio": "Frontend dev",
        "interests": ["ui"],
        "activity": {
            "days_active": 15,
            "projects_joined": 1,
            "contributions": 8,
            "last_active_days_ago": 3,
        },
        "joined_days_ago": 45,
    },
    {
        "id": 104,
        "name": "Real User 104",
        "skills": ["Java", "Spring Boot"],
        "level": "Advanced",
        "goal": "Mentor",
        "bio": "Platform engineer",
        "interests": ["system design"],
        "activity": {
            "days_active": 60,
            "projects_joined": 4,
            "contributions": 35,
            "last_active_days_ago": 0,
        },
        "joined_days_ago": 200,
    },
]


# ────────────────────────────────────────────────────────────────────
# 1. MATCH USERS — POST /api/v1/match-users
# ────────────────────────────────────────────────────────────────────

class TestV1MatchUsers:
    """Validate the match-users contract for Spring Boot."""

    def test_returns_flat_array(self) -> None:
        """Spring Boot expects a JSON array, not a nested wrapper."""
        resp = client.post("/api/v1/match-users", json={"userId": 1})
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list), "Response must be a flat JSON array"
        assert len(data) > 0

    def test_camelcase_keys(self) -> None:
        """Java Jackson expects camelCase, not snake_case."""
        resp = client.post("/api/v1/match-users", json={"userId": 1})
        first = resp.json()[0]
        # Must have camelCase keys.
        assert "userId" in first, "Missing camelCase 'userId'"
        assert "matchScore" in first, "Missing camelCase 'matchScore'"
        # Must NOT have snake_case keys.
        assert "user_id" not in first, "Should not have snake_case 'user_id'"
        assert "match_score" not in first, "Should not have snake_case 'match_score'"
        assert "similarity_score" not in first, "Should not have 'similarity_score'"

    def test_match_score_is_percentage(self) -> None:
        """matchScore must be 0–100, not 0–1."""
        resp = client.post("/api/v1/match-users", json={"userId": 1})
        for match in resp.json():
            assert 0 <= match["matchScore"] <= 100, (
                f"matchScore {match['matchScore']} is not in [0, 100]"
            )

    def test_self_not_in_results(self) -> None:
        """User should never match with themselves."""
        resp = client.post("/api/v1/match-users", json={"userId": 1})
        for match in resp.json():
            assert match["userId"] != 1

    def test_includes_name_and_skills(self) -> None:
        """Spring Boot maps these to the MatchDTO fields."""
        resp = client.post("/api/v1/match-users", json={"userId": 2})
        first = resp.json()[0]
        assert "name" in first and isinstance(first["name"], str)
        assert "skills" in first and isinstance(first["skills"], list)

    def test_results_sorted_by_score(self) -> None:
        resp = client.post("/api/v1/match-users", json={"userId": 1})
        scores = [m["matchScore"] for m in resp.json()]
        assert scores == sorted(scores, reverse=True)

    def test_invalid_user_returns_404(self) -> None:
        resp = client.post("/api/v1/match-users", json={"userId": 9999})
        assert resp.status_code == 404
        assert "detail" in resp.json()

    def test_missing_userid_returns_422(self) -> None:
        resp = client.post("/api/v1/match-users", json={})
        assert resp.status_code == 422

    def test_accepts_snake_case_input_too(self) -> None:
        """populate_by_name allows both camelCase and snake_case input."""
        resp = client.post("/api/v1/match-users", json={"user_id": 1})
        assert resp.status_code == 200

    def test_response_has_timing_header(self) -> None:
        resp = client.post("/api/v1/match-users", json={"userId": 1})
        assert "X-Process-Time" in resp.headers

    def test_real_mode_uses_request_users(self) -> None:
        resp = client.post(
            "/api/v1/match-users",
            json={"userId": 101, "users": REAL_MODE_USERS},
        )
        assert resp.status_code == 200
        returned_ids = {m["userId"] for m in resp.json()}
        allowed = {u["id"] for u in REAL_MODE_USERS if u["id"] != 101}
        assert returned_ids.issubset(allowed)


# ────────────────────────────────────────────────────────────────────
# 2. TRUST SCORE — POST /api/v1/trust-score
# ────────────────────────────────────────────────────────────────────

class TestV1TrustScore:
    """Validate the trust-score contract for Spring Boot."""

    def test_returns_object_not_array(self) -> None:
        resp = client.post("/api/v1/trust-score", json={"userId": 1})
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, dict), "Response must be a JSON object"

    def test_camelcase_keys(self) -> None:
        resp = client.post("/api/v1/trust-score", json={"userId": 4})
        data = resp.json()
        assert "userId" in data
        assert "trustScore" in data
        assert "label" in data
        # No snake_case.
        assert "user_id" not in data
        assert "trust_score" not in data

    def test_trust_score_in_range(self) -> None:
        resp = client.post("/api/v1/trust-score", json={"userId": 4})
        assert 0 <= resp.json()["trustScore"] <= 100

    def test_high_activity_user(self) -> None:
        """User 4 (Sneha) is very active — trust score should be high."""
        resp = client.post("/api/v1/trust-score", json={"userId": 4})
        assert resp.json()["trustScore"] >= 75

    def test_invalid_user_returns_404(self) -> None:
        resp = client.post("/api/v1/trust-score", json={"userId": 9999})
        assert resp.status_code == 404


# ────────────────────────────────────────────────────────────────────
# 3. SUGGEST PROJECTS — POST /api/v1/suggest-projects
# ────────────────────────────────────────────────────────────────────

class TestV1SuggestProjects:
    """Validate the suggest-projects contract for Spring Boot."""

    def test_returns_flat_array(self) -> None:
        resp = client.post(
            "/api/v1/suggest-projects", json={"skills": ["React", "JavaScript"]}
        )
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        assert len(data) > 0

    def test_camelcase_keys(self) -> None:
        resp = client.post(
            "/api/v1/suggest-projects", json={"skills": ["Python", "Machine Learning"]}
        )
        first = resp.json()[0]
        assert "title" in first
        assert "difficulty" in first
        # camelCase for multi-word fields.
        assert "requiredSkills" in first or "required_skills" not in first
        assert "matchScore" in first

    def test_has_required_fields(self) -> None:
        resp = client.post(
            "/api/v1/suggest-projects", json={"skills": ["Java", "Spring Boot"]}
        )
        for proj in resp.json():
            assert "title" in proj
            assert "difficulty" in proj

    def test_empty_skills_returns_results(self) -> None:
        """Even with unknown skills, the fallback should kick in."""
        resp = client.post(
            "/api/v1/suggest-projects", json={"skills": ["UnknownTech"]}
        )
        # May return empty or fallback — should not error.
        assert resp.status_code == 200

    def test_missing_skills_returns_422(self) -> None:
        resp = client.post("/api/v1/suggest-projects", json={})
        assert resp.status_code == 422


# ────────────────────────────────────────────────────────────────────
# 4. BUILD TEAM — POST /api/v1/build-team
# ────────────────────────────────────────────────────────────────────

class TestV1BuildTeam:
    """Validate the build-team contract for Spring Boot."""

    def test_returns_flat_array(self) -> None:
        resp = client.post(
            "/api/v1/build-team", json={"userId": 1, "teamSize": 3}
        )
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        assert len(data) >= 3

    def test_camelcase_keys(self) -> None:
        resp = client.post(
            "/api/v1/build-team", json={"userId": 1, "teamSize": 3}
        )
        first = resp.json()[0]
        assert "userId" in first
        assert "name" in first
        assert "role" in first
        # No snake_case.
        assert "user_id" not in first

    def test_seed_user_is_first(self) -> None:
        resp = client.post(
            "/api/v1/build-team", json={"userId": 1, "teamSize": 3}
        )
        assert resp.json()[0]["userId"] == 1

    def test_default_team_size(self) -> None:
        """When teamSize is omitted, default should be 3."""
        resp = client.post("/api/v1/build-team", json={"userId": 2})
        assert resp.status_code == 200
        assert len(resp.json()) >= 3

    def test_team_has_unique_members(self) -> None:
        resp = client.post(
            "/api/v1/build-team", json={"userId": 1, "teamSize": 5}
        )
        ids = [m["userId"] for m in resp.json()]
        assert len(ids) == len(set(ids)), "Team has duplicate members"

    def test_invalid_user_returns_404(self) -> None:
        resp = client.post(
            "/api/v1/build-team", json={"userId": 9999, "teamSize": 3}
        )
        assert resp.status_code == 404

    def test_real_mode_build_team_uses_request_users(self) -> None:
        resp = client.post(
            "/api/v1/build-team",
            json={"userId": 101, "teamSize": 3, "users": REAL_MODE_USERS},
        )
        assert resp.status_code == 200
        returned_ids = {m["userId"] for m in resp.json()}
        allowed = {u["id"] for u in REAL_MODE_USERS}
        assert returned_ids.issubset(allowed)


# ────────────────────────────────────────────────────────────────────
# 5. RECOMMENDATIONS — POST /api/v1/recommendations
# ────────────────────────────────────────────────────────────────────

class TestV1Recommendations:
    """Validate the recommendations contract for Spring Boot."""

    def test_returns_people_and_projects(self) -> None:
        resp = client.post("/api/v1/recommendations", json={"userId": 2})
        assert resp.status_code == 200
        data = resp.json()
        assert "people" in data
        assert "projects" in data
        assert isinstance(data["people"], list)
        assert isinstance(data["projects"], list)

    def test_people_have_camelcase(self) -> None:
        resp = client.post("/api/v1/recommendations", json={"userId": 2})
        if resp.json()["people"]:
            first = resp.json()["people"][0]
            assert "userId" in first
            assert "matchScore" in first
            assert "user_id" not in first

    def test_projects_have_camelcase(self) -> None:
        resp = client.post("/api/v1/recommendations", json={"userId": 1})
        if resp.json()["projects"]:
            first = resp.json()["projects"][0]
            assert "title" in first
            assert "difficulty" in first
            assert "matchScore" in first

    def test_people_scores_are_percentage(self) -> None:
        resp = client.post("/api/v1/recommendations", json={"userId": 2})
        for person in resp.json()["people"]:
            assert 0 <= person["matchScore"] <= 100

    def test_cold_start_user_gets_trending(self) -> None:
        """User 8 (joined 10 days ago) should get trending recommendations."""
        resp = client.post("/api/v1/recommendations", json={"userId": 8})
        assert resp.status_code == 200
        # Cold-start users may have trending=True.
        people = resp.json()["people"]
        if people:
            assert people[0].get("trending") is True

    def test_invalid_user_returns_404(self) -> None:
        resp = client.post("/api/v1/recommendations", json={"userId": 9999})
        assert resp.status_code == 404

    def test_real_mode_recommendations_uses_request_users(self) -> None:
        resp = client.post(
            "/api/v1/recommendations",
            json={"userId": 101, "users": REAL_MODE_USERS},
        )
        assert resp.status_code == 200
        people_ids = {p["userId"] for p in resp.json()["people"]}
        allowed = {u["id"] for u in REAL_MODE_USERS if u["id"] != 101}
        assert people_ids.issubset(allowed)


# ────────────────────────────────────────────────────────────────────
# CROSS-CUTTING: Backward compatibility
# ────────────────────────────────────────────────────────────────────

class TestBackwardCompatibility:
    """Internal /api/* routes must still work alongside /api/v1/*."""

    def test_internal_match_still_works(self) -> None:
        resp = client.post(
            "/api/match-users", json={"user_id": 1, "top_k": 3}
        )
        assert resp.status_code == 200
        data = resp.json()
        # Internal routes use snake_case wrapper.
        assert "matches" in data
        assert "user_id" in data

    def test_internal_trust_still_works(self) -> None:
        resp = client.get("/api/trust-score?user_id=4")
        assert resp.status_code == 200
        assert "trust_score" in resp.json()

    def test_health_endpoint_still_works(self) -> None:
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "healthy"

    def test_both_v1_and_internal_match_same_user(self) -> None:
        """V1 and internal should return the same users, different format."""
        v1_resp = client.post("/api/v1/match-users", json={"userId": 1})
        internal_resp = client.post(
            "/api/match-users", json={"user_id": 1, "top_k": 5}
        )
        v1_ids = {m["userId"] for m in v1_resp.json()}
        internal_ids = {m["user_id"] for m in internal_resp.json()["matches"]}
        assert v1_ids == internal_ids, "V1 and internal routes disagreed"
