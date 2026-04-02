"""Shared constants used across services."""

from __future__ import annotations

# ── Experience-level hierarchy (used for level-match scoring) ──────
LEVEL_ORDER: dict[str, int] = {
    "Beginner": 1,
    "Intermediate": 2,
    "Advanced": 3,
}

# Maximum value a level can take — used to normalise level distance.
MAX_LEVEL_DISTANCE: int = max(LEVEL_ORDER.values()) - min(LEVEL_ORDER.values())

# ── Semantic skill groups ──────────────────────────────────────────
# Skills in the same group receive a similarity bonus even when exact
# keyword matching fails.  This gives the AI a "smarter" feel.
SEMANTIC_SKILL_GROUPS: dict[str, list[str]] = {
    "Frontend Frameworks": ["React", "Vue.js", "Next.js", "Angular", "Svelte"],
    "Frontend Styling": ["CSS", "Tailwind CSS", "SCSS", "HTML"],
    "Backend Python": ["Python", "Django", "FastAPI", "Flask"],
    "Backend JVM": ["Java", "Spring Boot", "Kotlin"],
    "Backend Node": ["Node.js", "Express", "NestJS"],
    "Data Science": ["NumPy", "Pandas", "Matplotlib", "Jupyter"],
    "Machine Learning": ["Machine Learning", "TensorFlow", "PyTorch", "scikit-learn"],
    "Deep Learning": ["Deep Learning", "Computer Vision", "NLP", "GANs"],
    "Mobile": ["Flutter", "Dart", "React Native", "Swift", "Kotlin"],
    "DevOps": ["Docker", "Kubernetes", "CI/CD", "AWS", "GCP"],
    "Databases": ["SQL", "PostgreSQL", "MongoDB", "Redis", "Firebase"],
    "API & Integration": ["REST APIs", "GraphQL", "gRPC", "WebSocket"],
    "Languages": ["JavaScript", "TypeScript", "Python", "Java", "Go", "Rust"],
}

# Build a reverse look-up: skill → group name (lowercased keys).
_SKILL_TO_GROUP: dict[str, str] = {}
for _group, _skills in SEMANTIC_SKILL_GROUPS.items():
    for _skill in _skills:
        _SKILL_TO_GROUP[_skill.lower()] = _group


def get_skill_group(skill: str) -> str | None:
    """Return the semantic group a skill belongs to, or None."""
    return _SKILL_TO_GROUP.get(skill.lower())


def compute_group_overlap(skills_a: list[str], skills_b: list[str]) -> float:
    """Return fraction of shared semantic skill groups between two users.

    Result is in [0, 1].  Partial credit is given per shared group.
    """
    groups_a = {get_skill_group(s) for s in skills_a} - {None}
    groups_b = {get_skill_group(s) for s in skills_b} - {None}
    if not groups_a or not groups_b:
        return 0.0
    intersection = groups_a & groups_b
    union = groups_a | groups_b
    return len(intersection) / len(union)


# ── Skill → suggested project mapping (fallback) ──────────────────
SKILL_PROJECT_MAP: dict[str, list[str]] = {
    "React": ["Portfolio Website", "Todo App", "Dashboard UI"],
    "JavaScript": ["Browser Extension", "Quiz Game", "Chat Widget"],
    "Python": ["CLI Tool", "Web Scraper", "REST API"],
    "Machine Learning": ["Prediction System", "Chatbot", "Recommender"],
    "Node.js": ["REST API Server", "Real-time Chat", "Task Queue"],
    "Django": ["Blog Platform", "E-commerce Backend", "CMS"],
    "FastAPI": ["Microservice", "ML API", "Webhook Relay"],
    "Flutter": ["Expense Tracker", "Weather App", "Notes App"],
    "Java": ["Inventory System", "Banking App", "Scheduler"],
    "TypeScript": ["Component Library", "Form Builder", "CLI Tool"],
    "TensorFlow": ["Image Classifier", "Style Transfer", "Object Detector"],
    "Deep Learning": ["GAN Art Generator", "Speech Recognition", "Text Summariser"],
    "Docker": ["CI/CD Pipeline", "Container Orchestrator", "Dev Environment"],
    "SQL": ["Analytics Dashboard", "Reporting Tool", "Data Warehouse"],
    "GraphQL": ["Blog API", "E-commerce API", "Social Feed"],
    "Vue.js": ["Admin Panel", "Portfolio Site", "Landing Page"],
    "CSS": ["Design System", "Animation Library", "Theme Builder"],
    "MongoDB": ["User Directory", "Content Store", "Event Logger"],
}
