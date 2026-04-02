# SkillSync AI v2.0 🧠⚡

> AI-powered skill matching & recommendation platform built with FastAPI,
> SentenceTransformers, and scikit-learn.  Runs on a normal laptop — no GPU required.

---

## 🚀 Quick Start

```bash
# 1. Create virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS / Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the server
uvicorn app.main:app --reload

# 4. Open the docs
#    http://127.0.0.1:8000/docs
```

> **Note:** The first startup downloads the `all-MiniLM-L6-v2` model (~80 MB).
> Subsequent starts are instant.

---

## ✨ What's New in v2

| Feature | v1 | v2 |
|---------|----|----|
| Recommendation signals | 4 (skill, level, goal, activity) | **5** (+interest similarity) |
| Goal matching | Binary (same/different) | **Compatibility matrix** with partial credit |
| Embedding cache | ❌ Re-computed every request | ✅ **Global singleton** — compute once |
| Project scoring | Jaccard only | **Jaccard + Semantic** (cosine similarity) |
| Skill grouping | ❌ | ✅ **Semantic skill groups** (14 clusters) |
| User index lookup | O(N) linear scan | **O(1)** hash map |
| Team builder scoring | O(N²) Python loops | **Vectorised numpy** |
| Response timing | ❌ | ✅ `X-Process-Time` header |
| Data cache safety | `lru_cache` on mutable list | **Frozen tuples** + copy-on-read |
| Cold-start boundary | `<=` (false positives) | `<` (correct) |
| Health endpoint | Basic | **Shows cache stats** |

---

## 📡 API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Root — API status |
| `GET` | `/health` | Health check + cache stats |
| `GET` | `/stats` | System config & weights |
| `POST` | `/api/match-users` | Semantic user matching |
| `POST` | `/api/build-team` | Build team around a seed user |
| `POST` | `/api/build-teams` | Auto-partition users into teams |
| `POST` | `/api/recommend` | Hybrid people + project recommendations |
| `GET` | `/api/trust-score` | Calculate user trust score |
| `POST` | `/api/suggest-projects` | Project suggestions by skill match |

---

## 🧪 Example Requests

### Match Users
```bash
curl -X POST http://127.0.0.1:8000/api/match-users \
  -H "Content-Type: application/json" \
  -d '{"user_id": 1, "top_k": 5}'
```

### Get Recommendations (Paginated)
```bash
curl -X POST http://127.0.0.1:8000/api/recommend \
  -H "Content-Type: application/json" \
  -d '{"user_id": 2, "page": 1, "limit": 5}'
```

### Build Team
```bash
curl -X POST http://127.0.0.1:8000/api/build-team \
  -H "Content-Type: application/json" \
  -d '{"user_id": 1, "team_size": 4}'
```

### Trust Score
```bash
curl http://127.0.0.1:8000/api/trust-score?user_id=4
```

### System Stats
```bash
curl http://127.0.0.1:8000/stats
```

---

## 🏗️ Architecture

```
app/
├── main.py                      # FastAPI entry + lifespan + timing middleware
├── config.py                    # Centralised settings (all weights & thresholds)
├── dependencies.py              # Shared DI helpers
├── api/
│   ├── match_routes.py          # Matching endpoints
│   └── recommend_routes.py      # Recommendation endpoints
├── schemas/
│   ├── user_schema.py           # User profile schema
│   ├── match_schema.py          # Match request/response
│   └── recommend_schema.py      # Recommendation request/response
├── services/
│   ├── matching/
│   │   ├── vectorizer.py        # EmbeddingCache + SentenceTransformer
│   │   ├── similarity.py        # Cosine similarity (optimised argpartition)
│   │   ├── matcher.py           # Top-K match orchestrator
│   │   └── team_builder.py      # Vectorised greedy team clustering
│   ├── recommendation/
│   │   ├── recommender.py       # 5-signal hybrid scoring engine
│   │   ├── project_suggester.py # Jaccard + semantic project matching
│   │   └── trust_score.py       # Activity + ratings trust score
│   ├── llm_utils/
│   │   ├── prompt_templates.py  # Ready-to-use LLM prompts
│   │   └── llm_client.py       # Pluggable LLM client
│   └── common/
│       ├── constants.py         # Semantic skill groups + mappings
│       └── utils.py             # Safe data loading + O(1) index
├── background/
│   └── tasks.py                 # Startup warm-up + cache refresh
├── core/
│   ├── logger.py                # Structured logging
│   └── security.py              # Auth placeholder
└── ml_models/                   # Reserved for custom models
data/
├── users.json                   # Mock user database (10 users)
└── projects.json                # Mock project database (12 projects)
tests/
└── test_matching.py             # 25+ integration tests
```

---

## 🧠 How It Works

### Matching Engine
1. Each user profile → rich text description
2. SentenceTransformer encodes it into a 384-dim dense vector
3. **EmbeddingCache** stores all vectors + similarity matrix globally
4. Cosine similarity finds the closest matches
5. Results include **shared semantic skill groups**

### Recommendation Engine (5-Signal Hybrid)
```
Final Score = (Skill Similarity × 0.40)    ← semantic embeddings
            + (Interest Sim    × 0.20)    ← interest/goal/bio embeddings
            + (Level Match     × 0.15)    ← normalised distance
            + (Activity Weight × 0.15)    ← engagement signals
            + (Goal Match      × 0.10)    ← compatibility matrix
```

### Cold-Start Handling
New users (< 14 days) automatically receive **trending user**
recommendations ranked by activity score.

### Project Suggestions (Hybrid)
```
Score = (0.60 × Jaccard overlap) + (0.40 × Semantic cosine sim)
      + Difficulty bonus (if level matches)
```

### Team Builder
Vectorised greedy clustering using numpy matrix operations.  Assigns
roles (Team Lead, ML/Data, Frontend, Backend, Contributor) automatically.

---

## ⚙️ Tech Stack

| Component | Technology |
|-----------|-----------|
| Framework | FastAPI |
| Embeddings | SentenceTransformers (`all-MiniLM-L6-v2`) |
| Similarity | scikit-learn (cosine similarity) |
| Vectors | NumPy |
| Validation | Pydantic v2 |
| Database | JSON (mock) |

---

## 🧪 Running Tests

```bash
pip install pytest httpx
pytest tests/ -v
```

---

## 📝 License

MIT
