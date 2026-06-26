# LibraryMind

An AI-powered backend service for a public library. Patrons can search the catalogue by meaning, get grounded book recommendations, chat with an AI librarian, have support tickets classified automatically, and receive summarised analysis of book reviews.

## Architecture

LibraryMind follows a strict four-layer separation:

| Layer | Responsibility |
|---|---|
| **API** (`app/api/v1/`) | FastAPI routers — request validation, routing, error mapping |
| **Service** (`app/services/`) | Business logic — RAG engine, chat, classification, summarisation |
| **AI Provider** (`app/providers/`) | Multi-provider abstraction with automatic failover |
| **Infrastructure** (`app/infrastructure/`) | ChromaDB, Redis cache, rate limiter, usage tracker |

## Requirements

- Python 3.10+
- Redis (optional — the app falls back to an in-memory store if Redis is unavailable)
- API key for at least one provider (OpenAI or Anthropic via the Amali gateway)

## Setup

### 1. Clone and create a virtual environment

```bash
git clone <repo-url>
cd LibraryMind
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment variables

Copy the example file and fill in your keys:

```bash
cp .env.example .env
```

Edit `.env` — see the [Environment Variables](#environment-variables) section below.

### 4. Seed the vector database

Run once to embed all 20+ books and store them in ChromaDB:

```bash
python scripts/seed_db.py
```

### 5. Start the server

```bash
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

The interactive API docs are available at `http://localhost:8000/docs`.

## Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `OPENAI_API_KEY` | One of these two is required | — | OpenAI API key |
| `ANTHROPIC_API_KEY` | One of these two is required | — | Anthropic API key (or Amali gateway key) |
| `OPENAI_BASE_URL` | No | OpenAI default | Override base URL (e.g. Amali gateway) |
| `ANTHROPIC_BASE_URL` | No | Anthropic default | Override base URL |
| `PRIMARY_PROVIDER` | No | `openai` | Which provider to try first (`openai` or `anthropic`) |
| `RATE_LIMIT_PER_MINUTE` | No | `20` | Max AI requests per minute |
| `REDIS_URL` | No | `redis://localhost:6379` | Redis connection string |
| `CACHE_TTL` | No | `3600` | Response cache time-to-live in seconds |
| `RAG_RELEVANCE_THRESHOLD` | No | `0.4` | Cosine distance cut-off (lower = stricter) |
| `CHROMA_PATH` | No | `./chroma_db` | Where ChromaDB persists its data |

## API Endpoints

All endpoints live under `/api/v1/`. The health check is at `/health`.

---

### `GET /health`

Returns service status, today's AI spend, and total request count.

```bash
curl http://localhost:8000/health
```

```json
{
  "status": "ok",
  "daily_cost_usd": 0.003241,
  "total_requests_today": 17
}
```

---

### `POST /api/v1/search/books`

Semantic search over the catalogue — returns books ranked by meaning, not keywords.

```bash
curl -X POST http://localhost:8000/api/v1/search/books \
  -H "Content-Type: application/json" \
  -d '{"query": "space exploration adventure", "limit": 5}'
```

```json
[
  {
    "id": "book-003",
    "title": "The Martian",
    "author": "Andy Weir",
    "year": 2011,
    "genre": "Science Fiction",
    "description": "...",
    "relevance_score": 0.8712
  }
]
```

---

### `POST /api/v1/search/ask`

RAG-powered Q&A — answers grounded in the catalogue with source citations.

```bash
curl -X POST http://localhost:8000/api/v1/search/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What science fiction books do you have about desert planets?"}'
```

```json
{
  "answer": "We have 'Dune' by Frank Herbert, set on the desert planet Arrakis...",
  "sources": [
    {"title": "Dune", "author": "Frank Herbert", "relevance_score": 0.912}
  ],
  "cached": false
}
```

---

### `POST /api/v1/chat/`

Multi-turn AI librarian chatbot. Omit `conversation_id` on the first message — the server generates one.

**Turn 1 — start a new conversation:**

```bash
curl -X POST http://localhost:8000/api/v1/chat/ \
  -H "Content-Type: application/json" \
  -d '{"message": "Recommend a science fiction book"}'
```

```json
{
  "reply": "I'd recommend 'Dune' by Frank Herbert...",
  "conversation_id": "a3f1c2d4-...",
  "sources": [...]
}
```

**Turn 2 — continue the same conversation:**

```bash
curl -X POST http://localhost:8000/api/v1/chat/ \
  -H "Content-Type: application/json" \
  -d '{"message": "Tell me more about that one", "conversation_id": "a3f1c2d4-..."}'
```

---

### `POST /api/v1/classify/ticket`

Classifies a free-text library support ticket into structured JSON.

```bash
curl -X POST http://localhost:8000/api/v1/classify/ticket \
  -H "Content-Type: application/json" \
  -d '{"ticket": "My library card is not working at the self-checkout and I am very frustrated."}'
```

```json
{
  "category": "technical",
  "priority": "high",
  "sentiment": "negative",
  "suggested_department": "IT Support",
  "summary": "Patron cannot use their library card at the self-checkout machine."
}
```

Valid values:
- `category`: `account` | `borrowing` | `technical` | `complaint` | `suggestion` | `general`
- `priority`: `low` | `medium` | `high` | `urgent`
- `sentiment`: `positive` | `neutral` | `negative`

---

### `POST /api/v1/summarise/reviews`

Summarises a list of book reviews into a single holistic analysis.

```bash
curl -X POST http://localhost:8000/api/v1/summarise/reviews \
  -H "Content-Type: application/json" \
  -d '{
    "reviews": [
      "Absolutely loved it. The world-building is stunning.",
      "Dense and slow at first but incredibly rewarding.",
      "A masterpiece that redefined the genre.",
      "Some parts dragged but the characters were unforgettable."
    ]
  }'
```

```json
{
  "overall_sentiment": "positive",
  "average_rating": 4.2,
  "key_themes": ["world-building", "pacing", "character depth"],
  "praise": ["stunning world-building", "unforgettable characters"],
  "criticism": ["slow start", "dense prose"],
  "recommendation": "Highly recommended for readers who enjoy epic, immersive science fiction."
}
```

---

## Running the Tests

### Unit tests

```bash
python -m pytest
```

### HTTP smoke test (requires a running server)

```bash
python scripts/smoke_test_api.py
```

### Provider integration test (makes real AI calls)

```bash
python scripts/smoke_test.py
```

## Error Responses

| HTTP Status | Cause |
|---|---|
| `422 Unprocessable Entity` | Missing or invalid request fields (Pydantic validation) |
| `429 Too Many Requests` | Application-level rate limit exceeded |
| `503 Service Unavailable` | All AI providers failed |

## Project Structure

```
LibraryMind/
├── app/
│   ├── api/v1/          # FastAPI routers (books, chat, classify, query, search, summarise)
│   ├── core/            # Settings (pydantic-settings)
│   ├── infrastructure/  # ChromaDB, Redis cache, rate limiter, usage tracker
│   ├── providers/       # OpenAI + Anthropic providers, ResilientAIService
│   ├── services/        # RAG engine, chat, classification, summarisation, embedding
│   └── main.py          # FastAPI app, CORS, router registration, /health
├── data/
│   └── books.json       # 20+ book catalogue
├── scripts/
│   ├── seed_db.py       # Populate ChromaDB from books.json
│   ├── smoke_test.py    # Provider/infrastructure integration test
│   └── smoke_test_api.py # HTTP endpoint smoke test
├── tests/               # pytest unit tests
├── .env.example
└── requirements.txt
```
