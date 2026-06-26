from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.v1 import books, chat, classify, query, search, summarise

_DESCRIPTION = """
**LibraryMind** is an AI-powered library assistant that helps patrons discover books
through natural-language questions.

## How it works

1. Your question is embedded into a vector using a local sentence-transformer model.
2. ChromaDB finds the most semantically similar books in the catalogue.
3. Only relevant books are sent to the AI — the model is explicitly forbidden from
   inventing titles outside the provided context.
4. The answer is cached so repeated questions are served instantly.

## Endpoints

| Endpoint | Purpose |
|---|---|
| `POST /api/v1/search/books` | Semantic search over the catalogue |
| `POST /api/v1/search/ask` | RAG-powered question answering |
| `POST /api/v1/chat/` | Multi-turn chatbot conversation |
| `POST /api/v1/classify/ticket` | Classify a library support ticket |
| `POST /api/v1/summarise/reviews` | Summarise a collection of book reviews |
| `POST /api/v1/query/` | Ask the assistant a question (legacy path) |
| `GET /api/v1/books/` | List all books in the catalogue |
| `GET /api/v1/books/{id}` | Retrieve a single book by ID |
| `GET /health` | Service health check with daily cost and request count |
"""

app = FastAPI(
    title="LibraryMind",
    version="0.1.0",
    description=_DESCRIPTION,
    contact={
        "name": "LibraryMind",
        "email": "ssozijamillah@gmail.com",
    },
    openapi_tags=[
        {
            "name": "Search",
            "description": "Semantic book search and RAG-powered question answering.",
        },
        {
            "name": "Query",
            "description": "Natural-language questions answered from the book catalogue (legacy path).",
        },
        {
            "name": "Chat",
            "description": "Multi-turn conversational assistant with persistent session history.",
        },
        {
            "name": "Classify",
            "description": "Classify raw library support tickets into structured JSON.",
        },
        {
            "name": "Summarise",
            "description": "Summarise collections of patron book reviews holistically.",
        },
        {
            "name": "Books",
            "description": "Browse and retrieve catalogue entries.",
        },
        {
            "name": "Health",
            "description": "Service liveness check with usage statistics.",
        },
    ],
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(search.router, prefix="/api/v1")
app.include_router(query.router, prefix="/api/v1")
app.include_router(books.router, prefix="/api/v1")
app.include_router(chat.router, prefix="/api/v1")
app.include_router(classify.router, prefix="/api/v1")
app.include_router(summarise.router, prefix="/api/v1")


@app.get("/health", tags=["Health"], summary="Service health check")
def health() -> dict:
    """Returns service status, daily AI spend, and total requests made today."""
    from app.infrastructure.usage_tracker import usage_tracker
    return {
        "status": "ok",
        "daily_cost_usd": usage_tracker.total_cost_today(),
        "total_requests_today": usage_tracker.total_requests_today(),
    }
