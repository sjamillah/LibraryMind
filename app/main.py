from fastapi import FastAPI

from app.api.v1 import books, query

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
| `POST /api/v1/query/` | Ask the assistant a question |
| `GET /api/v1/books/` | List all books in the catalogue |
| `GET /api/v1/books/{id}` | Retrieve a single book by ID |
| `GET /health` | Service health check |
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
            "name": "Query",
            "description": "Natural-language questions answered from the book catalogue.",
        },
        {
            "name": "Books",
            "description": "Browse and retrieve catalogue entries.",
        },
        {
            "name": "Health",
            "description": "Service liveness check.",
        },
    ],
)

app.include_router(query.router, prefix="/api/v1")
app.include_router(books.router, prefix="/api/v1")


@app.get("/health", tags=["Health"], summary="Service health check")
def health() -> dict:
    """Returns `ok` when the service is running."""
    return {"status": "ok"}
