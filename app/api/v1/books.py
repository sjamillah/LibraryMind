import json
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter(prefix="/books", tags=["Books"])

_BOOKS_PATH = Path("data/books.json")


class BookSchema(BaseModel):
    id: str
    title: str
    author: str
    year: int
    genre: str
    description: str


@router.get(
    "/",
    response_model=list[BookSchema],
    summary="List all books in the catalogue",
)
def list_books() -> list[BookSchema]:
    """Return every book in the library catalogue."""
    if not _BOOKS_PATH.exists():
        raise HTTPException(
            status_code=503,
            detail="Book catalogue not found. Ensure data/books.json exists.",
        )
    books = json.loads(_BOOKS_PATH.read_text(encoding="utf-8"))
    return [BookSchema(**b) for b in books]


@router.get(
    "/{book_id}",
    response_model=BookSchema,
    summary="Get a single book by ID",
)
def get_book(book_id: str) -> BookSchema:
    """Return a single book by its ID (e.g. `book-001`)."""
    if not _BOOKS_PATH.exists():
        raise HTTPException(status_code=503, detail="Book catalogue not found.")
    books = json.loads(_BOOKS_PATH.read_text(encoding="utf-8"))
    for book in books:
        if book["id"] == book_id:
            return BookSchema(**book)
    raise HTTPException(status_code=404, detail=f"Book '{book_id}' not found.")
