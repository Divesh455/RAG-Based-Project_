from __future__ import annotations

from functools import lru_cache

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from rag_service import CourseAssistant


app = FastAPI(
    title="RAG Course Assistant",
    description="A FastAPI + Tailwind UI for asking questions about the course videos.",
)
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

SAMPLE_QUESTIONS = [
    "Where is CSS introduced in the course?",
    "Which video explains semantic tags in HTML?",
    "Where can I learn about inline and block elements?",
    "What lesson should I watch to understand the basic HTML structure?",
]


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=500)
    top_k: int = Field(default=3, ge=1, le=8)


@lru_cache
def get_assistant() -> CourseAssistant:
    return CourseAssistant()


@app.get("/", response_class=HTMLResponse)
async def home(request: Request) -> HTMLResponse:
    bootstrap_error = None
    stats = {
        "chunk_count": 0,
        "lesson_count": 0,
        "embeddings_path": "",
        "lesson_preview": [],
    }

    try:
        stats = get_assistant().get_stats()
    except Exception as exc:  # noqa: BLE001
        bootstrap_error = str(exc)

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "stats": stats,
            "sample_questions": SAMPLE_QUESTIONS,
            "bootstrap_error": bootstrap_error,
        },
    )


@app.get("/health")
async def health() -> dict[str, str]:
    try:
        get_assistant()
        return {"status": "ok"}
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/api/ask")
async def ask_question(payload: AskRequest) -> dict:
    question = payload.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Please enter a question.")

    try:
        return get_assistant().answer_question(question=question, top_k=payload.top_k)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected server error: {exc}",
        ) from exc
