"""Small FastAPI service for the separate (non-Netlify) Python backend.

Netlify hosts the web page and calls this service through its serverless proxy.
This endpoint is a transparent RULE-BASED demo, not an RL-trained model.
"""
import os
import random
import re
from pathlib import Path

from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel, Field

from src.data_prep import read_jsonl

app = FastAPI(title="CSCI 5350 Movie & Music Demo", version="0.1")


class RecommendationRequest(BaseModel):
    message: str = Field(min_length=3, max_length=300)
    previous_titles: list[str] = Field(default_factory=list, max_length=50)


def check_access(authorization):
    required = os.getenv("API_ACCESS_TOKEN", "").strip()
    if required and authorization != "Bearer " + required:
        raise HTTPException(status_code=401, detail="Not authorized")


def pick_question(message, history):
    file = Path("data/processed/demo.jsonl")
    if not file.exists():
        raise HTTPException(status_code=503, detail="Demo questions are not prepared yet. Run --mode prepare on the backend.")
    questions = read_jsonl(file)
    wanted = "music" if re.search(r"\b(music|song|songs|track|spotify|artist)\b", message, re.I) else "movie"
    candidate_questions = [q for q in questions if q["category"] == wanted]
    # The visible recommendation must honor a genre the visitor actually asked for.
    requested_genre = None
    genre_terms = {
        "sci-fi": ["sci-fi", "science fiction", "scifi"],
        "action": ["action"], "comedy": ["comedy"], "drama": ["drama"],
        "horror": ["horror"], "romance": ["romance"],
        "rock": ["rock"], "pop": ["pop"], "jazz": ["jazz"],
    }
    for genre, names in genre_terms.items():
        if any(re.search(r"\b" + re.escape(term) + r"\b", message, re.I) for term in names):
            requested_genre = genre
            break
    if requested_genre:
        filtered = [q for q in candidate_questions if requested_genre in q["correctness_rule"].casefold()]
        if not filtered:
            raise HTTPException(status_code=404, detail=(
                f"No {wanted} example with genre {requested_genre} in the reserved demo split. "
                "Try a more general request or prepare more data."
            ))
        candidate_questions = filtered
    if not candidate_questions:
        raise HTTPException(status_code=503, detail=f"No prepared {wanted} questions yet.")
    seen = {str(x).casefold() for x in history}
    unseen = [q for q in candidate_questions if q["expected_title"].casefold() not in seen]
    options = unseen if unseen else candidate_questions
    # Random sampling instead of always returning the globally top-rated title.
    picked = random.choice(options)
    return {
        "project": "Gemma 4 Option A",
        "engine": "rule-based dataset demo (NOT trained Gemma)",
        "category": picked["category"],
        "recommendation": picked["expected_title"],
        "reason": picked["correctness_rule"],
        "choice": picked["ground_truth"],
        "sample_question": picked["prompt"],
        "note": "For the research comparison, use the separate base/SFT/RL evaluation scripts.",
    }


@app.get("/api/health")
def health():
    return {"status": "ok", "demo_ready": Path("data/processed/demo.jsonl").exists()}


@app.post("/api/recommend")
def recommend(payload: RecommendationRequest, authorization: str | None = Header(default=None)):
    check_access(authorization)
    return pick_question(payload.message, payload.previous_titles)
