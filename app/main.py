from dotenv import load_dotenv
load_dotenv()

import psycopg
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.database import get_conn
from app.prompt import compose_story_prompt
from app.schemas import CrewStoryResponse, StoredStory, StoryRequest, StoryResponse
from app.services.gemini_service import call_gemini
from app.services.story_service import fetch_recent_stories, save_story
from app.services.crew_service import run_story_crew

app = FastAPI(title="Bedtime Story Generator")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/story", response_model=StoryResponse)
def story(payload: StoryRequest):
    if not payload.child_name.strip() or not payload.plot.strip():
        raise HTTPException(
            status_code=400,
            detail="Please fill in at least the child's name and the plot.",
        )
    story_text = call_gemini(compose_story_prompt(payload))
    save_story(payload, story_text)
    return StoryResponse(story=story_text)


@app.post("/story/crew", response_model=CrewStoryResponse)
def story_crew(payload: StoryRequest):
    """
    Multi-agent endpoint (capstone upgrade).
    Runs a 3-agent CrewAI pipeline:
      Agent 1 - picks theme + moral
      Agent 2 - writes the bedtime story
      Agent 3 - generates title + parent summary
    """
    if not payload.child_name.strip() or not payload.plot.strip():
        raise HTTPException(
            status_code=400,
            detail="Please fill in at least the child's name and the plot.",
        )
    result = run_story_crew(payload)
    save_story(payload, result["story"])
    return CrewStoryResponse(
        title=result["title"],
        story=result["story"],
        summary=result["summary"],
    )


@app.get("/stories", response_model=list[StoredStory])
def stories(child_name: str):
    if not child_name.strip():
        raise HTTPException(status_code=400, detail="child_name query parameter is required.")
    return fetch_recent_stories(child_name)


@app.get("/healthz")
def healthz():
    status = {"postgres": False}
    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
        status["postgres"] = True
    except psycopg.Error:
        pass
    return status
