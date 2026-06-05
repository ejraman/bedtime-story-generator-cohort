"""
crew_service.py
---------------
3-agent CrewAI pipeline for the Bedtime Story Generator capstone.

Agents (sequential):
  1. Theme Picker   - selects an age-appropriate theme and moral
  2. Story Writer   - writes the full bedtime story
  3. Title Maker    - crafts a catchy title and one-line parent summary

LLM: Gemini 2.5 Flash Lite (same model used by the existing /story route).
"""

import os
from crewai import Agent, Crew, Process, Task
from crewai.llm import LLM
from app.schemas import StoryRequest

def _make_llm() -> LLM:
    return LLM(
        model="gemini/gemini-2.5-flash-lite",
        api_key=os.environ["GEMINI_API_KEY"],
        temperature=0.7,
    )

def run_story_crew(req: StoryRequest) -> dict:
    llm = _make_llm()

    theme_agent = Agent(
        role="Children's Story Theme Specialist",
        goal="Choose a single age-appropriate theme and a clear moral lesson that fits the child's details and the requested plot.",
        backstory="You are a child psychologist and storyteller with 20 years of experience selecting themes that resonate with young readers. You always pick themes that are positive, imaginative, and suitable for the child's age.",
        llm=llm,
        verbose=True,
    )

    story_agent = Agent(
        role="Bedtime Story Writer",
        goal="Write a warm, engaging bedtime story of 150-200 words that incorporates the chosen theme, moral, characters, setting, and plot. The story must end peacefully to help the child sleep.",
        backstory="You are a beloved children's book author known for soothing bedtime stories. Your writing is simple, vivid, and always ends on a calm, sleepy note.",
        llm=llm,
        verbose=True,
    )

    title_agent = Agent(
        role="Story Title and Summary Specialist",
        goal="Create a catchy, child-friendly title for the story and write a single sentence that parents can read to understand the story's moral before bedtime.",
        backstory="You are a children's book editor who has titled hundreds of bestselling picture books. You know exactly what makes a title irresistible to both children and parents.",
        llm=llm,
        verbose=True,
    )

    task_theme = Task(
        description=(
            f"The child's name is {req.child_name.strip()}.\n"
            f"Characters: {req.characters.strip()}\n"
            f"Setting: {req.setting.strip()}\n"
            f"Plot idea: {req.plot.strip()}\n\n"
            "Select ONE theme (e.g. friendship, courage, kindness) and ONE moral lesson that fits perfectly. Output exactly two lines:\n"
            "Theme: <theme>\n"
            "Moral: <moral lesson>"
        ),
        expected_output="Two lines: 'Theme: ...' and 'Moral: ...'",
        agent=theme_agent,
    )

    task_story = Task(
        description=(
            f"Using the theme and moral from the previous task, write a bedtime story for {req.child_name.strip()}.\n"
            f"Characters: {req.characters.strip()}\n"
            f"Setting: {req.setting.strip()}\n"
            f"Plot: {req.plot.strip()}\n\n"
            "The story must be 150-200 words, written in simple language for a young child, and end with the child (or main character) falling peacefully asleep. Output the story text only."
        ),
        expected_output="A bedtime story of 150-200 words ending peacefully.",
        agent=story_agent,
        context=[task_theme],
    )

    task_title = Task(
        description=(
            "Read the completed bedtime story from the previous task and:\n"
            "1. Write a short, catchy title (max 8 words).\n"
            "2. Write one sentence for parents summarising the moral.\n\n"
            "Output exactly:\n"
            "Title: <title>\n"
            "Summary: <one sentence for parents>"
        ),
        expected_output="Two lines: 'Title: ...' and 'Summary: ...'",
        agent=title_agent,
        context=[task_story],
    )

    crew = Crew(
        agents=[theme_agent, story_agent, title_agent],
        tasks=[task_theme, task_story, task_title],
        process=Process.sequential,
        verbose=True,
    )

    result = crew.kickoff()

    story_text = str(task_story.output.raw).strip()
    title_raw  = str(task_title.output.raw).strip()
    title   = _extract_line(title_raw, "Title")
    summary = _extract_line(title_raw, "Summary")

    return {
        "title":   title   or "A Magical Bedtime Story",
        "story":   story_text,
        "summary": summary or "",
    }

def _extract_line(text: str, key: str) -> str:
    for line in text.splitlines():
        if line.strip().lower().startswith(key.lower() + ":"):
            return line.split(":", 1)[1].strip()
    return ""
