"""agent/agent.py
The GyanVerse ADK agent. Exposes `root_agent` so it can be run with the ADK CLI
(`adk run agent` / `adk web`) or driven programmatically from app.py.

The agent decides its own step sequence (agentic loop, not a fixed pipeline):
it retrieves context, checks for knowledge gaps, fetches supplementary material,
generates a study guide, and delivers it.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

load_dotenv()

from google.adk.agents import Agent
from google.adk.models import Gemini
from google.genai import types

import appconfig as config
from . import tools

# Retry transient rate-limit (429) and server errors with backoff. The free
# tier caps requests per day, but per-minute bursts can briefly 429; retrying
# smooths those out.
_MODEL = Gemini(
    model=config.GEMINI_MODEL,
    retry_options=types.HttpRetryOptions(
        attempts=5,
        initial_delay=2.0,
        exp_base=2.0,
        max_delay=30.0,
        http_status_codes=[408, 429, 500, 502, 503, 504],
    ),
)

INSTRUCTION = """You are the GyanVerse Study Agent, an autonomous study-guide builder for students.

A user will give you a topic (or a syllabus) to study. Your job is to produce a
complete, grounded study guide with notes and a quiz, and persist it.

You decide the order of steps and which tools to call based on the situation.
The recommended workflow is:

1. Call `retrieve_context(topic)` to pull the most relevant passages from the
   indexed course material. If it returns an error, say so honestly to the user
   and stop — do not guess.
2. Call `identify_gaps(topic, retrieved_docs)` passing the "context" text chunks
   returned in step 1. It tells you which important subtopics are missing or
   under-covered.
3. For the most important gap (or gaps), call `fetch_supplementary(gap_topic)`
   to obtain short external material. You may skip this if there are no gaps.
4. Call `generate_study_guide(topic, context)` passing a context object shaped as
   {"context": [...chunks from step 1...], "supplementary": [{subtopic, content, source}...]}.
   This produces the structured study guide (sections + quiz).
5. Call `deliver_output(content, destination="both")` passing the study guide
   returned in step 4, so it is saved to file and Firestore.

After the tools have run, reply to the user with a concise summary: the topic,
the number of sections and quiz questions, the sources used, any knowledge gaps
that were filled, and where the guide was saved.

Guidelines:
- Ground answers in the retrieved context. Never invent facts.
- If a tool returns status "error", report the error clearly and continue if
  possible, otherwise stop gracefully.
- Keep the final user-facing reply short and useful.
"""

root_agent = Agent(
    name="gyanverse_study_agent",
    model=_MODEL,
    description=(
        "Autonomous study-guide builder for students. Given a topic it retrieves "
        "context from indexed course material, identifies and fills knowledge gaps, "
        "and generates a structured study guide with notes and a quiz."
    ),
    instruction=INSTRUCTION,
    tools=[
        tools.retrieve_context,
        tools.identify_gaps,
        tools.fetch_supplementary,
        tools.generate_study_guide,
        tools.deliver_output,
    ],
)