"""smoke_test.py
End-to-end runner test for the GyanVerse ADK agent WITHOUT hitting Google Cloud.

It registers a tiny mock LLM (`mock-meta`) in the ADK model registry that plays
the part of Gemini: it calls `retrieve_context`, then `deliver_output`, then
answers with text. This exercises the real ADK pipeline:

    InMemoryRunner -> agent loop -> FunctionTool wrapping -> ToolContext
    injection -> tool execution -> result events -> final text event

Verification:
    * A known session_id is placed in the session state; after the run the
      local state file must show that session with step "delivered". That proves
      ToolContext injection worked (deliver_output reads session_id from
      tool_context.state). retrieve_context errors honestly when no index exists
      (offline), which is expected here.
    * output files outputs/transformers.json and .md must exist.

Run:  python smoke_test.py   (from the project root, uses the project venv)
"""
import asyncio
import os
import sys
from typing import ClassVar

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from google.adk.agents import Agent
from google.adk.events import Event
from google.adk.models import BaseLlm
from google.adk.models.llm_response import LlmResponse
from google.adk.models.registry import LLMRegistry
from google.adk.runners import InMemoryRunner
from google.genai import types

import appconfig as config
from agent import tools

GUIDE = {
    "topic": "Transformers",
    "summary": "A study guide about the Transformer architecture.",
    "sections": [
        {
            "heading": "Attention Mechanism",
            "content": "Self-attention lets each token attend to all others.",
            "source": "mock-source.pdf",
        }
    ],
    "quiz": [
        {
            "question": "What does self-attention compute?",
            "options": ["Importance of tokens to each other", "Nothing"],
            "answer_index": 0,
            "explanation": "Attention weights reflect token relevance.",
        }
    ],
}


class MockLlm(BaseLlm):
    """Stand-in model: retrieve_context -> deliver_output -> final text."""

    turn: ClassVar[int] = 0

    @classmethod
    def supported_models(cls):
        return [r"^mock-meta$"]

    async def generate_content_async(self, llm_request, stream=False):
        type(self).turn += 1
        if type(self).turn == 1:
            part = types.Part(
                function_call=types.FunctionCall(
                    name="retrieve_context", args={"topic": "Transformers"}
                )
            )
        elif type(self).turn == 2:
            part = types.Part(
                function_call=types.FunctionCall(
                    name="deliver_output",
                    args={"content": GUIDE, "destination": "both"},
                )
            )
        else:
            part = types.Part(
                text="Done! I built the Transformers study guide. It has 1 section "
                "and 1 quiz question, grounded in your course material, saved to "
                "outputs/."
            )
        yield LlmResponse(
            content=types.Content(role="model", parts=[part]),
            partial=False,
            finish_reason=types.FinishReason.STOP if type(self).turn >= 3 else None,
        )


def main():
    LLMRegistry.register(MockLlm)

    test_agent = Agent(
        name="gyanverse_study_agent",
        model="mock-meta",
        description="test",
        instruction="Build a study guide.",
        tools=[
            tools.retrieve_context,
            tools.identify_gaps,
            tools.fetch_supplementary,
            tools.generate_study_guide,
            tools.deliver_output,
        ],
    )

    runner = InMemoryRunner(agent=test_agent, app_name="gyanverse-agent")
    session = runner.session_service.create_session_sync(
        app_name="gyanverse-agent",
        user_id="smoke",
        session_id="smoke-session-1",
        state={"session_id": "smoke-session-1"},
    )

    message = types.Content(role="user", parts=[types.Part(text="Build a study guide on Transformers.")])
    for event in runner.run(user_id="smoke", session_id=session.id, new_message=message):
        if isinstance(event, Event):
            role = event.author or "?"
            content = event.content
            if content and getattr(content, "parts", None):
                for p in content.parts:
                    if p.function_call:
                        print(f"[CALL]  {role}: {p.function_call.name}{p.function_call.args}")
                    elif p.function_response:
                        out = str(p.function_response.response)[:100]
                        print(f"[RESULT] {role}: {p.function_response.name} -> {out}")
                    elif p.text:
                        print(f"[TEXT]  {role}: {p.text[:120]}")

    # Verification 1: tool_context injection persisted the delivered step.
    # The session key must be exactly "smoke-session-1" (the value read from
    # tool_context.state) - not a random id. retrieve_context legitimately
    # returns an error (no index offline), so it does not log a step.
    state_path = os.path.join(config.LOCAL_STATE_DIR, "sessions.json")
    ok = False
    if os.path.exists(state_path):
        import json
        data = json.load(open(state_path, encoding="utf-8"))
        snap = data.get("smoke-session-1")
        if snap:
            steps = snap.get("steps_completed", [])
            ok = "delivered" in steps
            print(f"\nSTATE STEPS for smoke-session-1: {steps}")
    print("TOOLCONTEXT_INJECTION:", "OK" if ok else "FAILED")

    # Verification 2: output files
    json_file = os.path.join(config.OUTPUTS_DIR, "transformers.json")
    md_file = os.path.join(config.OUTPUTS_DIR, "transformers.md")
    print("OUTPUT_FILES:", "OK" if os.path.exists(json_file) and os.path.exists(md_file) else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
