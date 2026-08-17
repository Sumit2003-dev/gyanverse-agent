"""
app.py
Streamlit frontend for the GyanVerse Agent.

Input: a topic/syllabus
Trigger: "Run Agent" -> kicks off the ADK agent loop and streams the step log
Output: rendered study guide + quiz, plus a status log proving agent autonomy.
"""
import asyncio
import contextlib
import glob
import json
import os
import queue
import sys
import threading
import uuid

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st

import appconfig as config
from agent.agent import root_agent
from agent import state as state_module

st.set_page_config(
    page_title="GyanVerse Agent",
    page_icon=":material/school:",
    layout="wide",
)

APP_NAME = "gyanverse-agent"

# ---------------------------------------------------------------- utils ----
def _make_runner():
    from google.adk.runners import InMemoryRunner

    return InMemoryRunner(agent=root_agent, app_name=APP_NAME)


# A single long-lived asyncio loop for the whole process. The google-genai
# async transport (aiohttp session + default executor) is bound to the loop it
# first runs on; ADK's sync runner.run() creates and tears down a fresh loop per
# invocation, which breaks subsequent calls with "cannot schedule new futures
# after shutdown". Driving run_async on one persistent loop avoids that.
_AGENT_LOOP = None
_AGENT_LOOP_THREAD = None


def _agent_loop():
    global _AGENT_LOOP, _AGENT_LOOP_THREAD
    if _AGENT_LOOP is None:
        _AGENT_LOOP = asyncio.new_event_loop()
        _AGENT_LOOP_THREAD = threading.Thread(
            target=_AGENT_LOOP.run_forever, daemon=True, name="gyanverse-agent-loop"
        )
        _AGENT_LOOP_THREAD.start()
    return _AGENT_LOOP


def _iter_run(runner, user_id, session_id, message):
    """Drive runner.run_async on the persistent loop, yielding events here."""
    loop = _agent_loop()
    event_queue: queue.Queue = queue.Queue()

    async def _consume():
        try:
            async with contextlib.aclosing(
                runner.run_async(
                    user_id=user_id, session_id=session_id, new_message=message
                )
            ) as agen:
                async for event in agen:
                    event_queue.put(event)
        finally:
            event_queue.put(None)

    future = asyncio.run_coroutine_threadsafe(_consume(), loop)
    while True:
        event = event_queue.get()
        if event is None:
            break
        yield event
    future.result()


def _create_session(runner, user_id, session_id, state):
    result = runner.session_service.create_session(
        app_name=APP_NAME, user_id=user_id, session_id=session_id, state=state
    )
    if asyncio.iscoroutine(result):
        return asyncio.run(result)
    return result


def _latest_guide():
    files = sorted(
        glob.glob(os.path.join(config.OUTPUTS_DIR, "*.json")),
        key=os.path.getmtime,
        reverse=True,
    )
    for path in files:
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f), path
        except Exception:
            continue
    return None, None


def _step_label(name):
    return {
        "retrieve_context": "Retrieving context from indexed course material",
        "identify_gaps": "Checking for knowledge gaps",
        "fetch_supplementary": "Fetching supplementary material",
        "generate_study_guide": "Generating study guide + quiz",
        "deliver_output": "Saving study guide",
    }.get(name, name)


def _fill_example(topic):
    st.session_state["topic_input"] = topic


EXAMPLES = [
    (
        "Transformers",
        "Transformer architecture for NLP — attention, positional encoding, self-attention, multi-head attention, training",
    ),
    (
        "Distributed systems",
        "Distributed systems — CAP theorem, consistency models, consensus algorithms (Paxos, Raft), logical clocks",
    ),
    (
        "Machine learning",
        "Intro to machine learning — supervised vs unsupervised, overfitting, evaluation metrics",
    ),
]


# ------------------------------------------------------------------ hero ----
st.title("GyanVerse Agent", text_alignment="center")
st.caption(
    "Autonomous study-guide builder — retrieves your course material, finds knowledge gaps, and generates a notes + quiz guide.",
    text_alignment="center",
)

# --------------------------------------------------------------- sidebar ----
with st.sidebar:
    st.markdown("### :material/school: GyanVerse")
    st.caption("Autonomous study-guide builder")

    st.space("small")
    st.markdown("**Runtime**")
    if config.GOOGLE_API_KEY:
        st.badge("Gemini API key", icon=":material/key:", color="green")
    elif config.GOOGLE_CLOUD_PROJECT:
        st.badge("Vertex AI", icon=":material/cloud:", color="blue")
    else:
        st.badge("Not configured", icon=":material/error:", color="red")
    st.caption(f"Model: `{config.GEMINI_MODEL}`")

    db = state_module._get_db()
    if db is not None:
        st.badge("Firestore memory", icon=":material/cloud_sync:", color="blue")
    else:
        st.badge("Local memory", icon=":material/database:", color="orange")

    history = state_module.list_sessions()
    if history:
        with st.expander(f"Past sessions ({len(history)})", icon=":material/history:"):
            for h in sorted(history, key=lambda x: x.get("updated_at", ""), reverse=True)[:10]:
                st.markdown(f"**{h.get('topic', '?')}**")
                st.caption(
                    f"Status: `{h.get('status', '')}` · "
                    f"{str(h.get('updated_at', ''))[:16].replace('T', ' ')}"
                )
                steps = ", ".join(h.get("steps_completed", []) or [])
                if steps:
                    st.caption("Steps: " + steps)

# --------------------------------------------------------------- main ----
with st.form("run_agent", border=False):
    st.text_area(
        "Topic or syllabus to study",
        key="topic_input",
        placeholder="e.g. Transformer architecture for NLP — attention, positional encoding, self-attention",
        height=110,
    )
    submitted = st.form_submit_button(
        "Run agent",
        type="primary",
        icon=":material/send:",
        disabled=not (st.session_state.get("topic_input") or "").strip(),
    )

with st.container(horizontal=True):
    st.caption("Quick start:")
    for label, topic_text in EXAMPLES:
        st.button(
            label,
            icon=":material/bolt:",
            type="tertiary",
            on_click=_fill_example,
            kwargs={"topic": topic_text},
        )

# ---------------------------------------------------------------- run -----
if submitted:
    topic = (st.session_state.get("topic_input") or "").strip()
    if not topic:
        st.info("Enter a topic to study.", icon=":material/info:")
    else:
        user_id = "student"
        session_id = uuid.uuid4().hex
        state_module.start_session(session_id, topic)

        runner = _make_runner()
        _create_session(
            runner, user_id, session_id, {"session_id": session_id, "topic": topic}
        )

        from google.genai import types

        user_content = types.Content(role="user", parts=[types.Part(text=topic)])

        final_text = ""
        run_failed = False
        run_error = ""

        with st.status(
            f"Running the study agent for “{topic}”…", state="running", expanded=True
        ) as status:
            try:
                for event in _iter_run(runner, user_id, session_id, user_content):
                    is_final = event.is_final_response()
                    if is_final and (not event.content or not event.content.parts):
                        run_failed = True
                        run_error = "The agent run ended with an error (no response)."
                    if event.content and event.content.parts:
                        for part in event.content.parts:
                            function_call = getattr(part, "function_call", None)
                            function_response = getattr(part, "function_response", None)
                            if function_call is not None and getattr(
                                function_call, "name", None
                            ):
                                status.write(
                                    f":material/route: **{_step_label(function_call.name)}**"
                                )
                            elif function_response is not None:
                                status.write(
                                    ":material/check_circle: tool result received"
                                )
                            elif getattr(part, "text", None):
                                if not is_final:
                                    status.write(
                                        f":material/chat: {part.text[:200]}"
                                    )
                                else:
                                    final_text = part.text
                    if is_final:
                        if run_failed:
                            status.update(label="Agent failed", state="error")
                        else:
                            status.update(
                                label="Agent finished", state="complete", expanded=False
                            )
            except Exception as exc:
                run_failed = True
                run_error = str(exc)
                status.update(label="Agent failed", state="error")

        session_snap = state_module.get_session(session_id)
        if session_snap and session_snap.get("error"):
            run_failed = True
            run_error = run_error or session_snap["error"]

        if run_failed:
            st.warning(
                f"The run did not complete. {run_error}",
                icon=":material/warning:",
            )
            st.info(
                "Tip: the free Gemini tier allows only ~20 model requests per day. "
                "Each run uses several, so if this happened repeatedly you may have "
                "hit the daily quota — wait for the reset or add a billing account.",
                icon=":material/lightbulb:",
            )
        else:
            if final_text:
                st.subheader("Agent summary", divider=False)
                st.markdown(final_text)

            guide, guide_path = _latest_guide()
            if guide and isinstance(guide, dict) and guide.get("topic"):
                st.subheader("Your study guide")
                st.markdown(f"### {guide.get('topic')}")
                if guide.get("summary"):
                    st.markdown(guide["summary"])

                tab_notes, tab_quiz = st.tabs(
                    [":material/menu_book: Notes", ":material/quiz: Quiz"]
                )
                with tab_notes:
                    for section in guide.get("sections", []):
                        with st.container(border=True):
                            st.markdown(f"#### {section.get('heading', 'Notes')}")
                            st.markdown(section.get("content", ""))
                            if section.get("source"):
                                st.caption(f"Source: `{section['source']}`")
                with tab_quiz:
                    quiz = guide.get("quiz", [])
                    if not quiz:
                        st.caption("No quiz included in this guide.")
                    for i, q in enumerate(quiz, 1):
                        with st.container(border=True):
                            st.markdown(f"**{i}. {q.get('question', '')}**")
                            options = q.get("options", [])
                            answer_index = q.get("answer_index", 0)
                            for j, opt in enumerate(options):
                                if j == answer_index:
                                    st.markdown(
                                        f":green-badge[Correct] {opt}",
                                    )
                                else:
                                    st.markdown(f"{opt}")
                            if q.get("explanation"):
                                st.caption(f"Explanation: {q['explanation']}")

                st.caption(f"Saved to: `{guide_path}`")

st.space("small")
st.caption(
    "Built with Google Agent Development Kit (ADK) + Gemini · Retrieval pipeline from GyanVerse AI."
)