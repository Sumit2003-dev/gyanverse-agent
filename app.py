"""
app.py
Thin Streamlit frontend for the GyanVerse Agent.

Input: a topic/syllabus
Trigger: "Run Agent" -> kicks off the ADK agent loop and streams the step log
Output: rendered study guide + quiz, plus a status log proving agent autonomy.
"""
import asyncio
import glob
import json
import os
import sys
import uuid

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st

import appconfig as config
from agent.agent import root_agent
from agent import state as state_module

st.set_page_config(page_title="GyanVerse Agent", page_icon="🎓", layout="wide")

APP_NAME = "gyanverse-agent"

st.title("🎓 GyanVerse Agent")
st.caption("Autonomous study-guide builder — retrieves your course material, finds knowledge gaps, and generates a notes + quiz guide.")

# ---------------------------------------------------------------- utils ----
def _make_runner():
    from google.adk.runners import InMemoryRunner

    return InMemoryRunner(agent=root_agent, app_name=APP_NAME)


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


# --------------------------------------------------------------- sidebar ----
with st.sidebar:
    st.subheader("ℹ️ GCP status")
    if config.GOOGLE_CLOUD_PROJECT:
        st.success(f"Project: `{config.GOOGLE_CLOUD_PROJECT}`")
    else:
        st.warning("`GOOGLE_CLOUD_PROJECT` not set. Add it to `.env` before running the agent.")

    db = state_module._get_db()
    if db is not None:
        st.info("Persistent memory: Firestore")
    else:
        st.info("Persistent memory: local JSON fallback (`.state/`)")

    history = state_module.list_sessions()
    if history:
        with st.expander(f"📚 Past sessions ({len(history)})", expanded=False):
            for h in sorted(history, key=lambda x: x.get("updated_at", ""), reverse=True)[:10]:
                st.markdown(f"**{h.get('topic', '?')}** — `{h.get('status', '')}`")
                steps = ", ".join(h.get("steps_completed", []) or [])
                if steps:
                    st.caption(steps)

# --------------------------------------------------------------- main ----
topic = st.text_area(
    "Topic or syllabus to study",
    placeholder="e.g. Transformer architecture for NLP — attention, positional encoding, self-attention, multi-head attention, training",
    height=110,
)

col_run, col_info = st.columns([1, 3])
with col_run:
    run_clicked = st.button("🚀 Run Agent", type="primary", disabled=not topic.strip())

if run_clicked:
    topic = topic.strip()
    user_id = "student"
    session_id = uuid.uuid4().hex
    state_module.start_session(session_id, topic)

    runner = _make_runner()
    _create_session(runner, user_id, session_id, {"session_id": session_id, "topic": topic})

    user_content = None
    from google.genai import types

    user_content = types.Content(role="user", parts=[types.Part(text=topic)])

    log_lines = []
    final_text = ""

    with st.container(border=True):
        st.subheader("🔄 Agent run log")
        log_placeholder = st.empty()
        status_placeholder = st.empty()

        for event in runner.run(user_id=user_id, session_id=session_id, new_message=user_content):
            if event.content and event.content.parts:
                for part in event.content.parts:
                    function_call = getattr(part, "function_call", None)
                    function_response = getattr(part, "function_response", None)
                    if function_call is not None and getattr(function_call, "name", None):
                        name = function_call.name
                        args = getattr(function_call, "args", None) or {}
                        arg_summary = str(args)[:120]
                        log_lines.append(f"⚙️ **{_step_label(name)}**  `{arg_summary}`")
                    elif function_response is not None:
                        log_lines.append("   ✅ tool result received")
                    elif getattr(part, "text", None):
                        if not event.is_final_response():
                            log_lines.append(f"💬 {part.text[:200]}")
                        else:
                            final_text = part.text

            with log_placeholder.container():
                for line in log_lines:
                    st.markdown(line)
            if event.is_final_response():
                status_placeholder.success("✅ Agent finished")
            else:
                status_placeholder.info("⏳ Agent working…")

    session_snap = state_module.get_session(session_id)
    if session_snap:
        steps = session_snap.get("steps_completed", [])
        st.caption("Completed steps: " + " → ".join(steps) if steps else "")

    if final_text:
        with st.container(border=True):
            st.subheader("🤖 Agent summary")
            st.markdown(final_text)

    guide, guide_path = _latest_guide()
    if guide and isinstance(guide, dict) and guide.get("topic"):
        with st.container(border=True):
            st.subheader("📖 Study guide")
            st.markdown(f"### {guide.get('topic')}")
            if guide.get("summary"):
                st.markdown(guide["summary"])
            for section in guide.get("sections", []):
                st.markdown(f"#### {section.get('heading', 'Notes')}")
                st.markdown(section.get("content", ""))
                if section.get("source"):
                    st.caption(f"Source: {section['source']}")
            st.markdown("### 📝 Quiz")
            for i, q in enumerate(guide.get("quiz", []), 1):
                st.markdown(f"**{i}. {q.get('question', '')}**")
                options = q.get("options", [])
                answer_index = q.get("answer_index", 0)
                for j, opt in enumerate(options):
                    marker = "✔" if j == answer_index else "•"
                    st.markdown(f"- {marker} {opt}")
                if q.get("explanation"):
                    st.caption(f"Explanation: {q['explanation']}")
            st.caption(f"Saved to: `{guide_path}`")

st.caption("Built with Google Agent Development Kit (ADK) + Gemini · Retrieval pipeline pre-existing from GyanVerse AI.")