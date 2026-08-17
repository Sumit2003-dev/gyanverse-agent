"""agent/state.py
Cross-session persistent state for the GyanVerse agent.

Uses Firestore when available (collection `sessions/{session_id}` with a
`sessions/{session_id}/gaps` subcollection) and transparently falls back to a
local JSON store so the agent can run without GCP during development.
"""
import json
import os
import sys
import time
import uuid

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import appconfig as config

STEPS = ["retrieved", "gaps_identified", "supplementing", "generating", "delivered"]

_db = None
_local_path = None


def _get_db():
    global _db
    if _db is not None:
        return _db
    try:
        from google.cloud import firestore

        if config.FIRESTORE_EMULATOR:
            # The emulator does not require (or verify) a real project id.
            os.environ.setdefault("FIRESTORE_EMULATOR_HOST", "localhost:8080")
            _db = firestore.Client(project=config.GOOGLE_CLOUD_PROJECT or "gyanverse-agent-dev")
            return _db
        if config.GOOGLE_CLOUD_PROJECT:
            _db = firestore.Client(project=config.GOOGLE_CLOUD_PROJECT)
            return _db
    except Exception as exc:
        print(f"[state] Firestore unavailable ({exc}); using local fallback.", file=sys.stderr)
    _db = None
    return None


def _local_store():
    global _local_path
    if _local_path is None:
        os.makedirs(config.LOCAL_STATE_DIR, exist_ok=True)
        _local_path = os.path.join(config.LOCAL_STATE_DIR, "sessions.json")
    data = {}
    if os.path.exists(_local_path):
        try:
            with open(_local_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            data = {}
    return data


def _write_local(data):
    os.makedirs(config.LOCAL_STATE_DIR, exist_ok=True)
    with open(os.path.join(config.LOCAL_STATE_DIR, "sessions.json"), "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def start_session(session_id: str, topic: str) -> str:
    """Create/update a session doc: topic, status, steps_completed, timestamps."""
    session_id = session_id or uuid.uuid4().hex
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    doc = {
        "topic": topic,
        "status": "started",
        "steps_completed": [],
        "created_at": now,
        "updated_at": now,
        "error": "",
    }
    db = _get_db()
    if db is not None:
        db.collection(config.SESSION_COLLECTION).document(session_id).set(doc, merge=True)
    else:
        data = _local_store()
        data[session_id] = {**doc, "gaps": []}
        _write_local(data)
    return session_id


def update_step(session_id: str, step: str, error: str = ""):
    """Append a completed step and update status/error, idempotently."""
    if step not in STEPS:
        return
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    db = _get_db()
    if db is not None:
        from google.cloud import firestore

        ref = db.collection(config.SESSION_COLLECTION).document(session_id)
        ref.update(
            {
                "steps_completed": firestore.ArrayUnion([step]),
                "status": "delivered" if step == "delivered" else step,
                "updated_at": now,
                "error": error or "",
            }
        )
    else:
        data = _local_store()
        session = data.setdefault(session_id, {})
        completed = list(session.get("steps_completed") or [])
        if step not in completed:
            completed.append(step)
        session["steps_completed"] = completed
        session["status"] = "delivered" if step == "delivered" else step
        session["updated_at"] = now
        session["error"] = error or ""
        _write_local(data)


def add_gap(session_id: str, subtopic: str):
    """Persist an identified knowledge gap with resolution status 'open'."""
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    gap = {"subtopic": subtopic, "status": "open", "created_at": now}
    db = _get_db()
    if db is not None:
        gap_id = uuid.uuid4().hex
        db.collection(config.SESSION_COLLECTION).document(session_id).collection("gaps").document(
            gap_id
        ).set(gap)
    else:
        data = _local_store()
        session = data.setdefault(session_id, {})
        session.setdefault("gaps", []).append(gap)
        _write_local(data)


def resolve_gap(session_id: str, subtopic: str):
    """Mark a gap resolved after supplementary material is gathered."""
    db = _get_db()
    if db is not None:
        gaps_ref = db.collection(config.SESSION_COLLECTION).document(session_id).collection("gaps")
        for doc in gaps_ref.where("subtopic", "==", subtopic).stream():
            doc.reference.update({"status": "resolved"})
    else:
        data = _local_store()
        session = data.setdefault(session_id, {})
        for gap in session.get("gaps", []):
            if gap.get("subtopic") == subtopic:
                gap["status"] = "resolved"
        _write_local(data)


def get_session(session_id: str):
    """Return the current session doc as a dict (or None)."""
    db = _get_db()
    if db is not None:
        snapshot = db.collection(config.SESSION_COLLECTION).document(session_id).get()
        if snapshot.exists:
            return snapshot.to_dict()
        return None
    return _local_store().get(session_id)


def list_sessions():
    """Return a summary list of sessions (for the UI sidebar)."""
    db = _get_db()
    if db is not None:
        out = []
        for doc in db.collection(config.SESSION_COLLECTION).stream():
            data = doc.to_dict()
            out.append({"session_id": doc.id, **data})
        return out
    data = _local_store()
    return [{"session_id": sid, **details} for sid, details in data.items()]


def delete_session(session_id: str):
    """Delete a session (and its gaps/artifacts subcollections in Firestore)."""
    db = _get_db()
    if db is not None:
        ref = db.collection(config.SESSION_COLLECTION).document(session_id)
        for sub in ("gaps", "artifacts"):
            for sub_doc in ref.collection(sub).stream():
                sub_doc.reference.delete()
        ref.delete()
    else:
        data = _local_store()
        data.pop(session_id, None)
        _write_local(data)