"""tests/test_state.py
Phase 3 smoke test for cross-session state (`agent/state.py`).

Runs the full CRUD cycle for a session:
    start_session → update_step → add_gap → resolve_gap → get_session
    → list_sessions → delete_session

It uses whichever backend is active:
  * FIRESTORE_EMULATOR=true and the emulator running on localhost:8080  → Firestore
  * GOOGLE_CLOUD_PROJECT set (real cloud Firestore)                     → Firestore
  * otherwise                                                            → local fallback (.state/sessions.json)

Run:
  # 1. local fallback (no GCP, no emulator)
  python tests/test_state.py

  # 2. against the Firestore emulator (needs gcloud CLI installed)
  gcloud components install beta cloud-firestore-emulator
  gcloud beta emulators firestore start --project=gyanverse-agent-dev
  # in a second terminal:
  $env:FIRESTORE_EMULATOR="true"
  python tests/test_state.py

Exit code 0 on success, 1 on failure. Prints the backend in use.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import appconfig as config
from agent import state as state_module

SESSION_ID = "state-test-session"


def main():
    mode = "local-fallback"
    db = state_module._get_db()
    if db is not None:
        mode = "firestore-emulator" if config.FIRESTORE_EMULATOR else "firestore-cloud"
    print(f"BACKEND: {mode}")

    state_module.delete_session(SESSION_ID)

    sid = state_module.start_session(SESSION_ID, "Unit Testing")
    assert sid == SESSION_ID

    state_module.update_step(SESSION_ID, "retrieved")
    state_module.update_step(SESSION_ID, "gaps_identified")
    state_module.add_gap(SESSION_ID, "mocking frameworks")
    state_module.resolve_gap(SESSION_ID, "mocking frameworks")
    state_module.update_step(SESSION_ID, "delivered")

    snap = state_module.get_session(SESSION_ID)
    assert snap is not None, "session doc missing"
    assert snap["steps_completed"] == ["retrieved", "gaps_identified", "delivered"], snap
    assert snap["status"] == "delivered", snap
    gaps = snap.get("gaps", [])
    assert gaps and gaps[0]["status"] == "resolved", gaps

    sessions = state_module.list_sessions()
    assert any(s["session_id"] == SESSION_ID for s in sessions), sessions

    state_module.delete_session(SESSION_ID)
    assert state_module.get_session(SESSION_ID) is None, "session still exists after delete"

    print("STATE_CRUD_TEST: OK")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except AssertionError as exc:
        print(f"STATE_CRUD_TEST: FAILED -> {exc}", file=sys.stderr)
        sys.exit(1)