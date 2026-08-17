# GyanVerse Agent — Architecture

## System diagram

```
                         ┌────────────────────────────────────────────┐
                         │              Streamlit UI (app.py)         │
                         │   topic input → "Run Agent" → step log     │
                         │   rendered study guide + quiz              │
                         └───────────────▲────────────────────────────┘
                                         │  events stream (ADK Runner)
                                         │
                         ┌───────────────┴────────────────────────────┐
                         │         ADK Agent (agent/agent.py)         │
                         │   decides the tool sequence (agentic loop) │
                         │   model: Gemini (gemini-3.5-flash)         │
                         └───┬────────┬────────┬────────┬────────┬────┘
                             │        │        │        │        │
                ┌────────────┘        │        │        │        └────────────┐
                ▼                     ▼        ▼        ▼                     ▼
        retrieve_context       identify_gaps  fetch_      generate_       deliver_output
        (retrieval/ FAISS)     (Gemini)       supplementary study_guide    (file + Firestore)
                                              (Wikipedia)  (Gemini)
                │                                        │                    │
                │                                        │                    │
        ┌───────┴───────┐   ┌────────────────────┐   ┌───▼────┐   ┌──────────▼───────────┐
        │  Vector index │   │  Vertex AI Gemini  │   │ outputs/│   │  Firestore           │
        │  FAISS/Chroma │   │  (embeddings + LLM)│   │ .json   │   │  sessions/{id}       │
        │  retrieval/   │   │                    │   │ .md     │   │  sessions/{id}/gaps  │
        └───────────────┘   └────────────────────┘   └─────────┘   └──────────────────────┘

   Hosting: Cloud Run (min-instances 0)  ·  Auth: gcloud ADC / service account
```

## Data flow (one run)

1. **User** enters a topic in Streamlit.
2. **app.py** creates an ADK session (state carries `session_id` + `topic`) and
   starts a Firestore `sessions/{id}` document.
3. **ADK Runner** drives the agent. The **agent** (not a script) picks tools:
   - `retrieve_context(topic)` → top-K chunks from the FAISS index built over the
     student's course PDFs/DOCX/TXT.
   - `identify_gaps(topic, chunks)` → Gemini returns missing subtopics; each is
     written to `sessions/{id}/gaps`.
   - `fetch_supplementary(gap)` → Wikipedia summary (falls back to a Gemini
     primer); gap marked `resolved`.
   - `generate_study_guide(topic, context)` → Gemini returns structured
     `{summary, sections[], quiz[]}` JSON.
   - `deliver_output(guide)` → writes `outputs/{topic}.json` + `.md` and a copy
     to `sessions/{id}/artifacts/{topic}` in Firestore.
4. Each step is appended to `steps_completed` on the session doc
   (`retrieved → gaps_identified → supplementing → generating → delivered`).
   Failures are logged on the session (`error` field) and the run can resume —
   this is the **failure-tolerant / persistent-memory** story.
5. **app.py** renders the event stream (tool calls) and the final study guide.

## Resilience

- Firestore is used when a project id is set **or** `FIRESTORE_EMULATOR=true`
  (the emulator needs no real project). Otherwise state falls back to a local
  `.state/sessions.json` so development works offline.
- Tools return `status: success | error` dicts; the agent reacts gracefully and
  stops honestly if retrieval has no index.
- `fetch_supplementary` degrades from live Wikipedia fetch to Gemini-generated
  primers if the network or topic is unavailable.

## Cost control

- Cloud Run `--min-instances 0` → scales to zero when idle.
- Gemini **Flash** model (cheaper than Pro).
- Optional Firestore emulator / local embedding backend for zero-cost dev.