# GyanVerse Agent

Autonomous study-guide builder for students. Give it a topic or syllabus and it
will **retrieve context** from your indexed course material, **identify knowledge
gaps**, **fetch supplementary material**, **generate a structured study guide**
(sectioned notes + quiz), and **deliver it** — deciding the sequence itself via
the Google Agent Development Kit (ADK) agent loop.

Built for the **All Things Agentic Hackathon · Taskmaster Track**.

> **Disclosure (required by submission rules):**
> - **Pre-existing component:** the FAISS retrieval pipeline in [`retrieval/`](retrieval/)
>   (document loading → chunking → embeddings → vector search) is carried over from
>   the author's earlier **GyanVerse AI** project. It was copied here as-is, then
>   lightly adapted to support a Google-native Gemini embedding backend in addition
>   to its original OpenAI backend.
> - **Newly built for this hackathon:** the ADK agent + five scoped tools
>   ([`agent/`](agent/)), Firestore cross-session state ([`agent/state.py`](agent/state.py)),
>   the Streamlit frontend ([`app.py`](app.py)), and the Cloud Run deployment
>   ([`Dockerfile`](Dockerfile), [`deploy.ps1`](deploy.ps1)).

---

## What it does

```
User topic → ADK Agent loop → Study guide + quiz
              ├─ retrieve_context       (FAISS vector search over course material)
              ├─ identify_gaps          (Gemini flags missing subtopics)
              ├─ fetch_supplementary    (Wikipedia / Gemini primer for gaps)
              ├─ generate_study_guide   (structured notes + quiz)
              └─ deliver_output         (saves to outputs/ + Firestore)
```

The agent decides *when* to call each tool (it is an agentic loop, not a fixed
script) and reports each step in the UI, so you can watch its autonomy live.

## Tech stack

| Piece | Technology |
|-------|-----------|
| Agent framework | Google Agent Development Kit (ADK 2.x) |
| Model | Gemini (`gemini-3.5-flash`) via Vertex AI |
| Retrieval | FAISS vector store (+ ChromaDB installed, ready to swap) |
| Embeddings | Gemini `text-embedding-005` (Vertex AI), OpenAI, or local ST — configurable |
| State / memory | Firestore (`sessions/{id}` + `sessions/{id}/gaps` subcollection), local fallback |
| Frontend | Streamlit |
| Hosting | Cloud Run (scales to zero) |

---

## Local setup

Prerequisites: Python 3.10+, a GCP project with **Vertex AI API** and
**Firestore API (Native mode)** enabled, and the `gcloud` CLI.

```powershell
# 1. Project
git clone https://github.com/Sumit2003-dev/gyanverse-agent.git   # or copy this folder
cd gyanverse-agent
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 2. Environment
Copy-Item .env.example .env
#   → edit .env: set GOOGLE_CLOUD_PROJECT, GOOGLE_CLOUD_LOCATION

# 3. Auth (local development)
gcloud auth application-default login

# 4. Build the retrieval index
#    drop your PDF/DOCX/TXT files into data/, then:
python retrieval/ingest.py

# 5. Run the app
python -m streamlit run app.py
```

Open http://localhost:8501, type a topic, hit **Run Agent**.

### Also try: the ADK dev UI / CLI

```powershell
adk run agent          # interactive CLI
adk web --port 8000    # browser dev UI (run from the project root)
```

## Testing (no GCP required)

```powershell
# End-to-end ADK agent loop using a built-in mock LLM (no credentials).
# Exercises the real runner: tool calls, ToolContext injection, state writes,
# and output file generation.
python smoke_test.py

# Phase 3 cross-session state CRUD (start → steps → gaps → resolve → delete).
# Uses the local fallback when no Firestore is reachable.
python tests\test_state.py
```

To exercise the **real Firestore** path locally (zero cost, no billing):

```powershell
.\scripts\firestore_emulator.ps1              # starts emulator on localhost:8080
# new terminal:
$env:FIRESTORE_EMULATOR = "true"
python tests\test_state.py                    # prints BACKEND: firestore-emulator
```

The emulator needs the `gcloud` CLI but **not** a real GCP project, and works
even with `GOOGLE_CLOUD_PROJECT` left blank.

---

## Cloud deployment (Cloud Run)

```powershell
.\deploy.ps1            # deploy.ps1 -ServiceName gyanverse-agent -ProjectId YOUR_PROJECT_ID
# or manually:
gcloud run deploy gyanverse-agent `
  --source . `
  --region us-central1 `
  --allow-unauthenticated `
  --min-instances 0 `
  --set-env-vars GOOGLE_CLOUD_PROJECT=YOUR_PROJECT_ID
```

Notes:
- The service uses Cloud Run's default service account; grant it
  **Vertex AI User** and **Cloud Datastore User** roles (or use a dedicated
  service account).
- `--min-instances 0` keeps cost near zero when idle.
- Firestore data persists across sessions, giving the agent **cross-session
  memory** (see `agent/state.py`).

## Configuration

See [`.env.example`](.env.example). Highlights:

| Variable | Default | Purpose |
|----------|---------|---------|
| `GOOGLE_CLOUD_PROJECT` | — | GCP project (Vertex AI + Firestore) |
| `GOOGLE_CLOUD_LOCATION` | `us-central1` | Vertex AI region |
| `GEMINI_MODEL` | `gemini-3.5-flash` | Model used by the agent |
| `EMBED_BACKEND` | `gemini` | `gemini` / `openai` / `local` |
| `FIRESTORE_EMULATOR` | `false` | Set `true` to use the local Firestore emulator |

## Troubleshooting

- **"GOOGLE_CLOUD_PROJECT is not set"** → fill `.env` and run
  `gcloud auth application-default login`.
- **"No FAISS index found"** → run `python retrieval/ingest.py` after adding
  files to `data/`.
- **HTTP 429 / quota** → your GCP trial credit is exhausted; check the billing
  page or switch `EMBED_BACKEND` to `openai`/`local`.

## Roadmap

- [ ] Swap to a local LLaMA/Gemma model (fully offline)
- [ ] Multi-document filtering (query specific files only)
- [ ] Chrome extension / mobile input
- [ ] Add an additional Google AI model (Gemma) for scoring — bonus criterion

See [`ARCHITECTURE.md`](ARCHITECTURE.md) for the system diagram.