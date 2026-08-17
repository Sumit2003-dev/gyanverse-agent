"""agent/tools.py
The five scoped tools the GyanVerse agent can call. Each tool is isolated and
single-purpose (maps to the "isolated / scoped tools" judging criterion):

    retrieve_context   -> FAISS/Chroma retrieval over the indexed study material
    identify_gaps      -> flags subtopics missing from the retrieved context
    fetch_supplementary-> optional external fetch for a missing gap
    generate_study_guide-> structured notes + quiz grounded in the context
    deliver_output     -> persist the study guide (file / Firestore)

Every tool returns a JSON-serializable dict with a "status" key so the LLM can
react to successes and failures (failure-tolerant agent design).
"""
import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request
import uuid

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import appconfig as config
from . import llm
from . import state as state_module
from retrieval.rag_engine import RAGEngine, engine_exists

_engine = None


def _get_engine():
    global _engine
    if _engine is None:
        _engine = RAGEngine()
    return _engine


def _session_id(tool_context):
    if tool_context is None:
        return None
    try:
        return tool_context.state.get("session_id")
    except Exception:
        return None


def _slugify(text, max_len=60):
    slug = re.sub(r"[^a-z0-9]+", "-", str(text).lower()).strip("-")
    return slug[:max_len] or "study-guide"


def _normalize_guide(result, topic):
    guide = {
        "topic": topic,
        "summary": result.get("summary", "") if isinstance(result, dict) else "",
        "sections": result.get("sections", []) if isinstance(result, dict) else [],
        "quiz": result.get("quiz", []) if isinstance(result, dict) else [],
    }
    if isinstance(guide["sections"], str):
        guide["sections"] = [{"heading": "Notes", "content": guide["sections"], "source": "synthesised"}]
    if isinstance(guide["quiz"], str):
        guide["quiz"] = []
    return guide


def _guide_to_markdown(guide):
    if not isinstance(guide, dict):
        return str(guide)
    lines = [f"# {guide.get('topic', 'Study Guide')}", "", guide.get("summary", ""), ""]
    for section in guide.get("sections", []):
        lines.append(f"## {section.get('heading', 'Notes')}")
        lines.append(section.get("content", ""))
        if section.get("source"):
            lines.append(f"*Source: {section['source']}*")
        lines.append("")
    quiz = guide.get("quiz", [])
    if quiz:
        lines.append("## Quiz")
        for i, q in enumerate(quiz, 1):
            lines.append(f"**{i}. {q.get('question', '')}**")
            options = q.get("options", [])
            answer_index = q.get("answer_index", 0)
            for j, opt in enumerate(options):
                marker = "✔" if j == answer_index else "•"
                lines.append(f"- {marker} {opt}")
            if q.get("explanation"):
                lines.append(f"  _{q['explanation']}_")
            lines.append("")
    return "\n".join(lines)


def _wikipedia_summary(title):
    query = urllib.parse.quote(str(title).strip().replace(" ", "_"))
    url = config.WIKIPEDIA_API.format(title=query)
    try:
        with urllib.request.urlopen(url, timeout=config.FETCH_TIMEOUT_SECONDS) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        extract = (data.get("extract") or "").strip()
        if extract:
            return extract, f"wikipedia:{data.get('title', title)}"
    except Exception:
        pass
    return None, None


# --------------------------------------------------------------------------- #
# Tool 1: retrieve_context
# --------------------------------------------------------------------------- #
def retrieve_context(topic: str, tool_context=None):
    """Search the indexed course documents and return the most relevant context chunks for the given topic.

    Args:
        topic (str): The subject or question to look up in the indexed study material.

    Returns:
        dict: {status, count, context: [{text, source, score}], sources: [source names]}
    """
    if not engine_exists():
        return {
            "status": "error",
            "error_message": (
                "No FAISS index found. Run `python retrieval/ingest.py` after adding "
                "documents to the data/ folder."
            ),
            "count": 0,
            "context": [],
            "sources": [],
        }
    try:
        results = _get_engine().retrieve(topic)
    except Exception as exc:
        return {
            "status": "error",
            "error_message": f"Retrieval failed: {exc}",
            "count": 0,
            "context": [],
            "sources": [],
        }
    context = [
        {
            "text": r["page_content"],
            "source": r["metadata"].get("source", "unknown"),
            "score": r["score"],
        }
        for r in results
    ]
    sid = _session_id(tool_context)
    if sid:
        state_module.update_step(sid, "retrieved")
    return {
        "status": "success",
        "count": len(context),
        "context": context,
        "sources": sorted({c["source"] for c in context}),
    }


# --------------------------------------------------------------------------- #
# Tool 2: identify_gaps
# --------------------------------------------------------------------------- #
def identify_gaps(topic: str, retrieved_docs: list, tool_context=None):
    """Analyse the retrieved documents and identify important subtopics that are missing or under-covered.

    Args:
        topic (str): The study topic being prepared.
        retrieved_docs (list): The context chunks returned by retrieve_context (each item is a document text string).

    Returns:
        dict: {status, count, gaps: [{subtopic, reason}]}
    """
    if isinstance(retrieved_docs, dict):
        docs_text = json.dumps(retrieved_docs.get("context", retrieved_docs), indent=2)[:20000]
    else:
        docs_text = "\n---\n".join(str(d) for d in (retrieved_docs or []))[:20000]

    prompt = f"""You are an expert curriculum reviewer.
Topic to teach: {topic}

The following text was retrieved from the student's course material:

<retrieved>
{docs_text}
</retrieved>

Analyse what a complete study guide for this topic MUST cover, then decide which
important subtopics are MISSING or only superficially covered in the retrieved text.

Respond ONLY with JSON of the form:
{{"gaps": [{{"subtopic": "<name>", "reason": "<one sentence why it is missing/important>"}}]}}
List at most 5 gaps. If the retrieved text already covers the topic well, return an empty list.
"""
    result = llm.generate_json(prompt)
    gaps_raw = result.get("gaps", []) if isinstance(result, dict) else []
    if isinstance(gaps_raw, str):
        gaps_raw = [gaps_raw]
    gaps = []
    for g in gaps_raw:
        if isinstance(g, dict):
            gaps.append({"subtopic": str(g.get("subtopic", "")), "reason": str(g.get("reason", ""))})
        else:
            gaps.append({"subtopic": str(g), "reason": ""})
    gaps = [g for g in gaps if g["subtopic"]][:5]

    sid = _session_id(tool_context)
    if sid:
        state_module.update_step(sid, "gaps_identified")
        for g in gaps:
            state_module.add_gap(sid, g["subtopic"])
    return {"status": "success", "count": len(gaps), "gaps": gaps}


# --------------------------------------------------------------------------- #
# Tool 3: fetch_supplementary
# --------------------------------------------------------------------------- #
def fetch_supplementary(gap_topic: str, tool_context=None):
    """Fetch short supplementary material for a subtopic that is missing from the indexed documents.

    Args:
        gap_topic (str): A subtopic that was identified as missing or under-covered.

    Returns:
        dict: {status, content, source}
    """
    content, source = _wikipedia_summary(gap_topic)
    if content is None:
        try:
            prompt = (
                f"Provide a concise, accurate 3-6 sentence primer on the topic: {gap_topic}. "
                "Focus on the core definitions and key points a student must know."
            )
            content = llm.generate(prompt, temperature=0.2, max_output_tokens=600)
            source = "gemini-generated"
        except Exception as exc:
            return {"status": "error", "error_message": str(exc), "content": "", "source": "none"}

    sid = _session_id(tool_context)
    if sid:
        state_module.update_step(sid, "supplementing")
        state_module.resolve_gap(sid, gap_topic)
    return {"status": "success", "content": content, "source": source}


# --------------------------------------------------------------------------- #
# Tool 4: generate_study_guide
# --------------------------------------------------------------------------- #
def generate_study_guide(topic: str, context: dict, tool_context=None):
    """Generate a structured study guide (sectioned notes + a quiz) grounded in the provided context.

    Args:
        topic (str): The study topic.
        context (dict): Retrieved context and any supplementary material, with keys
            context (list of chunks), supplementary (list of {subtopic, content, source}).

    Returns:
        dict: {status, topic, summary, sections: [{heading, content, source}],
               quiz: [{question, options, answer_index, explanation}]}
    """
    if isinstance(context, dict):
        chunks = context.get("context", [])
        supplementary = context.get("supplementary", [])
    else:
        chunks, supplementary = context, []

    chunks_text = "\n".join(
        f"[Source: {c.get('source', 'unknown')}]\n{c.get('text', c) if isinstance(c, dict) else c}"
        for c in (chunks or [])
    )
    extra_text = "\n".join(
        f"[Supplementary: {s.get('subtopic', s.get('source', 'web'))}]\n{s.get('content', '')}"
        for s in (supplementary or [])
    )

    prompt = f"""You are an expert study-guide author.
Create a high-quality study guide for the topic: {topic}

Use the retrieved course material below as the PRIMARY source of truth. Where the
material is thin, incorporate the supplementary material. Never invent facts;
mark uncertain points clearly.

<retrieved>
{chunks_text}
</retrieved>

<supplementary>
{extra_text}
</supplementary>

Respond ONLY with JSON matching this exact structure:
{{
  "summary": "<2-3 sentence overview>",
  "sections": [
    {{"heading": "<section title>", "content": "<thorough notes in markdown>", "source": "<source doc name or 'supplementary'>"}}
  ],
  "quiz": [
    {{"question": "<question>", "options": ["<a>", "<b>", "<c>", "<d>"], "answer_index": <0-3>, "explanation": "<why>"}}
  ]
}}
Include 3-6 sections and 3-5 quiz questions.
"""
    result = llm.generate_json(prompt)
    guide = _normalize_guide(result, topic)

    sid = _session_id(tool_context)
    if sid:
        state_module.update_step(sid, "generating")
        try:
            tool_context.state["temp:study_guide"] = guide
        except Exception:
            pass
    return {"status": "success", **guide}


# --------------------------------------------------------------------------- #
# Tool 5: deliver_output
# --------------------------------------------------------------------------- #
def deliver_output(content: dict, destination: str = "both", tool_context=None):
    """Persist the finished study guide to the requested destination.

    Args:
        content (dict): The study guide produced by generate_study_guide.
        destination (str): One of "file", "firestore", or "both". "file" saves JSON
            and Markdown to outputs/; "firestore" stores it under the session.

    Returns:
        dict: {status, destination, paths, session_id}
    """
    topic = content.get("topic", "study-guide") if isinstance(content, dict) else "study-guide"
    slug = _slugify(topic)
    sid = _session_id(tool_context) or uuid.uuid4().hex[:8]
    destination = (destination or "both").lower()
    paths = []

    try:
        os.makedirs(config.OUTPUTS_DIR, exist_ok=True)
        json_path = os.path.join(config.OUTPUTS_DIR, f"{slug}.json")
        md_path = os.path.join(config.OUTPUTS_DIR, f"{slug}.md")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(content, f, indent=2)
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(_guide_to_markdown(content))
        paths = [json_path, md_path]
    except Exception as exc:
        return {
            "status": "error",
            "error_message": f"Could not write output files: {exc}",
            "destination": destination,
            "paths": [],
            "session_id": sid,
        }

    if destination in ("firestore", "both"):
        try:
            db = state_module._get_db()
            if db is not None:
                now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                db.collection(config.SESSION_COLLECTION).document(sid).collection("artifacts").document(
                    slug
                ).set({"content": content, "created_at": now})
        except Exception as exc:
            print(f"[deliver_output] Firestore write skipped: {exc}", file=sys.stderr)

    state_module.update_step(sid, "delivered")
    return {"status": "success", "destination": destination, "paths": paths, "session_id": sid}