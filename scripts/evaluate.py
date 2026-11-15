#!/usr/bin/env python
"""
Evaluation Mode (SRS 3.2.9): run test questions through the full tutor and
its simpler versions, and compare the results.

Versions (SRS names):
    full         Full tutor: evidence (ECA) + learner state (LSE) -> action (DSC)
    baseline_a   Baseline A: normal RAG (retrieve and answer, no state awareness)
    baseline_b   Baseline B: learner state only (evidence is ignored)
    baseline_c   Baseline C: evidence confidence only (learner is always Uncertain)
    plain_llm    Optional extra: the LLM alone, no retrieval (not run by default)

The learner state of a question comes from the "learner_state" column of the
CSV, or is Uncertain when empty. No student data is used.

Run it from the project folder, with the web server STOPPED:

    cd backend
    python ..\\scripts\\evaluate.py

Quick test with 3 questions:   python ..\\scripts\\evaluate.py --limit 3
Only some versions:            python ..\\scripts\\evaluate.py --versions full,baseline_a
A stopped run is continued by running the same command again.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import logging
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = SCRIPT_DIR.parent / "backend"

DEFAULT_VERSIONS = ["full", "baseline_a", "baseline_b", "baseline_c"]
ALL_VERSIONS = DEFAULT_VERSIONS + ["plain_llm"]
VERSION_LABELS = {
    "full": "Full tutor",
    "baseline_a": "Baseline A: normal RAG",
    "baseline_b": "Baseline B: learner state only",
    "baseline_c": "Baseline C: evidence confidence only",
    "plain_llm": "Plain LLM (no retrieval)",
}

LEARNER_STATES = ("understanding", "uncertain", "confused", "repeated_failure")
_STATE_ALIASES = {
    "repeatedly_failing": "repeated_failure",
    "repeated_failing": "repeated_failure",
    "repeatedly_failure": "repeated_failure",
    "understand": "understanding",
    "uncertain": "uncertain",
    "confuse": "confused",
}

RESULT_FIELDS = [
    "id", "version", "question", "answerable", "reference_page", "learner_state",
    "evidence_state", "evidence_score", "teaching_action", "outcome",
    "source_pages", "source_match", "response_time_s", "answer", "error",
    "manual_label",
]

# Used by the baselines that have no decision table.
RAG_PROMPT = (
    "Answer the question using only the passages below. "
    "Keep the answer under 150 words.\n\n"
    "Passages:\n{passages}\n\n"
    "Question: {question}"
)
PLAIN_PROMPT = "Answer the question. Keep the answer under 150 words.\n\nQuestion: {question}"

# A generated answer that says the material does not contain the answer counts as a refusal.
# This is a rough text check; the answers are saved so they can be checked by hand.
REFUSAL_RE = re.compile(
    r"(could not find|couldn't find|can't find|cannot find|"
    r"not (?:found|mentioned|covered|provided|included|available|stated|specified|discussed)"
    r" (?:in|by|within) (?:the |your )?(?:provided |given |study )?(?:passages?|material|context|documents?|text|lectures?)|"
    r"(?:does not|doesn't|do not|don't) (?:contain|mention|provide|cover|include|specify|discuss|say|have|list|state|explain|describe|define)|"
    r"no (?:information|mention|reference|details)|not enough information|"
    r"insufficient information|unable to (?:find|answer|provide)|"
    r"(?:cannot|can't) (?:answer|provide)|will not guess|"
    r"not in the (?:provided |given |study )?(?:passages|material|context|document|text|lecture))",
    re.IGNORECASE,
)


# ──────────────────────────── helpers (no app imports) ────────────────────────────

@dataclass
class Question:
    id: str
    question: str
    answerable: bool
    reference_page: int | None
    learner_state: str


@dataclass
class Pipeline:
    """The backend pieces the evaluation needs (built from the real app, or faked in tests)."""
    retriever: Any
    evidence: Any
    decision: Any
    prompt_builder: Any
    llm: Any
    EvidenceState: Any
    make_message: Any
    refusal_message: str
    top_k: int = 5
    document_ids: list[str] | None = None
    retry_waits: tuple[float, ...] = (15.0, 30.0, 60.0)
    llm_called: bool = field(default=False)


def normalise_learner_state(value: str | None) -> str:
    text = (value or "").strip().lower().replace(" ", "_").replace("-", "_")
    if not text:
        return "uncertain"
    text = _STATE_ALIASES.get(text, text)
    if text not in LEARNER_STATES:
        raise ValueError(
            f"learner_state '{value}' is not one of: {', '.join(LEARNER_STATES)}"
        )
    return text


def load_questions(path: Path) -> list[Question]:
    if not path.exists():
        raise SystemExit(f"Question file not found: {path}")
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        missing = {"id", "question", "answerable"} - set(reader.fieldnames or [])
        if missing:
            raise SystemExit(
                f"The question file is missing the column(s): {', '.join(sorted(missing))}"
            )
        questions: list[Question] = []
        for line_no, row in enumerate(reader, start=2):
            qid = (row.get("id") or "").strip()
            text = (row.get("question") or "").strip()
            if not qid or not text:
                raise SystemExit(f"Line {line_no}: id and question must not be empty.")
            answerable = (row.get("answerable") or "").strip().lower() in {"yes", "y", "true", "1"}
            ref = (row.get("reference_page") or "").strip()
            try:
                learner = normalise_learner_state(row.get("learner_state"))
            except ValueError as exc:
                raise SystemExit(f"Line {line_no}: {exc}")
            questions.append(
                Question(qid, text, answerable, int(ref) if ref.isdigit() else None, learner)
            )
    if not questions:
        raise SystemExit("The question file has no questions.")
    return questions


def is_llm_error(text: str) -> bool:
    return "[LLM Error:" in text or text.startswith("Error: OPENAI_API_KEY")


def classify_outcome(action_value: str, answer: str) -> str:
    """refused | clarified | answered"""
    if action_value == "state_uncertainty":
        return "refused"
    if action_value == "clarify":
        return "clarified"
    return "refused" if REFUSAL_RE.search(answer or "") else "answered"


def format_passages(citations: list[Any]) -> str:
    return "\n\n".join(
        f"[{i}] (page {c.page_number}) {c.text}" for i, c in enumerate(citations, start=1)
    )


async def collect(llm: Any, messages: list[Any]) -> str:
    text = ""
    async for chunk in llm.generate_stream(messages, context=""):
        text += chunk
    return text


async def call_llm(p: Pipeline, messages: list[Any]) -> tuple[str, str]:
    """Returns (answer, error). Retries a few times (e.g. after a 503 or rate limit)."""
    attempt = 0
    while True:
        p.llm_called = True
        text = (await collect(p.llm, messages)).strip()
        if not is_llm_error(text):
            return text, ""
        if attempt >= len(p.retry_waits):
            return "", text
        await asyncio.sleep(p.retry_waits[attempt])
        attempt += 1


def _value(x: Any) -> str:
    return str(getattr(x, "value", x)) if x is not None else ""


async def answer_question(p: Pipeline, version: str, q: Question) -> dict[str, Any]:
    """Run one question through one version. Returns one result row."""
    started = time.perf_counter()
    conversation_id = f"eval-{version}-{q.id}"  # fresh, so the "Clarify twice" rule never carries over
    evidence_state = None
    evidence_score = ""
    action = None
    cited: list[Any] = []
    answer = ""
    error = ""

    if version == "plain_llm":
        messages = [p.make_message("user", PLAIN_PROMPT.format(question=q.question))]
        answer, error = await call_llm(p, messages)
    else:
        citations = await p.retriever.retrieve(
            query=q.question, top_k=p.top_k, document_ids=p.document_ids
        )
        ranked = sorted(citations, key=lambda c: c.score or 0.0, reverse=True)
        best = ranked[0].score if ranked else 0.0

        if version in ("full", "baseline_c"):
            evidence_state, best = await p.evidence.assess(q.question, citations)
        evidence_score = round(float(best or 0.0), 2)

        if version == "full":
            action = p.decision.decide(q.learner_state, evidence_state, conversation_id=conversation_id)
        elif version == "baseline_b":  # learner only: evidence is ignored (treated as Strong)
            action = p.decision.decide(q.learner_state, p.EvidenceState.STRONG, conversation_id=conversation_id)
        elif version == "baseline_c":  # evidence only: learner is always Uncertain
            action = p.decision.decide("uncertain", evidence_state, conversation_id=conversation_id)

        if _value(action) == "state_uncertainty":
            answer = p.refusal_message  # fixed message, no LLM call (SRS 3.2.7)
        else:
            cited = ranked
            if action is None:  # baseline A: normal RAG
                messages = [p.make_message(
                    "user", RAG_PROMPT.format(passages=format_passages(ranked), question=q.question)
                )]
            else:
                messages = p.prompt_builder.build_messages(
                    history=[], query=q.question, citations=ranked, action=action
                )
            answer, error = await call_llm(p, messages)

    elapsed = round(time.perf_counter() - started, 2)
    action_value = _value(action)

    if error:
        outcome = "error"
    else:
        outcome = classify_outcome(action_value, answer)

    pages = sorted({c.page_number for c in cited})  # kept for every row, so rows can be rescored
    if q.reference_page is not None and outcome == "answered":
        source_match = "yes" if q.reference_page in pages else "no"
    else:
        source_match = ""

    return {
        "id": q.id,
        "version": version,
        "question": q.question,
        "answerable": "yes" if q.answerable else "no",
        "reference_page": q.reference_page if q.reference_page is not None else "",
        "learner_state": q.learner_state,
        "evidence_state": _value(evidence_state),
        "evidence_score": evidence_score,
        "teaching_action": action_value,
        "outcome": outcome,
        "source_pages": " ".join(str(x) for x in pages),
        "source_match": source_match,
        "response_time_s": elapsed,
        "answer": answer.replace("\r", " ").strip(),
        "error": error,
        "manual_label": "",
    }


# ──────────────────────────── results file, resume, summary ────────────────────────────

def read_results(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def latest_by_key(rows: list[dict[str, str]]) -> dict[tuple[str, str], dict[str, str]]:
    """The last row of each (id, version) wins, so a retried question replaces its failed row."""
    return {(r["id"], r["version"]): r for r in rows}


def append_row(path: Path, row: dict[str, Any]) -> None:
    new_file = not path.exists() or path.stat().st_size == 0
    with open(path, "a", newline="", encoding="utf-8-sig" if new_file else "utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_FIELDS)
        if new_file:
            writer.writeheader()
        writer.writerow(row)
        f.flush()


def _pct(num: int, den: int) -> str:
    return f"{100.0 * num / den:.0f}%" if den else "-"


def summarise(rows: list[dict[str, str]], question_ids: set[str], versions: list[str]) -> list[dict[str, Any]]:
    latest = latest_by_key(rows)
    summary = []
    for v in versions:
        mine = [r for (qid, ver), r in latest.items() if ver == v and qid in question_ids]
        if not mine:
                       continue
        ok = [r for r in mine if r["outcome"] != "error"]
        for r in ok:
            # Re-check each saved answer with the current refusal rules
            r["outcome"] = classify_outcome(r["teaching_action"], r["answer"])
        ans_q = [r for r in ok if r["answerable"] == "yes"]
        unans_q = [r for r in ok if r["answerable"] == "no"]
        unsupported = sum(r["outcome"] == "answered" for r in unans_q)
        correct_ref = sum(r["outcome"] == "refused" for r in unans_q)
        wrong_ref = sum(r["outcome"] == "refused" for r in ans_q)
        answered_ans = [r for r in ans_q if r["outcome"] == "answered"]
        with_ref = [r for r in answered_ans if r["source_match"] in ("yes", "no")]
        src_ok = sum(r["source_match"] == "yes" for r in with_ref)
        times = [float(r["response_time_s"]) for r in ok if r["response_time_s"]]
        summary.append({
            "version": VERSION_LABELS.get(v, v),
            "questions": len(mine),
            "answerable": len(ans_q),
            "unanswerable": len(unans_q),
            "unsupported_answers": unsupported,
            "unsupported_rate": _pct(unsupported, len(unans_q)),
            "correct_refusals": correct_ref,
            "correct_refusal_rate": _pct(correct_ref, len(unans_q)),
            "wrong_refusals": wrong_ref,
            "wrong_refusal_rate": _pct(wrong_ref, len(ans_q)),
            "clarifying_questions": sum(r["outcome"] == "clarified" for r in ok),
            "source_page_matches": f"{src_ok}/{len(with_ref)}",
            "source_match_rate": _pct(src_ok, len(with_ref)),
            "avg_response_time_s": round(sum(times) / len(times), 2) if times else "",
            "errors": len(mine) - len(ok),
        })
    return summary


def write_summary(path: Path, summary: list[dict[str, Any]]) -> None:
    if not summary:
        return
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)


def print_summary(summary: list[dict[str, Any]]) -> None:
    if not summary:
        print("No results yet.")
        return
    print("\n=== Results (side by side) ===")
    cols = [
        ("version", "Version"),
        ("unsupported_answers", "Unsupported"),
        ("correct_refusals", "Correct refusals"),
        ("wrong_refusals", "Wrong refusals"),
        ("clarifying_questions", "Clarify"),
        ("source_page_matches", "Source page"),
        ("avg_response_time_s", "Avg time (s)"),
        ("errors", "Errors"),
    ]
    widths = [max(len(h), *(len(str(r[k])) for r in summary)) for k, h in cols]
    print("  ".join(h.ljust(w) for (k, h), w in zip(cols, widths)))
    print("  ".join("-" * w for w in widths))
    for r in summary:
        print("  ".join(str(r[k]).ljust(w) for (k, h), w in zip(cols, widths)))
    print(
        "\nUnsupported = answered a question the material cannot answer (lower is better).\n"
        "Correct refusals = refused an unanswerable question (higher is better).\n"
        "Wrong refusals = refused a question the material can answer (lower is better).\n"
        "Source page = answered questions whose reference page was among the sources."
    )


# ──────────────────────────── main run ────────────────────────────

async def run(p: Pipeline, questions: list[Question], versions: list[str],
              out_path: Path, delay: float) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = {key for key, row in latest_by_key(read_results(out_path)).items() if not row["error"]}
    total = len(questions) * len(versions)
    count = 0
    for q in questions:
        for v in versions:
            count += 1
            if (q.id, v) in done:
                print(f"[{count}/{total}] {q.id} {v}: already done, skipped")
                continue
            p.llm_called = False
            row = await answer_question(p, v, q)
            append_row(out_path, row)
            print(f"[{count}/{total}] {q.id} {v}: {row['outcome']}"
                  f" ({row['response_time_s']}s)" + (f"  ERROR: {row['error'][:80]}" if row["error"] else ""))
            if p.llm_called and delay > 0:
                await asyncio.sleep(delay)  # stay within the free AI plan limits


async def build_pipeline(document_name: str | None) -> Pipeline:
    os.chdir(BACKEND_DIR)  # so that backend/.env and backend/data are found
    sys.path.insert(0, str(BACKEND_DIR))
    logging.basicConfig(level=logging.WARNING)

    from app import dependencies  # noqa: E402
    from app.config import settings  # noqa: E402
    from app.core.models import ChatMessage  # noqa: E402
    from app.core.tutor_models import EvidenceState  # noqa: E402
    from app.services.chat_service import STATE_UNCERTAINTY_MESSAGE  # noqa: E402

    print("Starting the backend components (loading the embedding model can take a minute)...")
    await dependencies.startup()

    metadata = dependencies.get_metadata_store()
    docs, _ = await metadata.list_documents(limit=200, offset=0)
    ready = [d for d in docs if _value(d.status).lower() in ("completed", "ready")]
    if document_name:
        ready = [d for d in ready if document_name.lower() in d.original_name.lower()]
    if not ready:
        raise SystemExit(
            "No completed document found. Upload your PDF in the app first "
            "(Documents tab, status 'completed')."
        )
    print("Documents used: " + ", ".join(d.original_name for d in ready))

    chat = dependencies.get_chat_service()
    return Pipeline(
        retriever=chat._retriever,
        evidence=chat._evidence,
        decision=chat._decision,
        prompt_builder=chat._prompt_builder,
        llm=chat._llm,
        EvidenceState=EvidenceState,
        make_message=lambda role, content: ChatMessage(role=role, content=content),
        refusal_message=STATE_UNCERTAINTY_MESSAGE,
        top_k=settings.TOP_K_RETRIEVAL,
        document_ids=[d.id for d in ready],
    )


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="HoloTutor-X evaluation mode (SRS 3.2.9)")
    ap.add_argument("--questions", type=Path, default=SCRIPT_DIR / "test_questions.csv",
                    help="CSV with columns id, question, answerable, reference_page, learner_state")
    ap.add_argument("--versions", default=",".join(DEFAULT_VERSIONS),
                    help="comma list from: " + ", ".join(ALL_VERSIONS))
    ap.add_argument("--document", default=None,
                    help="use only documents whose name contains this text (default: all completed)")
    ap.add_argument("--out", type=Path, default=SCRIPT_DIR / "results" / "results.csv")
    ap.add_argument("--delay", type=float, default=4.0,
                    help="seconds to wait after each AI call (free plan limits)")
    ap.add_argument("--limit", type=int, default=None, help="only the first N questions (quick test)")
    args = ap.parse_args()
    versions = [v.strip() for v in args.versions.split(",") if v.strip()]
    bad = [v for v in versions if v not in ALL_VERSIONS]
    if bad:
        ap.error(f"unknown version(s): {', '.join(bad)}. Choose from: {', '.join(ALL_VERSIONS)}")
    args.versions = versions
    return args


def main() -> None:
    args = parse_args()
    questions = load_questions(args.questions.resolve())
    if args.limit:
        questions = questions[: args.limit]
    out_path = args.out.resolve()

    async def go() -> None:
        pipeline = await build_pipeline(args.document)
        await run(pipeline, questions, args.versions, out_path, args.delay)

    try:
        asyncio.run(go())
    except KeyboardInterrupt:
        print("\nStopped. Run the same command again to continue where it stopped.")

    summary = summarise(read_results(out_path), {q.id for q in questions}, args.versions)
    summary_path = out_path.with_name("results_summary.csv")
    write_summary(summary_path, summary)
    print_summary(summary)
    print(f"\nPer-question results: {out_path}\nSummary:              {summary_path}")


if __name__ == "__main__":
    main()
