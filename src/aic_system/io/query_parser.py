"""Parses query-N-{kis,qa,trake}.txt files.

The competition doc specifies the *filename* convention precisely (suffix
tells you the task type) but does not specify the internal line format of
these .txt files. This module is written defensively: it infers task type
from the filename (authoritative), and parses content assuming one query
per non-empty line, optionally prefixed with an explicit query ID
("qid<TAB or comma or space>text"). If the real organizer files differ
(e.g. JSON, or one-query-per-file), update `_parse_lines` -- everything
downstream consumes `Query` objects, not raw text, so the blast radius of
a format surprise is contained to this one function.

*** VERIFY AGAINST A REAL DOWNLOADED QUERY PACKAGE BEFORE TRUSTING THIS. ***
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

TASK_KIS = "kis"
TASK_QA = "qa"
TASK_TRAKE = "trake"
VALID_TASKS = {TASK_KIS, TASK_QA, TASK_TRAKE}

_FILENAME_TASK_RE = re.compile(r"-(kis|qa|trake)\b", re.IGNORECASE)


@dataclass
class Query:
    query_id: str          # e.g. "query-1-kis" or an explicit id from the file
    task: str              # "kis" | "qa" | "trake"
    text: str              # the query / question text
    num_events: int | None = None  # TRAKE only: expected number of key events
    source_file: Path | None = None


def infer_task_from_filename(path: Path) -> str:
    m = _FILENAME_TASK_RE.search(path.stem)
    if not m:
        raise ValueError(
            f"Cannot infer task type from filename: {path.name!r} "
            f"(expected it to contain -kis, -qa, or -trake)"
        )
    return m.group(1).lower()


def _guess_num_events(text: str) -> int:
    """Rough heuristic count of key events for TRAKE, used only as a
    fallback display/sanity value -- the real event count for scoring
    purposes comes from the organizer's ground truth, not from us. Counts
    sequence-marker phrases; defaults to 1 if none found so callers don't
    crash on an unexpected format.
    """
    markers = re.split(
        r"(?:,\s*then\b|\bthen\b|\bafter that\b|\bnext\b|;)",
        text,
        flags=re.IGNORECASE,
    )
    n = len([m for m in markers if m.strip()])
    return max(n, 1)


def _parse_lines(raw_text: str, task: str, base_id: str) -> list[Query]:
    queries: list[Query] = []
    for i, line in enumerate(raw_text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        # Optional "qid<TAB>text" or "qid,text" prefix.
        qid = f"{base_id}-{i}"
        text = line
        for sep in ("\t", "|"):
            if sep in line:
                left, right = line.split(sep, 1)
                if left.strip() and len(left.strip()) < 32:
                    qid, text = left.strip(), right.strip()
                break

        q = Query(query_id=qid, task=task, text=text, source_file=None)
        if task == TASK_TRAKE:
            q.num_events = _guess_num_events(text)
        queries.append(q)
    return queries


def load_query_file(path: Path) -> list[Query]:
    task = infer_task_from_filename(path)
    raw_text = path.read_text(encoding="utf-8")
    base_id = path.stem
    queries = _parse_lines(raw_text, task, base_id)
    for q in queries:
        q.source_file = path
    return queries


def load_query_dir(dir_path: Path) -> dict[str, list[Query]]:
    """Returns {file_stem: [Query, ...]} for every query-*.txt in dir_path."""
    out: dict[str, list[Query]] = {}
    for path in sorted(dir_path.glob("query-*.txt")):
        out[path.stem] = load_query_file(path)
    return out
