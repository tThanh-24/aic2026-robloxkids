"""Writes competition-format CSV submissions.

Format rules enforced here (see competition spec):
  - UTF-8, comma-separated, NO header row.
  - video filename WITHOUT .mp4 extension.
  - frame IDs are integers.
  - KIS row:   video,frame
  - Q&A row:   video,frame,answer        (answer always quoted; embedded
                                           quotes doubled; <=100 chars)
  - TRAKE row: video,frame_1,...,frame_N (N must match the query's event
                                           count; frames must be given in
                                           chronological order by the caller
                                           -- this module does not reorder)
  - max 100 rows per query (truncated here, with a warning, not silently
    dropped without notice)
"""
from __future__ import annotations

import csv
import warnings
from dataclasses import dataclass
from pathlib import Path

MAX_CANDIDATES = 100
ANSWER_MAX_CHARS = 100


def _strip_video_ext(video: str) -> str:
    for ext in (".mp4", ".MP4", ".mkv", ".avi"):
        if video.endswith(ext):
            return video[: -len(ext)]
    return video


@dataclass
class KISCandidate:
    video: str
    frame: int


@dataclass
class QACandidate:
    video: str
    frame: int
    answer: str


@dataclass
class TrakeCandidate:
    video: str
    frames: list[int]  # length N, chronological order


def _open_writer(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    f = open(path, "w", newline="", encoding="utf-8")
    # QUOTE_MINIMAL + doubled-quote escaping is Python csv's default
    # behavior and matches the spec exactly (quote only when needed for
    # KIS, always for Q&A since we pre-decide that below).
    writer = csv.writer(f, delimiter=",", quoting=csv.QUOTE_MINIMAL)
    return f, writer


def _truncate(candidates: list, query_id: str) -> list:
    if len(candidates) > MAX_CANDIDATES:
        warnings.warn(
            f"[{query_id}] {len(candidates)} candidates given, truncating to "
            f"{MAX_CANDIDATES} (rank order preserved -- make sure your list "
            f"is already sorted best-first before calling the writer)."
        )
        return candidates[:MAX_CANDIDATES]
    return candidates


def write_kis_csv(path: Path, candidates: list[KISCandidate], query_id: str = "") -> None:
    candidates = _truncate(candidates, query_id)
    f, writer = _open_writer(path)
    try:
        for c in candidates:
            writer.writerow([_strip_video_ext(c.video), int(c.frame)])
    finally:
        f.close()


def write_qa_csv(path: Path, candidates: list[QACandidate], query_id: str = "") -> None:
    # NOTE: deliberately does NOT use csv.writer for row assembly. We need
    # to always force-quote the answer column while leaving video/frame
    # unquoted (matches the spec's examples exactly), and csv.writer's
    # QUOTE_MINIMAL has no per-column quoting control -- it would either
    # quote nothing (answer has no delimiter) or double-quote an
    # already-quoted string. Building the line manually keeps quoting
    # rules explicit and easy to unit-test.
    candidates = _truncate(candidates, query_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        for c in candidates:
            answer = c.answer
            if len(answer) > ANSWER_MAX_CHARS:
                warnings.warn(
                    f"[{query_id}] answer exceeds {ANSWER_MAX_CHARS} chars, "
                    f"truncating: {answer!r}"
                )
                answer = answer[:ANSWER_MAX_CHARS]
            # "the safest approach is simply to quote every Q&A answer" --
            # always wrap in quotes, doubling any embedded quote chars.
            safe_answer = answer.replace('"', '""')
            video = _strip_video_ext(c.video)
            f.write(f'{video},{int(c.frame)},"{safe_answer}"\r\n')


def write_trake_csv(path: Path, candidates: list[TrakeCandidate], query_id: str = "") -> None:
    candidates = _truncate(candidates, query_id)
    f, writer = _open_writer(path)
    try:
        for c in candidates:
            row = [_strip_video_ext(c.video)] + [int(fr) for fr in c.frames]
            writer.writerow(row)
    finally:
        f.close()


def write_submission(
    task: str, path: Path, candidates: list, query_id: str = ""
) -> None:
    """Dispatch to the right writer by task type ('kis' | 'qa' | 'trake')."""
    task = task.lower()
    if task == "kis":
        write_kis_csv(path, candidates, query_id)
    elif task == "qa":
        write_qa_csv(path, candidates, query_id)
    elif task == "trake":
        write_trake_csv(path, candidates, query_id)
    else:
        raise ValueError(f"Unknown task type: {task!r} (expected kis/qa/trake)")
