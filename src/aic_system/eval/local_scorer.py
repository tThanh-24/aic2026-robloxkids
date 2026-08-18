"""Reimplements the competition's scoring formula so you can validate a
submission LOCALLY, against a held-out ground truth split, before ever
spending one of your 3 real Codabench attempts per query package.

Scoring recap (from the competition spec):
  R@k = max R-Score over ranks 1..k  (for k in {1, 5, 20, 50, 100})
  Score = mean(R@1, R@5, R@20, R@50, R@100)

R-Score per row:
  KIS:   1 if video matches AND frame in [s, e], else 0
  Q&A:   1 if video matches AND frame in [s, e] AND answer matches, else 0
  TRAKE: 0 if video doesn't match; else (# events with frame_j in
         [s_j, e_j]) / N

Ground truth format expected here (you supply this from your own held-out
split -- the organizer does not give you full ground truth):
    {
      "query_id": "query-1-kis-3",
      "task": "kis",
      "video": "L21_V004",
      "interval": [10800, 11200]
    }
    {
      "query_id": "query-3-qa-1",
      "task": "qa",
      "video": "L22_V010",
      "interval": [2900, 3100],
      "answer": "3"
    }
    {
      "query_id": "query-4-trake-2",
      "task": "trake",
      "video": "L23_V002",
      "intervals": [[1000,1200],[2000,2200],[3000,3200],[4000,4200]]
    }
"""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

R_AT_K_VALUES = (1, 5, 20, 50, 100)


def _normalize_video(name: str) -> str:
    for ext in (".mp4", ".MP4", ".mkv", ".avi"):
        if name.endswith(ext):
            return name[: -len(ext)]
    return name


def _normalize_answer(a: str) -> str:
    # Mirrors a conservative "exact match, case/whitespace-insensitive"
    # comparison. The real competition rule for semantic vs exact answer
    # comparison isn't fully specified in the doc provided -- if the
    # organizer publishes an exact rule (e.g. fuzzy/semantic match),
    # update this function; every other part of the scorer is unaffected.
    return " ".join(a.strip().lower().split())


@dataclass
class GTRow:
    query_id: str
    task: str
    video: str
    interval: tuple[int, int] | None = None          # kis / qa
    intervals: list[tuple[int, int]] | None = None    # trake
    answer: str | None = None                          # qa


def load_ground_truth(path: Path) -> dict[str, GTRow]:
    """Loads a JSONL file of ground truth rows (see module docstring)."""
    gt: dict[str, GTRow] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            row = GTRow(
                query_id=d["query_id"],
                task=d["task"].lower(),
                video=_normalize_video(d["video"]),
                interval=tuple(d["interval"]) if "interval" in d else None,
                intervals=[tuple(iv) for iv in d["intervals"]] if "intervals" in d else None,
                answer=d.get("answer"),
            )
            gt[row.query_id] = row
    return gt


def _read_submission_csv(path: Path, task: str) -> list[list[str]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter=",")
        for row in reader:
            if row:
                rows.append(row)
    return rows


def r_score_kis(row: list[str], gt: GTRow) -> float:
    video, frame = row[0], int(row[1])
    if _normalize_video(video) != gt.video:
        return 0.0
    s, e = gt.interval
    return 1.0 if s <= frame <= e else 0.0


def r_score_qa(row: list[str], gt: GTRow) -> float:
    video, frame, answer = row[0], int(row[1]), row[2]
    if _normalize_video(video) != gt.video:
        return 0.0
    s, e = gt.interval
    if not (s <= frame <= e):
        return 0.0
    return 1.0 if _normalize_answer(answer) == _normalize_answer(gt.answer or "") else 0.0


def r_score_trake(row: list[str], gt: GTRow) -> float:
    video = row[0]
    frames = [int(x) for x in row[1:]]
    if _normalize_video(video) != gt.video:
        return 0.0
    n = len(gt.intervals)
    if len(frames) != n:
        # Malformed row (wrong number of events) -- score 0 rather than
        # raising, so one bad row doesn't crash scoring of the whole file.
        return 0.0
    hits = sum(
        1 for fr, (s, e) in zip(frames, gt.intervals) if s <= fr <= e
    )
    return hits / n


_R_SCORE_FN = {"kis": r_score_kis, "qa": r_score_qa, "trake": r_score_trake}


def score_query(rows: list[list[str]], gt: GTRow) -> dict[str, float]:
    """Returns {'R@1': ..., 'R@5': ..., ..., 'score': ...} for one query."""
    fn = _R_SCORE_FN[gt.task]
    r_scores = [fn(row, gt) for row in rows[:100]]

    result = {}
    running_max = 0.0
    idx = 0
    for k in R_AT_K_VALUES:
        while idx < min(k, len(r_scores)):
            running_max = max(running_max, r_scores[idx])
            idx += 1
        result[f"R@{k}"] = running_max
    result["score"] = sum(result[f"R@{k}"] for k in R_AT_K_VALUES) / len(R_AT_K_VALUES)
    return result


def score_submission_dir(
    submission_dir: Path, ground_truth: dict[str, GTRow], query_task_map: dict[str, str]
) -> dict:
    """
    query_task_map: {query_id: task} so we know how to parse each CSV's
    columns even without ground truth (task is still needed to route rows).
    Returns a report dict with per-query and per-package aggregate scores.
    """
    per_query = {}
    for csv_path in sorted(submission_dir.glob("*.csv")):
        # One CSV file may contain rows for a single query in this repo's
        # convention (see scripts/run_submission.py) -- if your pipeline
        # instead writes one CSV per *package* with a query_id column,
        # adjust this loader accordingly.
        query_id = csv_path.stem
        if query_id not in ground_truth:
            continue
        gt = ground_truth[query_id]
        rows = _read_submission_csv(csv_path, gt.task)
        per_query[query_id] = score_query(rows, gt)

    if not per_query:
        return {"per_query": {}, "package_score": 0.0, "num_scored": 0}

    package_score = sum(q["score"] for q in per_query.values()) / len(per_query)
    return {
        "per_query": per_query,
        "package_score": package_score,
        "num_scored": len(per_query),
    }
