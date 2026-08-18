"""Main orchestration entrypoint: query file(s) -> CSV submission(s).

This is currently a skeleton wiring together the parts that ARE built
(query parsing, CSV writing, config) with clearly marked TODOs for the
retrieval/model logic from the build-order in the README. Fill in
`run_kis`, `run_qa`, `run_trake` as those pieces come online -- the I/O
contract (Query in, csv_writer.*Candidate out) is already fixed so you
can build/test each task's retrieval logic independently.

Usage (once retrieval is implemented):
    python -m aic_system.pipeline --config default --queries data/queries --out data/submission
"""
from __future__ import annotations

import argparse
from pathlib import Path

from aic_system.config import load_config, resolve_path
from aic_system.io.csv_writer import write_submission
from aic_system.io.query_parser import Query, load_query_dir


def run_kis(query: Query, cfg: dict) -> list:
    # TODO: embed query.text, search the FAISS visual/text index, group
    # hits into (video, frame) candidates, sort best-first, return
    # list[csv_writer.KISCandidate] of length <= 100.
    raise NotImplementedError("KIS retrieval not implemented yet")


def run_qa(query: Query, cfg: dict) -> list:
    # TODO: reuse run_kis's retrieval for top-K frame candidates, run the
    # VQA model per candidate, re-rank, return list[csv_writer.QACandidate].
    raise NotImplementedError("Q&A retrieval not implemented yet")


def run_trake(query: Query, cfg: dict) -> list:
    # TODO: split query.text into events, retrieve per event, lock to a
    # single video, order chronologically, return list[csv_writer.TrakeCandidate].
    raise NotImplementedError("TRAKE retrieval not implemented yet")


_TASK_RUNNERS = {"kis": run_kis, "qa": run_qa, "trake": run_trake}


def process_query_file(stem: str, queries: list[Query], cfg: dict, out_dir: Path) -> None:
    if not queries:
        return
    task = queries[0].task
    runner = _TASK_RUNNERS[task]

    all_candidates = []
    for q in queries:
        try:
            all_candidates.extend(runner(q, cfg))
        except NotImplementedError as e:
            print(f"  [{q.query_id}] skipped: {e}")
            return

    out_path = out_dir / f"{stem}.csv"
    write_submission(task, out_path, all_candidates, query_id=stem)
    print(f"  wrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="default")
    ap.add_argument("--queries", default=None, help="override configs paths.queries_dir")
    ap.add_argument("--out", default=None, help="override configs paths.submission_dir")
    args = ap.parse_args()

    cfg = load_config(args.config)
    queries_dir = Path(args.queries) if args.queries else resolve_path(cfg, "queries_dir")
    out_dir = Path(args.out) if args.out else resolve_path(cfg, "submission_dir")
    out_dir.mkdir(parents=True, exist_ok=True)

    query_files = load_query_dir(queries_dir)
    if not query_files:
        print(f"No query-*.txt files found in {queries_dir}")
        return

    for stem, queries in query_files.items():
        print(f"Processing {stem} ({len(queries)} queries, task={queries[0].task})...")
        process_query_file(stem, queries, cfg, out_dir)


if __name__ == "__main__":
    main()
