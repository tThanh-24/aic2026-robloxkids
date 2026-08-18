"""Main orchestration entrypoint: query file(s) -> CSV submission(s).

Loading strategy:
  - Index resources (FAISS + sidecar + BM25) load once, lazily, on the
    first query that needs them.
  - The CLIP text encoder loads on first query.
  - The VLM (Qwen2-VL, ~16 GB VRAM at fp16) also loads lazily — every task
    uses it now (KIS/TRAKE rerank + Q&A answering), so the first query of
    ANY type triggers the load. Pass --no-vlm to stay retrieval-only.

Usage:
    python -m aic_system.pipeline --config default --queries data/queries --out data/submission
    python -m aic_system.pipeline --alpha 0.8 --top-k-vqa 30 --vlm-model Qwen/Qwen2-VL-2B-Instruct
    python -m aic_system.pipeline --no-vlm          # pure CLIP+BM25, skips Q&A queries
"""
from __future__ import annotations

import argparse
from pathlib import Path

from aic_system.config import load_config, resolve_path
from aic_system.io.csv_writer import write_submission
from aic_system.io.query_parser import Query, load_query_dir
from aic_system.retrieval.search import run_kis, run_qa, run_trake


class Resources:
    """Lazily-initialized shared state for all task runners."""

    def __init__(self, cfg: dict, args):
        self.cfg = cfg
        self.args = args
        self._index: dict | None = None
        self._clip = None
        self._vlm = None

    @property
    def index(self) -> dict:
        if self._index is None:
            from aic_system.ingest.indexer import load_index_resources

            self._index = load_index_resources(self.cfg)
            # Let CLI args override the config's retrieval hyperparameters.
            if self.args.alpha is not None:
                self.cfg["retrieval"]["alpha"] = self.args.alpha
            if self.args.top_k_vqa is not None:
                self.cfg["retrieval"]["top_k_for_vqa"] = self.args.top_k_vqa
            if self.args.vlm_model:
                self.cfg["models"]["vqa"]["name"] = self.args.vlm_model
        return self._index

    @property
    def clip(self):
        if self._clip is None:
            from aic_system.models.clip_encoder import CLIPTextEncoder

            m = self.cfg["models"]["clip_text"]
            print(f"  [clip] loading {m['name']} on {m.get('device', 'cuda')} ...")
            self._clip = CLIPTextEncoder(model_name=m["name"], device=m.get("device", "cuda"))
        return self._clip

    @property
    def vlm(self):
        if self._vlm is None:
            from aic_system.models.vlm import VLM

            m = self.cfg["models"]["vqa"]
            print(f"  [vlm] loading {m['name']} ({m.get('dtype', 'float16')}) ...")
            self._vlm = VLM(
                model_name=m["name"],
                fallback_name=m.get("fallback_name"),
                device=m.get("device", "cuda"),
                dtype=m.get("dtype", "float16"),
                max_new_tokens=m.get("max_new_tokens", 32),
            )
        return self._vlm


def _alpha(resources: Resources) -> float:
    return resources.cfg["retrieval"].get("alpha", 0.7)


def _vlm(resources: Resources):
    """The VLM shared by all task runners, or None under --no-vlm."""
    if getattr(resources.args, "no_vlm", False):
        return None
    return resources.vlm


def process_query_file(stem: str, queries: list[Query], resources: Resources, out_dir: Path) -> None:
    if not queries:
        return
    task = queries[0].task
    vlm = _vlm(resources)

    all_candidates = []
    for q in queries:
        try:
            if task == "kis":
                all_candidates.extend(
                    run_kis(q, resources.index, resources.clip, alpha=_alpha(resources), vlm=vlm)
                )
            elif task == "qa":
                if vlm is None:
                    print(f"  [{q.query_id}] skipped: Q&A requires the VLM (--no-vlm disables it)")
                    continue
                all_candidates.extend(
                    run_qa(q, resources.index, resources.clip, vlm, alpha=_alpha(resources))
                )
            elif task == "trake":
                all_candidates.extend(
                    run_trake(q, resources.index, resources.clip, alpha=_alpha(resources), vlm=vlm)
                )
            else:
                raise ValueError(f"unknown task {task!r}")
        except Exception as e:  # one bad query must not kill the package
            print(f"  [{q.query_id}] FAILED: {type(e).__name__}: {e}")
            continue

    if not all_candidates:
        print(f"  [{stem}] no candidates produced; CSV not written")
        return
    out_path = out_dir / f"{stem}.csv"
    write_submission(task, out_path, all_candidates, query_id=stem)
    print(f"  wrote {out_path} ({len(all_candidates)} rows)")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="default")
    ap.add_argument("--queries", default=None, help="override configs paths.queries_dir")
    ap.add_argument("--out", default=None, help="override configs paths.submission_dir")
    ap.add_argument("--alpha", type=float, default=None,
                    help="visual (CLIP) weight in fusion; 1-alpha goes to BM25")
    ap.add_argument("--top-k-vqa", type=int, default=None,
                    help="how many retrieved frames the VLM answers per Q&A query")
    ap.add_argument("--vlm-model", default=None,
                    help="override models.vqa.name (e.g. Qwen/Qwen2-VL-2B-Instruct)")
    ap.add_argument("--no-vlm", action="store_true",
                    help="disable all VLM stages (pure CLIP+BM25; Q&A queries are skipped)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    queries_dir = Path(args.queries) if args.queries else resolve_path(cfg, "queries_dir")
    out_dir = Path(args.out) if args.out else resolve_path(cfg, "submission_dir")
    out_dir.mkdir(parents=True, exist_ok=True)

    query_files = load_query_dir(queries_dir)
    if not query_files:
        print(f"No query-*.txt files found in {queries_dir}")
        return

    resources = Resources(cfg, args)
    for stem, queries in query_files.items():
        print(f"Processing {stem} ({len(queries)} queries, task={queries[0].task})...")
        process_query_file(stem, queries, resources, out_dir)


if __name__ == "__main__":
    main()
