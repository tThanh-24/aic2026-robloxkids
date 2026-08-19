"""Main orchestration entrypoint: query file(s) -> CSV submission(s).

Loading strategy:
  - Index resources (FAISS + sidecar + BM25) load once, lazily, on the
    first query that needs them.
  - The CLIP text encoder loads on first query.
  - The VLM (Qwen2-VL, ~16 GB VRAM at fp16) loads ONLY when the first
    Q&A query arrives, so KIS/TRAKE-only runs never pay for it.

Usage:
    python -m aic_system.pipeline --config default --queries data/queries --out data/submission
    python -m aic_system.pipeline --alpha 0.8 --top-k-vqa 30 --vlm-model Qwen/Qwen2-VL-2B-Instruct
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
            if getattr(self.args, "rerank", None) is not None:
                self.cfg["retrieval"].setdefault("rerank", {})["enabled"] = self.args.rerank
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


def _maybe_vlm(resources: Resources):
    """The VLM for KIS/TRAKE re-ranking — None unless rerank is enabled, so
    retrieval-only runs never load the ~16 GB model (Q&A always needs it)."""
    rr = resources.cfg["retrieval"].get("rerank", {})
    return resources.vlm if rr.get("enabled", False) else None


def process_query_file(stem: str, queries: list[Query], resources: Resources, out_dir: Path) -> None:
    if not queries:
        return
    task = queries[0].task

    all_candidates = []
    for q in queries:
        try:
            if task == "kis":
                all_candidates.extend(run_kis(q, resources.index, resources.clip,
                                              alpha=_alpha(resources), vlm=_maybe_vlm(resources)))
            elif task == "qa":
                all_candidates.extend(
                    run_qa(q, resources.index, resources.clip, resources.vlm, alpha=_alpha(resources))
                )
            elif task == "trake":
                all_candidates.extend(run_trake(q, resources.index, resources.clip,
                                                alpha=_alpha(resources), vlm=_maybe_vlm(resources)))
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
    ap.add_argument("--rerank", dest="rerank", action="store_true", default=None,
                    help="force-enable Qwen2-VL re-ranking for KIS/TRAKE "
                         "(overrides configs)")
    ap.add_argument("--no-rerank", dest="rerank", action="store_false",
                    help="disable Qwen2-VL re-ranking (KIS/TRAKE never load the VLM)")
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
