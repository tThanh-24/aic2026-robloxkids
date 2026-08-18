# AIC-style Video Retrieval System (KIS / Q&A / TRAKE)

A pipeline for the AIC25-style video retrieval competition: given text
queries, return up to 100 ranked `(video, frame[, answer])` candidates per
query as CSV, scored by Mean of Top-k R-Score (`R@1,5,20,50,100`).

## Status

| Piece | Status |
|---|---|
| CSV writer (exact competition format) | ✅ built + tested |
| Submission validator (pre-upload gate) | ✅ built + tested |
| Submission packager (correct zip structure) | ✅ built + tested |
| Local scorer (reimplements R@k / Mean-of-Top-k) | ✅ built + tested, matches both spec worked examples |
| Query file parser | ✅ built (⚠️ format assumptions -- see below) |
| Dataset download script (aria2c) | ✅ built |
| Dataset inspector | ✅ built, **run before trusting ingestion code** |
| KIS / Q&A / TRAKE retrieval | ❌ not implemented -- `src/aic_system/pipeline.py` has the wiring + TODOs |
| Embedding index build (map-keyframes + clip-features -> FAISS) | ❌ not implemented |
| OCR / ASR / objects ingestion | ❌ not implemented |

## Quickstart

```bash
# 1. Set up the venv (torch installed first, pinned to your CUDA build)
./scripts/setup_venv.sh          # cuda (default, RTX 3090)
# ./scripts/setup_venv.sh cpu    # cpu-only, for dev without a GPU

source .venv/bin/activate

# 2. Download the dataset (edit scripts/download_links.txt to change what's fetched)
sudo apt install aria2 unzip     # if not already installed
./scripts/download_dataset.sh --list-only        # see the plan first
./scripts/download_dataset.sh --only keyframes,map,clip   # smallest useful subset
./scripts/download_dataset.sh                     # everything

# 3. VERIFY real schemas before writing/trusting ingestion code
python scripts/inspect_dataset.py

# 4. Run tests
pytest tests/

# 5. (once retrieval is implemented) run the pipeline
python -m aic_system.pipeline --config default

# 6. ALWAYS validate before uploading
python scripts/validate_submission.py data/submission/
python scripts/package_submission.py     # validates + zips correctly
```

## Repo layout

```
scripts/
  download_links.txt        <- EDIT THIS to add/remove/change dataset URLs
  download_dataset.sh        aria2c-based fetch + extract, driven by the list above
  inspect_dataset.py          peeks at real files to confirm schemas
  setup_venv.sh                creates .venv, installs torch first then the rest
  validate_submission.py      pre-upload format gate -- run this every time
  package_submission.py       zips data/submission/*.csv -> submission.zip correctly

src/aic_system/
  config.py                  loads configs/default.yaml (+ configs/local.yaml overrides)
  pipeline.py                 entrypoint: query files -> CSVs (retrieval logic is TODO)
  io/
    query_parser.py           query-N-{kis,qa,trake}.txt -> Query objects
    csv_writer.py               Query results -> competition-format CSV
  eval/
    local_scorer.py             reimplements R@k / Mean-of-Top-k scoring
  ingest/                      (empty -- see "Next steps")
  models/                      (empty -- see "Next steps")
  retrieval/                    (empty -- see "Next steps")

tests/                         pytest suite for csv_writer + local_scorer
configs/default.yaml            all paths, model names, competition constants
```

## Dataset assets (organizer-provided)

| Asset | Expected content | Confidence |
|---|---|---|
| `Keyframes_L2X.zip` | pre-extracted representative frames per video, `L2X_V0NN/*.jpg` | high (standard AIC convention) |
| `Videos_L2X_a.zip` | source `.mp4` files | high |
| `map-keyframes-aic25-b1.zip` | per-video CSV mapping keyframe index -> real frame index / timestamp | **verify with inspect_dataset.py** -- this is the join key between retrieval output and the frame IDs ground truth intervals are expressed in |
| `clip-features-32-aic25-b1.zip` | per-video `.npy`, one row per keyframe, CLIP ViT-B/32 embeddings (dim 512) | **verify** -- column order must match map-keyframes row order |
| `media-info-aic25-b1.zip` | per-video JSON: title/description/keywords/etc | medium |
| `objects-aic25-b1.zip` | per-keyframe JSON: detected object classes + boxes | medium |

Nothing in the pipeline should hardcode assumptions about these schemas
without `inspect_dataset.py` having confirmed them first -- that script
exists specifically to catch a wrong assumption before it's baked into
hours of ingestion code.

## Dependency strategy (single venv, no conflicts)

The main conflict risk in this stack is multiple libraries each wanting
their own CUDA runtime. The approach:

1. **torch is installed first**, pinned to a `cu121` wheel (matches the
   Docker base image `nvidia/cuda:12.1.1-...`, works with RTX 3090 given
   driver >=530). Every other GPU-touching library is chosen to be
   compatible with *this* runtime rather than bringing its own.
2. **EasyOCR instead of PaddleOCR** -- PaddleOCR's `paddlepaddle-gpu`
   ships its own CUDA expectations and has a history of fighting with a
   separately-installed torch. EasyOCR is torch-based, so it shares
   torch's CUDA runtime for free. (OCR isn't the bottleneck anyway; if you
   want it fully off the GPU, `configs/default.yaml -> models.ocr.gpu: false`
   runs it on CPU.)
3. **faiss-cpu instead of faiss-gpu** -- `faiss-gpu` has its own
   CUDA-linking quirks and, for a corpus in the hundreds-of-thousands to
   low-millions of frames, CPU FAISS (flat or IVF index) is fast enough
   that it's not worth the conflict risk. Revisit only if the corpus turns
   out to be much larger than expected.
4. **faster-whisper** uses `ctranslate2`, a self-contained C++ runtime
   that doesn't fight with torch's CUDA build.
5. Everything is one `pyproject.toml` -> one lockfile
   (`requirements.lock.txt`, generate with `pip-compile` once the venv is
   working) -> one Docker image built from that lockfile, so dev and
   container environments can't drift.

## Submission safety net

You get **3 submission attempts per query package, and a malformed file
still burns an attempt.** Every path from CSV generation to Codabench
upload goes through:

```
write_submission()  ->  validate_submission.py  ->  package_submission.py  ->  upload
                              ^
                    also run standalone anytime
```

`package_submission.py` refuses to produce a zip if validation fails
(`--force` overrides this, not recommended). Use `local_scorer.py` against
a self-made held-out ground-truth split (JSONL, see its docstring for the
schema) to estimate R@k *before* spending a real attempt -- the public
leaderboard only reflects 50% of ground truth, so don't over-index on it
either.

## Open questions / things to verify against real data

- **Query `.txt` internal format** -- `query_parser.py` currently assumes
  one query per line, optionally `qid<TAB>text`. Confirm against a real
  downloaded query package and adjust `_parse_lines` if it differs (JSON,
  one-query-per-file, etc.) -- everything downstream consumes `Query`
  objects, so the blast radius of being wrong is contained to that one
  function.
- **Q&A answer comparison rule** -- the competition doc says answers are
  compared "according to the competition's stated semantic/exact
  comparison rule" without giving the rule. `local_scorer.py` currently
  does case/whitespace-insensitive exact match as a conservative default
  -- update `_normalize_answer` once the real rule is published.
- **map-keyframes / clip-features exact schema** -- run
  `inspect_dataset.py` after the first download and update
  `configs/default.yaml -> models.provided_clip` and the (not yet written)
  `ingest/` modules accordingly.

## Next steps (build order, see earlier planning discussion)

1. `ingest/build_index.py` -- parse map-keyframes + clip-features into a
   FAISS index + metadata store (SQLite/Parquet).
2. `retrieval/kis.py` -- embed query text, search the index, group into
   ranked `(video, frame)` candidates.
3. `retrieval/qa.py` -- reuse KIS retrieval, add VQA re-ranking
   (Qwen2-VL-7B-Instruct fits comfortably in 3090 VRAM at fp16).
4. `retrieval/trake.py` -- event decomposition, per-event retrieval,
   single-video locking, chronological ordering.
5. Wire all three into `pipeline.py`'s `run_kis` / `run_qa` / `run_trake`.
6. Error-analyze against the local scorer, iterate.
