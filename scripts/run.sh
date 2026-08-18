#!/usr/bin/env bash
# run.sh — ONE command for the whole pipeline: venv check -> (optional)
# dataset download -> index build -> query processing -> validation ->
# packaging into submission.zip.
#
# Usage:
#   ./scripts/run.sh                          # index (if missing) + pipeline + validate + zip
#   ./scripts/run.sh --download               # also download dataset first (keyframes/map/clip/media_info, NO raw videos)
#   ./scripts/run.sh --rebuild-index          # force FAISS/BM25 rebuild
#   ./scripts/run.sh --alpha 0.8 --top-k-vqa 30        # hyperparams pass through to the pipeline
#   ./scripts/run.sh --vlm-model Qwen/Qwen2-VL-2B-Instruct
#   ./scripts/run.sh --queries data/queries --out data/submission
#
# Any unrecognized flag is forwarded verbatim to `python -m aic_system.pipeline`
# (see its --help), except the script-owned ones listed above.

set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# ---------- 0. venv ---------------------------------------------------------
if [[ -x "venv/bin/python" ]]; then
  PY="$ROOT_DIR/venv/bin/python"
elif [[ -x ".venv/bin/python" ]]; then
  PY="$ROOT_DIR/.venv/bin/python"
else
  echo "== No venv found — creating one (torch cu121 first, then the rest) =="
  ./scripts/setup_venv.sh
  PY="$ROOT_DIR/.venv/bin/python"
fi

# ---------- 1. parse args ---------------------------------------------------
DO_DOWNLOAD=0
REBUILD_INDEX=0
PIPELINE_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --download)      DO_DOWNLOAD=1; shift ;;
    --rebuild-index) REBUILD_INDEX=1; shift ;;
    *)               PIPELINE_ARGS+=("$1"); shift ;;
  esac
done

# ---------- 2. dataset (optional) ------------------------------------------
if [[ "$DO_DOWNLOAD" -eq 1 ]]; then
  echo "== [1/5] Downloading dataset (keyframes, map-keyframes, clip-features, media-info; NO raw videos) =="
  ./scripts/download_dataset.sh --only keyframes,map,clip,media_info
else
  echo "== [1/5] Dataset download skipped (use --download to fetch) =="
fi

# ---------- 3. index --------------------------------------------------------
if [[ "$REBUILD_INDEX" -eq 1 ]] || [[ ! -f "data/index/faiss.idx" ]]; then
  echo "== [2/5] Building FAISS + BM25 index (one-time, ~2 min) =="
  "$PY" -m aic_system.ingest.indexer --config default
else
  echo "== [2/5] Index already built (data/index/faiss.idx) — skipping; use --rebuild-index to force =="
fi

# ---------- 4. pipeline -----------------------------------------------------
echo "== [3/5] Running pipeline (queries -> CSVs) =="
"$PY" -m aic_system.pipeline --config default "${PIPELINE_ARGS[@]}"

# ---------- 5. validate + package ------------------------------------------
echo "== [4/5] Validating submission CSVs =="
"$PY" scripts/validate_submission.py data/submission/

echo "== [5/5] Packaging submission.zip =="
"$PY" scripts/package_submission.py

echo ""
echo "Done. Upload submission.zip to Codabench (last submission counts!)"
