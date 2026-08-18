#!/usr/bin/env bash
# Container entrypoint: builds the FAISS/BM25 index on first start (when
# data/index/faiss.idx is absent), then execs the container command
# (default, from the Dockerfile CMD: the full retrieval pipeline).
#
# Full run command for reference (adjust the host paths to your machine):
#
#   docker run --gpus all --rm \
#     -v /home/tthanh/Projects/raw_data/clip-features-32-aic25-b1/clip-features-32:/app/data/features/clip:ro \
#     -v /home/tthanh/Projects/raw_data/map-keyframes-aic25-b1/map-keyframes:/app/data/metadata/map_keyframes:ro \
#     -v /home/tthanh/Projects/raw_data/media-info-aic25-b1/media-info:/app/data/metadata/media_info:ro \
#     -v /root/aic2026-robloxkids/data/raw/keyframes:/app/data/raw/keyframes:ro \
#     -v /root/aic2026-robloxkids/data/queries:/app/data/queries:ro \
#     -v /root/aic2026-robloxkids/data/index:/app/data/index \
#     -v /root/aic2026-robloxkids/data/submission:/app/data/submission \
#     -v hf-cache:/cache/huggingface \
#     aic-system
#
# Pipeline flags go after the image name (they replace CMD), e.g.:
#   docker run --gpus all --rm [mounts...] aic-system \
#     python -m aic_system.pipeline --alpha 0.8
set -euo pipefail
cd /app

if [[ ! -f data/index/faiss.idx ]]; then
  echo "== entrypoint: data/index/faiss.idx not found — building index =="
  python -m aic_system.ingest.indexer --config default
fi

exec "$@"
