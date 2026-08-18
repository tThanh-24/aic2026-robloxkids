# Docker image for the AIC retrieval pipeline.
#
# Design notes:
#   - Base is plain python:3.11-slim, NOT nvidia/cuda: the cu121 torch
#     wheels bundle their own CUDA runtime (nvidia-* pip packages), so the
#     image only needs the host driver via `docker run --gpus all`. This
#     keeps the image several GB smaller than a cuda:devel base.
#   - Install order mirrors scripts/setup_venv.sh: torch FIRST (single
#     CUDA runtime), then everything else resolved against it.
#   - No dataset, queries, or model weights are baked in -- mount them
#     (see the run commands in the header of scripts/docker-entrypoint.sh
#     or the README "Docker" section).
#
# Build:
#   docker build -t aic-system .
#
# Run (see scripts/docker-entrypoint.sh header for the full volume set):
#   docker run --gpus all --rm [mounts...] aic-system
#
# The 3090 host driver (>=530) is all that's required on the machine.

FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/cache/huggingface

WORKDIR /app

# --- install: torch cu121 first, then the project --------------------------
COPY pyproject.toml README.md ./
RUN pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121

COPY src ./src
COPY scripts ./scripts
COPY configs ./configs
RUN pip install .

# --- runtime entrypoint -----------------------------------------------------
# Auto-builds data/index if missing, then execs the given command
# (default: the full pipeline).
RUN chmod +x scripts/docker-entrypoint.sh
ENTRYPOINT ["scripts/docker-entrypoint.sh"]
CMD ["python", "-m", "aic_system.pipeline", "--config", "default"]

# Mount points used at runtime (see entrypoint header):
#   /app/data/...        dataset, queries, index, submissions
#   /cache/huggingface   CLIP + Qwen2-VL weights (~17 GB, avoid re-downloads)
VOLUME ["/cache/huggingface"]
