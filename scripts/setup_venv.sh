#!/usr/bin/env bash
# setup_venv.sh — creates the single project venv in the right order:
# torch (pinned CUDA build) FIRST, then everything else. This ordering is
# what avoids dependency-resolver conflicts (torch's CUDA build is the one
# constraint every other package here needs to be compatible with).
#
# Usage:
#   ./scripts/setup_venv.sh            # CUDA 12.1 build (RTX 3090 default)
#   ./scripts/setup_venv.sh cpu        # CPU-only torch (for non-GPU dev/testing)

set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${ROOT_DIR}/.venv"
MODE="${1:-cuda}"

python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"
pip install --upgrade pip wheel setuptools

if [[ "$MODE" == "cpu" ]]; then
  pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
else
  # cu121 build — matches nvidia/cuda:12.1.1 base image, works fine on
  # RTX 3090 (Ampere, driver >=530 required for CUDA 12.1).
  pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
fi

# Install the rest against the already-installed torch.
if [[ -f "${ROOT_DIR}/requirements.lock.txt" ]]; then
  pip install -r "${ROOT_DIR}/requirements.lock.txt"
else
  pip install -e "${ROOT_DIR}[dev]"
fi

echo ""
echo "venv ready at ${VENV_DIR}"
echo "Activate with: source .venv/bin/activate"
python -c "import torch; print('torch', torch.__version__, '| CUDA available:', torch.cuda.is_available())"
