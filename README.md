# AIC25 Video Retrieval System (KIS / Q&A / TRAKE)

A pipeline for the AIC-style video retrieval competition. Given text queries,
it returns up to 100 ranked `(video, frame)` candidates per query as CSV,
packaged into `submission.zip` for Codabench.

Three task types are supported (inferred from the query filename suffix):

| Task | What it does | Output CSV columns |
|------|--------------|--------------------|
| **KIS** | Known-item search: CLIP text→image retrieval fused with BM25 over media-info text | `video,frame` |
| **QA** | KIS retrieval, then Qwen2-VL reads the top frames and answers the question | `video,frame,answer` (answer ≤ 100 chars) |
| **TRAKE** | Temporal ranking of key events: per-event retrieval + video voting | `video,frame1,frame2,...` (non-decreasing) |

The pipeline runs entirely on the **organizer's pre-extracted packages**
(CLIP features, map-keyframes, media-info, keyframe JPEGs). Raw videos are
never needed — don't download them unless you want them for debugging.

```
dataset packages ──▶ indexer (FAISS + BM25, one-time ~2 min)
                            │
queries (*.txt) ──▶ pipeline ──▶ data/submission/*.csv
                            │         │
                            │         └─▶ validate_submission.py ─▶ package_submission.py ─▶ submission.zip
                            └─▶ Q&A only: Qwen2-VL answers from keyframe JPEGs
```

## Requirements

- **OS**: Linux (developed on Ubuntu, Python 3.10; 3.11 also works)
- **GPU**: NVIDIA with driver ≥ 550 for the cu124 wheels (verified on an
  RTX 3090 / 24 GB). Qwen2-VL-7B at fp16 needs ~16–17 GB VRAM; it falls
  back to the 2B model automatically on OOM. CLIP falls back to CPU if no
  GPU is present. KIS/TRAKE-only runs never load the VLM at all.
- **Disk**: a few GB for keyframes/features/metadata + ~17 GB cached Hugging
  Face weights (CLIP + Qwen2-VL) + ~350 MB for the built index
- **System packages**: `aria2`, `unzip` (dataset download)

## Setup

Install order matters: **torch first** from the PyTorch index (so the CUDA
build is cu124, not the default PyPI one), then everything else resolved
against it.

```bash
git clone <repo-url> && cd aic2026-robloxkids

# 1. venv (run.sh finds either ./venv or ./.venv)
python3 -m venv venv
source venv/bin/activate

# 2. torch FIRST, pinned cu124 build
pip install --upgrade pip wheel setuptools
pip install torch==2.6.0 torchaudio==2.6.0 torchvision==0.21.0 \
  --index-url https://download.pytorch.org/whl/cu124

# 3. everything else + the package itself (editable, with dev tools)
pip install -r requirements.txt
pip install -e ".[dev]"

# 4. dataset download tools
sudo apt install aria2 unzip

# sanity check
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

> **Note on `scripts/setup_venv.sh`**: it predates `requirements.txt` — it
> pins the older torch 2.5.1/cu121 and, because `requirements.lock.txt`
> currently ships as a comments-only placeholder, its lockfile branch would
> install nothing beyond torch. Prefer the manual steps above until the
> script is refreshed (see "Lockfile" under Troubleshooting).

## Get the data

Download links live in `scripts/download_links.txt` — that's the only file
to edit to add/remove batches (e.g. a future `L31` set).

```bash
# everything the pipeline needs (NO raw videos)
./scripts/download_dataset.sh --only keyframes,map,clip,media_info

# other useful invocations
./scripts/download_dataset.sh --list-only     # print the plan, do nothing
./scripts/download_dataset.sh                 # everything incl. videos (~100 GB)
./scripts/download_dataset.sh --jobs 4 --conns 8
```

Archives are classified by filename pattern and extracted to:

| Category | Filename pattern | Extracted to |
|----------|------------------|--------------|
| keyframes | `Keyframes_*.zip` | `data/raw/keyframes/` |
| videos | `Videos_*.zip` | `data/raw/videos/` (not used by the pipeline) |
| clip | `clip-features-*.zip` | `data/features/clip/` |
| map | `map-keyframes-*.zip` | `data/metadata/map_keyframes/` |
| media_info | `media-info-*.zip` | `data/metadata/media_info/` |
| objects | `objects-*.zip` | `data/metadata/objects/` |

After the first download, `python scripts/inspect_dataset.py` prints one
real sample per category so you can verify schemas.

## Add queries

Query files go in `data/queries/` named `query-N-{kis,qa,trake}.txt`
(the suffix determines the task). One query per non-empty line; an optional
`qid<TAB>text` prefix is honored. Two examples ship in `data/queries/`.

## Run

One command does everything — venv check, (optional) download, index build
if missing, pipeline, validation, packaging:

```bash
./scripts/run.sh --download     # first run: fetch data too
./scripts/run.sh                # later runs (index is reused)
./scripts/run.sh --rebuild-index --alpha 0.8 --top-k-vqa 30
./scripts/run.sh --vlm-model Qwen/Qwen2-VL-2B-Instruct   # smaller VLM
```

Unrecognized flags pass through to the pipeline module, so `--help` there
lists everything (`python -m aic_system.pipeline --help`).

Step-by-step equivalents:

```bash
# 1. one-time index build (~2 min) -> data/index/{faiss.idx, sidecar.npz, video_names.json, bm25.pkl}
python -m aic_system.ingest.indexer --config default

# 2. queries -> CSVs in data/submission/
python -m aic_system.pipeline --config default

# 3. validate the CSVs against the competition format
python scripts/validate_submission.py data/submission/

# 4. package submission.zip (runs validation first; refuses to zip on errors)
python scripts/package_submission.py
```

Upload `submission.zip` to Codabench. The zip contains a top-level
`submission/` directory with the CSVs — this is a spec requirement, and the
packaging/validation scripts exist so a format mistake never burns one of
your limited submission attempts.

## Configuration

`configs/default.yaml` holds all defaults: dataset paths, retrieval
hyperparameters (`alpha` = CLIP weight in fusion, `1-alpha` goes to BM25;
`top_k_clip`, `top_k_for_vqa`, TRAKE settings), and model choices.

Put machine-specific overrides in `configs/local.yaml` (gitignored) — it is
deep-merged over the default, e.g.:

```yaml
models:
  vqa:
    device: "cpu"
```

## Tests

```bash
source venv/bin/activate
pytest
```

## Docker (alternative to the venv setup)

```bash
docker build -t aic-system .

docker run --gpus all --rm \
  -v /path/to/clip-features-32:/app/data/features/clip:ro \
  -v /path/to/map-keyframes:/app/data/metadata/map_keyframes:ro \
  -v /path/to/media-info:/app/data/metadata/media_info:ro \
  -v /path/to/keyframes:/app/data/raw/keyframes:ro \
  -v $PWD/data/queries:/app/data/queries:ro \
  -v $PWD/data/index:/app/data/index \
  -v $PWD/data/submission:/app/data/submission \
  -v hf-cache:/cache/huggingface \
  aic-system
```

The entrypoint builds the index on first start (if `data/index/faiss.idx`
is absent), then runs the pipeline. Flags go after the image name:
`docker run --gpus all --rm [mounts...] aic-system python -m aic_system.pipeline --alpha 0.8`.
Model weights (~17 GB) persist in the `hf-cache` volume.

## Troubleshooting

- **"aria2c not found" / "unzip not found"** — `sudo apt install aria2 unzip`.
- **torch installed the wrong CUDA build** — it must come from the
  `--index-url https://download.pytorch.org/whl/cu124` page *before*
  `requirements.txt`; check with `python -c "import torch; print(torch.__version__)"`
  (expect `2.6.0+cu124`).
- **VLM OOM / no GPU** — use `--vlm-model Qwen/Qwen2-VL-2B-Instruct` or a
  `configs/local.yaml` override; the pipeline also auto-falls-back to the
  2B model on OOM, and KIS/TRAKE-only runs skip the VLM entirely.
- **Q&A answers empty** — the VLM needs the keyframe JPEGs
  (`paths.raw_keyframes`); without them Q&A retrieval still works but
  answering is disabled.
- **Lockfile** — `requirements.lock.txt` is a placeholder on purpose. To get
  reproducible installs, generate a real one once your venv works
  (`pip install pip-tools && pip-compile pyproject.toml -o requirements.lock.txt --extra dev`)
  and commit it; `scripts/setup_venv.sh` will then prefer it. Until then,
  use the manual setup steps above.
- **faiss/numpy** — numpy is pinned `<2.0` because faiss-cpu wheels lag
  behind numpy 2.x; don't upgrade numpy independently.

## Project layout

```
configs/default.yaml        all defaults (paths, models, retrieval params)
scripts/
  download_links.txt        dataset URLs — the only file to edit for new batches
  download_dataset.sh       aria2c download + extract, category-aware
  run.sh                    one command: download? -> index -> pipeline -> validate -> zip
  setup_venv.sh             venv bootstrap (older torch pin — see note above)
  inspect_dataset.py        print real samples/schemas after first download
  validate_submission.py    competition-format checks (run before every upload)
  package_submission.py     zip CSVs under submission/ for Codabench
src/aic_system/
  pipeline.py               entrypoint: query files -> CSVs (lazy model loading)
  config.py                 default.yaml + optional local.yaml deep-merge
  ingest/indexer.py         one-time FAISS + BM25 index build
  retrieval/search.py       KIS / QA / TRAKE runners (fusion + VQA)
  models/                   CLIP text encoder, Qwen2-VL wrapper
  io/                       query parser, submission CSV writer
  eval/local_scorer.py      local R@k scoring against a ground truth
data/
  queries/                  query-N-{kis,qa,trake}.txt (input)
  index/                    built index artifacts (generated)
  submission/               output CSVs (generated)
```
```
