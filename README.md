# AIC Video Retrieval System — KIS / Q&A / TRAKE

Text-to-video retrieval pipeline for the AIC-style video retrieval competition.
Given text queries, it produces up to 100 ranked `(video, frame)` candidates per
query as CSV, validated and packaged into `submission.zip` for Codabench.

Retrieval is a fusion of **CLIP text→image search** and **BM25** over media-info
text; **Qwen2-VL** then re-ranks KIS/TRAKE candidates and answers Q&A queries.
Everything runs on the organizer's pre-extracted packages (CLIP features,
map-keyframes, media-info, keyframe JPEGs) — raw videos are never needed.

## Tasks

The task type is inferred from the query filename suffix:

| Task | What it does | Output CSV columns |
|------|--------------|--------------------|
| **KIS** | Known-item search: fusion retrieval, then optional Qwen2-VL verification of the top hits | `video,frame` |
| **QA** | Fusion retrieval, then Qwen2-VL answers the question from the top videos' keyframes | `video,frame,answer` (answer ≤ 100 chars) |
| **TRAKE** | Temporal ranking of key events: per-event retrieval + video voting, then optional per-event frame verification | `video,frame1,frame2,...` (non-decreasing) |

Standardized task terminology (KIS/AVS/VQA/KISC definitions) lives in
[`docs/tasks.md`](docs/tasks.md).

## Architecture

```
                    ┌────────────────────────────────────────────────┐
 dataset packages ─▶│ indexer (one-time, ~2 min)                     │
 (clip, map,        │   FAISS index + sidecar + BM25 + video names   │
  media-info)       └───────────────────┬────────────────────────────┘
                                        │
 queries (*.txt) ──▶ CLIP text encoder ─┴─▶ weighted fusion (alpha)
                          │                        │
                          │                 ┌──────┴────────────────────┐
                          │                 │ KIS: top-100 rows         │
                          │                 │  + rerank: Qwen2-VL       │
                          │                 │    verifies top-N,        │
                          │                 │    verified first         │
                          │                 │ QA: per-video multi-image │
                          │                 │  Qwen2-VL answer          │
                          │                 │ TRAKE: event split → per-│
                          │                 │  event retrieval → video │
                          │                 │  voting + frame verify   │
                          │                 └──────┬────────────────────┘
                          ▼                        ▼
              keyframe JPEGs (Q&A / rerank)   data/submission/*.csv
                                                       │
                              validate_submission.py ──┴──▶ package_submission.py ─▶ submission.zip
```

Key design choices:

- **Re-rank only reorders, never drops.** Verified KIS hits move to the front;
  everything beyond the verified window keeps its retrieval order — so
  recall@k can only improve, never degrade.
- **The VLM loads lazily and only when needed.** Q&A always needs it;
  KIS/TRAKE load it only when re-ranking is enabled (`--no-rerank` keeps
  retrieval-only runs entirely VLM-free).
- **One bad query never kills the package** — failures are logged per query
  and processing continues.

## Quickstart

```bash
./scripts/run.sh --download     # first run: also fetch the dataset packages
./scripts/run.sh                # later runs (index is reused)
```

`run.sh` does everything: venv check → optional download → index build if
missing → pipeline → validation → `submission.zip`. Upload the zip to Codabench
(the zip contains a top-level `submission/` directory — a spec requirement).

## Requirements

- **OS**: Linux (developed on Ubuntu, Python 3.10; 3.11 also works)
- **GPU**: NVIDIA with driver ≥ 550 for the cu124 wheels (verified on an
  RTX 3090 / 24 GB). Qwen2-VL-7B at fp16 needs ~16–17 GB VRAM and falls back
  to the 2B model automatically on OOM. CLIP falls back to CPU if no GPU is
  present. KIS/TRAKE-only runs with `--no-rerank` never load the VLM at all.
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

> **Note on `scripts/setup_venv.sh`**: it predates `requirements.txt` and pins
> the older torch 2.5.1/cu121 build. Prefer the manual steps above; the script
> is due a refresh.

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

Create `data/queries/` (gitignored, so it's empty in a fresh clone) and add
files named `query-N-{kis,qa,trake}.txt` — the suffix determines the task.
One query per non-empty line; an optional `qid<TAB>text` prefix is honored.

## Run

One command does everything — venv check, (optional) download, index build
if missing, pipeline, validation, packaging:

```bash
./scripts/run.sh --download     # first run: fetch data too
./scripts/run.sh                # later runs (index is reused)
./scripts/run.sh --rebuild-index --alpha 0.8 --top-k-vqa 30
./scripts/run.sh --vlm-model Qwen/Qwen2-VL-2B-Instruct   # smaller VLM
./scripts/run.sh --no-rerank    # VLM-free fast run (KIS/TRAKE skip the VLM)
```

Unrecognized flags pass through to the pipeline module, so `--help` there
lists everything (`python -m aic_system.pipeline --help`):

| Flag | Effect |
|------|--------|
| `--alpha FLOAT` | CLIP weight in the fusion; `1-alpha` goes to BM25 |
| `--top-k-vqa INT` | how many retrieved frames the VLM considers per Q&A query |
| `--vlm-model NAME` | override `models.vqa.name` (e.g. the 2B model) |
| `--rerank` | force-enable Qwen2-VL re-ranking for KIS/TRAKE (overrides config) |
| `--no-rerank` | disable re-ranking — KIS/TRAKE never load the VLM |
| `--queries DIR` / `--out DIR` | override the configured input/output directories |

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

The validation/packaging scripts exist so a format mistake never burns one of
your limited Codabench submission attempts.

## How each task is solved

**Shared retrieval.** Every task starts from the same fused ranking: the
query text is CLIP-encoded and searched against the FAISS index
(`top_k_clip` candidates), BM25 scores the media-info text, and the two are
blended with weight `alpha` (CLIP) and `1-alpha` (BM25).

**KIS.** Fusion produces the top-100 rows. With re-ranking enabled, Qwen2-VL
*verifies* the top `rerank.top_n` hits against the query text ("does this
image match this description?") and verified hits move to the front, ordered
by fused score + VLM confidence. Unverifiable replies count as *not*
verified, so they can never outrank an explicit match.

**Q&A.** Instead of answering per frame, the top retrieved hits are grouped
by video and Qwen2-VL gets **one multi-image call per distinct video** (up to
3 of its keyframes in a single prompt) — fewer calls than per-frame answering,
and counting/order questions get temporal context. Each output row carries
the answer generated from its own video; filler rows beyond the answered set
propagate their video's answer when available, else the globally best one.

**TRAKE.** The query is split into events; each event is retrieved
separately (`trake.per_event_clip_k` candidates) and the videos accumulate
votes to pick the host video. Per event, the top `trake.per_event_frames`
frames are greedily chained in chronological order — rows must have exactly
N frames in non-decreasing order or the evaluator's parser may reject them.
With re-ranking enabled, each event's candidate frames are verified by
Qwen2-VL against the event text and verified matches are ordered first.

## Configuration

`configs/default.yaml` holds all defaults:

| Key | Default | Meaning |
|-----|---------|---------|
| `retrieval.alpha` | `0.7` | CLIP weight in fusion; `1-alpha` → BM25 |
| `retrieval.top_k_clip` | `200` | FAISS candidates fetched per query before fusion |
| `retrieval.top_k_for_vqa` | `20` | candidates passed to the VLM for Q&A answering |
| `retrieval.rerank.enabled` | `true` | Qwen2-VL verifies top KIS hits + TRAKE event frames |
| `retrieval.rerank.top_n` | `20` | KIS candidates verified per query |
| `retrieval.trake.per_event_clip_k` | `50` | keyframe candidates per event for video voting |
| `retrieval.trake.per_event_frames` | `3` | frames per event used to build candidate rows |
| `models.vqa.name` | `Qwen/Qwen2-VL-7B-Instruct` | answering/verification VLM |
| `models.vqa.fallback_name` | `Qwen/Qwen2-VL-2B-Instruct` | automatic fallback on OOM |

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
`docker run --gpus all --rm [mounts...] aic-system python -m aic_system.pipeline --no-rerank`.
Model weights (~17 GB) persist in the `hf-cache` volume.

## Troubleshooting

- **"aria2c not found" / "unzip not found"** — `sudo apt install aria2 unzip`.
- **torch installed the wrong CUDA build** — it must come from the
  `--index-url https://download.pytorch.org/whl/cu124` page *before*
  `requirements.txt`; check with `python -c "import torch; print(torch.__version__)"`
  (expect `2.6.0+cu124`).
- **VLM OOM / no GPU** — use `--vlm-model Qwen/Qwen2-VL-2B-Instruct` or a
  `configs/local.yaml` override; the pipeline also auto-falls-back to the
  2B model on OOM. Retrieval-only runs (`--no-rerank`, no Q&A queries) skip
  the VLM entirely.
- **Re-ranking is slow** — lower `retrieval.rerank.top_n`, or run with
  `--no-rerank` while iterating; each verified candidate is one VLM call.
- **Q&A answers empty** — the VLM needs the keyframe JPEGs
  (`paths.raw_keyframes`); without them Q&A retrieval still works but
  answering is disabled.
- **faiss/numpy** — numpy is pinned `<2.0` because faiss-cpu wheels lag
  behind numpy 2.x; don't upgrade numpy independently.

## Project layout

```
configs/default.yaml        all defaults (paths, models, retrieval params)
docs/tasks.md               standardized task/terminology catalog
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
  retrieval/search.py       KIS / QA / TRAKE runners (fusion, rerank, VQA)
  models/                   CLIP text encoder, Qwen2-VL wrapper (answer + verify)
  io/                       query parser, submission CSV writer
  eval/local_scorer.py      local R@k scoring against a ground truth
data/
  queries/                  query-N-{kis,qa,trake}.txt (input, gitignored)
  index/                    built index artifacts (generated)
  submission/               output CSVs (generated)
```
