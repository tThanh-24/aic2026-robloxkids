"""One-time index building: FAISS over pre-computed CLIP features + BM25
over media_info text.

Everything here runs from the organizer's pre-extracted packages
(clip-features / map-keyframes / media-info) — no raw videos required.

Outputs (written to paths.index_dir):
  faiss.idx        — IndexFlatIP over L2-normalized float32 keyframe vectors
  sidecar.npz      — per-FAISS-position arrays: video_ids (index into
                     video_names.json), keyframe_n (1-based, matches the
                     map_keyframes CSV), frame_idx (the REAL frame number
                     that must appear in submissions)
  video_names.json — ordered list of video names
  bm25.pkl         — pickled BM25Okapi + the tokenized corpus + metadata

CLI:
    python -m aic_system.ingest.indexer --config default
"""
from __future__ import annotations

import argparse
import json
import pickle
import re
from pathlib import Path

import numpy as np

from aic_system.config import load_config, resolve_path

# Vietnamese letters keep their diacritics; split tokens on anything else.
_TOKEN_RE = re.compile(r"[\wÀ-ỹà-ỹ]+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    """Language-agnostic whitespace/punct tokenizer, lowercase output.

    Good enough for BM25 over titles/keywords: single-syllable Vietnamese
    words ("tin", "tuc") still overlap across diacritic variants because we
    keep both the accented token and, cheaply, nothing else — callers that
    need diacritic folding should add it here, not at call sites.
    """
    return [t.lower() for t in _TOKEN_RE.findall(text or "")]


def _video_stem(path: Path) -> str:
    name = path.name
    for ext in (".npy", ".csv", ".json", ".mp4", ".MP4"):
        if name.endswith(ext):
            return name[: -len(ext)]
    return name


def iter_video_files(directory: Path, suffix: str) -> dict[str, Path]:
    """Map video_name -> file for every *.{suffix} under directory
    (recursive — organizer zips sometimes nest one level)."""
    out: dict[str, Path] = {}
    for path in sorted(directory.rglob(f"*{suffix}")):
        out[_video_stem(path)] = path
    return out


def load_metadata(media_info_dir: Path) -> dict[str, dict]:
    """{video_name: {"title": str, "keywords": [str], "description": str}}.

    Missing fields default to empty — some videos have sparse JSON.
    """
    metadata: dict[str, dict] = {}
    for video, path in iter_video_files(media_info_dir, ".json").items():
        try:
            d = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            print(f"  [media_info] skipping {path.name}: {e}")
            continue
        metadata[video] = {
            "title": d.get("title") or "",
            "keywords": d.get("keywords") or [],
            "description": d.get("description") or "",
        }
    return metadata


def metadata_document(meta: dict) -> str:
    """BM25 document per video. Title + keywords only: the description is
    YouTube channel boilerplate (subscribe links, hashtags) shared by every
    video of a channel, which would drown the discriminative tokens."""
    return f"{meta['title']} {' '.join(meta['keywords'])}"


def build_bm25_index(metadata: dict[str, dict]):
    """Returns (BM25Okapi, corpus_tokens, video_list) over metadata docs."""
    from rank_bm25 import BM25Okapi

    video_list = sorted(metadata.keys())
    corpus = [tokenize(metadata_document(metadata[v])) for v in video_list]
    bm25 = BM25Okapi(corpus)
    return bm25, corpus, video_list


def build_faiss_index(clip_dir: Path, map_kf_dir: Path, out_dir: Path) -> tuple[int, list[str], int]:
    """Concatenate per-video .npy features into one IndexFlatIP.

    Keyframe alignment: organizer .npy rows are ordered by the map_keyframes
    CSV's `n` column (1..num_keyframes). We join on that and keep the CSV's
    `frame_idx` (real frame number) in the sidecar — submissions must
    contain frame_idx values, NOT keyframe numbers.
    """
    import faiss

    feature_files = iter_video_files(clip_dir, ".npy")
    map_files = iter_video_files(map_kf_dir, ".csv")
    if not feature_files:
        raise FileNotFoundError(f"No .npy feature files under {clip_dir}")

    videos = sorted(v for v in feature_files if v in map_files)
    skipped = sorted(set(feature_files) - set(videos))
    if skipped:
        print(f"  [faiss] {len(skipped)} videos without map CSV (skipped): {skipped[:5]}...")

    video_names: list[str] = []
    video_ids: list[np.ndarray] = []
    kf_ns: list[np.ndarray] = []
    frame_idxs: list[np.ndarray] = []
    vec_blocks: list[np.ndarray] = []
    total_vecs = 0

    for vi, video in enumerate(videos):
        feats = np.load(feature_files[video])
        if feats.ndim != 2:
            print(f"  [faiss] {video}: unexpected shape {feats.shape}, skipping")
            continue
        map_df = np.genfromtxt(
            map_files[video], delimiter=",", names=True, dtype=None, encoding="utf-8"
        )
        n_rows = len(map_df)
        if n_rows != feats.shape[0]:
            print(
                f"  [faiss] {video}: map CSV has {n_rows} rows but .npy has "
                f"{feats.shape[0]} vectors — trusting the first {min(n_rows, feats.shape[0])}"
            )
        n_rows = min(n_rows, feats.shape[0])

        vecs = feats[:n_rows].astype(np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        vec_blocks.append(vecs / norms)

        video_names.append(video)
        video_ids.append(np.full(n_rows, vi, dtype=np.int32))
        kf_ns.append(np.arange(1, n_rows + 1, dtype=np.int32))
        frame_idxs.append(np.asarray(map_df["frame_idx"], dtype=np.int64)[:n_rows])
        total_vecs += n_rows

    if total_vecs == 0:
        raise RuntimeError("No usable feature/map pairs found")

    matrix = np.vstack(vec_blocks)
    index = faiss.IndexFlatIP(matrix.shape[1])
    index.add(matrix)

    sidecar = {
        "video_ids": np.concatenate(video_ids),
        "keyframe_n": np.concatenate(kf_ns),
        "frame_idx": np.concatenate(frame_idxs),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(out_dir / "faiss.idx"))
    np.savez(out_dir / "sidecar.npz", **sidecar)
    (out_dir / "video_names.json").write_text(
        json.dumps(video_names, ensure_ascii=False), encoding="utf-8"
    )
    return total_vecs, video_names, matrix.shape[1]


def build_all(cfg: dict) -> None:
    clip_dir = resolve_path(cfg, "clip_features")
    map_kf_dir = resolve_path(cfg, "map_keyframes")
    media_dir = resolve_path(cfg, "media_info")
    out_dir = resolve_path(cfg, "index_dir")

    print(f"[1/2] Building FAISS index from {clip_dir} ...")
    total_vecs, video_names, dim = build_faiss_index(clip_dir, map_kf_dir, out_dir)
    print(f"      {len(video_names)} videos, {total_vecs} keyframe vectors "
          f"({matrix_bytes(total_vecs, dim):.0f} MB index)")

    print(f"[2/2] Building BM25 index from {media_dir} ...")
    metadata = load_metadata(media_dir)
    bm25, corpus, bm25_videos = build_bm25_index(metadata)
    with open(out_dir / "bm25.pkl", "wb") as f:
        pickle.dump({"bm25": bm25, "corpus": corpus, "videos": bm25_videos}, f)
    print(f"      {len(bm25_videos)} videos with metadata")
    print(f"Done. Index artifacts in {out_dir}")


def matrix_bytes(n: int, dim: int) -> float:
    return n * dim * 4 / 1e6


def load_index_resources(cfg: dict) -> dict:
    """Load every retrieval artifact once. Called a single time at pipeline
    startup; the returned dict is passed to all task runners."""
    import faiss

    index_dir = resolve_path(cfg, "index_dir")
    faiss_path = index_dir / "faiss.idx"
    if not faiss_path.exists():
        raise FileNotFoundError(
            f"{faiss_path} not found — run `python -m aic_system.ingest.indexer` first"
        )

    index = faiss.read_index(str(faiss_path))
    sidecar = dict(np.load(index_dir / "sidecar.npz"))
    video_names = json.loads((index_dir / "video_names.json").read_text(encoding="utf-8"))
    video_to_id = {v: i for i, v in enumerate(video_names)}

    with open(index_dir / "bm25.pkl", "rb") as f:
        bm25_data = pickle.load(f)

    return {
        "faiss": index,
        "sidecar": sidecar,
        "video_names": video_names,
        "video_to_id": video_to_id,
        "bm25": bm25_data["bm25"],
        "bm25_videos": bm25_data["videos"],
        "bm25_corpus": bm25_data["corpus"],
        "clip_features_dir": resolve_path(cfg, "clip_features"),
        "raw_keyframes_dir": resolve_path(cfg, "raw_keyframes"),
        "cfg": cfg,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="default")
    args = ap.parse_args()
    build_all(load_config(args.config))


if __name__ == "__main__":
    main()
