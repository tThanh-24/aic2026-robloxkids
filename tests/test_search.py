"""Regression tests for the Q&A no-output bugs.

Covers the two failure modes that made QA produce no CSV:
  1. organizer keyframe zips nest one extra directory level, so
     {raw_keyframes}/{video}/ didn't exist and every VLM lookup was skipped;
  2. with no answerable frames, run_qa returned [] and the pipeline wrote
     nothing — it must now still emit retrieval-ranked rows.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from aic_system.ingest.indexer import resolve_keyframes_root
from aic_system.io.query_parser import Query
from aic_system.retrieval.search import run_qa


def _make_keyframes(root: Path, nested: bool) -> None:
    target = root / "keyframes" if nested else root
    (target / "L01_V001").mkdir(parents=True)
    (target / "L01_V001" / "001.jpg").write_bytes(b"jpg")


def test_resolve_keyframes_root_flat_layout(tmp_path):
    _make_keyframes(tmp_path, nested=False)
    assert resolve_keyframes_root(tmp_path, ["L01_V001"]) == tmp_path


def test_resolve_keyframes_root_nested_layout(tmp_path):
    _make_keyframes(tmp_path, nested=True)
    assert resolve_keyframes_root(tmp_path, ["L01_V001"]) == tmp_path / "keyframes"


def test_resolve_keyframes_root_missing_videos_returns_root(tmp_path):
    assert resolve_keyframes_root(tmp_path, ["L01_V001"]) == tmp_path


class _StubEncoder:
    def encode(self, text: str) -> np.ndarray:
        v = np.zeros(8, dtype=np.float32)
        v[0] = 1.0
        return v


def _fake_resources(tmp_path: Path) -> dict:
    import faiss

    vecs = np.eye(8, dtype=np.float32)[:2]  # 2 keyframes, dim 8
    index = faiss.IndexFlatIP(8)
    index.add(vecs)
    sidecar = {
        "video_ids": np.array([0, 0], dtype=np.int32),
        "keyframe_n": np.array([1, 2], dtype=np.int32),
        "frame_idx": np.array([10, 20], dtype=np.int64),
    }
    return {
        "faiss": index,
        "sidecar": sidecar,
        "video_names": ["L01_V001"],
        "video_to_id": {"L01_V001": 0},
        "bm25": None,
        "bm25_videos": [],
        "bm25_corpus": [],
        "clip_features_dir": tmp_path,
        # Empty dir -> no keyframe images -> answering impossible.
        "raw_keyframes_dir": tmp_path / "nowhere",
        "cfg": {"retrieval": {"top_k_clip": 5, "top_k_for_vqa": 2, "alpha": 0.7}},
    }


def test_run_qa_without_keyframes_still_emits_rows(tmp_path):
    q = Query(query_id="query-1-qa", task="qa", text="ai đố wearing what?")
    cands = run_qa(q, _fake_resources(tmp_path), _StubEncoder(), vlm=None)
    assert len(cands) == 2  # all index rows, ranked
    assert all(c.video == "L01_V001" for c in cands)
    assert all(c.answer == "" for c in cands)
    assert [c.frame for c in cands] == [10, 20]  # best-first by CLIP score


def test_run_qa_top_k_vqa_from_config(tmp_path, monkeypatch):
    """cfg top_k_for_vqa must apply when run_qa is called without the arg
    (the pipeline's call path)."""
    called = []

    class _VLM:
        def answer_question(self, image, question, lang="vi"):
            called.append(image)
            return "vest", 0.9

    res = _fake_resources(tmp_path)
    kf_dir = tmp_path / "kf" / "L01_V001"
    kf_dir.mkdir(parents=True)
    (kf_dir / "001.jpg").write_bytes(b"jpg")
    (kf_dir / "002.jpg").write_bytes(b"jpg")
    res["raw_keyframes_dir"] = tmp_path / "kf"

    q = Query(query_id="query-1-qa", task="qa", text="ao dai hay vest?")
    cands = run_qa(q, res, _StubEncoder(), vlm=_VLM())
    assert len(called) == 2          # every retrieved frame got answered
    assert all(c.answer == "vest" for c in cands)
