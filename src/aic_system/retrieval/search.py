"""All retrieval logic: CLIP search, BM25 search, fusion, and the three
task pipelines (KIS / Q&A / TRAKE).

Contracts with `ingest.indexer.load_index_resources`:
  resources["faiss"]                    IndexFlatIP (cosine, vectors L2-normed)
  resources["sidecar"]["video_ids"]     FAISS row -> video index
  resources["sidecar"]["keyframe_n"]    FAISS row -> keyframe number (1-based)
  resources["sidecar"]["frame_idx"]     FAISS row -> REAL frame id for submission
  resources["video_names"]              video index -> name
  resources["bm25"] / ["bm25_videos"]   BM25Okapi + doc order
  resources["clip_features_dir"]        per-video .npy (for TRAKE in-video search)
  resources["raw_keyframes_dir"]        keyframe JPEGs (for the VLM)

Scoring-driven design choices (see doc.md):
  - R@k takes the MAX R-Score over the first k rows -> always emit the
    full 100 rows; deeper rows are free attempts, never harmful.
  - Q&A rows 21..100 repeat the most confident answer: answer correctness
    is only rewarded when the row's frame is also in [s,e], so reusing the
    best answer on filler frames maximizes the chance that some row inside
    the interval carries a semantically-correct answer.
  - TRAKE rows must have EXACTLY N frames in chronological order; a row
    with the wrong count risks failing the evaluator's parser entirely.
  - Every task accepts an optional `vlm`; when given, KIS/TRAKE rerank
    their shortlists with visual verification (yes/no relevance) and
    TRAKE may also split implicit multi-event queries with it. With
    vlm=None all three runners are pure CLIP+BM25.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from aic_system.io.csv_writer import KISCandidate, QACandidate, TrakeCandidate
from aic_system.io.query_parser import Query
from aic_system.models.clip_encoder import CLIPTextEncoder

from aic_system.ingest.indexer import tokenize

# ---------------------------------------------------------------------------
# Language detection (a query is entirely VI or entirely EN, per the docs)
# ---------------------------------------------------------------------------
_VI_CHARS = set("ăâêôơưđáàảãạấầẩẫậắằẳẵặéèẻẽẹếềểễệíìỉĩịóòỏõọốồổỗộớờởỡợúùủũụứừửữựýỳỷỹỵ")


def detect_language(text: str) -> str:
    """'vi' if any Vietnamese-specific character appears, else 'en'."""
    return "vi" if any(ch in _VI_CHARS for ch in text.lower()) else "en"


# ---------------------------------------------------------------------------
# Search primitives
# ---------------------------------------------------------------------------
def clip_search(
    query_vec: np.ndarray, resources: dict, top_k: int = 200
) -> list[dict]:
    """FAISS search -> [{video, video_id, keyframe_n, frame_idx, score}] best-first."""
    k = min(top_k, resources["faiss"].ntotal)
    scores, positions = resources["faiss"].search(
        query_vec.astype(np.float32).reshape(1, -1), k
    )
    side = resources["sidecar"]
    hits = []
    for score, pos in zip(scores[0], positions[0]):
        if pos < 0:  # FAISS pads with -1 when k > ntotal
            continue
        vid = int(side["video_ids"][pos])
        hits.append(
            {
                "video": resources["video_names"][vid],
                "video_id": vid,
                "keyframe_n": int(side["keyframe_n"][pos]),
                "frame_idx": int(side["frame_idx"][pos]),
                "score": float(score),
            }
        )
    return hits


def bm25_search(query_text: str, resources: dict) -> dict[str, float]:
    """Video-level BM25 scores normalized to [0, 1] (max-normalized).

    BM25 scores are only a boost on top of CLIP: min-max style normalization
    against the max keeps the top video at 1.0 and everything else relative,
    and returns all-zeros (not garbage) when nothing matches.
    """
    bm25, videos = resources["bm25"], resources["bm25_videos"]
    if bm25 is None or not videos:
        return {}
    raw = bm25.get_scores(tokenize(query_text))
    top = float(raw.max()) if raw.size else 0.0
    if top <= 0.0:
        return {v: 0.0 for v in videos}
    return {v: float(s) / top for v, s in zip(videos, raw)}


def fused_rank(
    clip_hits: list[dict],
    bm25_scores: dict[str, float],
    alpha: float = 0.7,
    top_k: int = 100,
) -> list[dict]:
    """final = alpha * clip + (1 - alpha) * bm25(video of the hit).

    CLIP scores are keyframe-level; BM25 scores are video-level and get
    attached to every keyframe hit of that video. Clips without a BM25
    entry score 0 on the text side.
    """
    for hit in clip_hits:
        bm = bm25_scores.get(hit["video"], 0.0)
        hit["fused"] = alpha * hit["score"] + (1.0 - alpha) * bm
    ranked = sorted(clip_hits, key=lambda h: h["fused"], reverse=True)
    return ranked[:top_k]


def _retrieve(query_text: str, resources: dict, encoder: CLIPTextEncoder,
              alpha: float, top_k_clip: int, top_k: int) -> list[dict]:
    """Shared KIS-style retrieval: encode -> CLIP search -> fuse -> rank."""
    query_vec = encoder.encode(query_text)
    hits = clip_search(query_vec, resources, top_k=top_k_clip)
    return fused_rank(hits, bm25_search(query_text, resources), alpha=alpha, top_k=top_k)


# ---------------------------------------------------------------------------
# Shared VLM-stage helpers (used by KIS rerank, Q&A answering, TRAKE verify)
# ---------------------------------------------------------------------------
def _keyframe_image(resources: dict, video: str, keyframe_n: int) -> Path | None:
    """Locate {raw_keyframes}/{video}/{nnn}.jpg, tolerating naming variants
    (001.jpg / 01.jpg / 1.jpg). Returns None if the video dir is absent."""
    video_dir = Path(resources["raw_keyframes_dir"]) / video
    if not video_dir.is_dir():
        return None
    for pattern in (f"{keyframe_n:03d}.jpg", f"{keyframe_n:02d}.jpg", f"{keyframe_n}.jpg"):
        candidate = video_dir / pattern
        if candidate.exists():
            return candidate
    return None


def _blend_ranked(hits: list[dict], conf_key: str) -> list[dict]:
    """Rank by 0.5 * normalized retrieval score + 0.5 * VLM score.

    The ranking rule run_qa has always used, factored out so the KIS rerank
    applies the same combination. Hits without the VLM key (e.g. an image
    that could not be located -> neutral 0.5) keep their retrieval order.
    """
    max_fused = max((h["fused"] for h in hits), default=0.0) or 1.0
    return sorted(
        hits,
        key=lambda h: 0.5 * (h["fused"] / max_fused) + 0.5 * h.get(conf_key, 0.5),
        reverse=True,
    )


# ---------------------------------------------------------------------------
# KIS
# ---------------------------------------------------------------------------
def run_kis(query: Query, resources: dict, encoder: CLIPTextEncoder,
            alpha: float = 0.7, vlm=None) -> list[KISCandidate]:
    cfg = resources["cfg"]["retrieval"]
    ranked = _retrieve(query.text, resources, encoder, alpha,
                       top_k_clip=cfg.get("top_k_clip", 200), top_k=100)

    # VLM rerank: visually verify the head of the list; the remaining rows
    # stay in retrieval order (deeper rows are free attempts under
    # max-R@k scoring, so verification effort is spent only up front).
    if vlm is not None and ranked:
        rerank_n = min(int(cfg.get("rerank", {}).get("kis_top_k", 40)), len(ranked))
        for hit in ranked[:rerank_n]:
            image = _keyframe_image(resources, hit["video"], hit["keyframe_n"])
            hit["vlm"] = vlm.verify(image, query.text) if image else 0.5
        ranked = _blend_ranked(ranked[:rerank_n], "vlm") + ranked[rerank_n:]

    return [KISCandidate(video=h["video"], frame=h["frame_idx"]) for h in ranked]


# ---------------------------------------------------------------------------
# Q&A
# ---------------------------------------------------------------------------
def run_qa(query: Query, resources: dict, encoder: CLIPTextEncoder, vlm,
           alpha: float = 0.7, top_k_vqa: int | None = None) -> list[QACandidate]:
    cfg = resources["cfg"]["retrieval"]
    if top_k_vqa is None:
        top_k_vqa = int(cfg.get("top_k_for_vqa", 20))
    lang = detect_language(query.text)

    ranked = _retrieve(query.text, resources, encoder, alpha,
                       top_k_clip=cfg.get("top_k_clip", 200), top_k=top_k_vqa)

    answered: list[dict] = []
    for hit in ranked:
        image = _keyframe_image(resources, hit["video"], hit["keyframe_n"])
        if image is None:
            continue  # no keyframes downloaded -> QA answering impossible
        answer, conf = vlm.answer_question(image, query.text, lang=lang)
        answered.append({**hit, "answer": answer.strip()[:100], "conf": conf})

    if not answered:
        print(f"  [{query.query_id}] no keyframe images found for VLM; "
              f"answering disabled (check paths.raw_keyframes)")
        return []

    # Rank answered candidates by retrieval score + VLM confidence, then
    # keep extending the list with the remaining retrieval filler rows.
    answered = _blend_ranked(answered, "conf")
    primary_answer = answered[0]["answer"]

    candidates = [QACandidate(video=h["video"], frame=h["frame_idx"], answer=h["answer"])
                  for h in answered]

    seen = {(c.video, c.frame) for c in candidates}
    for hit in _retrieve(query.text, resources, encoder, alpha,
                         top_k_clip=cfg.get("top_k_clip", 200), top_k=100):
        if len(candidates) >= 100:
            break
        if (hit["video"], hit["frame_idx"]) in seen:
            continue
        candidates.append(
            QACandidate(video=hit["video"], frame=hit["frame_idx"], answer=primary_answer)
        )
    return candidates


# ---------------------------------------------------------------------------
# TRAKE
# ---------------------------------------------------------------------------
_EVENT_SPLIT_RE = re.compile(
    r"(?:;\s*|\.\s+|,\s*(?:then|rồi|sau đó|tiếp theo)\b"
    r"|\b(?:then|after that|next|afterwards|finally|sau đó|sau nữa|tiếp theo|cuối cùng)\b)",
    re.IGNORECASE,
)


def split_events(text: str) -> list[str]:
    """Split a TRAKE query into per-event descriptions.

    Conservative marker set: splitting too aggressively creates events the
    ground truth doesn't have, and a wrong N invalidates the whole row.
    """
    parts = [p.strip(" ,.") for p in _EVENT_SPLIT_RE.split(text)]
    return [p for p in parts if len(p) >= 3] or [text.strip()]


def _video_vote(event_hits: list[list[dict]]) -> list[str]:
    """Videos ranked by summed CLIP score across all events' hit lists."""
    votes: dict[str, float] = {}
    for hits in event_hits:
        for hit in hits:
            votes[hit["video"]] = votes.get(hit["video"], 0.0) + hit["score"]
    return [v for v, _ in sorted(votes.items(), key=lambda kv: kv[1], reverse=True)]


def _event_frames_in_video(
    event_vec: np.ndarray, video: str, resources: dict, per_event_frames: int
) -> list[dict]:
    """Top CLIP hits for one event, restricted to one video.

    Returns full hit dicts (keyframe_n lets the VLM locate the JPEG,
    frame_idx is what goes into the submission).

    Implemented as a wide FAISS search + sidecar filter rather than loading
    the video's .npy: same result, and it reuses the shared normalized index
    (k a few thousand is still a sub-10ms flat search).
    """
    hits = clip_search(event_vec, resources, top_k=4000)
    return [h for h in hits if h["video"] == video][:per_event_frames]


def _verified_order(vlm, resources: dict, video: str, event_text: str,
                    options: list[dict]) -> list[dict]:
    """Re-order one event's frame options by VLM relevance.

    Stable sort, so equal scores keep CLIP order; frames whose image is
    missing get the neutral 0.5 and effectively stay at their CLIP rank.
    """
    scored = []
    for hit in options:
        image = _keyframe_image(resources, video, hit["keyframe_n"])
        conf = vlm.verify(image, event_text) if image else 0.5
        scored.append((conf, hit))
    scored.sort(key=lambda pair: -pair[0])
    return [hit for _, hit in scored]


def _greedy_chronological(frame_options: list[list[int]]) -> list[int] | None:
    """Pick the highest-priority frame per event such that frames are
    strictly increasing (events must be output in chronological order).
    Returns None if no monotone assignment exists."""
    chosen: list[int] = []
    prev = -1
    for options in frame_options:
        pick = next((f for f in options if f > prev), None)
        if pick is None:
            return None
        chosen.append(pick)
        prev = pick
    return chosen


def run_trake(query: Query, resources: dict, encoder: CLIPTextEncoder,
              alpha: float = 0.7, vlm=None) -> list[TrakeCandidate]:
    cfg = resources["cfg"]["retrieval"]
    trake_cfg = cfg.get("trake", {})
    max_videos = int(cfg.get("rerank", {}).get("trake_videos", 3))
    clip_k = int(trake_cfg.get("per_event_clip_k", 50))
    per_event_frames = int(trake_cfg.get("per_event_frames", 3))

    events = split_events(query.text)
    # The regex only splits on explicit transition markers. When it found
    # none, let the VLM look for an implicit event sequence — but only
    # accept a plausible split: a wrong event count invalidates every row.
    if vlm is not None and len(events) == 1:
        llm_events = vlm.split_events(query.text)
        if llm_events and 2 <= len(llm_events) <= 6:
            events = llm_events
            print(f"  [{query.query_id}] VLM split into {len(events)} events (regex found 1)")
    event_vecs = encoder.encode_batch(events)
    event_hits = [clip_search(v, resources, top_k=clip_k) for v in event_vecs]

    voted_videos = _video_vote(event_hits)
    if not voted_videos:
        return []

    candidates: list[TrakeCandidate] = []
    seen_rows: set[tuple[str, tuple[int, ...]]] = set()

    def _emit(video: str, frames: list[int]) -> None:
        key = (video, tuple(frames))
        if len(frames) == len(events) and key not in seen_rows and len(candidates) < 100:
            seen_rows.add(key)
            candidates.append(TrakeCandidate(video=video, frames=list(frames)))

    # Try the top-voted videos; usually the first is right and the rest are
    # cheap hedges against a wrong video lock (a wrong video scores 0).
    for video in voted_videos[:max_videos]:
        options = [
            _event_frames_in_video(vec, video, resources, per_event_frames)
            for vec in event_vecs
        ]
        if vlm is not None:
            options = [
                _verified_order(vlm, resources, video, event, frames)
                for event, frames in zip(events, options)
            ]
        frame_options = [[h["frame_idx"] for h in opts] for opts in options]
        primary = _greedy_chronological(frame_options)
        if primary is None:
            continue
        _emit(video, primary)

        # Variants: swap the j-th event to its 2nd/3rd best frame. Only
        # emit when the swapped assignment is STILL chronological —
        # re-sorting instead would silently permute frames across events.
        for swap_depth in range(1, per_event_frames):
            for j in range(len(events)):
                if swap_depth >= len(frame_options[j]):
                    continue
                variant = list(primary)
                variant[j] = frame_options[j][swap_depth]
                if all(b > a for a, b in zip(variant, variant[1:])):
                    _emit(video, variant)
        if len(candidates) >= 100:
            break

    return candidates

