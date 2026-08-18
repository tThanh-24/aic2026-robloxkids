#!/usr/bin/env python3
"""Peeks at one real sample from each downloaded asset category and prints
its actual schema -- run this ONCE after your first download batch and
before writing/trusting any ingestion code, since map-keyframes column
names, clip-features shape/dtype, and objects/media-info JSON schema are
all currently assumptions in this repo (see README "Open questions").

Usage:
    python scripts/inspect_dataset.py
    python scripts/inspect_dataset.py --config configs/local.yaml
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aic_system.config import load_config, resolve_path  # noqa: E402


def _peek_first(dir_path: Path, pattern: str) -> Path | None:
    matches = sorted(dir_path.rglob(pattern))
    return matches[0] if matches else None


def inspect_keyframes(cfg):
    d = resolve_path(cfg, "raw_keyframes")
    print(f"\n=== Keyframes ({d}) ===")
    if not d.exists() or not any(d.iterdir()):
        print("  (not downloaded yet)")
        return
    sample_img = _peek_first(d, "*.jpg") or _peek_first(d, "*.webp") or _peek_first(d, "*.png")
    if sample_img:
        rel = sample_img.relative_to(d)
        print(f"  sample file: {rel}")
        print(f"  parent dir naming (video id?): {sample_img.parent.name}")
        siblings = sorted(sample_img.parent.glob("*"))
        print(f"  siblings in that dir: {len(siblings)} files, "
              f"e.g. {[s.name for s in siblings[:5]]}")
    else:
        print("  no image files found under expected extensions (.jpg/.webp/.png)")


def inspect_videos(cfg):
    d = resolve_path(cfg, "raw_videos")
    print(f"\n=== Videos ({d}) ===")
    if not d.exists() or not any(d.iterdir()):
        print("  (not downloaded yet)")
        return
    sample = _peek_first(d, "*.mp4")
    if sample:
        print(f"  sample file: {sample.relative_to(d)}")
        print(f"  size: {sample.stat().st_size / 1e6:.1f} MB")


def inspect_map_keyframes(cfg):
    d = resolve_path(cfg, "map_keyframes")
    print(f"\n=== map-keyframes ({d}) ===")
    if not d.exists() or not any(d.iterdir()):
        print("  (not downloaded yet)")
        return
    sample = _peek_first(d, "*.csv")
    if sample:
        print(f"  sample file: {sample.relative_to(d)}")
        with open(sample, "r", encoding="utf-8") as f:
            lines = [next(f) for _ in range(5) if f]
        for line in lines:
            print(f"    {line.rstrip()}")
    else:
        print("  no .csv files found -- check actual extension (might be .txt or .json)")


def inspect_clip_features(cfg):
    d = resolve_path(cfg, "clip_features")
    print(f"\n=== clip-features ({d}) ===")
    if not d.exists() or not any(d.iterdir()):
        print("  (not downloaded yet)")
        return
    sample = _peek_first(d, "*.npy")
    if sample:
        try:
            import numpy as np
            arr = np.load(sample)
            print(f"  sample file: {sample.relative_to(d)}")
            print(f"  shape: {arr.shape}, dtype: {arr.dtype}")
        except ImportError:
            print(f"  sample file: {sample.relative_to(d)} (install numpy to inspect shape)")
    else:
        print("  no .npy files found -- check actual format (might be .bin, .pt, .json)")


def inspect_media_info(cfg):
    d = resolve_path(cfg, "media_info")
    print(f"\n=== media-info ({d}) ===")
    if not d.exists() or not any(d.iterdir()):
        print("  (not downloaded yet)")
        return
    sample = _peek_first(d, "*.json")
    if sample:
        print(f"  sample file: {sample.relative_to(d)}")
        with open(sample, "r", encoding="utf-8") as f:
            data = json.load(f)
        print(f"  top-level keys: {list(data.keys())}")
        print(f"  sample content: {json.dumps(data, ensure_ascii=False, indent=2)[:500]}")


def inspect_objects(cfg):
    d = resolve_path(cfg, "objects")
    print(f"\n=== objects ({d}) ===")
    if not d.exists() or not any(d.iterdir()):
        print("  (not downloaded yet)")
        return
    sample = _peek_first(d, "*.json")
    if sample:
        print(f"  sample file: {sample.relative_to(d)}")
        with open(sample, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            print(f"  list of {len(data)} entries; first entry keys: "
                  f"{list(data[0].keys()) if data else '(empty)'}")
            if data:
                print(f"  sample entry: {json.dumps(data[0], ensure_ascii=False, indent=2)[:500]}")
        elif isinstance(data, dict):
            print(f"  top-level keys: {list(data.keys())}")
            print(f"  sample content: {json.dumps(data, ensure_ascii=False, indent=2)[:500]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="default")
    args = ap.parse_args()
    cfg = load_config(Path(args.config).stem if "/" in args.config else args.config)

    inspect_keyframes(cfg)
    inspect_videos(cfg)
    inspect_map_keyframes(cfg)
    inspect_clip_features(cfg)
    inspect_media_info(cfg)
    inspect_objects(cfg)

    print("\nDone. Update src/aic_system/ingest/*.py and configs/default.yaml "
          "if any of the above differs from what's currently assumed there.")


if __name__ == "__main__":
    main()
