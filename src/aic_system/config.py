"""Loads configs/default.yaml, optionally merged with configs/local.yaml.

configs/local.yaml is gitignored -- put machine-specific overrides
(paths, device names, etc.) there rather than editing default.yaml.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIGS_DIR = REPO_ROOT / "configs"


def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(name: str = "default") -> dict[str, Any]:
    default_path = CONFIGS_DIR / "default.yaml"
    with open(default_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if name != "default":
        override_path = CONFIGS_DIR / f"{name}.yaml"
        if override_path.exists():
            with open(override_path, "r", encoding="utf-8") as f:
                cfg = _deep_merge(cfg, yaml.safe_load(f) or {})

    local_path = CONFIGS_DIR / "local.yaml"
    if local_path.exists():
        with open(local_path, "r", encoding="utf-8") as f:
            cfg = _deep_merge(cfg, yaml.safe_load(f) or {})

    return cfg


def resolve_path(cfg: dict, key: str) -> Path:
    """Resolve a paths.* config entry to an absolute path under REPO_ROOT."""
    rel = cfg["paths"][key]
    return (REPO_ROOT / rel).resolve()
