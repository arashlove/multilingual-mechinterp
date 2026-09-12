"""YAML config loading."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config file into a dict."""
    path = Path(path)
    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config root must be a mapping: {path}")
    return data


def merge_configs(*configs: dict[str, Any]) -> dict[str, Any]:
    """Shallow-merge configs left-to-right (later keys win)."""
    merged: dict[str, Any] = {}
    for cfg in configs:
        merged.update(cfg)
    return merged
