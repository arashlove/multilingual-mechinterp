"""Load prompt files from disk."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_prompt_file(path: str | Path) -> list[dict[str, Any]]:
    """Load prompts from a JSON or JSONL file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    if path.suffix == ".jsonl":
        rows: list[dict[str, Any]] = []
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows

    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and "prompts" in data:
        return list(data["prompts"])
    raise ValueError(f"Unsupported prompt file format: {path}")


def load_prompts(
    path: str | Path | None = None,
    *,
    language: str | None = None,
) -> list[dict[str, Any]]:
    """Load prompts, optionally filtering by language code."""
    if path is None:
        path = Path(__file__).resolve().parents[3] / "data" / "prompts" / "examples.json"
    rows = load_prompt_file(path)
    if language is None:
        return rows
    return [r for r in rows if r.get("language") == language]
