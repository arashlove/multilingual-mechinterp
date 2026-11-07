"""Run directories, provenance manifests and stage status for Placement 2."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any

from multilingual_mechinterp.utils.config import load_config
from multilingual_mechinterp.utils.io import ensure_dir, save_json

APPROVAL_REQUIRED = "approval_required"

STAGE_STATES = ("pending", "running", "ok", "failed", "skipped", "block_missing_data")

# Phase order: human preprocessing must finish before scenarios are authored.
STAGES = (
    "inspect_human_source",
    "preprocess_human",
    "summarize_human",
    "export_scenario_guide",
    "init_templates",
    "validate_data",
    "check_human_coverage",
    "finalize_human_labels",
    "run_baseline",
    "evaluate_human_agreement",
    "select_cases",
    "fit_jlens",
    "analyze_jlens",
    "intervene",
    "report",
)

REQUIRED_SECTIONS = ("run", "human", "inputs", "model")

TRACKED_PACKAGES = (
    "multilingual-mechinterp",
    "torch",
    "transformers",
    "numpy",
    "matplotlib",
    "pyyaml",
    "datasets",
    "polars",
    "pyarrow",
)

# Weights are too large to hash on every run; these small files pin the snapshot.
MODEL_SNAPSHOT_FILES = (
    "config.json",
    "generation_config.json",
    "model.safetensors.index.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
)


def load_placement2_config(path: str | Path) -> dict[str, Any]:
    cfg = load_config(path)
    missing = [name for name in REQUIRED_SECTIONS if not isinstance(cfg.get(name), dict)]
    if missing:
        raise ValueError(f"{path}: missing config sections: {', '.join(missing)}")
    return cfg


def pending_approvals(cfg: dict[str, Any]) -> list[str]:
    """Dotted keys whose value still needs researcher sign-off."""
    pending: list[str] = []

    def walk(node: Any, prefix: str) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                walk(value, f"{prefix}.{key}" if prefix else str(key))
        elif node == APPROVAL_REQUIRED:
            pending.append(prefix)

    walk(cfg, "")
    if not cfg.get("model", {}).get("revision"):
        pending.append("model.revision")
    return pending


def sha256_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        while chunk := f.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(obj: Any) -> str:
    text = json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def input_paths(cfg: dict[str, Any]) -> dict[str, str]:
    human = cfg.get("human", {})
    paths = {
        key: human[key]
        for key in ("source_manifest", "country_culture_map", "language_reference_map")
        if human.get(key)
    }
    paths.update({key: value for key, value in cfg.get("inputs", {}).items() if value})
    return paths


def hash_inputs(cfg: dict[str, Any], root: str | Path = ".") -> dict[str, dict[str, Any]]:
    root = Path(root)
    hashed: dict[str, dict[str, Any]] = {}
    for name, rel in input_paths(cfg).items():
        path = root / rel
        exists = path.is_file()
        hashed[name] = {
            "path": str(rel),
            "exists": exists,
            "sha256": sha256_file(path) if exists else None,
        }
    return hashed


def git_info(root: str | Path = ".") -> dict[str, Any] | None:
    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=root, capture_output=True, text=True, check=True
        ).stdout.strip()

    try:
        return {"commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain"))}
    except (OSError, subprocess.CalledProcessError):
        return None


def package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in TRACKED_PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def environment_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": package_versions(),
        "cuda": None,
    }
    try:
        import torch
    except ImportError:
        return info
    cuda: dict[str, Any] = {"torch_cuda": torch.version.cuda, "available": torch.cuda.is_available()}
    if cuda["available"]:
        props = torch.cuda.get_device_properties(0)
        cuda.update({"device_name": props.name, "total_memory_bytes": props.total_memory})
    info["cuda"] = cuda
    return info


def local_commit(local: str | Path) -> str | None:
    """Hub commit of a ``huggingface-cli download --local-dir`` folder, read from the
    download metadata it leaves in ``.cache/huggingface/download`` (or from a cache
    ``snapshots/<commit>`` path). None when neither is present."""
    local = Path(local)
    if local.parent.name == "snapshots":
        return local.name
    meta_dir = local / ".cache" / "huggingface" / "download"
    for meta in sorted(meta_dir.glob("*.metadata")) if meta_dir.is_dir() else []:
        lines = meta.read_text(encoding="utf-8", errors="ignore").splitlines()
        if lines and len(lines[0].strip()) == 40:
            return lines[0].strip()
    return None


def model_identity(cfg: dict[str, Any]) -> dict[str, Any]:
    model = cfg.get("model", {})
    identity: dict[str, Any] = {
        "model_id": model.get("model_id"),
        "revision": model.get("revision"),
        "local_path": model.get("local_path"),
        "dtype": model.get("dtype"),
        "local_commit": None,
        "local_snapshot": None,
    }
    local = Path(model["local_path"]) if model.get("local_path") else None
    if local is not None and local.is_dir():
        identity["local_commit"] = local_commit(local)
        identity["local_snapshot"] = {
            name: sha256_file(local / name)
            for name in MODEL_SNAPSHOT_FILES
            if (local / name).is_file()
        }
    return identity


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class RunContext:
    """One run directory under ``run.output_root`` with a manifest and stage status.

    Nothing here deletes artifacts; lens checkpoints and caches are kept.
    """

    def __init__(
        self,
        cfg: dict[str, Any],
        *,
        config_path: str | Path | None = None,
        run_id: str | None = None,
        root: str | Path = ".",
    ) -> None:
        self.cfg = cfg
        self.root = Path(root)
        self.config_path = Path(config_path) if config_path else None
        self.run_id = run_id or cfg["run"]["name"]
        self.run_dir = self.root / cfg["run"]["output_root"] / self.run_id
        self.manifest_path = self.run_dir / "run_manifest.json"
        self.status_path = self.run_dir / "status.json"

    @property
    def config_sha256(self) -> str:
        return sha256_json(self.cfg)

    def build_manifest(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "created_at": _now(),
            "config_path": str(self.config_path) if self.config_path else None,
            "config_sha256": self.config_sha256,
            "seed": self.cfg["run"].get("seed"),
            "pending_approvals": pending_approvals(self.cfg),
            "inputs": hash_inputs(self.cfg, self.root),
            "model": model_identity(self.cfg),
            "environment": environment_info(),
            "git": git_info(self.root),
        }

    def write_manifest(self) -> dict[str, Any]:
        manifest = self.build_manifest()
        save_json(manifest, self.manifest_path)
        status = self.load_status()
        for stage in STAGES:
            status.setdefault(stage, {"state": "pending"})
        self._save_status(status)
        return manifest

    def load_status(self) -> dict[str, dict[str, Any]]:
        if not self.status_path.is_file():
            return {}
        with self.status_path.open(encoding="utf-8") as f:
            return json.load(f)

    def _save_status(self, status: dict[str, dict[str, Any]]) -> None:
        save_json(status, self.status_path)

    def set_status(self, stage: str, state: str, **fields: Any) -> dict[str, Any]:
        if stage not in STAGES:
            raise ValueError(f"Unknown stage: {stage}")
        if state not in STAGE_STATES:
            raise ValueError(f"Unknown stage state: {state}")
        status = self.load_status()
        entry = {"state": state, "updated_at": _now(), **fields}
        status[stage] = entry
        self._save_status(status)
        return entry

    def block(self, stage: str, reason: str) -> dict[str, Any]:
        return self.set_status(stage, "block_missing_data", reason=reason)

    def stage_dir(self, stage: str) -> Path:
        return ensure_dir(self.run_dir / stage)

    def is_fresh(self, stage: str, fingerprint: str) -> bool:
        """True when ``stage`` finished ok with the same inputs/config fingerprint."""
        entry = self.load_status().get(stage, {})
        return entry.get("state") == "ok" and entry.get("fingerprint") == fingerprint

    @contextmanager
    def stage(self, name: str, *, fingerprint: str | None = None) -> Iterator[Path]:
        """Mark ``name`` running, then ok or failed; yields the stage output folder."""
        torch = sys.modules.get("torch")
        track_gpu = torch is not None and torch.cuda.is_available()
        if track_gpu:
            torch.cuda.reset_peak_memory_stats()

        started_at = _now()
        start = time.perf_counter()
        self.set_status(name, "running", started_at=started_at, fingerprint=fingerprint)
        out_dir = self.stage_dir(name)

        def timings() -> dict[str, Any]:
            fields: dict[str, Any] = {
                "started_at": started_at,
                "finished_at": _now(),
                "wall_seconds": round(time.perf_counter() - start, 3),
                "fingerprint": fingerprint,
                "config_sha256": self.config_sha256,
                "output_dir": str(out_dir),
            }
            if track_gpu:
                fields["peak_gpu_bytes"] = torch.cuda.max_memory_allocated()
            return fields

        try:
            yield out_dir
        except BaseException as exc:
            self.set_status(name, "failed", error=f"{type(exc).__name__}: {exc}", **timings())
            raise
        self.set_status(name, "ok", **timings())
