"""Colab / local bootstrap for multilingual-mechinterp notebooks.

Expected Google Drive layout (copy the whole project folder)::

    MyDrive/multilingual-mechinterp/
      dist/*.whl          # built wheels (and deps if you vendor them)
      data/               # culture JSON, activations, …
      configs/
      notebooks/
      results/

Usage in a notebook (first code cell)::

    from pathlib import Path
    import runpy
    setup = runpy.run_path(str(Path("colab_setup.py")))  # if cwd is notebooks/
    # or:
    # setup = runpy.run_path("/content/drive/MyDrive/multilingual-mechinterp/notebooks/colab_setup.py")
    globals().update({k: setup[k] for k in setup["EXPORTS"]})
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# EDIT THESE for your Colab run
# ---------------------------------------------------------------------------

# Folder name (or absolute path) under Google Drive
DRIVE_FOLDER_NAME = "multilingual-mechinterp"
DRIVE_SEARCH_ROOTS = (
    Path("/content/drive/MyDrive"),
    Path("/content/drive/MyDrive/Documents"),
    Path("/content/drive/MyDrive/GitHub"),
)

# Swappable HF model — change this one line to switch later (e.g. Gemma)
MODEL_NAME = "Qwen/Qwen2.5-1.5B"  # later: "google/gemma-2-2b" / "google/gemma-2-9b"
MODEL_DTYPE = "auto"  # "bfloat16" | "float16" | "float32" | "auto"
MODEL_TRUST_REMOTE_CODE = True  # Qwen often needs this; Gemma usually False
HF_TOKEN_ENV = "HF_TOKEN"  # optional gated models

USE_TINY_OFFLINE = False  # True → skip HF download, use TinyCausalLM / TinyDecoder demos
INSTALL_WHEELS = True
MOUNT_DRIVE = True

EXPORTS = (
    "ROOT",
    "DATA_DIR",
    "DIST_DIR",
    "CONFIG_DIR",
    "RESULTS_DIR",
    "MODEL_NAME",
    "MODEL_DTYPE",
    "USE_TINY_OFFLINE",
    "IN_COLAB",
    "load_experiment_model",
    "culture_json_path",
)


def _in_colab() -> bool:
    try:
        import google.colab  # noqa: F401

        return True
    except ImportError:
        return False


def _mount_drive() -> None:
    if not MOUNT_DRIVE or not _in_colab():
        return
    from google.colab import drive

    mount_point = Path("/content/drive")
    if not (mount_point / "MyDrive").exists():
        drive.mount(str(mount_point))


def _find_project_root() -> Path:
    """Locate repo root: cwd, parent, or Drive search."""
    cwd = Path.cwd().resolve()
    for candidate in (cwd, cwd.parent):
        if (candidate / "src" / "multilingual_mechinterp").exists() or (candidate / "pyproject.toml").exists():
            return candidate
        if (candidate / "data").exists() and (candidate / "notebooks").exists():
            return candidate

    if _in_colab():
        _mount_drive()
        # Absolute path override
        env = os.environ.get("MMI_PROJECT_ROOT")
        if env:
            p = Path(env)
            if p.exists():
                return p

        name = DRIVE_FOLDER_NAME
        if Path(name).is_absolute() and Path(name).exists():
            return Path(name)

        for root in DRIVE_SEARCH_ROOTS:
            if not root.exists():
                continue
            direct = root / name
            if direct.exists():
                return direct
            # shallow search
            matches = list(root.glob(f"**/{name}"))
            for m in matches:
                if m.is_dir() and ((m / "data").exists() or (m / "dist").exists() or (m / "notebooks").exists()):
                    return m

    # Fall back to repo-ish cwd
    return cwd if (cwd / "notebooks").exists() else cwd.parent


def _pip_install_wheels(dist_dir: Path) -> None:
    if not INSTALL_WHEELS:
        return
    wheels = sorted(dist_dir.glob("*.whl"))
    if not wheels:
        print(f"[colab_setup] No wheels in {dist_dir} — trying editable / requirements.")
        req = ROOT / "requirements.txt"
        pyproject = ROOT / "pyproject.toml"
        if (ROOT / "src").exists() and pyproject.exists():
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-e", str(ROOT), "-q"])
        elif req.exists():
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", str(req), "-q"])
            if (ROOT / "src").exists():
                sys.path.insert(0, str(ROOT / "src"))
        return

    print(f"[colab_setup] Installing {len(wheels)} wheel(s) from {dist_dir}")
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "--upgrade", *[str(w) for w in wheels], "-q"]
    )
    # common notebook extras
    subprocess.check_call([sys.executable, "-m", "pip", "install", "matplotlib", "-q"])


def load_experiment_model(model_name: str | None = None, **kwargs):
    """Load the configured HF model, or a tiny offline stand-in."""
    name = model_name or MODEL_NAME
    if USE_TINY_OFFLINE:
        from multilingual_mechinterp.patching import TinyCausalLM

        print("[colab_setup] USE_TINY_OFFLINE=True → TinyCausalLM")
        return TinyCausalLM()

    from multilingual_mechinterp.models import load_model

    token = os.environ.get(HF_TOKEN_ENV) or kwargs.pop("token", None)
    load_kwargs = {
        "dtype": kwargs.pop("dtype", MODEL_DTYPE),
        "trust_remote_code": kwargs.pop("trust_remote_code", MODEL_TRUST_REMOTE_CODE),
        **kwargs,
    }
    if token:
        load_kwargs["token"] = token
    print(f"[colab_setup] Loading model: {name}")
    return load_model(name, **load_kwargs)


def culture_json_path() -> Path:
    return DATA_DIR / "all200questions_persianMiddleEastCulture.json"


# ---- run on import / runpy ----
IN_COLAB = _in_colab()
ROOT = _find_project_root()
DATA_DIR = ROOT / "data"
DIST_DIR = ROOT / "dist"
CONFIG_DIR = ROOT / "configs"
RESULTS_DIR = ROOT / "results"

if str(ROOT / "src") not in sys.path and (ROOT / "src").exists():
    sys.path.insert(0, str(ROOT / "src"))

print(f"[colab_setup] IN_COLAB={IN_COLAB}")
print(f"[colab_setup] ROOT={ROOT}")
print(f"[colab_setup] MODEL_NAME={MODEL_NAME}  USE_TINY_OFFLINE={USE_TINY_OFFLINE}")

if DIST_DIR.exists() or (ROOT / "pyproject.toml").exists():
    try:
        _pip_install_wheels(DIST_DIR)
    except subprocess.CalledProcessError as exc:
        print(f"[colab_setup] wheel install failed: {exc}")

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
for sub in ("sae", "jlens", "patching", "comparison"):
    (RESULTS_DIR / sub).mkdir(parents=True, exist_ok=True)

if not culture_json_path().exists():
    print(f"[colab_setup] WARNING: culture JSON not found at {culture_json_path()}")
else:
    print(f"[colab_setup] culture data OK: {culture_json_path()}")
