"""Rewrite notebook setup cells to use notebooks/colab_setup.py."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB_DIR = ROOT / "notebooks"

SETUP_SOURCE = r'''# Drive layout expected:
#   MyDrive/multilingual_mech/
#     data/all200questions_persianMiddleEastCulture.json
#     dist/*.whl   (wheel builds)
#     configs/  notebooks/  output/
#
# Edit MODEL_NAME in notebooks/colab_setup.py (Qwen2.5 now; Gemma later),
# or override below after bootstrap.

from pathlib import Path
import runpy

def _resolve_setup_script() -> Path:
    here = Path.cwd()
    candidates = [
        here / "colab_setup.py",
        here / "notebooks" / "colab_setup.py",
        here.parent / "notebooks" / "colab_setup.py",
        Path("/content/drive/MyDrive/multilingual_mech/notebooks/colab_setup.py"),
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        "colab_setup.py not found. Mount Drive and use folder multilingual_mech/, "
        "or open the notebook from the repo."
    )

_setup = runpy.run_path(str(_resolve_setup_script()))
globals().update({k: _setup[k] for k in _setup["EXPORTS"]})

# Session overrides (uncomment as needed):
# MODEL_NAME = "google/gemma-2-2b"
# MODEL_TRUST_REMOTE_CODE = False
# USE_TINY_OFFLINE = True   # demos without downloading HF weights

import matplotlib.pyplot as plt
import torch

from multilingual_mechinterp.utils import ensure_dir, load_config

cfg_path = CONFIG_DIR / "qwen25.yaml"
cfg = load_config(cfg_path) if cfg_path.exists() else {}
if "model" in cfg and not USE_TINY_OFFLINE:
    # keep notebook MODEL_NAME as source of truth; cfg is fallback metadata
    pass

print("Ready.")
print(" ROOT   =", ROOT)
print(" DATA   =", DATA_DIR)
print(" DIST   =", DIST_DIR)
print(" OUTPUT =", OUTPUT_DIR)  # Drive exports go here
print(" MODEL  =", MODEL_NAME, "| tiny=", USE_TINY_OFFLINE)
'''

# Per-notebook extras appended after shared setup
EXTRAS = {
    "01_sae.ipynb": '''
from multilingual_mechinterp.metrics import evaluate_sae
from multilingual_mechinterp.sae import TiedSAE, sweep_l1, train_tied_sae

OUT = ensure_dir(OUTPUT_DIR / "sae")
sae_cfg = cfg.get("sae", {"ratio": 2, "alpha": 8.6e-4, "lr": 1e-3,
                          "alphas": [1e-4, 3e-4, 8.6e-4, 1.4e-3, 3e-3]})
sae_cfg
''',
    "02_jlens.ipynb": '''
from multilingual_mechinterp.jlens import TinyDecoder, fit, run_jlens, JacobianLens, from_hf

OUT = ensure_dir(OUTPUT_DIR / "jlens")
''',
    "03_activation_patching.ipynb": '''
from multilingual_mechinterp.patching import (
    TinyCausalLM, recommend_layers, run_patching,
)
from multilingual_mechinterp.data import culture_prompt_pairs, load_culture_questions

OUT = ensure_dir(OUTPUT_DIR / "patching")
''',
    "04_common_example.ipynb": '''
from multilingual_mechinterp.data import culture_prompt_pairs, load_culture_questions
from multilingual_mechinterp.jlens import TinyDecoder, fit, run_jlens
from multilingual_mechinterp.metrics import evaluate_sae
from multilingual_mechinterp.patching import TinyCausalLM, recommend_layers, run_patching
from multilingual_mechinterp.sae import train_tied_sae

OUT = ensure_dir(OUTPUT_DIR / "comparison")
''',
    "05_compare_methods.ipynb": '''
from multilingual_mechinterp.data import culture_prompt_pairs, load_culture_questions
from multilingual_mechinterp.jlens import TinyDecoder, fit, run_jlens
from multilingual_mechinterp.metrics import evaluate_sae
from multilingual_mechinterp.patching import TinyCausalLM, recommend_layers, run_patching
from multilingual_mechinterp.sae import train_tied_sae
from multilingual_mechinterp.utils import save_json

OUT = ensure_dir(OUTPUT_DIR / "comparison")
N_ITEMS = 5
''',
}

# Also add a "load model" markdown+cell after setup for HF notebooks guidance
LOAD_MODEL_MD = (
    "## Load experiment model\n\n"
    "Uses `MODEL_NAME` from `colab_setup.py` (default **Qwen2.5**). "
    "Set `USE_TINY_OFFLINE=True` for demos without HF downloads.\n"
)

LOAD_MODEL_CODE = '''# Real model (Qwen now; change MODEL_NAME for Gemma later) OR tiny offline
# model = load_experiment_model()
# For gated Gemma: export HF_TOKEN=... or pass token=...

# Figures / JSON / checkpoints are written under OUTPUT_DIR on Drive:
#   multilingual_mech/output/{sae,jlens,patching,comparison,...}
print("To load HF weights:", f"load_experiment_model({MODEL_NAME!r})")
print("Culture JSON:", culture_json_path(), "exists=", culture_json_path().exists())
print("Exports go to:", OUTPUT_DIR)
'''


def as_nb_source(text: str) -> list[str]:
    text = text.strip("\n") + "\n"
    lines = text.splitlines(keepends=True)
    return lines


def patch_notebook(name: str) -> None:
    path = NB_DIR / name
    nb = json.loads(path.read_text(encoding="utf-8"))
    full_setup = SETUP_SOURCE + "\n" + EXTRAS[name].lstrip("\n")

    # Find first code cell after title (setup) and replace
    code_idxs = [i for i, c in enumerate(nb["cells"]) if c["cell_type"] == "code"]
    if not code_idxs:
        raise RuntimeError(f"no code cells in {name}")
    setup_idx = code_idxs[0]
    nb["cells"][setup_idx]["source"] = as_nb_source(full_setup)
    nb["cells"][setup_idx]["outputs"] = []
    nb["cells"][setup_idx]["execution_count"] = None

    # Insert load-model section after setup if not already present
    already = any(
        "Load experiment model" in "".join(c.get("source", []))
        for c in nb["cells"]
        if c["cell_type"] == "markdown"
    )
    if not already:
        insert_at = setup_idx + 1
        nb["cells"].insert(
            insert_at,
            {"cell_type": "markdown", "metadata": {}, "source": as_nb_source(LOAD_MODEL_MD)},
        )
        nb["cells"].insert(
            insert_at + 1,
            {
                "cell_type": "code",
                "metadata": {},
                "execution_count": None,
                "outputs": [],
                "source": as_nb_source(LOAD_MODEL_CODE),
            },
        )

    path.write_text(json.dumps(nb, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("updated", name)


def main() -> None:
    for name in EXTRAS:
        patch_notebook(name)


if __name__ == "__main__":
    main()
