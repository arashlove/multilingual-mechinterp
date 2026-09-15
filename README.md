# multilingual-mechinterp

Mechanistic interpretability toolkit for **multilingual** LLMs — Sparse Autoencoders (SAE), JLens, and activation patching — as an installable Python package with notebooks as experiments.

## Install

### Local (uv)

```bash
uv sync
# or: uv pip install -e ".[dev]"
```

Activate the venv (optional — prefer `uv run`):

```powershell
.\.venv\Scripts\Activate.ps1
```

### Colab / Drive wheels

Prefer installing wheels from Drive `dist/` via the notebook setup cell. Editable fallback:

```python
!pip install -e .
```

Or with the lock-free requirements file:

```bash
uv pip install -r requirements.txt
uv pip install -e .
```

## Quick start

```python
from multilingual_mechinterp.models import load_model
from multilingual_mechinterp.sae import run_sae, train_tied_sae, TiedSAE
from multilingual_mechinterp.jlens import run_jlens
from multilingual_mechinterp.patching import run_patching

model = load_model("EleutherAI/pythia-70m-deduped")

sae_result = run_sae(model=model, prompt="Hello world", layer=1)
jlens_result = run_jlens(model=model, prompt="Hello world")
patch_result = run_patching(
    model=model,
    source_prompt="The capital of France is",
    target_prompt="پایتخت فرانسه است",
)
```

### SAE Phase 0–1 (Cunningham et al.)

Paper guide: [`docs/GETTING_STARTED.md`](docs/GETTING_STARTED.md).

```bash
# extract Pythia-70M residual activations (layer 1)
python scripts/extract_activations.py --config configs/pythia70m_sae.yaml

# train tied SAE L1 sweep (FVU / L0 logged)
python scripts/train_sae_l1_sweep.py --config configs/pythia70m_sae.yaml
```

```python
from multilingual_mechinterp.sae import train_tied_sae, sweep_l1
from multilingual_mechinterp.metrics import evaluate_sae

result = train_tied_sae("data/processed/activations/layer_1", ratio=2, alpha=8.6e-4)
evaluate_sae(result.sae, ...)  # FVU + mean L0
```

### Activation patching

Guide: [`docs/activation_patching_reimpl/`](docs/activation_patching_reimpl/).

```python
from multilingual_mechinterp.patching import run_patching, recommend_layers

result = run_patching(
    model,
    source_prompt=prompt_en,
    target_prompt=prompt_fa,
    answer="Divan-e Hafez",
)
layers_to_swap = recommend_layers(result, top_k=3)  # from ΔP curve
```


Guide: [`docs/jlensreimplement/`](docs/jlensreimplement/).

```python
from multilingual_mechinterp.jlens import TinyDecoder, fit, run_jlens, from_hf

# Offline closed-form toy model
model = TinyDecoder()
lens = fit(model, ["long enough prompt for the jacobian estimator.."], skip_first=0)
result = run_jlens(model, "The capital of France is", lens=lens, use_jacobian=True)
```


```python
from multilingual_mechinterp import analyze

results = analyze(
    model=model,
    prompts=["Hello world"],
    methods=["sae", "jlens", "patching"],
)
```

## Repo layout

Reusable code lives under `src/multilingual_mechinterp/`. Notebooks only load config, call the package, and visualize.

```text
src/multilingual_mechinterp/   # installable package
notebooks/                     # experiments / demos
configs/                       # YAML configs
data/                          # prompts + processed artefacts
results/                       # method outputs
tests/                         # unit tests
```

## Notebooks

| Notebook | Purpose |
|----------|---------|
| `01_sae.ipynb` | Tied SAE train + FVU/L0 Pareto plots |
| `02_jlens.ipynb` | Fit/apply Jacobian lens + readout plots |
| `03_activation_patching.ipynb` | Causal ΔP curve + `recommend_layers` |
| `04_common_example.ipynb` | One culture pair through all three methods |
| `05_compare_methods.ipynb` | Multi-item comparison + swap-layer summary |

### Colab (Google Drive + wheels)

1. Copy the project to Drive as **`multilingual_mech`** (include wheels + data):

```text
MyDrive/multilingual_mech/
  data/
    all200questions_persianMiddleEastCulture.json   # 200 culture questions
  dist/
    *.whl                                           # your wheel builds (e.g. 2 wheels)
  configs/
  notebooks/
  output/                                           # auto-created; all exports go here
```

2. Build the wheel locally with uv (then upload `dist/`):

```bash
uv build --out-dir dist
# → dist/multilingual_mechinterp-*.whl
```

3. In Colab, open a notebook and run the first cell. It will:
   - mount Drive
   - locate `MyDrive/multilingual_mech/`
   - `pip install` wheels from `dist/`
   - point `DATA_DIR` at Drive data
   - set **`OUTPUT_DIR`** → `multilingual_mech/output/` (plots, JSON, checkpoints)
   - set **`MODEL_NAME`** (default `Qwen/Qwen2.5-1.5B`)

Swap models later by editing `notebooks/colab_setup.py` or overriding in the notebook:

```python
MODEL_NAME = "google/gemma-2-2b"   # or Qwen/Qwen2.5-7B, etc.
model = load_experiment_model(MODEL_NAME)
```

Set `USE_TINY_OFFLINE = True` in `colab_setup.py` for demos without downloading HF weights.

## Development

```bash
uv run pytest
uv run ruff check src tests
uv build --out-dir dist
```

## License

MIT
