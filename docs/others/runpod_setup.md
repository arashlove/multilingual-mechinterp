# RunPod setup

Build the package on your PC, push it, then install that wheel on the pod and open the notebook. Model weights stay at `/models/gemma-2-9b` and are not part of the wheel.

Current package version: **0.1.3** (`pyproject.toml` and `src/multilingual_mechinterp/__init__.py`).

`notebooks/runpod_setup.py` points the notebooks at `/models/gemma-2-9b`. Do not change that path on the pod.

## 1. On your PC — build the wheel

From the repo root in PowerShell. Bump both version strings first whenever `src/` changed, and use the same number in both files.

```powershell
# 1. Delete older builds so dist/ holds one wheel.
Get-ChildItem dist\multilingual_mechinterp-* | Remove-Item -Force

# 2. Build into dist/. Writes dist\multilingual_mechinterp-0.1.3-py3-none-any.whl
poetry build

# 3. Confirm one wheel.
Get-ChildItem dist\*.whl
```

Commit and push the wheel, the version files, `notebooks/runpod_setup.py`, and `notebooks/placement_1.ipynb`. The pod only sees what is on GitHub. Do not push a commit that deletes `dist/*.whl` without adding the new wheel.

## 2. On the pod — pull Gemma 2 9B

Container disk is wiped when the pod is terminated. Pull the weights again after that. A stop/start of the same pod keeps `/models` only if the container disk survived; if `config.json` is missing, run this again.

```bash
mkdir -p /models/gemma-2-9b
s5cmd --no-sign-request cp \
  's3://arash-llm-models-790189127408-ap-southeast-2-an/gemma-2-9b/*' \
  /models/gemma-2-9b/
ls /models/gemma-2-9b/config.json
```

## 3. On the pod — install `dist/`

```bash
cd /workspace
git clone https://github.com/arashlove/multilingual-mechinterp.git
# later runs, from the repo:
# git pull

cd /workspace/multilingual-mechinterp
ls dist/*.whl
python -m pip install --upgrade dist/multilingual_mechinterp-*.whl
python -c "import multilingual_mechinterp as m; print(m.__version__)"
```

The print must match the version you just pushed (`0.1.3` until you bump it). If `dist/` has more than one wheel, delete the old ones on your PC, rebuild, push, pull, and install again.

## 4. Run Placement 1

1. Jupyter on the pod → open `notebooks/placement_1.ipynb`.
2. Run all cells. Leave `N_ITEMS = 8` for the first pass.
3. Notes and figures are written to `output/placement_1/`.
4. Stop the GPU pod when the run finishes. The GPU keeps billing while the pod is up.

A community L40 (48 GB) at $0.69/hr fits Gemma 2 9B plus the Jacobian stage. There is no pod on the account until you start one.
