# RunPod setup

Build the package on your PC, push it, then install that wheel on the pod and open the notebook. Model weights stay at `/models/gemma-2-9b` and are not part of the wheel.

Current package version: **0.1.4** (`pyproject.toml` and `src/multilingual_mechinterp/__init__.py`).

`notebooks/runpod_setup.py` points the notebooks at `/models/gemma-2-9b`. Do not change that path on the pod.

## 1. On your PC — build the wheel

From the repo root in PowerShell. Bump both version strings first whenever `src/` changed, and use the same number in both files.

```powershell
# 1. Delete older builds so dist/ holds one wheel.
Get-ChildItem dist\multilingual_mechinterp-* | Remove-Item -Force

# 2. Build into dist/. Writes dist\multilingual_mechinterp-0.1.4-py3-none-any.whl
poetry build

# 3. Confirm one wheel.
Get-ChildItem dist\*.whl
```

Commit and push the wheel, the version files, `notebooks/runpod_setup.py`, and `notebooks/placement_1.ipynb`. The pod only sees what is on GitHub. Do not push a commit that deletes `dist/*.whl` without adding the new wheel.

## 2. On the pod — pull Gemma 2 9B

Download from Hugging Face first. Use the S3 copy only after Hub returns HTTP 429 or another rate-limit error. Both commands write the same folder, `/models/gemma-2-9b`, which is what `notebooks/runpod_setup.py` loads. That folder is on the pod disk and is wiped when the pod is terminated, so the next pod downloads again.

### Hugging Face (default)

Gemma 2 9B is gated. Accept the license at [google/gemma-2-9b](https://huggingface.co/google/gemma-2-9b) before the download. The weights are free to download. Hub limits how many file requests an account can make, not how many times you may download the model. Published file-resolver caps are about 3,000 requests per 5 minutes anonymous, 5,000 on a free account, and 12,000 on Pro. Those caps change, and a shared RunPod IP can still be throttled sooner. A normal Placement run that downloads Gemma a few times stays inside them.

| | Hugging Face | S3 bucket |
| --- | --- | --- |
| Weight download | Free | Outbound transfer can be charged on every pull |
| Account | Required for this gated model | Not required for the public `s5cmd` read |
| Pod path | `/models/gemma-2-9b` | `/models/gemma-2-9b` |
| When to use | First | After a Hub rate limit |

```bash
pip install -U huggingface_hub
hf auth login
mkdir -p /models/gemma-2-9b
hf download google/gemma-2-9b --local-dir /models/gemma-2-9b
ls /models/gemma-2-9b/config.json
```

`hf auth login` asks for a read token in the pod terminal. Do not put that token in the notebook or the git repo.

Pin the Hub revision so a later file change on the model page does not move the checkpoint under you. Copy the commit hash from the model page and pass it:

```bash
hf download google/gemma-2-9b \
  --revision <commit-from-the-model-page> \
  --local-dir /models/gemma-2-9b
```

Record that hash in the run notes. If the files are already in the local Hub cache, the download reuses them.

### S3 (after a Hub rate limit)

```bash
pip install -U s5cmd
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

The print must match the version you just pushed (`0.1.4` until you bump it). If `dist/` has more than one wheel, delete the old ones on your PC, rebuild, push, pull, and install again.

## 4. Run Placement 1

1. Jupyter on the pod → open `notebooks/placement_1.ipynb`.
2. Run all cells. Leave `N_ITEMS = 8` for the first pass.
3. Notes and figures are written to `output/placement_1/`.
4. Stop the GPU pod when the run finishes. The GPU keeps billing while the pod is up.

For this placement, download Gemma from Hugging Face, run SAE, Jacobian Lens, and activation patching on an A100 80 GB, and keep trained SAE weights, figures, and `output/placement_1/` as the research record. A community L40 (48 GB) at $0.69/hr also fits the same run. There is no pod on the account until you start one. The GPU and the pod disk bill while the pod is up. Hub downloads of the weights stay free until a rate limit.
