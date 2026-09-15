# RunPod + S3 model workflow (Phase 1: two models)

How we store LLM weights in **Amazon S3**, pull them onto a **RunPod** GPU pod with `s5cmd`, and point the notebook at local paths.

## Phase 1 model roles

| Role | Model on S3 | Local path on pod | Used for |
|------|-------------|-------------------|----------|
| **Subject** (interpret this) | `.../gemma-2-9b/` | `/models/gemma-2-9b` | SAE, JLens, activation patching, baseline |
| **Autointerp labeler** | `.../Qwen2.5-32B/` | `/models/Qwen2.5-32B` | Label SAE features from max-activating snippets |

Do **not** use Gemma as both subject and labeler. Load **one at a time** if VRAM is tight: finish Gemma experiments → free GPU → load Qwen for labeling.

Bucket (region `ap-southeast-2`):

```text
s3://arash-llm-models-790189127408-ap-southeast-2-an/
  gemma-2-9b/
  Qwen2.5-32B/
```

---

## Why this path

- Hugging Face from shared RunPod IPs often hits **HTTP 429**.
- `scp` of large checkpoints is slow; **PC → S3 → pod** with public `s5cmd --no-sign-request` is fast (~minutes).

---

## 1. Download models on your PC (already done if uploaded)

```powershell
# Subject
$env:HF_TOKEN = "hf_xxxxxxxx"   # Gemma is gated — accept license on Hub
hf download google/gemma-2-9b --local-dir E:\models\gemma-2-9b --token $env:HF_TOKEN

# Autointerp (ungated)
hf download Qwen/Qwen2.5-32B --local-dir E:\models\Qwen2.5-32B
# Prefer Instruct for labeling if you have it:
# hf download Qwen/Qwen2.5-32B-Instruct --local-dir E:\models\Qwen2.5-32B-Instruct
```

Confirm each folder has `config.json` before upload.

---

## 2. Upload to S3

```powershell
aws s3 sync "E:\models\gemma-2-9b" s3://arash-llm-models-790189127408-ap-southeast-2-an/gemma-2-9b/
aws s3 sync "E:\models\Qwen2.5-32B" s3://arash-llm-models-790189127408-ap-southeast-2-an/Qwen2.5-32B/
```

---

## 3. Public read (anonymous `s5cmd`)

Same as before: bucket policy allowing `s3:GetObject` (+ optional `ListBucket`) for anonymous principals.

**Security:** public weights are fine for open models. Gemma has a license — only publish if your Hub license allows redistribution; otherwise keep Gemma private and use signed URLs / AWS keys on the pod.

Check:

```bash
s5cmd --no-sign-request ls s3://arash-llm-models-790189127408-ap-southeast-2-an/gemma-2-9b/
s5cmd --no-sign-request ls s3://arash-llm-models-790189127408-ap-southeast-2-an/Qwen2.5-32B/
```

---

## 4. Deploy a RunPod GPU pod

### VRAM (both models, ~140 GB card)

| Workload | Rough need | On ~140 GB |
|----------|------------|------------|
| Gemma 2 9B bf16 + SAE / JLens (last 7 layers) / patching | ~25–45 GB | Easy |
| Qwen2.5-32B bf16 autointerp only | ~70–90 GB | Easy |
| Both loaded at once | ~90–110+ GB | Usually OK; safer sequential |

Attach a **network volume** at `/workspace` for the **repo + `output/`**.  
Put **weights on root disk** `/models/...` if the volume quota is small (re-`s5cmd` after terminate is fine).

---

## 5. On the pod: download **both** models

```bash
pip install -U s5cmd   # if needed

mkdir -p /models/gemma-2-9b /models/Qwen2.5-32B

s5cmd --no-sign-request cp \
  's3://arash-llm-models-790189127408-ap-southeast-2-an/gemma-2-9b/*' \
  /models/gemma-2-9b/

s5cmd --no-sign-request cp \
  's3://arash-llm-models-790189127408-ap-southeast-2-an/Qwen2.5-32B/*' \
  /models/Qwen2.5-32B/

# verify
ls /models/gemma-2-9b/config.json
ls /models/Qwen2.5-32B/config.json
du -sh /models/gemma-2-9b /models/Qwen2.5-32B
nvidia-smi
```

If a path is wrong (Transformers “Repo id must be…”), find the folder that **contains** `config.json`:

```bash
find /models -name config.json 2>/dev/null
```

---

## 6. Build + push the wheel **before** deploying the pod

**Do this on your PC every time package code changes, before you create/start a pod (or before `git pull` on an existing pod).**  
The wheel lives in GitHub under `dist/`; the notebook’s `colab_setup` installs `dist/*.whl` automatically — no manual `pip install` / `scp` needed on the pod.

### On your PC (required first)

```powershell
# from repo root
# 1) bump version in pyproject.toml + src/multilingual_mechinterp/__init__.py (e.g. 0.1.0 → 0.1.1)
uv build --out-dir dist
# → dist/multilingual_mechinterp-0.1.1-py3-none-any.whl

# 2) remove older wheels so only the new one is in dist/
Get-ChildItem dist\*.whl

# 3) commit and push (include dist/*.whl)
git add dist/*.whl pyproject.toml src/multilingual_mechinterp/__init__.py
git commit -m "Build wheel 0.1.1 for RunPod notebook install"
git push
```

An outdated `dist/*.whl` on GitHub causes errors like:

```text
cannot import name 'score_mcq_prompt' from 'multilingual_mechinterp.data'
(.../dist-packages/multilingual_mechinterp/...)
```

### Then on the pod (after models are pulled)

```bash
cd /workspace
git clone https://github.com/arashlove/multilingual-mechinterp.git
# existing checkout: cd multilingual-mechinterp && git pull
cd /workspace/multilingual-mechinterp
ls dist/*.whl   # confirm the version you just pushed
```

Open `notebooks/Full_mechanistic_interp.ipynb` → **restart kernel** → run the first cell.  
`colab_setup` installs the wheel from `dist/`.

**Fallback only** if there is no wheel: `pip install -e .`

---

## 7. Point `colab_setup` at **Gemma** (subject)

In `notebooks/colab_setup.py`:

```python
MODEL_NAME = "/models/gemma-2-9b"
MODEL_TRUST_REMOTE_CODE = False
USE_TINY_OFFLINE = False
# HF_TOKEN not needed for local path
```

Smoke test:

```python
from multilingual_mechinterp.models import load_model
m = load_model("/models/gemma-2-9b", trust_remote_code=False)
print(m.n_layers, m.device)
```

### Autointerp with Qwen 32B (later cell / separate kernel)

After SAE max-activating snippets are saved (`output/sae/autointerp_top_features.json`):

1. Free Gemma from GPU (`del model`; `torch.cuda.empty_cache()`), or restart kernel and **don’t** reload Gemma.
2. Load Qwen only for labeling:

```python
labeler = load_model("/models/Qwen2.5-32B", trust_remote_code=True)
# then generate short labels from snippets (or set AUTOINTERP_BACKEND in the notebook)
```

Prefer **Instruct** weights if that prefix exists on S3 (`Qwen2.5-32B-Instruct`).

---

## 8. Run the notebook

1. Open `notebooks/Full_mechanistic_interp.ipynb`.
2. Restart kernel after install.
3. Run Parts **1 → 6** (baseline, SAE, JLens last-7 layers, patching with option-letter margin, common example).
4. Optional: Qwen autointerp pass.

---

## Ops checklist

| Step | Done when |
|------|-----------|
| Fresh wheel on GitHub | `uv build` + push `dist/*.whl` **before** deploying / before pod `git pull` |
| Both prefixes on S3 | `s5cmd ls` shows `gemma-2-9b/` and `Qwen2.5-32B/` |
| Pod pull | `/models/gemma-2-9b/config.json` and `/models/Qwen2.5-32B/config.json` |
| Repo on pod | `git clone` / `git pull`; `ls dist/*.whl` shows the pushed version |
| Subject setup | `MODEL_NAME = "/models/gemma-2-9b"` |
| Notebook | First cell installs wheel; Parts 1–6 run without Hub downloads |
| Autointerp | Qwen loaded separately; labels in `autointerp_top_features.json` |

**Idle tip:** **Stop** the GPU pod when idle; keep the network volume for `/workspace`. Re-pull `/models` from S3 after terminate if root disk was wiped.
