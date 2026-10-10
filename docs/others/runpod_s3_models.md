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

## 6. Build the wheel and install it on the pod

Follow [runpod_setup.md](runpod_setup.md). Short form: on your PC, bump the version, `poetry build`, and push `dist/*.whl`. On the pod, `git pull`, then `pip install --upgrade dist/multilingual_mechinterp-*.whl`.

---

## 7. Point `runpod_setup` at **Gemma** (subject)

In `notebooks/runpod_setup.py`:

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

### Autointerp with Qwen 32B (§2.5)

In `Full_mechanistic_interp.ipynb` cell **2.5**:

```python
AUTOINTERP_BACKEND = "hf"                 # local Qwen labels
AUTOINTERP_MODEL = "/models/Qwen2.5-32B"   # uses *-Instruct if that folder exists
UNLOAD_SUBJECT_FOR_AUTOINTERP = False      # 140GB: keep Gemma loaded; set True if OOM
```

Flow: Gemma extracts max-activating snippets → Qwen proposes labels → Gemma returns to GPU for Parts 3–6.  
Gemma-only run: set `AUTOINTERP_BACKEND = "manual"` (skip Qwen download).  
Pull Qwen from S3 before that cell if you want labels.

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
| Fresh wheel on GitHub | `poetry build` + push `dist/*.whl` **before** deploying / before pod `git pull` |
| Both prefixes on S3 | `s5cmd ls` shows `gemma-2-9b/` and `Qwen2.5-32B/` |
| Pod pull | `/models/gemma-2-9b/config.json` and `/models/Qwen2.5-32B/config.json` |
| Repo on pod | `git clone` / `git pull`; `ls dist/*.whl` shows the pushed version |
| Subject setup | `MODEL_NAME = "/models/gemma-2-9b"` in `notebooks/runpod_setup.py` |
| Notebook | `pip install` the wheel from `dist/`, then run the notebook |
| Autointerp | Qwen loaded separately; labels in `autointerp_top_features.json` |

**Idle tip:** **Stop** the GPU pod when idle; keep the network volume for `/workspace`. Re-pull `/models` from S3 after terminate if root disk was wiped.
