# RunPod + S3 model workflow

How we store large LLM weights in **Amazon S3**, make the prefix readable without AWS credentials, pull them onto a **RunPod** GPU pod, and load them from `notebooks/colab_setup.py`.

## Why this path

- Hugging Face downloads from shared RunPod IPs often hit **HTTP 429**.
- `scp` of a 32B checkpoint (~60–80+ GB) over SSH is slow and fragile.
- **PC → S3 → pod** is resume-friendly and works well with a public (or signed) bucket.

Current example bucket / prefix:

```text
s3://arash-llm-models-790189127408-ap-southeast-2-an/Qwen2.5-32B/
```

Region: `ap-southeast-2`.

---

## 1. Download models on your PC

Use a drive with enough free space (e.g. `E:\models`).

```powershell
# Public / ungated
hf download Qwen/Qwen2.5-32B --local-dir E:\models\Qwen2.5-32B
hf download Qwen/Qwen2.5-7B --local-dir E:\models\Qwen2.5-7B
hf download deepseek-ai/DeepSeek-R1-Distill-Qwen-32B --local-dir E:\models\DeepSeek-R1-Distill-Qwen-32B

# Gated (Gemma): accept the license on Hugging Face, then:
$env:HF_TOKEN = "hf_xxxxxxxx"
hf download google/gemma-2-9b --local-dir E:\models\gemma-2-9b --token $env:HF_TOKEN
hf download google/gemma-2-27b --local-dir E:\models\gemma-2-27b --token $env:HF_TOKEN
```

Confirm the folder has `config.json` before uploading.

---

## 2. Create the S3 bucket and upload

On the PC (AWS CLI configured once with `aws configure`):

```powershell
# Example: create bucket in ap-southeast-2 (name must be globally unique)
aws s3 mb s3://arash-llm-models-790189127408-ap-southeast-2-an --region ap-southeast-2

# Upload (sync resumes if interrupted)
aws s3 sync "E:\models\Qwen2.5-32B" s3://arash-llm-models-790189127408-ap-southeast-2-an/Qwen2.5-32B/
```

Other models:

```powershell
aws s3 sync "E:\models\Qwen2.5-7B" s3://arash-llm-models-790189127408-ap-southeast-2-an/Qwen2.5-7B/
aws s3 sync "E:\models\DeepSeek-R1-Distill-Qwen-32B" s3://arash-llm-models-790189127408-ap-southeast-2-an/DeepSeek-R1-Distill-Qwen-32B/
aws s3 sync "E:\models\gemma-2-9b" s3://arash-llm-models-790189127408-ap-southeast-2-an/gemma-2-9b/
```

Prefer **`aws s3 sync`** over a single `cp` for large trees.

---

## 3. Make the bucket (or prefix) public for anonymous download

We use **public read** so the pod can pull with `s5cmd --no-sign-request` (no AWS keys on the pod).

### Security note

Anyone with the URL can download the weights. That is fine for **open** models (Qwen Apache-2.0, etc.). Do **not** put private data, secrets, or license-restricted weights you are not allowed to redistribute in a public prefix. Prefer a private bucket + IAM / presigned URLs if anything is sensitive.

### Typical console steps

1. Open **S3** → bucket `arash-llm-models-790189127408-ap-southeast-2-an`.
2. **Permissions** → turn **off** “Block all public access” (or allow public ACLs/policies as needed).
3. Attach a **bucket policy** that allows `s3:GetObject` (and optionally `s3:ListBucket`) for anonymous principals on the model prefixes you want shared.

Example policy (adjust ARNs / prefixes as needed):

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "PublicReadModelObjects",
      "Effect": "Allow",
      "Principal": "*",
      "Action": ["s3:GetObject"],
      "Resource": "arn:aws:s3:::arash-llm-models-790189127408-ap-southeast-2-an/*"
    },
    {
      "Sid": "PublicListBucket",
      "Effect": "Allow",
      "Principal": "*",
      "Action": ["s3:ListBucket"],
      "Resource": "arn:aws:s3:::arash-llm-models-790189127408-ap-southeast-2-an"
    }
  ]
}
```

4. Confirm anonymous access works from any machine:

```bash
s5cmd --no-sign-request ls s3://arash-llm-models-790189127408-ap-southeast-2-an/Qwen2.5-32B/
```

---

## 4. Deploy a RunPod GPU pod

### VRAM for Qwen2.5-32B

| Use | Rough need |
|-----|------------|
| bf16 / fp16 load | ~64 GB weights + overhead → **~80 GB** minimum |
| Mech interp (SAE / JLens / patching) | **96–140+ GB** preferred |
| 48 GB cards | Only with quantization — not ideal for full bf16 |

Attach a **network volume** at `/workspace` so models survive stop/restart.

Copy **SSH over TCP** from Connect, e.g.:

```bash
ssh root@<IP> -p <PORT> -i ~/.ssh/id_ed25519
```

Optional: JupyterLab in the browser is enough to run the notebook and a terminal (no Cursor required).

---

## 5. On the pod: download with `s5cmd` (public bucket)

In the **JupyterLab terminal** (or SSH):

```bash
# Install s5cmd if missing (pick one approach)
pip install -U s5cmd
# or download a release binary into PATH

mkdir -p /workspace/models/Qwen2.5-32B
s5cmd --no-sign-request cp \
  's3://arash-llm-models-790189127408-ap-southeast-2-an/Qwen2.5-32B/*' \
  /workspace/models/Qwen2.5-32B/
```

Verify:

```bash
ls /workspace/models/Qwen2.5-32B/config.json
du -sh /workspace/models/Qwen2.5-32B
nvidia-smi
```

Same pattern for other prefixes:

```bash
mkdir -p /workspace/models/Qwen2.5-7B
s5cmd --no-sign-request cp \
  's3://arash-llm-models-790189127408-ap-southeast-2-an/Qwen2.5-7B/*' \
  /workspace/models/Qwen2.5-7B/
```

---

## 6. Clone the repo and install

```bash
cd /workspace
git clone https://github.com/arashlove/multilingual-mechinterp.git
cd /workspace/multilingual-mechinterp
pip install -U pip
pip install dist/*.whl
# fallback: pip install -e .
```

---

## 7. Point `colab_setup` at the local path

In `notebooks/colab_setup.py`:

```python
MODEL_NAME = "/workspace/models/Qwen2.5-32B"
MODEL_TRUST_REMOTE_CODE = True
USE_TINY_OFFLINE = False
```

The folder must contain `config.json`. If Transformers says `Repo id must be in the form...`, the path is wrong or incomplete.

---

## 8. Run the notebook in JupyterLab

1. Open `/workspace/multilingual-mechinterp/notebooks/Full_mechanistic_interp.ipynb`.
2. Restart the kernel (after installing wheels).
3. Run cells from the top (setup / `colab_setup` first).

Smoke test:

```python
from multilingual_mechinterp.models import load_model
m = load_model("/workspace/models/Qwen2.5-32B", trust_remote_code=True)
print(m.n_layers, m.device)
```

---

## Ops checklist

| Step | Done when |
|------|-----------|
| Local HF download | `E:\models\...\config.json` exists |
| S3 upload | `aws s3 ls` / `s5cmd ls` shows objects |
| Public policy | `s5cmd --no-sign-request` works without keys |
| Volume mounted | `/workspace` persists across pod restarts |
| Pod pull | `/workspace/models/Qwen2.5-32B/config.json` exists |
| Notebook | Loads local path; no Hub download |

**Cost tip:** run the S3 pull on a cheap **CPU** pod with the same volume, then start the **GPU** pod to train / interpret.

**Idle tip:** **Stop** the GPU pod when not using it; do not delete the network volume unless you intend to wipe `/workspace`.
