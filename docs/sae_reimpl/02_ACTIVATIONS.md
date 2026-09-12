# 02 — Activation Extraction

Sparse autoencoders train on **cached LM activations**, not on tokens directly. This doc specifies how to extract them exactly as this repo does.

---

## 1. Pipeline overview

```
text corpus  →  tokenize  →  LM forward with hooks  →  flatten tokens  →  save .pt chunks
                                                          [N, d]
```

Then SAE training only reads the `.pt` files.

---

## 2. Models and dimensions

| Model | Residual `d_model` | MLP `d_mlp` |
|-------|--------------------|-------------|
| `EleutherAI/pythia-70m-deduped` | 512 | 2048 |
| `EleutherAI/pythia-410m-deduped` | 1024 | 4096 |
| `gpt2` | 768 | 3072 |

Use TransformerLens `HookedTransformer.from_pretrained(model_name)`.

Paper focus: **residual stream** of Pythia-70M / 410M.

---

## 3. Hook names (TransformerLens)

Repo helper: `activation_dataset.make_tensor_name`.

| `layer_loc` | Hook name | Dim |
|-------------|-----------|-----|
| `residual` | `blocks.{L}.hook_resid_post` | `d_model` |
| `mlp` | `blocks.{L}.mlp.hook_post` | `d_mlp` |
| `mlpout` | `blocks.{L}.hook_mlp_out` | `d_model` |
| `attn_concat` | `blocks.{L}.attn.hook_z` then flatten heads | `n_heads * d_head` |

**Default for paper replication:** `layer_loc="residual"`.

```69:81:activation_dataset.py
def make_tensor_name(layer: int, layer_loc: str, model_name: str) -> str:
    ...
    if layer_loc == "residual":
        if check_transformerlens_model(model_name):
            tensor_name = f"blocks.{layer}.hook_resid_post"
```

---

## 4. Tokenization / context

Repo defaults:

| Setting | Value |
|---------|-------|
| Max sequence length | `256` (`MAX_SENTENCE_LEN`) |
| Model batch size | `4` (`MODEL_BATCH_SIZE`) |
| Token packing | GPT-style concat with EOS separators (`chunk_and_tokenize`) |

Each forward yields activations of shape:

```text
[batch, seq, d]  →  rearrange to  [batch * seq, d]
```

Every token position is one training example for the SAE.

---

## 5. Exact extraction snippet (TransformerLens)

```python
from einops import rearrange
from transformer_lens import HookedTransformer

model = HookedTransformer.from_pretrained("EleutherAI/pythia-70m-deduped", device=device)
layer = 1
hook = f"blocks.{layer}.hook_resid_post"

tokens = batch["input_ids"].to(device)          # [B, S]
_, cache = model.run_with_cache(tokens, stop_at_layer=layer + 1)
acts = cache[hook].to(torch.float16)            # [B, S, d]
acts = rearrange(acts, "b s n -> (b s) n")      # [B*S, d]
```

Repo equivalent path: `make_activation_dataset_tl` in `activation_dataset.py`.

---

## 6. Caching format

### On disk

```
activation_data/
  layer_1/
    0.pt
    1.pt
    ...
```

Each `k.pt` is a `torch.Tensor` of shape `[N_k, d]` (often float16).

### Chunk size

Configured as `chunk_size_gb` (default `2.0`). Number of rows per chunk ≈

\[
N \approx \frac{\text{chunk\_bytes}}{d \cdot \text{bytes\_per\_element}}
\]

For float16 residual `d=512`, 2 GB ≈ millions of token activations.

### Save helper

```499:503:activation_dataset.py
def save_activation_chunk(dataset, n_saved_chunks, dataset_folder):
    dataset_t = torch.cat(dataset, dim=0).to("cpu")
    os.makedirs(dataset_folder, exist_ok=True)
    with open(dataset_folder + "/" + str(n_saved_chunks) + ".pt", "wb") as f:
        torch.save(dataset_t, f)
```

---

## 7. Datasets

| Name | When to use |
|------|-------------|
| `NeelNanda/pile-10k` | Smoke tests / development |
| `EleutherAI/pile` | Paper-scale (large download) |
| `openwebtext` | Autointerp text fragments |

Paper Appendix B: train on Pile activations, ~5–50M vectors.

CLI already in repo:

```powershell
python generate_test_data.py `
  --model="EleutherAI/pythia-70m-deduped" `
  --dataset="NeelNanda/pile-10k" `
  --layers 1 `
  --location="residual" `
  --n_chunks=1 `
  --chunk_size_gb=0.01 `
  --device="cpu"
```

---

## 8. Optional centering

Some experiments subtract a mean activation vector before SAE training (`center_dataset=True`). Paper residual results are typically **without** fancy whitening; if you center:

- compute mean on training activations
- save `means.pt`
- apply the same mean at eval time

Tied SAE also supports affine centering buffers; leave them identity unless you intentionally enable centering.

---

## 9. Acceptance checks

After extraction:

```python
x = torch.load("activation_data/layer_1/0.pt")
assert x.ndim == 2
assert x.shape[1] == 512  # pythia-70m residual
assert torch.isfinite(x.float()).all()
print(x.shape, x.dtype)
```

---

## 10. Porting to another project

Minimum API your project needs:

```python
def extract_activations(
    model_name: str,
    dataset_name: str,
    layer: int,
    layer_loc: str = "residual",
    n_tokens_target: int = 1_000_000,
    device: str = "cuda",
) -> torch.Tensor:
    """Return float32 tensor [N, d] of hooked activations."""
    ...
```

You can keep TransformerLens or re-hook HuggingFace modules yourself; **hook location and flatten convention must match**.

Next: [`03_TRAINING.md`](03_TRAINING.md).
