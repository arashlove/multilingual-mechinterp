# 03 — `JacobianLens`: Transport, Apply, Persist

**File to mirror:** [`jlens/lens.py`](../../jlens/lens.py)

---

## Data model

```python
class JacobianLens:
    """
    jacobians:     {layer_index: Tensor[d_model, d_model]}  # float
    source_layers: sorted list of fitted layer indices
    n_prompts:     number of prompts averaged
    d_model:       residual-stream width
    """

    def __init__(
        self,
        jacobians: dict[int, torch.Tensor],
        *,
        n_prompts: int,
        d_model: int,
    ) -> None:
        self.jacobians = {layer: J.float() for layer, J in jacobians.items()}
        self.source_layers = sorted(self.jacobians)
        self.n_prompts = n_prompts
        self.d_model = d_model
```

---

## Transport — the only math at apply time

\(J_l\) is stored with rows = \(\partial\text{out\_dim}/\partial\text{in}\).
For a row-vector residual `h` of shape `[..., d_model]`:

```
J @ h  ≡  h @ J.T
```

```python
def transport(self, residual: torch.Tensor, layer: int) -> torch.Tensor:
    J_bar = self.jacobians[layer].to(residual.device)
    return residual @ J_bar.T
```

Do **not** use `J @ residual` unless you treat residuals as column vectors and
reshape accordingly.

---

## `apply`

```python
@torch.no_grad()
def apply(
    self,
    model: LensModel,
    prompt: str,
    *,
    layers: Sequence[int] | None = None,
    positions: Sequence[int] | None = None,
    max_seq_len: int = 512,
    use_jacobian: bool = True,
) -> tuple[dict[int, torch.Tensor], torch.Tensor, torch.Tensor]:
    """
    Returns:
        lens_logits:  {layer: [n_positions, vocab]} on CPU float
        model_logits: [n_positions, vocab] final-layer unembed (no J)
        input_ids:    encoded prompt
    """
```

### Pipeline

```python
if layers is None:
    layers = self.source_layers
# validate layers ⊆ [0, n_layers) and (if use_jacobian) ⊆ source_layers

final_layer = model.n_layers - 1
record_at = sorted(set(layers) | {final_layer})

input_ids = model.encode(prompt, max_length=max_seq_len)
with ActivationRecorder(model.layers, at=record_at) as recorder:
    model.forward(input_ids)
    activations = {i: recorder.activations[i].detach() for i in record_at}

def select(layer: int) -> torch.Tensor:
    full = activations[layer][0]  # [seq_len, d_model]
    return (full if positions is None else full[list(positions)]).float()

lens_logits: dict[int, torch.Tensor] = {}
for layer in layers:
    residual = select(layer)
    if use_jacobian:
        residual = self.transport(residual, layer)
    lens_logits[layer] = model.unembed(residual).float().cpu()

model_logits = model.unembed(select(final_layer)).float().cpu()
return lens_logits, model_logits, input_ids
```

### Notes

- `positions` uses Python indexing (negatives from end). `None` → all positions.
- `use_jacobian=False` → vanilla **logit lens** baseline (identity transport).
- Default `max_seq_len` for apply is **512** (fit uses **128**).
- Always return CPU float logits for portability.

---

## Persistence

### Save format (`lens.pt`)

```python
def save(self, path: str, *, dtype: torch.dtype = torch.float16) -> None:
    torch.save(
        {
            "J": {layer: J.to(dtype) for layer, J in self.jacobians.items()},
            "n_prompts": self.n_prompts,
            "source_layers": self.source_layers,
            "d_model": self.d_model,
        },
        path,
    )
```

fp16 by default: halves size; entries are O(1), so range is fine; fp16 mantissa
beats bf16 for this use.

### Load

```python
@classmethod
def load(cls, path: str) -> JacobianLens:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if "J" not in checkpoint:
        raise ValueError(...)  # likely a fit() running checkpoint, not a lens
    return cls(
        jacobians=checkpoint["J"],
        n_prompts=checkpoint["n_prompts"],
        d_model=checkpoint["d_model"],
    )
```

Distinguish from fit checkpoints (`jacobian_sum` / `n_done` / `next_idx`).

### Hub / directory load

```python
@classmethod
def from_pretrained(
    cls,
    name_or_path: str,
    *,
    filename: str = "lens.pt",
    revision: str | None = None,
) -> JacobianLens:
    if os.path.isfile(name_or_path):
        return cls.load(name_or_path)
    if not os.path.isdir(name_or_path):
        from huggingface_hub import snapshot_download
        name_or_path = snapshot_download(
            name_or_path, allow_patterns=[filename], revision=revision
        )
    return cls.load(os.path.join(name_or_path, filename))
```

One Hub repo can host many model lenses via different `filename=` paths.

---

## `merge` — sharded fits

```python
@classmethod
def merge(cls, lenses: Sequence[JacobianLens]) -> JacobianLens:
    # Require identical source_layers and d_model
    n_total = sum(lens.n_prompts for lens in lenses)
    merged = {}
    for layer in first.source_layers:
        weighted_sum = sum(
            lens.jacobians[layer] * lens.n_prompts for lens in lenses
        )
        merged[layer] = weighted_sum / n_total
    return cls(jacobians=merged, n_prompts=n_total, d_model=first.d_model)
```

---

## Public package exports

From [`jlens/__init__.py`](../../jlens/__init__.py):

```python
__all__ = [
    "ActivationRecorder",
    "HFLensModel",
    "JacobianLens",
    "Layout",
    "LensModel",
    "configure_logging",
    "fit",
    "from_hf",
    "jacobian_for_prompt",
]
```

Vis / examples are importable but not re-exported (`jlens.vis`, `jlens.examples`).

---

## Acceptance criteria for this stage

- [ ] `transport` uses `@ J.T`, not `@ J`
- [ ] `apply` returns lens logits + model logits + input_ids with matching position dims
- [ ] `use_jacobian=False` matches logit-lens baseline
- [ ] Round-trip save (fp16) → load → apply preserves shapes
- [ ] `merge` is n_prompts-weighted; empty / mismatched inputs raise
