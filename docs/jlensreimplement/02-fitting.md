# 02 — Fitting the Jacobian

**File to mirror:** [`jlens/fitting.py`](../../jlens/fitting.py)

This is the mathematical core. Get the estimator wrong and the lens will still
produce ranked tokens — but they will not match the paper.

---

## Formula

```
lens_l(h) = unembed( J_l @ h )
J_l       = E[ ∂h_final / ∂h_l ]
```

### Per-prompt estimator (paper reduction)

For each output dimension \(d\):

1. Inject a **one-hot cotangent** at dimension \(d\) on **every valid target
   position at once**.
2. Backprop to get \(\partial h_{\text{final}} / \partial h_l\).
3. At source position \(p\), the gradient is
   \(\sum_{p' \ge p} \partial h_{\text{final}}[p'] / \partial h_l[p]\)
   (sum over later targets via the single multi-position cotangent).
4. **Mean over valid source positions** \(p\).

That mean is one row of \(J_l\) (shape `[d_model, d_model]`, row =
\(\partial\text{target\_dim}/\partial\text{source\_vector}\)).

A strict per-position estimator \(\partial h_{\text{final}}[p]/\partial h_l[p]\)
averaged over \(p\) also works as a lens; this repo implements the summed-
targets reduction above.

### Cost

- 1 forward + \(\lceil d_{\text{model}} / \text{dim\_batch}\rceil\) backwards per prompt
- Memory scales with `dim_batch` (prompt is replicated that many times)

---

## Constants and position mask

```python
SKIP_FIRST_N_POSITIONS = 16  # attention sinks; atypical residual stats
```

```python
def valid_position_mask(
    seq_len: int, *, skip_first: int = SKIP_FIRST_N_POSITIONS
) -> torch.Tensor:
    if skip_first < 0:
        raise ValueError(f"skip_first must be >= 0, got {skip_first}")
    mask = torch.zeros(seq_len, dtype=torch.bool)
    mask[skip_first : seq_len - 1] = True  # exclude final token too
    if mask.sum() == 0:
        raise ValueError(
            f"prompt too short: seq_len={seq_len}, need > {skip_first + 1} tokens"
        )
    return mask
```

Valid indices: `[skip_first, seq_len - 2]` inclusive. Need `seq_len > skip_first + 1`.

---

## Layer index resolution

```python
def _check_layer_indices(
    source_layers: Sequence[int] | None, target_layer: int | None, n_layers: int
) -> tuple[list[int], int]:
    target = n_layers - 1 if target_layer is None else target_layer
    if target < 0:
        target += n_layers
    # bounds-check target ∈ [0, n_layers)

    if source_layers is None:
        return list(range(target)), target  # every layer below target

    sources = sorted({l + n_layers if l < 0 else l for l in source_layers})
    # bounds-check; enforce sources[-1] < target
    return sources, target
```

Negative indices count from the end. Targeting the **penultimate** layer
sometimes yields a better-conditioned \(J_l\).

---

## `jacobian_for_prompt` — core algorithm

Signature:

```python
def jacobian_for_prompt(
    model: LensModel,
    prompt: str,
    source_layers: Sequence[int],
    *,
    target_layer: int | None = None,
    dim_batch: int = 8,
    max_seq_len: int = 128,
    skip_first: int = SKIP_FIRST_N_POSITIONS,
) -> tuple[dict[int, torch.Tensor], int, int]:
    """Returns (jacobians, seq_len, n_valid_positions).
    Each jacobian is [d_model, d_model] fp32 on CPU.
    """
```

### Algorithm (implement exactly)

```python
n_layers, d_model = model.n_layers, model.d_model
source_layers, target_layer = _check_layer_indices(
    source_layers, target_layer, n_layers
)

input_ids = model.encode(prompt, max_length=max_seq_len)  # [1, seq]
seq_len = input_ids.shape[1]
position_mask = valid_position_mask(seq_len, skip_first=skip_first)
n_valid_positions = int(position_mask.sum())

jacobians = {
    layer: torch.zeros(d_model, d_model, dtype=torch.float32)
    for layer in source_layers
}
n_passes = math.ceil(d_model / dim_batch)

with (
    ActivationRecorder(
        model.layers,
        at=[*source_layers, target_layer],
        start_graph_at=min(source_layers),
    ) as recorder,
    torch.enable_grad(),
):
    # One forward; retained graph reused for every backward
    replicated_ids = input_ids.expand(dim_batch, -1)
    model.forward(replicated_ids)

    target_activation = recorder.activations[target_layer]
    # shape: [dim_batch, seq_len, d_model]
    source_activations = [recorder.activations[layer] for layer in source_layers]

    valid_positions = position_mask.nonzero(as_tuple=True)[0].to(
        target_activation.device
    )
    batch_indices = torch.arange(dim_batch, device=target_activation.device)
    cotangent = torch.zeros_like(target_activation)

    for pass_idx, dim_start in enumerate(range(0, d_model, dim_batch)):
        n_dims_this_pass = min(dim_batch, d_model - dim_start)

        # One-hot at dim (dim_start + b) for batch element b,
        # at EVERY valid target position → rows dim_start.. of J_l
        cotangent.zero_()
        cotangent[
            batch_indices[:n_dims_this_pass, None],
            valid_positions[None, :],
            dim_start + batch_indices[:n_dims_this_pass, None],
        ] = 1.0

        grads = torch.autograd.grad(
            outputs=target_activation,
            inputs=source_activations,
            grad_outputs=cotangent,
            retain_graph=(pass_idx < n_passes - 1),
        )

        for layer, grad in zip(source_layers, grads, strict=True):
            # grad: [dim_batch, seq_len, d_model]
            positions_on_device = valid_positions.to(grad.device, non_blocking=True)
            rows = (
                grad[:n_dims_this_pass, positions_on_device, :].float().mean(dim=1)
            )  # [n_dims_this_pass, d_model]
            jacobians[layer][dim_start : dim_start + n_dims_this_pass, :] = (
                rows.cpu()
            )
        del grads

return jacobians, seq_len, n_valid_positions
```

### Cotangent indexing diagram

```
cotangent[batch_b, valid_pos_p, dim_(dim_start+b)] = 1

batch axis  ↔  which output dimension's row of J
pos axis    ↔  all valid targets lit at once (sum over p')
dim axis    ↔  one-hot output coordinate
```

After `autograd.grad`, mean over `valid_positions` collapses the position
axis into one row per batch element.

---

## `fit` — corpus average

```python
def fit(
    model: LensModel,
    prompts: Sequence[str],
    *,
    source_layers: Sequence[int] | None = None,
    target_layer: int | None = None,
    dim_batch: int = 8,
    max_seq_len: int = 128,
    skip_first: int = SKIP_FIRST_N_POSITIONS,
    checkpoint_path: str | None = None,
    checkpoint_every: int | None = 1,
    resume: bool = True,
) -> JacobianLens:
```

### Running state

| Field | Meaning |
|-------|---------|
| `jacobian_sum` | `{layer: Tensor[d,d]}` sum of per-prompt Js |
| `n_done` | Successful prompt count (denominator of mean) |
| `next_idx` | Next prompt list index to process |

**Why `next_idx` ≠ `n_done`:** too-short prompts are skipped. On resume you
must advance past them without double-counting successes.

### Loop body

```python
for prompt_idx, prompt in enumerate(prompts):
    if prompt_idx < next_idx:
        continue
    try:
        per_prompt_J, seq_len, n_valid = jacobian_for_prompt(...)
    except ValueError as exc:
        # too short, etc. — skip, advance next_idx, do NOT increment n_done
        next_idx = prompt_idx + 1
        continue

    for layer in source_layers:
        jacobian_sum[layer] += per_prompt_J[layer]
    n_done += 1
    next_idx = prompt_idx + 1
    # optional checkpoint every checkpoint_every prompts
```

Final:

```python
jacobian_mean = {layer: jacobian_sum[layer] / n_done for layer in source_layers}
return JacobianLens(jacobians=jacobian_mean, n_prompts=n_done, d_model=d_model)
```

### Atomic checkpoints

```python
def _atomic_save(obj: object, path: str) -> None:
    tmp_path = f"{path}.tmp.{os.getpid()}"
    torch.save(obj, tmp_path)
    os.replace(tmp_path, path)
```

Checkpoint payload:

```python
{
    "jacobian_sum": jacobian_sum,
    "n_done": n_done,
    "next_idx": next_idx,
    "source_layers": source_layers,
    "target_layer": target_layer,
    "skip_first": skip_first,
}
```

On resume, reject if `source_layers` / `target_layer` / `skip_first` disagree
with the current call (unless `resume=False`).

Load with `torch.load(..., map_location="cpu", weights_only=True)`.

### Diagnostics (optional but useful)

Per prompt, max over layers:

- `||J|| / sqrt(d_model)` — flags heavy-tailed outliers
- Relative shift of running mean — falls ~`1/n` once settled

### Parallel / sharded fitting

Run `fit()` on disjoint prompt slices → `JacobianLens.merge()`
(`n_prompts`-weighted mean). See [03-lens-apply.md](03-lens-apply.md).

---

## Knobs

| Knob | Default | Notes |
|------|---------|-------|
| `dim_batch` | 8 | ↑ memory, same FLOPs |
| `max_seq_len` | 128 | Fit truncation |
| `skip_first` | 16 | Attention sinks |
| `checkpoint_every` | 1 | Raise for large `d_model` (ckpt size ≈ `n_layers * d² * 4` bytes) |
| `source_layers` | all below target | Subset to save compute |
| `target_layer` | final | Try penultimate if ill-conditioned |

---

## Acceptance criteria for this stage

- [ ] Exact late-layer \(J\) matches closed form on `TinyDecoder` (see tests)
- [ ] Too-short prompts raise / skip cleanly
- [ ] Checkpoint resume does not double-count skipped prompts
- [ ] Returned matrices are fp32 CPU `[d_model, d_model]`
- [ ] `retain_graph=False` on the last backward pass
