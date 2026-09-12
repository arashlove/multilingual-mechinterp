# Phase 2 — Core Patch Loop

Goal: during a **target** forward pass, overwrite selected activations with values from Phase 1, then read next-token probabilities.

---

## 1. The intervention primitive

At a chosen site `(layer L, batch b, position t)`:

```text
h_target[b, t, :]  ←  h_source[b, t_src, :]     # or mean source vector
```

All higher layers then compute **as if** that state had been produced by the target prompt’s earlier computation. Residual connections mean later layers still see the patched vector unless you patch them too.

### Why windows matter

Patching a **single** residual at layer \(L\) lets layers \(> L\) overwrite / transform it. For small modules (attn/MLP) the effect is often tiny unless you restore a **window**:

```text
for L in range(start, start + k):
    patch layer L
```

This repo’s `object_lens` defaults to `num_patches = n_layers` (patch from `start` through the top).

---

## 2. Same-layer object lens (this repo’s workhorse)

Exact behavior of `object_lens` in `src/interventions.py`:

```python
@th.no_grad()
def object_lens(
    nn_model,
    target_prompts,          # list[str]
    idx,                     # token position to overwrite (int, usually negative)
    source_prompts=None,
    hiddens=None,            # optional precomputed [n_layers, B, d]
    steering_vectors=None,   # optional add to hiddens
    num_patches=-1,          # -1 → patch through last layer
    scan=True,
    remote=False,
):
    if isinstance(target_prompts, str):
        target_prompts = [target_prompts]
    n_layers = get_num_layers(nn_model)
    if num_patches == -1:
        num_patches = n_layers

    if hiddens is None:
        hiddens = collect_activations(nn_model, source_prompts, remote=remote)
        # hiddens: [n_layers, B, d]  — last-token by default

    if steering_vectors is not None:
        hiddens = [h + s for h, s in zip(hiddens, steering_vectors)]

    probs_l = []
    for start in range(n_layers):
        with nn_model.trace(target_prompts, scan=(start == 0 and scan), remote=remote):
            for L in range(start, min(start + num_patches, n_layers)):
                # CRITICAL: same layer index on source and target
                get_layer_output(nn_model, L)[:, idx] = hiddens[L]
            probs_l.append(get_next_token_probs(nn_model).cpu().save())

    # [B, n_layers, vocab]
    return (
        th.cat([p.value for p in probs_l], dim=0)
        .reshape(n_layers, len(target_prompts), -1)
        .transpose(0, 1)
    )
```

### What the nested loops mean

| Outer `start` | Inner patch set | Interpretation |
|---------------|-----------------|----------------|
| `0` | layers `0 … min(k,L)-1` | Early injection; later layers free |
| `ℓ` | layers `ℓ … ℓ+k-1` | “Information must enter at/after ℓ” |
| last | only final layer(s) | Late readout / unembed path |

Plotting \(P(y)\) vs `start` produces the layer-wise curves in the paper.

---

## 3. Portable PyTorch hooks

```python
from contextlib import contextmanager
import torch as th

@contextmanager
def patch_residuals(
    model,
    patch_map: dict[int, th.Tensor],
    positions: th.Tensor,
):
    """
    patch_map: {layer_idx: Tensor[B, d]} values to write
    positions: Tensor[B] token indices on the TARGET sequence
    """
    handles = []

    def make_hook(layer_idx, values):
        def hook(_module, _inp, out):
            is_tuple = isinstance(out, tuple)
            h = out[0] if is_tuple else out
            h = h.clone()  # avoid in-place on a view shared with autograd/cache
            b = th.arange(h.size(0), device=h.device)
            h[b, positions] = values.to(h.device, dtype=h.dtype)
            return (h, *out[1:]) if is_tuple else h
        return hook

    for L, values in patch_map.items():
        handles.append(
            model.model.layers[L].register_forward_hook(make_hook(L, values))
        )
    try:
        yield
    finally:
        for h in handles:
            h.remove()


@th.no_grad()
def patch_lens_hf(
    model,
    tokenizer,
    target_prompts: list[str],
    source_acts: th.Tensor,   # [n_layers, B, d]
    target_pos: int,          # e.g. object index (-k)
    window: int | None = None,
):
    n_layers = source_acts.size(0)
    if window is None:
        window = n_layers

    tok = tokenizer(target_prompts, return_tensors="pt", padding=True).to(model.device)
    last = tok.attention_mask.flip(1).cumsum(1).bool().int().sum(1)
    positions = last + target_pos if target_pos < 0 else th.full_like(last, target_pos)

    all_probs = []
    for start in range(n_layers):
        patch_map = {
            L: source_acts[L]
            for L in range(start, min(start + window, n_layers))
        }
        with patch_residuals(model, patch_map, positions):
            logits = model(**tok).logits  # [B, T, V]
        # next-token = last real token of each sequence
        gather_pos = last - 1
        b = th.arange(logits.size(0), device=logits.device)
        probs = logits[b, gather_pos].softmax(-1).cpu()
        all_probs.append(probs)

    return th.stack(all_probs, dim=1)  # [B, n_layers, V]
```

---

## 4. Single-site vs multi-site

### Single site (classic causal trace cell)

```python
patch_map = {L: source_acts[L]}  # only one layer
# optionally also fix token t only — positions already select t
```

### Multi-token (rare; e.g. whole subject span)

```python
# values: [B, span_len, d]
h[b, span_start:span_end] = values
```

Keep span lengths equal across batch or loop per example.

### Cross-layer patch (advanced)

Source layer \(L_s\) written into target layer \(L_t\):

```python
get_layer_output(model, L_t)[:, idx] = hiddens[L_s]
```

Only do this when you have a reason (e.g. Patchscopes-style “read at L_s, write into fixed target layer”). Default object patching is **same-index**.

---

## 5. Reading the metric from the patched run

Always evaluate at the **prediction position** (usually last real token), not the patched object position (unless they coincide).

```python
def get_next_token_probs_from_logits(logits, attention_mask):
    last = attention_mask.flip(1).cumsum(1).bool().int().sum(1) - 1
    b = th.arange(logits.size(0), device=logits.device)
    return logits[b, last].softmax(-1)
```

For a set of acceptable answer token ids `y_ids` (prefix variants, space-prefixed, etc.):

```python
def answer_prob(probs, y_ids: list[int]) -> th.Tensor:
    # probs: [B, V] or [B, n_layers, V]
    return probs[..., y_ids].sum(dim=-1)
```

This matches `Prompt.get_target_probs` in `src/prompt_tools.py`.

---

## 6. Baselines you should always compute

| Run | Purpose |
|-----|---------|
| Target, no patch | \(P(y_\text{target})\) — task works at all |
| Source, no patch | \(P(y_\text{source})\) — source actually encodes the concept |
| Target + patch | Intervention effect |
| Target + random / wrong-concept patch | Control for “any vector changes logits” |

Causal effect of interest for object patching:

\[
\Delta = P(y_\text{source} \mid \text{patched target}) - P(y_\text{source} \mid \text{unpatched target})
\]

---

## 7. Performance tips

1. **`scan=True` only on first nnsight trace** — graph discovery is expensive; subsequent layer sweeps reuse it (`scan=False`).
2. **Batch targets**; keep source cache aligned with batch order.
3. **Don’t retain the full vocab tensor** if you only need a few token ids — gather those columns inside the hook context and `.save()` only them.
4. **Clone before write** in PyTorch hooks — silent corruption otherwise.
5. **fp16/bf16**: cast patched vectors to the activation dtype.

Next: [03_object_patching_recipe.md](03_object_patching_recipe.md)
