# Phase 1 — Collecting Activations

Goal: from a **source** forward pass, extract tensors you will later write into a **target** forward pass.

---

## 1. What to cache

For residual-stream patching (default), cache the **layer block output** — the residual after attn+MLP for that layer:

```text
h^(\ell) = LayerNorm / block output at layer ℓ   shape [batch, seq, d_model]
```

Optional alternatives (store with the same interface):

| Module | Typical path (HF Llama-like) | When |
|--------|------------------------------|------|
| Residual out | `model.layers[ℓ].output[0]` | Default |
| Attention out | `model.layers[ℓ].self_attn.output[0]` | Attention circuit studies |
| Attn input hidden | `self_attn` kwargsarg `hidden_states` | Object-attention patching |
| MLP out | `model.layers[ℓ].mlp` output | Fact-storage studies |

Always record: **`(layer, position_index, batch_index) → vector[d]`**.

---

## 2. Padding-safe position index

Most bugs live here. With **left or right padding**, “last token” is **not** always `seq_len - 1`.

### Last non-pad token (matches this repo)

```python
import torch as th

def last_token_index(attention_mask: th.Tensor) -> th.Tensor:
    """
    attention_mask: [B, T] with 1 = real token, 0 = pad
    returns: [B] index of last real token
    Works for both left and right padding.
    """
    # flip → cumsum of 1s from the right → count of real tokens
    return attention_mask.flip(1).cumsum(1).bool().int().sum(1) - 1
```

### Relative negative indices

If you want “2nd-to-last real token”:

```python
def resolve_index(attention_mask: th.Tensor, idx: int) -> th.Tensor:
    last = last_token_index(attention_mask) + 1  # length of real prefix/span
    if idx >= 0:
        raise ValueError("Prefer negative idx with padding; absolute pos is ambiguous")
    return last + idx  # e.g. idx=-1 → last real token
```

### Object / subject span end

For prompts like:

```text
Français: "chien" - English: "
```

the **object** to patch is the last token of `"chien"` (inside the quotes), **not** the final `"` of the prompt. See `get_obj_id` in `src/prompt_tools.py`:

```python
def get_obj_id(sample_prompt: str, tokenizer) -> int:
    """
    Prompt format: ...'\"object\" - X: \"'
    Return negative index of the last token of `object`.
    """
    split = sample_prompt.split('"')
    start = '"'.join(split[:-2])
    end = '"' + '"'.join(split[-2:])
    tok_start = tokenizer.encode(start, add_special_tokens=False)
    tok_end = tokenizer.encode(end, add_special_tokens=False)
    full = tokenizer.encode(sample_prompt, add_special_tokens=False)
    if tok_start + tok_end != full:
        raise ValueError("Tokenizer split does not recompose; fix prompt formatting")
    return -len(tok_end) - 1
```

**Rule:** compute the index once on an unpadded single example, verify by decoding `ids[idx]`, then reuse that **relative** index for the batch (same template ⇒ same relative offset from the end).

---

## 3. Reference implementation (nnsight, as in this repo)

```python
# Adapted from src/nnsight_utils.py — collect_activations

@th.no_grad()
def collect_activations(
    nn_model,
    prompts: list[str],
    layers=None,
    get_activations=None,   # default: layer residual output
    idx=None,               # None → last real token; negative → from end
    remote=False,
):
    if get_activations is None:
        get_activations = get_layer_output  # model.model.layers[L].output[0]

    tok = nn_model.tokenizer(prompts, return_tensors="pt", padding=True)
    last = tok.attention_mask.flip(1).cumsum(1).bool().int().sum(1)
    if idx is None:
        pos = last.sub(1)
    elif idx < 0:
        pos = last + idx
    else:
        raise ValueError("positive index unsupported with left padding")

    if layers is None:
        layers = range(get_num_layers(nn_model))

    with nn_model.trace(prompts, remote=remote):
        acts = [
            get_activations(nn_model, layer)[
                th.arange(len(tok.input_ids)), pos
            ].cpu().save()
            for layer in layers
        ]
    return th.stack([a.value for a in acts])  # [n_layers, B, d]
```

**Return convention used throughout this codebase:**  
`acts[layer]` has shape `[batch, d]` — already sliced to one position.

---

## 4. Pure PyTorch hook version (portable)

```python
from contextlib import contextmanager
from typing import Callable

@contextmanager
def capture_residual(model, layers: list[int], positions: th.Tensor):
    """
    Yields a dict layer -> Tensor[B, d] filled during forward.
    `positions`: [B] token indices to capture.
    Assumes HF CausalLM: model.model.layers[i]
    """
    cache = {}
    handles = []

    def make_hook(layer_idx):
        def hook(_module, _inp, out):
            h = out[0] if isinstance(out, tuple) else out  # [B, T, d]
            b = th.arange(h.size(0), device=h.device)
            cache[layer_idx] = h[b, positions].detach()
            return out
        return hook

    for L in layers:
        handles.append(model.model.layers[L].register_forward_hook(make_hook(L)))
    try:
        yield cache
    finally:
        for h in handles:
            h.remove()


@th.no_grad()
def collect_activations_hf(model, tokenizer, prompts, layers=None, pos_idx=-1):
    tok = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
    if layers is None:
        layers = list(range(len(model.model.layers)))
    last = tok.attention_mask.flip(1).cumsum(1).bool().int().sum(1)
    positions = last + pos_idx if pos_idx < 0 else last - 1

    with capture_residual(model, layers, positions) as cache:
        model(**tok)

    return th.stack([cache[L].cpu() for L in layers])  # [n_layers, B, d]
```

---

## 5. Mean source concept vector

For object patching you often want a **concept** vector, not one example:

```python
def mean_over_sources(acts: th.Tensor, n_sources_per_target: int) -> th.Tensor:
    """
    acts: [n_layers, B_flat, d] where B_flat = n_targets * n_sources
    returns: [n_layers, n_targets, d]
    """
    n_layers, B, d = acts.shape
    assert B % n_sources_per_target == 0
    n_targets = B // n_sources_per_target
    return acts.view(n_layers, n_targets, n_sources_per_target, d).mean(dim=2)
```

This is what `object_patching` in `notebooks/obj_patch_translation.ipynb` does before calling `object_lens`.

---

## 6. Checklist before leaving Phase 1

1. Decode `tokenizer.decode(input_ids[b, pos[b]])` — is it the intended token?
2. Shapes: `[n_layers, batch, d]` with `d == config.hidden_size`.
3. Same tokenizer flags as inference (`add_prefix_space=False` matters for Llama).
4. `torch.no_grad()` — never build a graph for patching experiments.
5. Device: keep cache on CPU if large; move slice to GPU only when writing the hook.

Next: [02_core_patch_loop.md](02_core_patch_loop.md)
