# Object Patching — End-to-End Recipe

This is the precise procedure used in *Separating Tongue from Thought* / `notebooks/obj_patch_translation.ipynb`, distilled so you can re-implement it for any concept-transfer question.

---

## 1. Scientific claim under test

> Mid-layer residual states at the **object** token encode the **concept** largely independently of the surface language, and can be transplanted into another translation prompt to change which concept is expressed in the answer language.

---

## 2. Prompt construction

### Template

Few-shot translation, cut so the model must predict the translation of the final object:

```text
Français: "maison" - English: "house"
Français: "livre" - English: "book"
Français: "chien" - English: "
                 ▲
                 prediction starts here
```

The **object token(s)** to patch are the last subword(s) of `chien` (inside the quotes), **not** the final quotation mark.

Built by `translation_prompts(...)` / `prompts_from_df(...)` in `src/prompt_tools.py`.

### Source vs target pairs

| Role | Prompt content | Concept |
|------|----------------|---------|
| Source | e.g. `DE→EN` or `FR→JA` few-shot, **optionally truncated after the object** | Concept \(c_A\) (donor) |
| Target | e.g. `ZH→EN` full prediction prompt for a **different** concept \(c_B\) | Concept \(c_B\) (host) |

Critical pairing rule from the notebook:

```python
# Do NOT pair the same word_original — you want transfer of a *different* concept
if source_row["word_original"] == target_row["word_original"]:
    continue
```

Also filter token **collisions**: target answer tokens must not overlap latent/source answer tokens (`Prompt.has_no_collisions`), or your metric becomes ambiguous.

### Finding the object index

```python
idx = get_obj_id(target_prompts[0].prompt, tokenizer)  # negative int
# Verify once:
ids = tokenizer.encode(target_prompts[0].prompt, add_special_tokens=False)
print(tokenizer.decode([ids[idx]]))  # should be last piece of the object word
```

---

## 3. Algorithm (exact)

```text
Inputs:
  source_prompts[i][s]   # i = target index, s = source exemplar for concept c_A_i
  target_prompts[i]      # asks for concept c_B_i in answer language L_out
  idx                    # object token index on target (and source if aligned)
  num_patches            # window size; -1 = through top layer

For each batch of targets:
  1. Flatten source strings for the batch; collect residual acts
        acts: [n_layers, batch * n_sources, d]   # usually last token of truncated source
  2. Reshape → mean over sources
        hiddens: [n_layers, batch, d]
  3. For start_layer = 0 .. n_layers-1:
        Run target batch
        For L in [start_layer, start_layer + num_patches):
            residual[L][:, idx] ← hiddens[L]
        Record softmax at last token → probs[batch, start_layer, vocab]
  4. Metrics:
        P_source_lang  = sum probs over token ids for c_A in L_out
        P_target_lang  = sum probs over token ids for c_B in L_out
```

### Code skeleton matching the notebook

```python
def object_patching(nn_model, prompt_batch, scan, source_prompt_batch, only_first=False):
    """
    source_prompt_batch: np.ndarray [batch, n_sources] of strings
                         (sources truncated after object)
    prompt_batch:        list[str] target prompts
    """
    if only_first:
        source_prompt_batch = source_prompt_batch[:, :1]

    flat = source_prompt_batch.flatten().tolist()
    acts = collect_activations_batched(nn_model, flat, batch_size=len(prompt_batch))
    # acts: [n_layers, batch * n_sources, d]

    bsz, n_src = source_prompt_batch.shape[:2]
    hiddens = acts.view(get_num_layers(nn_model), bsz, n_src, -1).mean(2)

    return object_lens(
        nn_model,
        prompt_batch,
        idx=idx,
        hiddens=hiddens,
        num_patches=num_patches,
        scan=scan,
    )  # [batch, n_layers, vocab]
```

Then score with `run_prompts` which sums over `Prompt.target_tokens` and each `latent_tokens[lang]`.

---

## 4. Metrics and plots

For each layer start \(\ell\):

| Curve | Token set | Expected if claim is true |
|-------|-----------|---------------------------|
| Source concept in output lang | ids for \(c_A\) in \(L_\text{out}\) | **Rises** in middle layers when patched |
| Target concept in output lang | ids for \(c_B\) in \(L_\text{out}\) | **Falls** relative to baseline |
| Baseline target (no patch) | \(c_B\) | High if task is solvable |
| Baseline source (no patch) | \(c_A\) on source prompt | High |

Also compare **mean-over-sources** vs **only-first-source** (`only_first=True`) to check stability.

---

## 5. Minimal self-contained toy reimplementation

Use this if you are porting to a new repo without nnsight:

```python
"""
Toy object-patching: swap the concept vector at the object token.
Requires: transformers, torch
"""
import torch as th
from transformers import AutoModelForCausalLM, AutoTokenizer

@th.no_grad()
def run_object_patch(
    model,
    tokenizer,
    source_texts: list[str],
    target_texts: list[str],
    object_index: int,          # negative index into target tokens
    answer_token_ids: list[int],
    window: int | None = None,
):
    assert len(source_texts) == len(target_texts)
    device = model.device
    n_layers = len(model.model.layers)

    # --- Phase 1: source last-token residuals ---
    src = tokenizer(source_texts, return_tensors="pt", padding=True).to(device)
    src_last = src.attention_mask.flip(1).cumsum(1).bool().int().sum(1) - 1
    cache = {}

    def capture(L):
        def hook(_m, _i, out):
            h = out[0]
            b = th.arange(h.size(0), device=h.device)
            cache[L] = h[b, src_last].detach()
            return out
        return hook

    hs = [model.model.layers[L].register_forward_hook(capture(L)) for L in range(n_layers)]
    model(**src)
    for h in hs:
        h.remove()
    source_acts = th.stack([cache[L] for L in range(n_layers)])  # [L, B, d]

    # --- Phase 2: patch lens on targets ---
    tgt = tokenizer(target_texts, return_tensors="pt", padding=True).to(device)
    tgt_len = tgt.attention_mask.flip(1).cumsum(1).bool().int().sum(1)
    obj_pos = tgt_len + object_index
    pred_pos = tgt_len - 1
    if window is None:
        window = n_layers

    layer_probs = []
    for start in range(n_layers):
        patch_layers = list(range(start, min(start + window, n_layers)))

        def make_hook(L):
            def hook(_m, _i, out):
                h = out[0].clone()
                b = th.arange(h.size(0), device=h.device)
                h[b, obj_pos] = source_acts[L].to(h.dtype)
                return (h, *out[1:])
            return hook

        handles = [
            model.model.layers[L].register_forward_hook(make_hook(L))
            for L in patch_layers
        ]
        logits = model(**tgt).logits
        for h in handles:
            h.remove()
        b = th.arange(logits.size(0), device=device)
        probs = logits[b, pred_pos].softmax(-1)
        layer_probs.append(probs[:, answer_token_ids].sum(-1).cpu())

    return th.stack(layer_probs, dim=1)  # [B, n_layers]
```

Wire your own prompts + `object_index` + answer ids into this function; the rest is experiment design.

---

## 6. Implementation checklist

- [ ] Source and target share answer language \(L_\text{out}\) for the metric (or you knowingly measure cross-lingual leakage).
- [ ] Object index verified by decoding.
- [ ] Different concepts in source vs target.
- [ ] No token-id collision between metrics.
- [ ] Same layer index when writing `hiddens[L] → layer L`.
- [ ] Window size documented (`num_patches`).
- [ ] Baselines (unpatched target, unpatched source) logged.
- [ ] Batch order of sources aligned with targets after `mean(2)`.

---

## 7. Mapping onto this repository

| Step | File / symbol |
|------|----------------|
| Build prompts | `src/prompt_tools.py` → `translation_prompts`, `Prompt` |
| Object index | `get_obj_id` |
| Collect acts | `src/nnsight_utils.py` → `collect_activations` |
| Patch + layer sweep | `src/interventions.py` → `object_lens` |
| Batch scoring | `src/exp_tools.py` → `run_prompts` |
| Full experiment | `notebooks/obj_patch_translation.ipynb` |

Next: [04_variants.md](04_variants.md)
