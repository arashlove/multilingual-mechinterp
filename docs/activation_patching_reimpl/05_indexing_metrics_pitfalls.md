# Indexing, Metrics, and Pitfalls

Precision checklist for a correct re-implementation. Most failed patching experiments are indexing or metric bugs, not model bugs.

---

## 1. Token indexing rules

### Always prefer relative (negative) indices with padding

| Situation | Do |
|-----------|----|
| Batched left-padded inputs | Resolve via `attention_mask`, never hardcode `T-1` |
| Shared prompt template | Compute negative index once; reuse |
| Absolute index from an unpadded encode | Convert: `pos = real_length + neg_idx` |

### Verify by decoding

```python
ids = tokenizer.encode(prompt, add_special_tokens=False)
pos = get_obj_id(prompt, tokenizer)  # e.g. -4
print(repr(tokenizer.decode([ids[pos]])))
print(repr(tokenizer.decode(ids[pos - 2 : pos + 3])))  # context
```

### Chat templates / BOS

If the model adds BOS or a chat header only at runtime, compute indices on the **exact** string (or `input_ids`) the model sees inside the patched forward—not on a pre-template string.

### Patch position ≠ prediction position

| Position | Role |
|----------|------|
| Object / subject token | **Write** site |
| Last prompt token | **Read** logits / probs |

Mixing these up is the #1 silent failure mode.

---

## 2. Tokenizer footguns

1. **`add_prefix_space`**: Llama tokenizers often need `add_prefix_space=False` so `" dog"` and `"dog"` differ. This repo sets that in `load_model`.
2. **Leading space in answers**: Score both `house` and `Ġhouse` / `▁house` token ids; sum their probabilities.
3. **Multi-token answers**: Either score only the **first** token, or constrain generation; do not pretend a multi-token word is one id.
4. **Unicode / CJK**: First-byte tokens and byte fallbacks differ across Llama2/3/Qwen—see `process_tokens` / `unicode_prefixes` in `prompt_tools.py`.

```python
# Robust answer set
y_ids = process_tokens_with_tokenization(["house", "House"], tokenizer)
score = probs[..., y_ids].sum(-1)
```

---

## 3. Metric definitions (be explicit)

### Probability mass on answer set

\[
P(\mathcal{Y}) = \sum_{y \in \mathcal{Y}} \mathrm{softmax}(z)_y
\]

### Logit difference

\[
\mathrm{LD} = z_{y^+} - z_{y^-}
\]

More stable than raw probs when both are tiny.

### Indirect effect (causal tracing)

\[
\mathrm{IE} = P(y \mid \text{corrupt + restore}) - P(y \mid \text{corrupt})
\]

### Normalized recovery

\[
\frac{P_\text{patched} - P_\text{corrupt}}{P_\text{clean} - P_\text{corrupt}}
\]

Clips noise when clean/corrupt gap varies by example.

### Report all of

- Unpatched target \(P(y_\text{target})\)
- Unpatched source \(P(y_\text{source})\)
- Patched \(P(y_\text{source})\), \(P(y_\text{target})\)
- Optional control patch (shuffled concepts)

---

## 4. Causal / methodological pitfalls

| Pitfall | Why it hurts | Fix |
|---------|--------------|-----|
| Same concept in source & target | Trivial copy, not transfer | Pair different `word_original` |
| Token collision in metrics | Inflated overlap | `has_no_collisions` |
| Patching after information already moved | Late layers may be “readout only” | Sweep all start layers |
| Window = 1 on MLP/attn | Effect ≈ 0 | Use window ≥ 5–10 |
| Interpreting correlation as necessity | Patch success ≠ only pathway | Pair with ablation / path-specific freeze |
| Mean vector from mixed concepts | Washes signal | Mean only within one concept |
| Batch misalignment after flatten/mean | Wrong vector → wrong example | Assert shapes; keep indices |
| In-place hook without `.clone()` | Corrupts cache / gradients | Always clone before write |
| Evaluating at patched index | Wrong softmax position | Use last real token |
| Mixing dtypes | Silent cast / overflow | `.to(dtype=h.dtype)` |

---

## 5. Shape contract (recommended)

Adopt one convention project-wide:

```text
source_acts:  Float[n_layers, batch, d]
probs:        Float[batch, n_layers, vocab]   # after layer sweep
answer_ids:   Long[n_answers]                 # per example or shared
scores:       Float[batch, n_layers]          # probs summed over answer_ids
```

Layer dimension is always axis that encodes **start layer of the window**.

---

## 6. Debug protocol (15 minutes)

1. **Identity patch**: source = target, same concept → \(P(y)\) should stay ≈ baseline (or improve slightly). If it collapses, hooks are broken.
2. **Shuffle patch**: source concept random → source-answer prob should **not** systematically rise.
3. **Single layer print**: decode top-5 tokens after patch at mid layer; should look like source concept language.
4. **Index dump**: print patched token string for 3 batch items.
5. **No-op hook**: register hook that returns `out` unchanged; probs must match vanilla forward bit-for-bit (fp16 aside).

```python
@th.no_grad()
def assert_hook_noop(model, batch):
    base = model(**batch).logits
    handle = model.model.layers[0].register_forward_hook(lambda m, i, o: o)
    probed = model(**batch).logits
    handle.remove()
    assert th.allclose(base, probed, atol=0, rtol=0)
```

---

## 7. Minimal test vector

Before scaling to multilingual datasets, run one English→English control:

```text
Source: "The capital of France is"
Target: "The capital of Germany is"
Patch last-token state of "France" path into "Germany" subject end
Expect: P("Paris") ↑ on target, P("Berlin") ↓ in mid layers
```

If this fails, fix infrastructure before adding translation templates.

---

## 8. Library notes

| Stack | Patch API | Gotcha |
|-------|-----------|--------|
| **nnsight** | `with model.trace: tensor[...] = ...` | Set `scan=False` after first layer |
| **TransformerLens** | `run_with_hooks` / `ActivationCache` + `hook_fn` | Hook names `blocks.{L}.hook_resid_post` |
| **HF + PyTorch** | `register_forward_hook` | Outputs often `(hidden, *extras)` tuples |
| **pyvene / baukit** | Declarative interventions | Learn their site naming |

This repo uses **nnsight** with optional TransformerLens via `UnifiedTransformer` (`use_tl=True`).

---

Back to [00_overview.md](00_overview.md)
