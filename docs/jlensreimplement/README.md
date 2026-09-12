# Re-implementing the Jacobian Lens (`jlens`)

Precise guide for rebuilding this library in another project. Source of truth:
[`jlens/`](../../jlens/). Paper:
[Verbalizable Representations Form a Global Workspace in Language Models](https://transformer-circuits.pub/2026/workspace/index.html).

## What you are building

A **Jacobian lens** that reads what an internal residual-stream activation is
disposed to make the model say. For each layer \(l\):

```
lens_l(h) = unembed( J_l @ h )
J_l       = E[ ∂h_final / ∂h_l ]
```

The expectation is over prompts, source positions, and current+future target
positions on generic web text. Fitting averages per-prompt Jacobian estimators;
applying does one forward pass, multiplies residuals by \(J_l\), and unembeds.

**In scope (this package):** fit → apply → optional slice visualization.

**Out of scope (prompt JSON only under `data/`):** paper experiment runners
(probe-swap, ignition, steering, etc.). Reimplement those separately if needed;
see [07-paper-experiments.md](07-paper-experiments.md).

## Doc map

| Doc | Contents |
|-----|----------|
| [01-protocol-and-hooks.md](01-protocol-and-hooks.md) | `LensModel` contract + `ActivationRecorder` |
| [02-fitting.md](02-fitting.md) | Position mask, cotangent estimator, `fit`, checkpoints |
| [03-lens-apply.md](03-lens-apply.md) | `JacobianLens`: transport, apply, save/load/merge |
| [04-hf-adapter.md](04-hf-adapter.md) | HuggingFace layout autodetection + wrapper |
| [05-visualization.md](05-visualization.md) | Slice computation + HTML packaging |
| [06-checklist.md](06-checklist.md) | Rebuild order, pitfalls, verification tests |
| [07-paper-experiments.md](07-paper-experiments.md) | Experiment/eval conventions (not in Python) |

## Minimal dependency surface

| Dependency | Role |
|------------|------|
| `torch` | Autograd Jacobians, hooks, tensors |
| `transformers` (≥5.5) | Causal LM + tokenizer (caller loads; you wrap) |
| `huggingface_hub` | Optional: load Hub lenses via `snapshot_download` |
| `numpy` | Slice arrays / binary packing |
| `datasets` | Optional: WikiText corpus for fitting |

Python ≥ 3.10. No CLI — library API only.

## Suggested rebuild order

1. **Protocol** — `encode` / `forward` / `unembed` + toy residual decoder
2. **Hooks** — capture residuals; `start_graph_at` for activation-leaf graphs
3. **Fitting core** — `valid_position_mask` + `jacobian_for_prompt`
4. **Fit loop** — running mean, atomic checkpoints, `merge`
5. **Lens object** — `transport` (`h @ J.T`), `apply`, save/load
6. **HF adapter** — layout autodetection, freeze params, optional per-block compile
7. **Visualization** — ranks helper → `compute_slice` → HTML page

Verify at each step against the property tests in [06-checklist.md](06-checklist.md).

## Critical invariants (memorize these)

1. **Orientation:** \(J\) stores rows as \(\partial\text{output\_dim}/\partial\text{input}\). Apply with `residual @ J.T`.
2. **Estimator reduction:** one-hot cotangents at *all valid target positions at once*; mean grads over *valid source positions*. Not independent per-position Jacobians.
3. **Graph rooting:** model params stay `requires_grad=False`; activations at `start_graph_at` get `requires_grad_(True)`.
4. **Determinism:** `forward` must be identical across batch clones (eval, dropout off) — fitting replicates the prompt along batch.
5. **Compile:** wrap *each block*, never the whole module, or hooks break.
6. **Skip positions:** first 16 (attention sinks) and the final token (no next-token target).

## Typical pipelines

```text
# Fit
HF model → from_hf → fit(prompts) → lens.save(...)

# Apply
HF model → from_hf → JacobianLens.load|from_pretrained → apply / compute_slice
```

```python
# Apply (reference usage)
import transformers, jlens

hf = transformers.AutoModelForCausalLM.from_pretrained("org/model").cuda()
tok = transformers.AutoTokenizer.from_pretrained("org/model")
model = jlens.from_hf(hf, tok)

lens = jlens.JacobianLens.from_pretrained("org/lens-repo", filename="model/lens.pt")
lens_logits, model_logits, _ = lens.apply(
    model, "Fact: The currency used in the country shaped like a boot is",
    positions=[-2],
)

# Fit
lens = jlens.fit(model, prompts=my_prompts, checkpoint_path="out/ckpt.pt")
lens.save("out/jacobian_lens.pt")
```

Paper-scale: ~1000 × 128-token sequences; ~100 usable; quality saturates (§9.3).
Parallelize with disjoint prompt shards + `JacobianLens.merge()`.
