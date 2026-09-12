# Activation Patching — Precise Overview

Activation patching (also called *causal tracing*, *causal mediation analysis*, or *interchange intervention*) measures whether a hidden state **causally** carries information used for a downstream prediction.

This guide is written so you can re-implement the method in another project. Code snippets mirror the working patterns in this repo (`src/nnsight_utils.py`, `src/interventions.py`) and the classic Meng et al. causal-tracing formulation.

---

## 1. What you are measuring

A transformer computes a residual stream \(h^{(\ell)}_t \in \mathbb{R}^d\) at layer \(\ell\) and token position \(t\). Activation patching asks:

> If I replace \(h^{(\ell)}_t\) in run **B** with the value it had in run **A**, does the model’s output move toward A’s answer?

That change in output probability is the **causal effect** of that state.

### Formal definition

Let:

| Symbol | Meaning |
|--------|---------|
| \(x^\text{src}\) | Source prompt (donor of activations) |
| \(x^\text{tgt}\) | Target prompt (recipient; run you intervene on) |
| \(h^{(\ell)}_t(x)\) | Residual-stream activation at layer \(\ell\), position \(t\) |
| \(y\) | Metric token(s) of interest (e.g. correct next-token id) |
| \(P(y \mid x)\) | Softmax probability of \(y\) at the prediction position |

**Single-site patch** at \((\ell, t)\):

\[
\tilde{h}^{(\ell)}_t
=
\begin{cases}
h^{(\ell)}_t(x^\text{src}) & \text{at the patched site} \\
h^{(\ell)}_{t'}(x^\text{tgt}) & \text{elsewhere}
\end{cases}
\]

Causal effect:

\[
\text{CE}(\ell, t)
=
P\!\bigl(y \mid x^\text{tgt};\; \text{patch }(\ell,t)\bigr)
-
P\!\bigl(y \mid x^\text{tgt}\bigr)
\]

Positive CE means the patched state **pushed** the target run toward \(y\).

---

## 2. Three standard experimental designs

All three share the same intervention primitive (overwrite a tensor mid-forward). They differ only in how \(x^\text{src}\) and \(x^\text{tgt}\) are constructed.

### A. Clean → Corrupt → Restore (Meng / ROME causal tracing)

1. **Clean run** on factual prompt \(x\): cache all \(h^{(\ell)}_t\).
2. **Corrupt run** on \(x^*\) (noise on subject embeddings, or a counterfactual subject).
3. **Restore**: during the corrupt run, overwrite one (or a window of) state(s) with the clean cache.
4. Metric: recovery of \(P(\text{correct attribute})\).

Use this when you want a **heatmap over (token, layer)** for a single fact.

### B. Cross-prompt / interchange intervention (this repo’s main method)

1. Run **source** prompt(s) that encode concept \(c_A\) (e.g. FR→EN translation of “chien”).
2. Run **target** prompt that asks for concept \(c_B\) (e.g. ZH→EN of “猫”).
3. Patch source activations for \(c_A\) into the **object token** position of the target prompt.
4. Metric: does \(P(\text{English for }c_A)\) rise on the target prompt?

Use this when you want to test whether a representation is **content-bearing and transferable** (language-agnostic concept vectors, circuit reuse, etc.).

### C. Patchscopes / latent-prompt readout

1. Collect \(h^{(\ell)}_{-1}(x^\text{src})\).
2. Patch it into a **readout prompt** (e.g. `"king king\n1135 1135\nhello hello\n?"`) at the `?` position, often at every layer.
3. Metric: next-token distribution under the readout prompt.

Use this when you want a **natural-language description** of what a latent “means.”

---

## 3. The only two phases you must implement

Every variant reduces to:

```
PHASE 1 — CACHE
  forward(source) → store activations at chosen (module, layer, position)

PHASE 2 — PATCH
  forward(target) with hooks that overwrite those sites
  → read logits / probs at the prediction position
```

Everything else (windows, mean over sources, attention vs residual, metrics) is configuration around these two phases.

---

## 4. Design choices you must decide up front

| Choice | Options | Recommendation |
|--------|---------|----------------|
| **What to patch** | residual out, attn out, MLP out, attn inputs | Start with residual stream (`layer.output`) |
| **Where (token)** | last subject / object token, last prompt token, all tokens | Index carefully; see [05_indexing_metrics_pitfalls.md](05_indexing_metrics_pitfalls.md) |
| **Where (layer)** | single \(\ell\), window \([\ell, \ell+k)\), all layers from \(\ell\) | Windows are often required for small modules |
| **Source aggregation** | single example, mean over N sources | Mean is more stable for concept vectors |
| **Same-layer vs cross-layer** | patch layer \(\ell\) source → layer \(\ell\) target (same), or source \(\ell_s\) → target \(\ell_t\) | Same-layer is the default and safest |
| **Metric** | \(P(y)\), logit difference, KL, generation string | Prefer summed \(P\) over token variants of \(y\) |

---

## 5. Minimal algorithm (pseudocode)

```text
# PHASE 1
src_acts = {}  # (layer, pos) -> Tensor[batch, d]
with no_grad:
  out = model.forward(source_ids)
  for layer in layers:
    src_acts[layer] = residual[layer][arange(B), source_pos]  # [B, d]

# PHASE 2 — scan start layer s
for s in range(n_layers):
  def hook(module, inputs, output, layer=s):
    # output: [B, T, d]  (or tuple; take hidden states)
    h = output[0] if isinstance(output, tuple) else output
    for L in range(s, min(s + window, n_layers)):
      if L == layer:
        h = h.clone()
        h[arange(B), target_pos] = src_acts[L]
    return (h, *output[1:]) if isinstance(output, tuple) else h

  register hooks for layers in [s, s+window)
  probs[s] = softmax(model(target_ids).logits[:, -1, :])
  remove hooks

effect[s] = probs[s, y] - baseline_probs[y]
```

This is intentionally framework-agnostic. Concrete PyTorch hooks and nnsight versions are in the sibling docs.

---

## 6. Doc map

| Doc | Contents |
|-----|----------|
| [01_collect_activations.md](01_collect_activations.md) | Phase 1: caching residuals / attn, padding-safe last-token index |
| [02_core_patch_loop.md](02_core_patch_loop.md) | Phase 2: single-site, windows, nnsight & raw hooks |
| [03_object_patching_recipe.md](03_object_patching_recipe.md) | End-to-end recipe matching this paper’s object patching |
| [04_variants.md](04_variants.md) | Clean/corrupt restore, Patchscopes, attention patching, steering |
| [05_indexing_metrics_pitfalls.md](05_indexing_metrics_pitfalls.md) | Token indices, metrics, common bugs |

---

## 7. References

- Meng et al., *Locating and Editing Factual Associations in GPT* (ROME / causal tracing): https://arxiv.org/abs/2202.05262
- Vig et al., causal mediation for language models (2020)
- Ghandeharioun et al., *Patchscopes* (2024)
- This paper: *Separating Tongue from Thought* — https://arxiv.org/abs/2411.08745
- Working code in this repo: `src/interventions.py` (`object_lens`, `patchscope_lens`), `src/nnsight_utils.py` (`collect_activations`)
