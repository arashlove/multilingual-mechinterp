# 04 — Evaluation & Interpretability Metrics

Once you have a trained dictionary, evaluate it with the same metrics used in this repo / paper.

All metrics assume a `LearnedDict`-like object with `encode`, `predict`, `get_learned_dict`, and optional `center`.

---

## 1. Fraction of Variance Unexplained (FVU)

**Paper Fig 2 vertical axis concept:** how much variance reconstruction fails to explain.

Repo definition:

```310:314:standard_metrics.py
def fraction_variance_unexplained(model: LearnedDict, batch):
    x_hat = model.predict(batch)
    residuals = (batch - x_hat).pow(2).mean()
    total = (batch - batch.mean(dim=0)).pow(2).mean()
    return residuals / total
```

Interpretation:

- `FVU = 0` → perfect reconstruction
- `FVU = 1` → no better than predicting the mean
- Lower is better (at fixed sparsity)

Standalone:

```python
def fvu(model, batch):
    x_hat = model.predict(batch)
    resid = (batch - x_hat).pow(2).mean()
    total = (batch - batch.mean(0)).pow(2).mean()
    return (resid / total).item()
```

---

## 2. Sparsity / active features

Repo:

```305:308:standard_metrics.py
def mean_nonzero_activations(model: LearnedDict, batch):
    batch_centered = model.center(batch)
    c = model.encode(batch_centered)
    return (c != 0).float().mean(dim=0)
```

- Per-feature firing rate: `mean_nonzero_activations` → shape `[n_feat]`
- **Average number of active features per token** (Fig 2 x-ish axis):

```python
l0 = (c > 0).float().sum(dim=-1).mean().item()
# or
l0 = mean_nonzero_activations(model, batch).sum().item()
```

Dead feature count (Appendix E style): features activating fewer than \(T\) times on a large set (repo often uses threshold ~10).

```python
def count_dead_features(model, batch, threshold=10):
    c = model.encode(model.center(batch))
    fires = (c > 0).sum(dim=0)
    return int((fires < threshold).sum())
```

---

## 3. FVU–sparsity Pareto (Fig 2)

For each trained \(\alpha\) (and fixed \(R\)):

1. Sample eval activations (e.g. 50k rows held out or same chunk)
2. Compute `fvu`, `mean_l0`
3. Plot points; compare SAE vs PCA/ICA/etc.

Acceptance: curve should be smooth; no requirement of a sharp “knee” (paper notes absence of a clear knee).

---

## 4. Dictionary geometry — MMCS

Mean Max Cosine Similarity between two dictionaries (toy recovery / dict comparison):

```276:285:standard_metrics.py
def mmcs(model: LearnedDict, model2: LearnedDict):
    return mcs_duplicates(model, model2).mean()

def mmcs_to_fixed(model: LearnedDict, truth):
    cosine_sim = torch.einsum("md,gd->mg", model.get_learned_dict(), truth)
    max_cosine_sim = cosine_sim.max(dim=-1).values
    return max_cosine_sim.mean()
```

Use for:

- toy ground-truth recovery (`replicate_toy_models.py`)
- comparing small vs large dictionaries

---

## 5. Qualitative feature interpretability (Section 5)

For feature index `k`:

### A. Top activating contexts

1. Run `c = encode(acts)` on many tokens with known text alignment
2. Select tokens with largest \(c_k\)
3. Show local windows (e.g. ±10 tokens)
4. Optionally histogram which vocab tokens fire at different activation ranges (paper Fig 5)

You need a parallel array of token ids / strings for each activation row. When extracting, save metadata or re-run the model on the same ordered dataset.

### B. Ablation / logit effect

Paper uses less-than-rank-one ablation along feature direction and measures Δlogits (e.g. apostrophe feature suppresses `"s"`).

Sketch:

```python
# at a chosen position, residual x, feature f (unit), activation c_k
# remove feature contribution from residual
x_abl = x - c_k * f   # or clamp removal so feature no longer active
# continue LM forward from that layer with x_abl and compare logits
```

### C. Autointerpretability (Section 3)

Protocol (Appendix A):

1. Collect top-activating 64-token fragments
2. Ask an LLM to explain the feature
3. Ask a simulator LLM to predict activations on held-out fragments
4. Score = correlation(predicted, true)

Repo: `interpret.py` (OpenAI). For an open stack, swap in a local instruct model. Scores will not match paper exactly if the simulator model differs.

---

## 6. Baselines to beat (same interface)

Implement or wrap:

| Method | Expectation |
|--------|-------------|
| Neuron / identity (+ ReLU for residual) | weak interpretability |
| Random directions + ReLU | weak |
| PCA | good reconstruction, weaker monosemanticity |
| ICA | best classical baseline in paper |
| SAE (tied) | better autointerp in early–mid layers |

Repo baselines: `autoencoders/pca.py`, `ica.py`, `nmf.py`, `learned_dict.py`.

---

## 7. Causal / IOI evaluation (Section 4) — advanced

Paper patches residual features on Indirect Object Identification:

\[
x'_i = x_i + \sum_{j \in F}(\bar{c}_{i,j}-c_{i,j}) f_j
\]

and measures KL to target logits. Feature order from ACDC.

**This repo does not fully ship the IOI driver** (`ioi_feature_ident.py` missing). Dataset helpers exist in `test_datasets/ioi.py`. Treat IOI as a later phase unless you reimplement ACDC patching yourself.

---

## 8. Eval checklist for your project

Minimum:

- [ ] FVU on held-out activations
- [ ] Mean L0 / active features
- [ ] Dead feature count
- [ ] Pareto plot across \(\alpha\)
- [ ] Top-5 activating snippets for a few features

Stronger:

- [ ] Same metrics for PCA/ICA/neurons on identical data
- [ ] Manual monosemantic case study
- [ ] Optional autointerp scores
- [ ] Optional causal patching

Next: [`05_REFERENCE_IMPLEMENTATION.md`](05_REFERENCE_IMPLEMENTATION.md).
