# SAE Reimplementation Docs — Index

Precise guide to **re-implement sparse autoencoders for interpretability**, matching this repository and the ICLR 2024 paper:

> Cunningham, Ewart, Riggs, Huben, Sharkey — *Sparse Autoencoders Find Highly Interpretable Features in Language Models*  
> https://arxiv.org/abs/2309.08600  
> Upstream code: https://github.com/HoagyC/sparse_coding  
> This fork: https://github.com/arashlove/sparse_coding

Use these docs when porting SAE interpretability into another project. Prefer **paper-faithful Tied SAE** unless you explicitly need untied encoders/decoders (MLP layers).

---

## Reading order

| # | Doc | Purpose |
|---|-----|---------|
| 1 | [`01_ARCHITECTURE.md`](01_ARCHITECTURE.md) | Math, tensors, forward pass, loss, tied vs untied |
| 2 | [`02_ACTIVATIONS.md`](02_ACTIVATIONS.md) | Extract LM activations (hooks, shapes, caching) |
| 3 | [`03_TRAINING.md`](03_TRAINING.md) | Init, optimizer, batching, L1 sweeps, checkpoints |
| 4 | [`04_EVALUATION.md`](04_EVALUATION.md) | FVU, sparsity, MMCS, interpretability checks |
| 5 | [`05_REFERENCE_IMPLEMENTATION.md`](05_REFERENCE_IMPLEMENTATION.md) | Minimal standalone PyTorch modules to copy |

Also useful:

- [`../GETTING_STARTED.md`](../GETTING_STARTED.md) — env setup, smoke tests, paper→repo map
- Paper PDF: `../SPARSE AUTOENCODERS FIND HIGHLY INTERPRETABLE FEATURES IN LANGUAGE MODELS.pdf`

---

## What you must re-implement (checklist)

### Core (required for SAE method)

- [ ] Tied SAE: encoder matrix \(M \in \mathbb{R}^{n_\text{feat}\times d}\), bias \(b\), ReLU codes
- [ ] Row-normalize dictionary before encode/decode
- [ ] Loss = MSE reconstruction + \(\alpha \|c\|_1\)
- [ ] Adam optimizer, default lr `1e-3`
- [ ] Train on cached activations shaped `[N, d]`
- [ ] Dict size \(n_\text{feat} = R \cdot d\) (ratio \(R\))

### Data (required)

- [ ] Hook residual stream (or MLP) of an open LM (Pythia-70M default)
- [ ] Flatten `(batch, seq, d) → (batch*seq, d)`
- [ ] Store float16/float32 chunks on disk

### Evaluation (required for interpretability comparison)

- [ ] Fraction of Variance Unexplained (FVU)
- [ ] Mean number of active features (L0-style sparsity)
- [ ] Optional: top-activating token contexts per feature
- [ ] Optional: baselines behind the same interface (PCA / ICA / neurons)

### Optional (paper extras)

- [ ] Autointerpretability scoring (OpenAI or local LLM)
- [ ] IOI causal patching (this repo’s IOI driver script is incomplete)
- [ ] Untied SAE for MLP layers (reduces dead features)

---

## Canonical hyperparameters (paper)

| Setting | Typical paper value |
|---------|---------------------|
| Model | `EleutherAI/pythia-70m-deduped` (`d=512`) |
| Location | residual stream post-block |
| Architecture | **tied** weights + ReLU |
| Ratio \(R\) | `2` (autointerp), `4` (IOI / case studies) |
| L1 \(\alpha\) | `8.6e-4` (autointerp residual), `1.4e-3` (some case studies) |
| Optimizer | Adam, lr `1e-3` |
| Data | Pile activations, ~5–50M vectors, 1–10 epochs |
| Init | Xavier uniform on \(M\), zeros on \(b\) |

---

## Source of truth in this repo

| Concept | File |
|---------|------|
| Tied training loss | `autoencoders/sae_ensemble.py` → `FunctionalTiedSAE` |
| Untied training loss | `autoencoders/sae_ensemble.py` → `FunctionalSAE` |
| Inference API | `autoencoders/learned_dict.py` → `TiedSAE` / `UntiedSAE` |
| Simple trainer | `basic_l1_sweep.py` |
| Ensemble trainer | `big_sweep.py`, `autoencoders/ensemble.py` |
| Hooks / caching | `activation_dataset.py` |
| Metrics | `standard_metrics.py` |

---

## Design contract for your project

Expose every method (SAE, PCA, ICA, …) behind one interface:

```python
class LearnedDict:
    def get_learned_dict(self) -> Tensor:  # [n_feat, d], unit-norm rows
        ...
    def encode(self, x: Tensor) -> Tensor:  # [B, d] -> [B, n_feat]
        ...
    def decode(self, c: Tensor) -> Tensor:  # [B, n_feat] -> [B, d]
        ...
    def predict(self, x: Tensor) -> Tensor: # encode→decode (with optional centering)
        ...
```

Then all metrics and interpretability tools depend only on this interface.
