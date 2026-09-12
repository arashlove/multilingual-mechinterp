# 01 — SAE Architecture (exact)

This document specifies the **exact sparse autoencoder** used for interpretability in the paper and this repo.

---

## 1. Problem statement

Given activation vectors \(x \in \mathbb{R}^d\) from a language model (e.g. residual stream), learn an overcomplete dictionary of features

\[
\{f_k\}_{k=1}^{n_\text{feat}} \subset \mathbb{R}^d,\qquad n_\text{feat} = R\,d
\]

such that each activation is approximately a **sparse nonnegative** combination of dictionary features:

\[
x \approx \sum_{k=1}^{n_\text{feat}} c_k f_k,\qquad c_k \ge 0,\quad c \text{ sparse}.
\]

\(R\) is the **dictionary ratio** (expansion factor).

---

## 2. Paper equations (canonical)

### Tied SAE (paper default)

Parameters: \(M \in \mathbb{R}^{n_\text{feat}\times d}\), \(b \in \mathbb{R}^{n_\text{feat}}\).

\[
c = \mathrm{ReLU}(M x + b)
\]

\[
\hat{x} = M^\top c = \sum_i c_i f_i
\]

where rows of \(M\) are the dictionary features \(f_i\), and rows are **L2-normalized** during use so the L1 penalty cannot be gamed by shrinking \(c\) and growing \(M\).

### Loss (paper)

\[
\mathcal{L}(x)
=
\frac{\|x-\hat{x}\|_2^2}{\dim(x)}
+
\alpha \|c\|_1
\]

In batched PyTorch this is implemented as elementwise MSE mean + mean L1 over the batch:

```text
recon = ((x_hat - x)**2).mean()
l1    = alpha * c.abs().sum(dim=-1).mean()
loss  = recon + l1
```

`mean()` over all elements of `(B, d)` is equivalent to averaging reconstruction error across batch **and** dividing by \(d\).

---

## 3. This repo’s Tied SAE (source of truth)

### Files

- Training: `autoencoders/sae_ensemble.py` → `FunctionalTiedSAE`
- Inference: `autoencoders/learned_dict.py` → `TiedSAE`

### Parameters / buffers

| Name | Shape | Init |
|------|-------|------|
| `encoder` \(M\) | `[n_feat, d]` | `xavier_uniform_` |
| `encoder_bias` \(b\) | `[n_feat]` | zeros |
| `l1_alpha` | scalar buffer | hyperparameter \(\alpha\) |

Optional centering buffers (identity by default):

| Buffer | Default | Role |
|--------|---------|------|
| `center_trans` | zeros `[d]` | subtract mean / offset |
| `center_rot` | `I_d` | rotate |
| `center_scale` | ones `[d]` | rescale |

For a paper-faithful first implementation, leave centering as identity (no centering).

### Forward (training) — exact logic

```python
# M: [n_feat, d], b: [n_feat], batch: [B, d]

# 1) normalize dictionary rows
decoder_norms = M.norm(dim=-1)                         # [n_feat]
learned_dict = M / decoder_norms.clamp_min(1e-8)[:, None]  # [n_feat, d]

# 2) optional center (identity if unused)
batch_c = center(batch)                                # [B, d]

# 3) encode
c = einsum("nd,bd->bn", learned_dict, batch_c) + b     # [B, n_feat]
c = c.clamp_min(0.0)                                   # ReLU

# 4) decode (in centered space)
x_hat_c = einsum("nd,bn->bd", learned_dict, c)         # [B, d]

# 5) losses (compute in centered space for recon)
l_recon = (x_hat_c - batch_c).pow(2).mean()
l_l1 = alpha * c.norm(p=1, dim=-1).mean()
loss = l_recon + l_l1
```

Repo reference:

```135:162:autoencoders/sae_ensemble.py
    def loss(params, buffers, batch):
        decoder_norms = torch.norm(params["encoder"], 2, dim=-1)
        learned_dict = params["encoder"] / torch.clamp(decoder_norms, 1e-8)[:, None]

        batch_centered = FunctionalTiedSAE.center(buffers, batch)

        c = torch.einsum("nd,bd->bn", learned_dict, batch_centered)
        c = c + params["encoder_bias"]
        c = torch.clamp(c, min=0.0)

        x_hat_centered = torch.einsum("nd,bn->bd", learned_dict, c)
        ...
        l_reconstruction = (x_hat_centered - batch_centered).pow(2).mean()
        l_l1 = buffers["l1_alpha"] * torch.norm(c, 1, dim=-1).mean()
```

### Inference `TiedSAE.encode`

```205:215:autoencoders/learned_dict.py
    def encode(self, batch):
        if self.norm_encoder:
            norms = torch.norm(self.encoder, 2, dim=-1)
            encoder = self.encoder / torch.clamp(norms, 1e-8)[:, None]
        else:
            encoder = self.encoder

        c = torch.einsum("nd,bd->bn", encoder, batch)
        c = c + self.encoder_bias
        c = torch.clamp(c, min=0.0)
        return c
```

Decode is always:

```python
x_hat = einsum("nd,bn->bd", unit_norm_dict, c)
```

(`LearnedDict.decode`)

---

## 4. Untied SAE (use for MLP; optional)

Paper Appendix C: for MLP layers, untied encoder/decoder reduces dead features.

\[
c = \mathrm{ReLU}(M_e x + b),\qquad \hat{x} = M_d^\top c
\]

with **decoder rows** normalized (not encoder).

Repo: `FunctionalSAE` / `UntiedSAE`.

```53:78:autoencoders/sae_ensemble.py
    def loss(params, buffers, batch):
        c = torch.einsum("nd,bd->bn", params["encoder"], batch)
        c = c + params["encoder_bias"]
        c = torch.clamp(c, min=0.0)

        decoder_norms = torch.norm(params["decoder"], 2, dim=-1)
        learned_dict = params["decoder"] / torch.clamp(decoder_norms, 1e-8)[:, None]

        x_hat = torch.einsum("nd,bn->bd", learned_dict, c)

        l_reconstruction = (x_hat - batch).pow(2).mean()
        l_l1 = buffers["l1_alpha"] * torch.norm(c, 1, dim=-1).mean()
        l_bias_decay = buffers["bias_decay"] * torch.norm(params["encoder_bias"], 2)
        return l_reconstruction + l_l1 + l_bias_decay, ...
```

Notes:

- Untied has separate `encoder` and `decoder` matrices (both Xavier init).
- Optional `bias_decay` regularizer on \(b\) (default `0.0` in many sweeps).
- Dictionary features for interpretability = **normalized decoder rows**.

---

## 5. Critical implementation details (do not skip)

1. **Normalize every forward** (or keep a projected/normalized copy). Do not only normalize at init.
2. **ReLU after bias** — \(c=\mathrm{ReLU}(Mx+b)\), not \(\mathrm{ReLU}(Mx)+b\).
3. **Nonnegative codes only** — negatives are clamped to 0; this is part of monosemantic feature design.
4. **Tied means encoder = decoder direction** — one matrix \(M\); decode uses \(M^\top\) after row-norm.
5. **\(\alpha\) is per-training-run** — sweep it; results are a Pareto curve of sparsity vs FVU.
6. **`n_feat = int(R * d)`** — e.g. Pythia-70M residual `d=512`, `R=2` → 1024 features.
7. **Dtype** — train in float32; activations may be cached float16 then cast.

---

## 6. Shapes cheat sheet

| Tensor | Shape |
|--------|-------|
| activation batch `x` | `[B, d]` |
| dictionary / encoder `M` | `[n_feat, d]` |
| bias `b` | `[n_feat]` |
| codes `c` | `[B, n_feat]` |
| reconstruction `x_hat` | `[B, d]` |

Einsum convention in this repo: `"nd,bd->bn"` means rows of `M` are features.

---

## 7. What “a feature” is

Feature \(k\) is:

- **Direction:** unit-norm row \(f_k = M_{k,:}/\|M_{k,:}\|\)
- **Activation on input \(x\):** \(c_k = \mathrm{ReLU}(\langle f_k, x\rangle + b_k)\) (tied, normalized)
- **Contribution to reconstruction:** \(c_k f_k\)

Interpretability analyses (token histograms, ablations, autointerp) operate on \(c_k\) over text windows and/or on interventions along \(f_k\).

---

## 8. Minimal acceptance test for your reimplementation

Given random `x ~ N(0,I)` of shape `[64, 512]`, `n_feat=1024`, \(\alpha=1e-3\):

1. `encode(x)` returns `[64, 1024]` with all entries \(\ge 0\).
2. `get_learned_dict()` rows have L2 norm \(\approx 1\).
3. `predict(x)` returns `[64, 512]`.
4. `loss` is finite and decreases after a few Adam steps on the same batch.

Next: [`02_ACTIVATIONS.md`](02_ACTIVATIONS.md).
