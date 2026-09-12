# 03 — Training Sparse Autoencoders

How to train paper-faithful SAEs on cached activations.

---

## 1. Goal

Given activation chunks `[N, d]`, learn Tied SAE parameters \((M, b)\) for one or many L1 values \(\alpha\), with dictionary size \(n_\text{feat}=R\,d\).

Output: a list of `(LearnedDict, hyperparams)` checkpoints, typically `learned_dicts.pt`.

---

## 2. Hyperparameters (copy these)

| Name | Recommended start | Paper notes |
|------|-------------------|-------------|
| `ratio` \(R\) | `2` or `4` | autointerp uses 2; IOI/case studies often 4 |
| `l1_alpha` \(\alpha\) | sweep `logspace(-4, -2)` | paper residual autointerp \(\approx 8.6\times 10^{-4}\) |
| `lr` | `1e-3` | Adam |
| `batch_size` | `256`–`2048` | repo default often 256 / 1024 |
| `n_repetitions` (epochs over chunks) | `1`–`10` | paper Fig 3: 10 epochs |
| `dtype` | `float32` for training | cache may be float16 |
| init | Xavier on \(M\), zeros on \(b\) | |

**Important repo bug:** some configs set `n_epochs` but `big_sweep.py` uses `n_repetitions`. Prefer the latter.

---

## 3. Simplest trainer in this repo

`basic_l1_sweep.py` is the cleanest entrypoint:

1. Load `0.pt` to read `d`
2. Set `latent_dim = int(d * ratio)`
3. Init one `FunctionalTiedSAE` per \(\alpha\)
4. Stack into `FunctionalEnsemble` with Adam
5. For each epoch / chunk: random batches → `ensemble.step_batch`
6. Save `learned_dicts_epoch_*.pt`

Core init + train sketch:

```46:115:basic_l1_sweep.py
def basic_l1_sweep(
    dataset_dir, output_dir,
    ratio, l1_values=np.logspace(-4, -2, 16), batch_size=256,
    device="cuda", adam_kwargs={"lr": 1e-3},
    n_repetitions=1,
    save_after_every=False, 
):
    dataset = torch.load(os.path.join(dataset_dir, '0.pt'))
    activation_dim = dataset.shape[1]
    latent_dim = int(activation_dim * ratio)
    ...
    models = [FunctionalTiedSAE.init(activation_dim, latent_dim, l1, device=device)
              for l1 in l1_values]
    ensemble = FunctionalEnsemble(
        models, FunctionalTiedSAE,
        torchopt.adam, adam_kwargs,
        device=device
    )
    ...
    for epoch_idx in range(n_repetitions):
        for chunk in random_order(chunks):
            dataset = torch.load(...).to(torch.float32)
            sampler = BatchSampler(RandomSampler(...), batch_size, drop_last=False)
            ensemble_train_loop(ensemble, cfg, args, "ensemble", sampler, dataset, bar)
        torch.save(learned_dicts, ...)
```

CLI:

```powershell
python basic_l1_sweep.py `
  --dataset_dir="activation_data/layer_1" `
  --output_dir="dicts_l1_r2" `
  --ratio=2 `
  --n_repetitions=1
```

---

## 4. Single-step training loop (what to reimplement)

Per batch:

```python
batch = activations[idxs].to(device)          # [B, d], float32
loss, (loss_data, aux) = TiedSAE.loss(params, buffers, batch)
# aux["c"] = codes [B, n_feat]
optimizer.step(grads of params)
```

Repo ensemble path (`big_sweep.ensemble_train_loop`):

```python
for batch_idxs in sampler:
    batch = dataset[batch_idxs].to(args["device"])
    losses, aux_buffer = ensemble.step_batch(batch)
    # aux_buffer["c"] available for sparsity logging
```

You do **not** need `FunctionalEnsemble` / `torchopt` in your project. A plain `nn.Module` + `torch.optim.Adam` is fine if the math matches [`01_ARCHITECTURE.md`](01_ARCHITECTURE.md).

---

## 5. Minimal standalone training loop

```python
import torch
import torch.nn as nn

class TiedSAE(nn.Module):
    def __init__(self, d, n_feat, alpha):
        super().__init__()
        self.encoder = nn.Parameter(torch.empty(n_feat, d))
        nn.init.xavier_uniform_(self.encoder)
        self.bias = nn.Parameter(torch.zeros(n_feat))
        self.alpha = alpha

    def dictionary(self):
        return self.encoder / self.encoder.norm(dim=-1, keepdim=True).clamp_min(1e-8)

    def forward(self, x):
        D = self.dictionary()
        c = torch.relu(x @ D.T + self.bias)
        x_hat = c @ D
        return x_hat, c

    def loss(self, x):
        x_hat, c = self.forward(x)
        recon = (x_hat - x).pow(2).mean()
        l1 = self.alpha * c.abs().sum(dim=-1).mean()
        return recon + l1, {"recon": recon.detach(), "l1": l1.detach(), "c": c.detach()}


def train_sae(acts_path, R=2, alpha=8.6e-4, steps=5000, batch_size=1024, lr=1e-3, device="cpu"):
    data = torch.load(acts_path).to(dtype=torch.float32)
    d = data.shape[1]
    model = TiedSAE(d, int(R * d), alpha).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    for step in range(steps):
        idx = torch.randint(0, data.shape[0], (batch_size,))
        batch = data[idx].to(device)
        opt.zero_grad()
        loss, stats = model.loss(batch)
        loss.backward()
        opt.step()
        if step % 200 == 0:
            l0 = (stats["c"] > 0).float().sum(dim=-1).mean().item()
            print(step, float(loss), float(stats["recon"]), l0)
    return model
```

This is enough to drop into another codebase.

---

## 6. Sweeping \(\alpha\) (required for Fig-2-style analysis)

Do **not** pick a single \(\alpha\) blindly. Train a family:

```python
alphas = [0.0] + list(np.logspace(-4, -2, 8))
```

For each \(\alpha\), record:

- mean active features \(\mathbb{E}[\|c\|_0]\)
- FVU

Plot FVU (x) vs active features (y). Paper Fig 2.

Paper residual autointerp operating point: \(R=2\), \(\alpha\approx 0.00086\).

---

## 7. Checkpoint schema

This repo saves:

```python
learned_dicts = [
  (TiedSAE(...), {"dict_size": 1024, "l1_alpha": 0.00086}),
  (TiedSAE(...), {"dict_size": 1024, "l1_alpha": 0.001}),
  ...
]
torch.save(learned_dicts, "learned_dicts.pt")
```

In your project, prefer explicit state dicts:

```python
torch.save({
  "encoder": model.encoder.detach().cpu(),
  "bias": model.bias.detach().cpu(),
  "alpha": alpha,
  "ratio": R,
  "d": d,
  "model_name": "EleutherAI/pythia-70m-deduped",
  "layer": 1,
  "layer_loc": "residual",
}, "sae.pt")
```

---

## 8. Logging during training

Useful scalars per step / epoch:

| Metric | How |
|--------|-----|
| `loss` | total |
| `recon` | MSE mean |
| `l1` | \(\alpha\)-scaled L1 |
| `mean_l0` | `(c>0).sum(-1).mean()` |
| `dead_features` | features with &lt;10 fires over a large eval set |

Dead features are common in MLP SAEs; residual tied SAEs are more stable at moderate \(\alpha\).

---

## 9. Compute expectations

| Scale | Rough time |
|-------|------------|
| 0.01–0.1 GB acts, 1k–5k steps | minutes (CPU/GPU) |
| ~5–50M acts, 1–3 epochs, one \(\alpha\) | &lt;1 hour on A40 (paper) |
| Full \(\alpha\) sweep × layers | hours–days |

---

## 10. Training pitfalls

1. **Forgot row-normalization** → L1 becomes meaningless; features blow up.
2. **Train in float16 without care** → unstable; use float32 params.
3. **Too large \(\alpha\)** → high sparsity, high FVU, many dead features.
4. **Too small \(\alpha\) / \(\alpha=0\)** → dense codes, worse monosemanticity (paper IOI shows \(\alpha=0\) fails).
5. **Mismatch train/eval centering** → garbage reconstructions.
6. **Using `n_epochs` in this repo’s `big_sweep`** without setting `n_repetitions` → only one pass.

Next: [`04_EVALUATION.md`](04_EVALUATION.md).
