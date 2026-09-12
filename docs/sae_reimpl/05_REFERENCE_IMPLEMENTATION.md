# 05 — Reference Implementation (copy into your project)

Minimal, paper-faithful **Tied Sparse Autoencoder** + metrics + training utilities.  
No `torchopt`, no ensemble stacking — plain PyTorch.

Compatible with the math in [`01_ARCHITECTURE.md`](01_ARCHITECTURE.md) and behavior of `FunctionalTiedSAE` / `TiedSAE` in this repo.

---

## File layout suggestion for your project

```
your_project/
  interpretability/
    sae/
      model.py          # TiedSAE module
      train.py          # train + sweep
      metrics.py        # FVU, L0, dead features
      activations.py    # optional TL extraction wrapper
      interface.py      # LearnedDict protocol
```

---

## `interface.py`

```python
from __future__ import annotations
from typing import Protocol
import torch


class LearnedDict(Protocol):
    n_feats: int
    activation_size: int

    def get_learned_dict(self) -> torch.Tensor:
        """[n_feat, d], preferably unit-norm rows."""
        ...

    def encode(self, batch: torch.Tensor) -> torch.Tensor:
        """[B, d] -> [B, n_feat], nonnegative codes."""
        ...

    def decode(self, code: torch.Tensor) -> torch.Tensor:
        """[B, n_feat] -> [B, d]."""
        ...

    def predict(self, batch: torch.Tensor) -> torch.Tensor:
        """Reconstruction [B, d]."""
        ...
```

---

## `model.py`

```python
from __future__ import annotations
import torch
import torch.nn as nn


class TiedSAE(nn.Module):
    """
    Paper / HoagyC sparse_coding Tied SAE.

    c = ReLU(D x + b)
    x_hat = D^T c
    where D is row-normalized encoder matrix.
    """

    def __init__(self, activation_size: int, n_feats: int, l1_alpha: float):
        super().__init__()
        self.activation_size = activation_size
        self.n_feats = n_feats
        self.l1_alpha = float(l1_alpha)

        self.encoder = nn.Parameter(torch.empty(n_feats, activation_size))
        nn.init.xavier_uniform_(self.encoder)
        self.encoder_bias = nn.Parameter(torch.zeros(n_feats))

    @torch.no_grad()
    def get_learned_dict(self) -> torch.Tensor:
        return self._normalized_dict()

    def _normalized_dict(self) -> torch.Tensor:
        norms = torch.norm(self.encoder, p=2, dim=-1)
        return self.encoder / torch.clamp(norms, min=1e-8)[:, None]

    def encode(self, batch: torch.Tensor) -> torch.Tensor:
        # batch: [B, d]
        D = self._normalized_dict()
        c = torch.einsum("nd,bd->bn", D, batch)
        c = c + self.encoder_bias
        return torch.clamp(c, min=0.0)

    def decode(self, code: torch.Tensor) -> torch.Tensor:
        D = self._normalized_dict()
        return torch.einsum("nd,bn->bd", D, code)

    def predict(self, batch: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(batch))

    def loss(self, batch: torch.Tensor):
        D = self._normalized_dict()
        c = torch.einsum("nd,bd->bn", D, batch) + self.encoder_bias
        c = torch.clamp(c, min=0.0)
        x_hat = torch.einsum("nd,bn->bd", D, c)

        recon = (x_hat - batch).pow(2).mean()
        l1 = self.l1_alpha * torch.norm(c, p=1, dim=-1).mean()
        total = recon + l1
        stats = {
            "loss": total.detach(),
            "recon": recon.detach(),
            "l1": l1.detach(),
            "mean_l0": (c > 0).float().sum(dim=-1).mean().detach(),
            "c": c.detach(),
        }
        return total, stats

    def state_dict_cpu(self) -> dict:
        return {
            "encoder": self.encoder.detach().cpu().clone(),
            "encoder_bias": self.encoder_bias.detach().cpu().clone(),
            "l1_alpha": self.l1_alpha,
            "activation_size": self.activation_size,
            "n_feats": self.n_feats,
        }

    @classmethod
    def from_state_dict_cpu(cls, sd: dict, device: str | torch.device = "cpu") -> "TiedSAE":
        model = cls(sd["activation_size"], sd["n_feats"], sd["l1_alpha"])
        with torch.no_grad():
            model.encoder.copy_(sd["encoder"])
            model.encoder_bias.copy_(sd["encoder_bias"])
        return model.to(device)


class UntiedSAE(nn.Module):
    """Optional: better for MLP layers (paper Appendix C)."""

    def __init__(self, activation_size: int, n_feats: int, l1_alpha: float, bias_decay: float = 0.0):
        super().__init__()
        self.activation_size = activation_size
        self.n_feats = n_feats
        self.l1_alpha = float(l1_alpha)
        self.bias_decay = float(bias_decay)

        self.encoder = nn.Parameter(torch.empty(n_feats, activation_size))
        self.decoder = nn.Parameter(torch.empty(n_feats, activation_size))
        nn.init.xavier_uniform_(self.encoder)
        nn.init.xavier_uniform_(self.decoder)
        self.encoder_bias = nn.Parameter(torch.zeros(n_feats))

    def get_learned_dict(self) -> torch.Tensor:
        norms = torch.norm(self.decoder, p=2, dim=-1)
        return self.decoder / torch.clamp(norms, min=1e-8)[:, None]

    def encode(self, batch: torch.Tensor) -> torch.Tensor:
        c = torch.einsum("nd,bd->bn", self.encoder, batch) + self.encoder_bias
        return torch.clamp(c, min=0.0)

    def decode(self, code: torch.Tensor) -> torch.Tensor:
        D = self.get_learned_dict()
        return torch.einsum("nd,bn->bd", D, code)

    def predict(self, batch: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(batch))

    def loss(self, batch: torch.Tensor):
        c = self.encode(batch)
        x_hat = self.decode(c)
        recon = (x_hat - batch).pow(2).mean()
        l1 = self.l1_alpha * torch.norm(c, p=1, dim=-1).mean()
        bias_pen = self.bias_decay * torch.norm(self.encoder_bias, p=2)
        total = recon + l1 + bias_pen
        return total, {"loss": total.detach(), "recon": recon.detach(), "c": c.detach()}
```

---

## `metrics.py`

```python
from __future__ import annotations
import torch


def fraction_variance_unexplained(model, batch: torch.Tensor) -> float:
    x_hat = model.predict(batch)
    resid = (batch - x_hat).pow(2).mean()
    total = (batch - batch.mean(dim=0)).pow(2).mean()
    return float(resid / total)


def mean_l0(model, batch: torch.Tensor) -> float:
    c = model.encode(batch)
    return float((c > 0).float().sum(dim=-1).mean())


def feature_firing_rates(model, batch: torch.Tensor) -> torch.Tensor:
    c = model.encode(batch)
    return (c > 0).float().mean(dim=0)  # [n_feat]


def count_dead_features(model, batch: torch.Tensor, threshold: int = 10) -> int:
    c = model.encode(batch)
    fires = (c > 0).sum(dim=0)
    return int((fires < threshold).sum().item())


def mmcs_to_fixed(model, truth: torch.Tensor) -> float:
    """truth: [n_true, d] unit-ish features."""
    D = model.get_learned_dict()
    cos = torch.einsum("md,gd->mg", D, truth)
    return float(cos.max(dim=-1).values.mean())
```

---

## `train.py`

```python
from __future__ import annotations
import os
from typing import Iterable, List, Sequence

import numpy as np
import torch

from .model import TiedSAE
from .metrics import fraction_variance_unexplained, mean_l0, count_dead_features


def iter_batches(data: torch.Tensor, batch_size: int):
    n = data.shape[0]
    perm = torch.randperm(n)
    for i in range(0, n, batch_size):
        yield data[perm[i : i + batch_size]]


def train_tied_sae(
    data: torch.Tensor,
    ratio: float = 2.0,
    alpha: float = 8.6e-4,
    lr: float = 1e-3,
    batch_size: int = 1024,
    n_epochs: int = 1,
    device: str = "cpu",
    log_every: int = 100,
) -> TiedSAE:
    assert data.ndim == 2
    data = data.to(dtype=torch.float32)
    d = data.shape[1]
    n_feats = int(ratio * d)

    model = TiedSAE(d, n_feats, alpha).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    step = 0
    for epoch in range(n_epochs):
        for batch in iter_batches(data, batch_size):
            batch = batch.to(device)
            opt.zero_grad(set_to_none=True)
            loss, stats = model.loss(batch)
            loss.backward()
            opt.step()
            if step % log_every == 0:
                print(
                    f"epoch={epoch} step={step} loss={float(stats['loss']):.6f} "
                    f"recon={float(stats['recon']):.6f} l0={float(stats['mean_l0']):.2f}"
                )
            step += 1
    return model


def sweep_l1(
    data: torch.Tensor,
    ratio: float = 2.0,
    alphas: Sequence[float] | None = None,
    device: str = "cpu",
    **train_kwargs,
) -> List[dict]:
    if alphas is None:
        alphas = [0.0] + list(np.logspace(-4, -2, 8))

    # hold out a bit for eval
    n = data.shape[0]
    n_eval = min(50_000, max(1_000, n // 10))
    eval_data = data[:n_eval].to(device)
    train_data = data[n_eval:] if n > n_eval else data

    results = []
    for alpha in alphas:
        model = train_tied_sae(train_data, ratio=ratio, alpha=float(alpha), device=device, **train_kwargs)
        with torch.no_grad():
            results.append(
                {
                    "alpha": float(alpha),
                    "ratio": float(ratio),
                    "fvu": fraction_variance_unexplained(model, eval_data),
                    "mean_l0": mean_l0(model, eval_data),
                    "dead": count_dead_features(model, eval_data),
                    "state": model.state_dict_cpu(),
                }
            )
        print("DONE", results[-1]["alpha"], results[-1]["fvu"], results[-1]["mean_l0"])
    return results


def train_from_chunk_dir(
    chunk_dir: str,
    out_path: str,
    ratio: float = 2.0,
    alpha: float = 8.6e-4,
    device: str = "cpu",
    **kwargs,
):
    chunks = sorted(
        [os.path.join(chunk_dir, f) for f in os.listdir(chunk_dir) if f.endswith(".pt")],
        key=lambda p: int(os.path.splitext(os.path.basename(p))[0]),
    )
    assert chunks, f"No .pt chunks in {chunk_dir}"
    data = torch.cat([torch.load(p, map_location="cpu").float() for p in chunks], dim=0)
    model = train_tied_sae(data, ratio=ratio, alpha=alpha, device=device, **kwargs)
    torch.save(model.state_dict_cpu(), out_path)
    return model
```

---

## `activations.py` (optional TransformerLens helper)

```python
from __future__ import annotations
from typing import List

import torch
from einops import rearrange
from datasets import load_dataset
from transformer_lens import HookedTransformer
from transformers import AutoTokenizer


def hook_name(layer: int, layer_loc: str = "residual") -> str:
    if layer_loc == "residual":
        return f"blocks.{layer}.hook_resid_post"
    if layer_loc == "mlp":
        return f"blocks.{layer}.mlp.hook_post"
    if layer_loc == "mlpout":
        return f"blocks.{layer}.hook_mlp_out"
    raise ValueError(layer_loc)


@torch.no_grad()
def extract_residual_activations(
    model_name: str = "EleutherAI/pythia-70m-deduped",
    dataset_name: str = "NeelNanda/pile-10k",
    layer: int = 1,
    max_sequences: int = 64,
    seq_len: int = 128,
    batch_size: int = 4,
    device: str = "cpu",
) -> torch.Tensor:
    model = HookedTransformer.from_pretrained(model_name, device=device)
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    ds = load_dataset(dataset_name, split="train")
    texts = [ds[i]["text"] for i in range(min(max_sequences, len(ds)))]
    hook = hook_name(layer, "residual")

    rows: List[torch.Tensor] = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        enc = tok(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=seq_len,
        )
        tokens = enc["input_ids"].to(device)
        _, cache = model.run_with_cache(tokens, stop_at_layer=layer + 1)
        acts = cache[hook]
        # mask padding if desired; here we keep all positions for simplicity
        rows.append(rearrange(acts, "b s d -> (b s) d").float().cpu())
    return torch.cat(rows, dim=0)
```

---

## End-to-end smoke script

```python
# scripts/sae_smoke.py
import torch
from interpretability.sae.activations import extract_residual_activations
from interpretability.sae.train import train_tied_sae, sweep_l1
from interpretability.sae.metrics import fraction_variance_unexplained, mean_l0

device = "cuda" if torch.cuda.is_available() else "cpu"

acts = extract_residual_activations(
    max_sequences=32, seq_len=64, batch_size=2, device=device
)
print("acts", acts.shape)

model = train_tied_sae(acts, ratio=2, alpha=8.6e-4, n_epochs=1, batch_size=512, device=device)
print("FVU", fraction_variance_unexplained(model, acts[:2000].to(device)))
print("L0", mean_l0(model, acts[:2000].to(device)))

# optional sweep
# results = sweep_l1(acts, ratio=2, device=device, n_epochs=1, batch_size=512)
```

---

## Parity checklist vs this repo

| Behavior | Required |
|----------|----------|
| Xavier init on encoder | yes |
| Zero bias init | yes |
| Row-normalize **inside** forward | yes |
| ReLU after bias | yes |
| MSE mean + α mean L1 | yes |
| Adam lr 1e-3 | yes |
| `n_feats = int(R * d)` | yes |
| Tied decode uses same normalized rows | yes |
| Centering | optional (off by default) |
| Bias decay | tied: off; untied: optional |

If these match, your reimplementation is faithful enough to use as the SAE method in a multi-method interpretability codebase.

---

## Mapping back to this repository

| Your module | Repo equivalent |
|-------------|-----------------|
| `TiedSAE.loss` | `FunctionalTiedSAE.loss` |
| `TiedSAE.encode` | `TiedSAE.encode` in `learned_dict.py` |
| `train_tied_sae` | simplified `basic_l1_sweep.py` |
| `fraction_variance_unexplained` | `standard_metrics.fraction_variance_unexplained` |
| `extract_residual_activations` | `activation_dataset.setup_data` / `generate_test_data.py` |

---

## Next steps after copying

1. Wire SAE behind your shared method registry.
2. Add PCA/ICA wrappers with the same `LearnedDict` protocol.
3. Build FVU–L0 plots across \(\alpha\).
4. Add top-activating token export for qualitative interpretability.
5. Only then consider autointerp / IOI.

Return to [`00_INDEX.md`](00_INDEX.md) for the full checklist.
