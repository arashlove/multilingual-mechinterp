"""Train tied SAEs and run L1 (α) sweeps."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import torch
from tqdm import tqdm

from multilingual_mechinterp.sae.activations import iter_activation_dir, load_activation_chunk
from multilingual_mechinterp.sae.learned_dict import TiedSAE
from multilingual_mechinterp.utils.io import ensure_dir, save_json


@dataclass
class TrainConfig:
    d_sae: int | None = None
    ratio: float = 2.0
    alpha: float = 8.6e-4
    lr: float = 1e-3
    batch_size: int = 512
    n_epochs: int = 1
    device: str = "cpu"
    seed: int = 0


@dataclass
class TrainResult:
    sae: TiedSAE
    history: list[dict[str, float]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


def train_tied_sae(
    activations: torch.Tensor | str | Path,
    *,
    config: TrainConfig | None = None,
    **overrides: Any,
) -> TrainResult:
    """Train one tied SAE on an activation tensor or ``.pt`` path."""
    cfg = config or TrainConfig()
    for key, value in overrides.items():
        if not hasattr(cfg, key):
            raise TypeError(f"Unknown TrainConfig field: {key}")
        setattr(cfg, key, value)

    torch.manual_seed(cfg.seed)
    acts = _as_tensor(activations, device="cpu")
    d_model = int(acts.shape[-1])
    d_sae = cfg.d_sae if cfg.d_sae is not None else int(cfg.ratio * d_model)

    device = torch.device(cfg.device)
    sae = TiedSAE(d_model=d_model, d_sae=d_sae, alpha=cfg.alpha).to(device)
    opt = torch.optim.Adam(sae.parameters(), lr=cfg.lr)

    history: list[dict[str, float]] = []
    n = acts.shape[0]
    steps_per_epoch = max(1, (n + cfg.batch_size - 1) // cfg.batch_size)

    sae.train()
    for epoch in range(cfg.n_epochs):
        perm = torch.randperm(n)
        epoch_loss = 0.0
        epoch_recon = 0.0
        epoch_l1 = 0.0
        for step in tqdm(range(steps_per_epoch), desc=f"train epoch {epoch+1}/{cfg.n_epochs}"):
            idx = perm[step * cfg.batch_size : (step + 1) * cfg.batch_size]
            if idx.numel() == 0:
                continue
            batch = acts[idx].to(device)
            opt.zero_grad(set_to_none=True)
            loss, parts = sae.loss(batch)
            loss.backward()
            opt.step()
            epoch_loss += float(parts["loss"].detach())
            epoch_recon += float(parts["l_reconstruction"].detach())
            epoch_l1 += float(parts["l_l1"].detach())
        denom = max(steps_per_epoch, 1)
        history.append(
            {
                "epoch": float(epoch),
                "loss": epoch_loss / denom,
                "l_reconstruction": epoch_recon / denom,
                "l_l1": epoch_l1 / denom,
            }
        )

    sae.eval()
    return TrainResult(
        sae=sae,
        history=history,
        metadata={
            "d_model": d_model,
            "d_sae": d_sae,
            "alpha": cfg.alpha,
            "lr": cfg.lr,
            "n_tokens": n,
            "n_epochs": cfg.n_epochs,
        },
    )


def train_from_activation_dir(
    dataset_dir: str | Path,
    *,
    config: TrainConfig | None = None,
    **overrides: Any,
) -> TrainResult:
    """Concatenate all chunks in ``dataset_dir`` and train one SAE."""
    chunks = list(iter_activation_dir(dataset_dir, device="cpu"))
    acts = torch.cat(chunks, dim=0)
    return train_tied_sae(acts, config=config, **overrides)


def sweep_l1(
    activations: torch.Tensor | str | Path,
    alphas: Sequence[float],
    *,
    ratio: float = 2.0,
    output_dir: str | Path = "results/sae/l1_sweep",
    lr: float = 1e-3,
    batch_size: int = 512,
    n_epochs: int = 1,
    device: str = "cpu",
    save_checkpoints: bool = True,
    save_learned_dicts: bool = False,
) -> list[dict[str, Any]]:
    """Train tied SAEs across L1 coefficients and save metrics (+ optional weights).

    By default writes ``sweep_metrics.json`` and one ``.pt`` per alpha. Set
    ``save_checkpoints=False`` to keep only metrics (figures/numbers workflow).
    ``learned_dicts.pt`` is off by default — it duplicated every checkpoint.
    """
    from multilingual_mechinterp.metrics.sae import evaluate_sae

    acts = _as_tensor(activations, device="cpu")
    output_dir = ensure_dir(output_dir)
    rows: list[dict[str, Any]] = []

    for alpha in alphas:
        result = train_tied_sae(
            acts,
            ratio=ratio,
            alpha=float(alpha),
            lr=lr,
            batch_size=batch_size,
            n_epochs=n_epochs,
            device=device,
        )
        metrics = evaluate_sae(result.sae, acts[: min(5000, acts.shape[0])])
        ckpt_path: str | None = None
        if save_checkpoints:
            ckpt = output_dir / f"tied_sae_alpha_{alpha:.2e}.pt"
            result.sae.save(ckpt)
            ckpt_path = str(ckpt)
        row = {
            "alpha": float(alpha),
            "checkpoint": ckpt_path,
            **metrics,
            "final_loss": result.history[-1]["loss"] if result.history else None,
        }
        rows.append(row)
        print(
            f"alpha={alpha:.2e}  FVU={metrics['fvu']:.4f}  "
            f"mean_l0={metrics['mean_l0']:.2f}  dead={metrics['dead_features']}  "
            f"-> {ckpt_path or 'metrics-only'}"
        )

    save_json(rows, output_dir / "sweep_metrics.json")
    if save_learned_dicts and save_checkpoints:
        torch.save(
            [(r["alpha"], TiedSAE.load(r["checkpoint"])) for r in rows if r["checkpoint"]],
            output_dir / "learned_dicts.pt",
        )
    return rows


def _as_tensor(activations: torch.Tensor | str | Path, *, device: str | torch.device) -> torch.Tensor:
    if isinstance(activations, torch.Tensor):
        return activations.detach().float().to(device)
    path = Path(activations)
    if path.is_dir():
        return torch.cat(list(iter_activation_dir(path, device=device)), dim=0).float()
    return load_activation_chunk(path, device=device).float()
