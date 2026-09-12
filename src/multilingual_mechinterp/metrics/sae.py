"""SAE reconstruction / sparsity metrics (paper-style)."""

from __future__ import annotations

from typing import Any

import torch

from multilingual_mechinterp.sae.learned_dict import LearnedDict


def mean_nonzero_activations(model: LearnedDict, batch: torch.Tensor) -> torch.Tensor:
    """Per-feature fraction of examples with nonzero code (shape ``[n_feats]``)."""
    batch_centered = model.center(batch)
    code = model.encode(batch_centered)
    return (code != 0).float().mean(dim=0)


def feature_firing_rates(model: LearnedDict, batch: torch.Tensor) -> torch.Tensor:
    """Alias of :func:`mean_nonzero_activations` (per-feature fire rate)."""
    return mean_nonzero_activations(model, batch)


def mean_l0(model: LearnedDict, batch: torch.Tensor) -> float:
    """Mean number of active features per example."""
    batch_centered = model.center(batch)
    code = model.encode(batch_centered)
    return float((code > 0).float().sum(dim=-1).mean().item())


def count_dead_features(
    model: LearnedDict,
    batch: torch.Tensor,
    *,
    threshold: int = 10,
) -> int:
    """Count features that fire on fewer than ``threshold`` examples (paper App. E)."""
    batch_centered = model.center(batch)
    code = model.encode(batch_centered)
    fires = (code > 0).sum(dim=0)
    return int((fires < threshold).sum().item())


def fraction_variance_unexplained(model: LearnedDict, batch: torch.Tensor) -> float:
    """FVU = residual variance / total variance (paper Fig. 2)."""
    x_hat = model.predict(batch)
    residuals = (batch - x_hat).pow(2).mean()
    total = (batch - batch.mean(dim=0)).pow(2).mean()
    if float(total) < 1e-12:
        return 0.0
    return float((residuals / total).item())


def r_squared(model: LearnedDict, batch: torch.Tensor) -> float:
    return 1.0 - fraction_variance_unexplained(model, batch)


def mmcs_to_fixed(model: LearnedDict, truth: torch.Tensor) -> float:
    """Mean max cosine similarity of each learned feature to a fixed dictionary.

    ``truth`` shaped ``[n_true, d]`` (rows = features). Used for toy recovery
    and dictionary comparison (paper / ``standard_metrics.mmcs_to_fixed``).
    """
    learned = model.get_learned_dict()
    # normalize both for cosine
    learned = learned / torch.clamp(learned.norm(dim=-1, keepdim=True), min=1e-8)
    truth = truth / torch.clamp(truth.norm(dim=-1, keepdim=True), min=1e-8)
    cosine = torch.einsum("md,gd->mg", learned, truth.to(learned.device, learned.dtype))
    return float(cosine.max(dim=-1).values.mean().item())


def mmcs(model: LearnedDict, model2: LearnedDict) -> float:
    """Mean max cosine similarity between two learned dictionaries."""
    return mmcs_to_fixed(model, model2.get_learned_dict())


def _batch_on_model_device(model: LearnedDict, batch: torch.Tensor) -> torch.Tensor:
    """Move ``batch`` onto the same device as ``model`` parameters when possible."""
    if isinstance(model, torch.nn.Module):
        try:
            device = next(model.parameters()).device
            return batch.to(device)
        except StopIteration:
            pass
    return batch


def evaluate_sae(
    model: LearnedDict,
    batch: torch.Tensor,
    *,
    dead_threshold: int = 10,
) -> dict[str, Any]:
    """Return FVU / L0 / dead-feature summary for logging and sweeps."""
    batch = _batch_on_model_device(model, batch)
    fvu = fraction_variance_unexplained(model, batch)
    l0 = mean_l0(model, batch)
    dead = count_dead_features(model, batch, threshold=dead_threshold)
    n_feats = int(model.get_learned_dict().shape[0])
    return {
        "fvu": fvu,
        "r_squared": 1.0 - fvu,
        "mean_l0": l0,
        "n_active_features_mean": l0,
        "dead_features": dead,
        "dead_fraction": dead / max(n_feats, 1),
        "n_feats": n_feats,
    }
