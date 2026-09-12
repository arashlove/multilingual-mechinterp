"""SAE-based feature ablation and steering interventions."""

from __future__ import annotations

from typing import Sequence

import torch

from multilingual_mechinterp.sae.learned_dict import LearnedDict, TiedSAE


def ablate_features(
    activations: torch.Tensor,
    sae: LearnedDict | TiedSAE,
    feature_ids: Sequence[int],
) -> torch.Tensor:
    """Zero selected SAE features and decode back to residual space."""
    squeeze = activations.ndim == 1
    x = activations.unsqueeze(0) if squeeze else activations
    with torch.no_grad():
        z = sae.encode(sae.center(x)).clone()
        for fid in feature_ids:
            z[..., fid] = 0.0
        out = sae.uncenter(sae.decode(z))
    return out.squeeze(0) if squeeze else out


def steer_features(
    activations: torch.Tensor,
    sae: LearnedDict | TiedSAE,
    feature_ids: Sequence[int],
    *,
    strength: float = 1.0,
) -> torch.Tensor:
    """Add ``strength`` to selected SAE features and decode."""
    squeeze = activations.ndim == 1
    x = activations.unsqueeze(0) if squeeze else activations
    with torch.no_grad():
        z = sae.encode(sae.center(x)).clone()
        for fid in feature_ids:
            z[..., fid] = z[..., fid] + strength
        out = sae.uncenter(sae.decode(z))
    return out.squeeze(0) if squeeze else out
