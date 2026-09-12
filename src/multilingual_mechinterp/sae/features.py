"""SAE feature ranking and multi-prompt analysis helpers."""

from __future__ import annotations

from typing import Any, Sequence

import torch

from multilingual_mechinterp.models.loader import LoadedModel
from multilingual_mechinterp.sae.learned_dict import TiedSAE
from multilingual_mechinterp.sae.model import SAEResult, run_sae


def top_activating_features(latent: torch.Tensor, k: int = 10) -> list[dict[str, Any]]:
    """Return the top-k latent indices by activation value."""
    if latent.ndim != 1:
        raise ValueError(f"Expected 1D latent, got shape {tuple(latent.shape)}")
    k = min(k, latent.numel())
    values, indices = torch.topk(latent, k=k)
    return [
        {"feature_id": int(idx), "activation": float(val)}
        for idx, val in zip(indices.tolist(), values.tolist())
    ]


def analyze_features(
    model: LoadedModel | Any,
    prompts: str | Sequence[str],
    layer: int | None = None,
    *,
    sae: TiedSAE | None = None,
    top_k: int = 10,
    **kwargs: Any,
) -> list[SAEResult]:
    """Run SAE analysis over one or more prompts."""
    if isinstance(prompts, str):
        prompt_list = [prompts]
    else:
        prompt_list = list(prompts)
    return [
        run_sae(model=model, prompt=p, layer=layer, sae=sae, top_k=top_k, **kwargs)
        for p in prompt_list
    ]
