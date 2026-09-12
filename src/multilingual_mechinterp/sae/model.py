"""Sparse autoencoder model and high-level ``run_sae`` entry point."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from multilingual_mechinterp.models.loader import LoadedModel
from multilingual_mechinterp.sae.learned_dict import TiedSAE

# Backward-compatible name: paper architecture is the tied SAE.
SAE = TiedSAE


@dataclass
class SAEResult:
    """Output of a single SAE forward analysis."""

    prompt: str
    layer: int
    activations: torch.Tensor
    latent: torch.Tensor
    reconstruction: torch.Tensor
    top_features: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


def load_sae(
    path: str | Path | None = None,
    *,
    d_model: int = 512,
    d_sae: int = 1024,
    alpha: float = 1e-3,
    device: str | torch.device = "cpu",
) -> TiedSAE:
    """Load a tied SAE checkpoint, or return a randomly initialized one."""
    if path is not None:
        return TiedSAE.load(path, device=device)
    sae = TiedSAE(d_model=d_model, d_sae=d_sae, alpha=alpha)
    sae.to(device)
    sae.eval()
    return sae


def run_sae(
    model: LoadedModel | Any,
    prompt: str,
    layer: int | None = None,
    *,
    sae: TiedSAE | None = None,
    sae_path: str | Path | None = None,
    top_k: int = 10,
    device: str | None = None,
) -> SAEResult:
    """Encode residual activations at ``layer`` with a tied SAE."""
    from multilingual_mechinterp.models.hooks import hooked_forward
    from multilingual_mechinterp.sae.features import top_activating_features

    if not isinstance(model, LoadedModel):
        raise ValueError("model must be a LoadedModel (with tokenizer) for run_sae")

    loaded = model
    hf_model = loaded.model
    tokenizer = loaded.tokenizer
    n_layers = loaded.n_layers

    if layer is None:
        layer = max(0, n_layers // 2)
    if layer < 0 or layer >= n_layers:
        raise ValueError(f"layer must be in [0, {n_layers - 1}], got {layer}")

    run_device = torch.device(device or loaded.device)

    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(run_device) for k, v in inputs.items()}

    with torch.no_grad(), hooked_forward(hf_model, layers=[layer]) as cache:
        hf_model(**inputs)
        acts = cache.get(f"blocks.{layer}")

    token_act = acts[0, -1].float()
    d_model = token_act.shape[-1]

    if sae is None:
        sae = load_sae(sae_path, d_model=d_model, d_sae=2 * d_model, device=run_device)
    else:
        sae = sae.to(run_device)

    with torch.no_grad():
        latent, recon = sae(token_act)

    tops = top_activating_features(latent, k=top_k)

    return SAEResult(
        prompt=prompt,
        layer=layer,
        activations=token_act.cpu(),
        latent=latent.cpu(),
        reconstruction=recon.cpu(),
        top_features=tops,
        metadata={
            "model": loaded.name,
            "d_sae": sae.d_sae,
            "alpha": sae.alpha,
            "architecture": "tied_sae",
        },
    )
