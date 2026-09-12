"""Simple residual interventions for JLens-style experiments."""

from __future__ import annotations

from typing import Callable

import torch


def intervene_jlens(
    residual: torch.Tensor,
    *,
    direction: torch.Tensor | None = None,
    scale: float = 1.0,
    transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> torch.Tensor:
    """Apply a directional or custom transform to a residual vector."""
    out = residual.clone()
    if transform is not None:
        out = transform(out)
    if direction is not None:
        out = out + scale * direction.to(out.device, dtype=out.dtype)
    return out
