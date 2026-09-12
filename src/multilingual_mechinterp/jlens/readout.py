"""Residual-to-vocabulary readout helpers."""

from __future__ import annotations

from typing import Any

import torch


def project_to_vocab(residual: torch.Tensor, unembed: torch.nn.Module) -> torch.Tensor:
    """Map a residual vector to vocabulary logits via the LM head."""
    if residual.ndim == 1:
        residual = residual.unsqueeze(0)
    with torch.no_grad():
        logits = unembed(residual)
    return logits.squeeze(0)


def decode_residual(
    residual: torch.Tensor,
    unembed: torch.nn.Module,
    tokenizer: Any,
    *,
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """Return top-k decoded tokens for a residual vector."""
    logits = project_to_vocab(residual, unembed)
    values, indices = torch.topk(logits, k=min(top_k, logits.numel()))
    return [
        {
            "token_id": int(idx),
            "token": tokenizer.decode([idx]),
            "logit": float(val),
        }
        for idx, val in zip(indices.tolist(), values.tolist())
    ]
