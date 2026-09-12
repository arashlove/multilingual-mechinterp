"""Behavioural scoring helpers."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def next_token_prob(logits: torch.Tensor, token_id: int) -> float:
    """Softmax probability of ``token_id`` from a logits vector."""
    if logits.ndim != 1:
        raise ValueError(f"Expected 1D logits, got shape {tuple(logits.shape)}")
    probs = F.softmax(logits.float(), dim=-1)
    return float(probs[token_id].item())


def logit_diff(logits: torch.Tensor, correct_id: int, incorrect_id: int) -> float:
    """Logit(correct) - logit(incorrect)."""
    if logits.ndim != 1:
        raise ValueError(f"Expected 1D logits, got shape {tuple(logits.shape)}")
    return float((logits[correct_id] - logits[incorrect_id]).item())
