"""Causal-effect metrics for patching / interventions."""

from __future__ import annotations


def effect_size(patched: float, baseline: float) -> float:
    """Absolute change from baseline after an intervention."""
    return float(patched - baseline)


def normalized_effect(patched: float, baseline: float, clean: float) -> float:
    """Normalize patched effect between baseline and clean performance.

    Returns ``(patched - baseline) / (clean - baseline)`` when the denominator
    is non-zero; otherwise ``0.0``.
    """
    denom = clean - baseline
    if abs(denom) < 1e-12:
        return 0.0
    return float((patched - baseline) / denom)
