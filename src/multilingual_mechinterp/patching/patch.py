"""High-level activation-patching API."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import torch

from multilingual_mechinterp.models.loader import LoadedModel
from multilingual_mechinterp.patching.indexing import answer_token_ids
from multilingual_mechinterp.patching.lens import causal_effect_curve, patch_lens


@dataclass
class PatchResult:
    """Layer-wise causal effects from activation patching."""

    source_prompt: str
    target_prompt: str
    layers: list[int]
    scores: dict[int, float] = field(default_factory=dict)
    """Per start-layer mean causal effect ΔP(answer)."""
    baseline_score: float | None = None
    """Unpatched P(answer | target)."""
    best_layer: int | None = None
    """Start layer with largest causal effect (for choosing swap sites)."""
    patched_probs: dict[int, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


def run_patching(
    model: LoadedModel | Any,
    source_prompt: str,
    target_prompt: str,
    *,
    answer: str | Sequence[str] | Sequence[int] | None = None,
    source_pos: int | None = -1,
    target_pos: int | None = -1,
    window: int | None = None,
    layers: list[int] | None = None,
    device: str | None = None,
    metric: Any = None,
) -> PatchResult:
    """Patch source residuals into the target run and score ΔP(answer) by layer.

    This is the real intervention loop from ``docs/activation_patching_reimpl/``:
    cache source activations → overwrite target residuals in a start-layer
    window sweep → read next-token probs.

    Parameters
    ----------
    answer:
        Answer string(s) or precomputed token id(s) whose probability mass is
        the metric. Defaults to the first token of ``source_prompt``'s last word
        when omitted (smoke-test convenience only).
    target_pos / source_pos:
        Relative token indices (``-1`` = last real token). Object patching uses
        a more specific negative index for the object span.
    window:
        Number of consecutive layers to patch from each start (``None`` = through top).
    layers:
        If set, only those start layers appear in ``scores`` (still runs full sweep
        unless you pass a custom path later).
    metric:
        Ignored (kept for API compatibility with the old residual-diff stub).
    """
    del metric
    if not isinstance(model, LoadedModel) and not hasattr(model, "tokenizer"):
        raise ValueError("model must be a LoadedModel (or expose .tokenizer)")

    if answer is None:
        # Weak default: last whitespace-separated token of the source string
        approx = source_prompt.strip().split()[-1] if source_prompt.strip() else "a"
        answer = approx

    tokenizer = model.tokenizer if isinstance(model, LoadedModel) else model.tokenizer
    if answer and isinstance(answer, (list, tuple)) and answer and isinstance(answer[0], int):
        y_ids = [int(x) for x in answer]  # type: ignore[arg-type]
    else:
        y_ids = answer_token_ids(tokenizer, answer)  # type: ignore[arg-type]

    curve = causal_effect_curve(
        model,
        source_prompt,
        target_prompt,
        y_ids,
        source_pos=source_pos,
        target_pos=target_pos,
        window=window,
        device=device,
    )

    effect = curve["effect"][0]  # [n_layers]
    patched = curve["patched"][0]
    baseline = float(curve["baseline"][0].item())
    n_layers = int(effect.numel())
    layer_ids = list(range(n_layers))
    if layers is not None:
        layer_ids = [L for L in layers if 0 <= L < n_layers]

    scores = {L: float(effect[L].item()) for L in layer_ids}
    patched_probs = {L: float(patched[L].item()) for L in layer_ids}
    best = int(curve["best_layer"][0].item())

    return PatchResult(
        source_prompt=source_prompt,
        target_prompt=target_prompt,
        layers=layer_ids,
        scores=scores,
        baseline_score=baseline,
        best_layer=best,
        patched_probs=patched_probs,
        metadata={
            "model": getattr(model, "name", type(model).__name__),
            "answer_ids": y_ids,
            "window": window,
            "source_pos": source_pos,
            "target_pos": target_pos,
            "effect_all_layers": [float(x) for x in effect.tolist()],
        },
    )


def recommend_layers(
    result: PatchResult,
    *,
    top_k: int = 3,
    min_effect: float = 0.0,
) -> list[int]:
    """Pick start layers with the strongest positive causal effect."""
    ranked = sorted(result.scores.items(), key=lambda kv: kv[1], reverse=True)
    return [L for L, score in ranked if score > min_effect][:top_k]
