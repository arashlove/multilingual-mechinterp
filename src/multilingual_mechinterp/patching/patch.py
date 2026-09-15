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
    metric: str | Any = "prob",
    option_letters: Sequence[str] | None = None,
) -> PatchResult:
    """Patch source residuals into the target run and score by layer.

    Parameters
    ----------
    answer:
        Answer string(s) or token id(s). For culture MCQs prefer the correct
        **option letter** (``\"C\"``), not the answer text (``\"Divan-e Hafez\"``).
    metric:
        ``\"prob\"`` (default): ΔP(answer).
        ``\"margin\"``: Δ(logit_correct − max logit_wrong) over A–D (or
        ``option_letters``). Requires ``answer`` to be the correct letter.
    option_letters:
        Letters used for margin scoring (default ``ABCD``).
    """
    if not isinstance(model, LoadedModel) and not hasattr(model, "tokenizer"):
        raise ValueError("model must be a LoadedModel (or expose .tokenizer)")

    if answer is None:
        approx = source_prompt.strip().split()[-1] if source_prompt.strip() else "a"
        answer = approx

    tokenizer = model.tokenizer if isinstance(model, LoadedModel) else model.tokenizer
    if answer and isinstance(answer, (list, tuple)) and answer and isinstance(answer[0], int):
        y_ids = [int(x) for x in answer]  # type: ignore[arg-type]
    else:
        y_ids = answer_token_ids(tokenizer, answer)  # type: ignore[arg-type]

    metric_name = metric if isinstance(metric, str) else "prob"

    if metric_name == "margin":
        from multilingual_mechinterp.data.mcq import (
            margin_from_probs,
            option_token_ids,
        )

        letters = list(option_letters) if option_letters is not None else list("ABCD")
        opt_map = option_token_ids(tokenizer, letters)
        opt_ids = [opt_map[L] for L in letters]
        # correct letter must be in answer / y_ids
        correct_id = y_ids[0]
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
        # Re-score from full vocab probs → margin
        probs = curve["probs"]  # [B, n_layers, V]
        # baseline from unpatched: reconstruct via first patched? use dedicated forward
        # causal_effect_curve baseline is P(answer); recompute margins
        B, nL, V = probs.shape
        patched_m = torch.stack(
            [
                margin_from_probs(probs[:, L, :], correct_id=correct_id, option_ids=opt_ids)
                for L in range(nL)
            ],
            dim=1,
        )  # [B, nL]

        # baseline margin: run target once
        from multilingual_mechinterp.data.mcq import score_mcq_prompt

        base_info = score_mcq_prompt(
            model, target_prompt, correct_letter=str(answer).strip()[0].upper(), device=device
        )
        baseline = float(base_info["margin"])
        effect = patched_m - baseline
        best = int(effect[0].argmax().item())
        layer_ids = list(range(nL))
        if layers is not None:
            layer_ids = [L for L in layers if 0 <= L < nL]
        scores = {L: float(effect[0, L].item()) for L in layer_ids}
        patched_probs = {L: float(patched_m[0, L].item()) for L in layer_ids}
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
                "metric": "margin",
                "option_ids": opt_map,
                "baseline_mcq": base_info,
                "window": window,
                "source_pos": source_pos,
                "target_pos": target_pos,
            },
        )

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
            "metric": "prob",
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
