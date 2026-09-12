"""Layer-window activation patching lens (causal effect curves)."""

from __future__ import annotations

from typing import Sequence

import torch

from multilingual_mechinterp.models.loader import LoadedModel
from multilingual_mechinterp.patching.collect import collect_activations, get_blocks
from multilingual_mechinterp.patching.indexing import resolve_positions
from multilingual_mechinterp.patching.intervene import (
    answer_prob,
    next_token_probs_from_logits,
    patch_residuals,
)


@torch.no_grad()
def patch_lens(
    model: LoadedModel | torch.nn.Module,
    target_prompts: str | Sequence[str],
    source_acts: torch.Tensor,
    *,
    target_pos: int | None = -1,
    window: int | None = None,
    device: str | torch.device | None = None,
) -> torch.Tensor:
    """Sweep patch start-layer; return next-token probs ``[B, n_layers, V]``.

    For each start layer ``s``, overwrites residuals at layers
    ``[s, s+window)`` with ``source_acts[L]`` (same-layer index), then reads
    softmax at the last real target token.
    """
    if isinstance(target_prompts, str):
        target_prompts = [target_prompts]

    if isinstance(model, LoadedModel):
        hf_model = model.model
        tokenizer = model.tokenizer
        run_device = torch.device(device or model.device)
        n_layers = model.n_layers
    else:
        hf_model = model
        tokenizer = model.tokenizer
        run_device = torch.device(device or next(hf_model.parameters()).device)
        n_layers = len(get_blocks(hf_model))

    if source_acts.ndim != 3:
        raise ValueError(f"source_acts must be [n_layers, B, d], got {tuple(source_acts.shape)}")
    if source_acts.shape[0] != n_layers:
        raise ValueError(
            f"source_acts has {source_acts.shape[0]} layers but model has {n_layers}"
        )
    if source_acts.shape[1] != len(target_prompts):
        raise ValueError("batch size of source_acts must match number of target prompts")

    if window is None:
        window = n_layers

    tok = tokenizer(list(target_prompts), return_tensors="pt", padding=True)
    tok = {k: v.to(run_device) for k, v in tok.items()}
    attention_mask = tok.get("attention_mask")
    if attention_mask is None:
        attention_mask = torch.ones_like(tok["input_ids"])
    positions = resolve_positions(attention_mask, target_pos)

    all_probs: list[torch.Tensor] = []
    for start in range(n_layers):
        patch_map = {
            L: source_acts[L]
            for L in range(start, min(start + window, n_layers))
        }
        with patch_residuals(hf_model, patch_map, positions):
            try:
                out = hf_model(**tok)
            except TypeError:
                out = hf_model(tok["input_ids"])
            logits = out.logits if hasattr(out, "logits") else out
        probs = next_token_probs_from_logits(logits, attention_mask).cpu()
        all_probs.append(probs)

    return torch.stack(all_probs, dim=1)  # [B, n_layers, V]


@torch.no_grad()
def causal_effect_curve(
    model: LoadedModel | torch.nn.Module,
    source_prompts: str | Sequence[str],
    target_prompts: str | Sequence[str],
    answer_ids: Sequence[int],
    *,
    source_pos: int | None = -1,
    target_pos: int | None = -1,
    window: int | None = None,
    device: str | torch.device | None = None,
) -> dict[str, torch.Tensor]:
    """Full pipeline: cache source → patch lens → ΔP(answer) vs start layer.

    Returns dict with:
      - ``probs``: ``[B, n_layers, V]`` patched next-token probs
      - ``baseline``: ``[B]`` unpatched P(answer | target)
      - ``patched``: ``[B, n_layers]`` P(answer | patched target)
      - ``effect``: ``[B, n_layers]`` patched − baseline
      - ``best_layer``: ``[B]`` start layer with max effect
    """
    if isinstance(source_prompts, str):
        source_prompts = [source_prompts]
    if isinstance(target_prompts, str):
        target_prompts = [target_prompts]
    if len(source_prompts) != len(target_prompts):
        raise ValueError("source_prompts and target_prompts must have the same length")

    source_acts = collect_activations(
        model, source_prompts, pos_idx=source_pos, device=device
    )
    probs = patch_lens(
        model,
        target_prompts,
        source_acts,
        target_pos=target_pos,
        window=window,
        device=device,
    )
    patched = answer_prob(probs, answer_ids)  # [B, n_layers]

    # unpatched baseline
    if isinstance(model, LoadedModel):
        hf_model = model.model
        tokenizer = model.tokenizer
        run_device = torch.device(device or model.device)
    else:
        hf_model = model
        tokenizer = model.tokenizer
        run_device = torch.device(device or next(hf_model.parameters()).device)

    tok = tokenizer(list(target_prompts), return_tensors="pt", padding=True)
    tok = {k: v.to(run_device) for k, v in tok.items()}
    attention_mask = tok.get("attention_mask")
    if attention_mask is None:
        attention_mask = torch.ones_like(tok["input_ids"])
    with torch.no_grad():
        try:
            out = hf_model(**tok)
        except TypeError:
            out = hf_model(tok["input_ids"])
        logits = out.logits if hasattr(out, "logits") else out
        baseline_probs = next_token_probs_from_logits(logits, attention_mask).cpu()
    baseline = answer_prob(baseline_probs, answer_ids)  # [B]

    effect = patched - baseline.unsqueeze(1)
    best_layer = effect.argmax(dim=1)

    return {
        "probs": probs,
        "baseline": baseline,
        "patched": patched,
        "effect": effect,
        "best_layer": best_layer,
    }
