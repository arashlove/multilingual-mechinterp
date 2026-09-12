"""High-level JLens helpers used by notebooks / ``analyze``."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import torch

from multilingual_mechinterp.jlens.hf import HFLensModel, from_hf
from multilingual_mechinterp.jlens.lens import JacobianLens
from multilingual_mechinterp.jlens.protocol import LensModel
from multilingual_mechinterp.models.loader import LoadedModel


@dataclass
class JLensResult:
    """Layer-wise vocabulary projections from the Jacobian (or logit) lens."""

    prompt: str
    layers: list[int]
    top_tokens: dict[int, list[dict[str, Any]]] = field(default_factory=dict)
    logits: dict[int, torch.Tensor] = field(default_factory=dict)
    model_logits: torch.Tensor | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def as_lens_model(model: LensModel | LoadedModel | HFLensModel) -> LensModel:
    """Coerce a LoadedModel into an HFLensModel when needed."""
    if isinstance(model, LoadedModel):
        return from_hf(model.model, model.tokenizer)
    return model  # type: ignore[return-value]


def run_jlens(
    model: LensModel | LoadedModel | Any,
    prompt: str,
    *,
    layers: list[int] | None = None,
    top_k: int = 5,
    positions: Sequence[int] | None = (-1,),
    lens: JacobianLens | None = None,
    lens_path: str | Path | None = None,
    use_jacobian: bool = True,
    max_seq_len: int = 512,
    device: str | None = None,
) -> JLensResult:
    """Apply a fitted Jacobian lens (or identity / logit-lens baseline).

    Parameters
    ----------
    lens / lens_path:
        Fitted ``JacobianLens``. If both are omitted and ``use_jacobian`` is
        True, an identity transport is used (equivalent to logit lens for the
        requested layers). Pass a real fitted lens for paper-faithful readouts.
    positions:
        Token positions to score (default: last token). ``None`` = all positions.
    """
    del device  # LensModel carries its own device via parameters / embeddings
    lens_model = as_lens_model(model)

    if lens is None and lens_path is not None:
        lens = JacobianLens.load(str(lens_path))
    if lens is None:
        use_jacobian = False
        layer_ids = layers if layers is not None else list(range(lens_model.n_layers - 1))
        lens = JacobianLens.identity(
            n_layers=lens_model.n_layers,
            d_model=lens_model.d_model,
            layers=layer_ids,
        )

    lens_logits, model_logits, input_ids = lens.apply(
        lens_model,
        prompt,
        layers=layers,
        positions=positions,
        max_seq_len=max_seq_len,
        use_jacobian=use_jacobian,
    )

    tokenizer = lens_model.tokenizer
    top_tokens: dict[int, list[dict[str, Any]]] = {}
    logits_cpu: dict[int, torch.Tensor] = {}
    for layer, logits in lens_logits.items():
        # logits: [n_pos, vocab] — use last selected position for ranking
        row = logits[-1] if logits.ndim == 2 else logits
        logits_cpu[layer] = row
        values, indices = torch.topk(row, k=min(top_k, row.numel()))
        top_tokens[layer] = [
            {
                "token_id": int(idx),
                "token": tokenizer.decode([idx]),
                "logit": float(val),
            }
            for idx, val in zip(indices.tolist(), values.tolist())
        ]

    return JLensResult(
        prompt=prompt,
        layers=sorted(lens_logits),
        top_tokens=top_tokens,
        logits=logits_cpu,
        model_logits=model_logits[-1] if model_logits.ndim == 2 else model_logits,
        metadata={
            "use_jacobian": use_jacobian,
            "n_prompts_fit": lens.n_prompts,
            "seq_len": int(input_ids.shape[-1]),
        },
    )
