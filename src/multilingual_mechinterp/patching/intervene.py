"""Phase 2 — overwrite residuals during a target forward pass."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator, Mapping, Sequence

import torch

from multilingual_mechinterp.patching.collect import get_blocks


@contextmanager
def patch_residuals(
    model: torch.nn.Module,
    patch_map: Mapping[int, torch.Tensor],
    positions: torch.Tensor,
) -> Iterator[None]:
    """Overwrite residual positions during forward.

    ``patch_map``: ``{layer_idx: Tensor[B, d]}`` values to write.
    ``positions``: ``Tensor[B]`` absolute token indices on the **target** sequence.
    """
    blocks = get_blocks(model)
    handles: list[torch.utils.hooks.RemovableHandle] = []

    def make_hook(layer_idx: int, values: torch.Tensor):
        def hook(_module, _inp, out):
            is_tuple = isinstance(out, tuple)
            h = out[0] if is_tuple else out
            h = h.clone()
            b = torch.arange(h.size(0), device=h.device)
            pos = positions.to(h.device)
            h[b, pos] = values.to(device=h.device, dtype=h.dtype)
            return (h, *out[1:]) if is_tuple else h

        return hook

    try:
        for layer_idx, values in patch_map.items():
            handles.append(
                blocks[layer_idx].register_forward_hook(make_hook(int(layer_idx), values))
            )
        yield
    finally:
        for handle in handles:
            handle.remove()


def next_token_probs_from_logits(
    logits: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Softmax next-token distribution at the last real token. ``[B, V]``."""
    from multilingual_mechinterp.patching.indexing import last_token_index

    last = last_token_index(attention_mask)
    b = torch.arange(logits.size(0), device=logits.device)
    return logits[b, last].float().softmax(-1)


def answer_prob(probs: torch.Tensor, y_ids: Sequence[int]) -> torch.Tensor:
    """Sum probability mass over an answer token-id set.

    ``probs``: ``[B, V]`` or ``[B, n_layers, V]``.
    """
    ids = list(y_ids)
    return probs[..., ids].sum(dim=-1)
