"""Phase 1 — collect residual activations at chosen token positions."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator, Sequence

import torch

from multilingual_mechinterp.models.hooks import _default_layer_modules
from multilingual_mechinterp.models.loader import LoadedModel
from multilingual_mechinterp.patching.indexing import resolve_positions


def get_blocks(model: torch.nn.Module) -> list[torch.nn.Module]:
    """Return residual blocks in layer order."""
    # Tiny / custom models with top-level .layers
    if hasattr(model, "layers") and not hasattr(model, "model") and not hasattr(model, "gpt_neox"):
        layers = getattr(model, "layers")
        if isinstance(layers, (torch.nn.ModuleList, list)) and len(layers) > 0:
            return list(layers)
    return [module for _, module in _default_layer_modules(model)]


@contextmanager
def capture_residuals(
    model: torch.nn.Module,
    layers: Sequence[int],
    positions: torch.Tensor,
) -> Iterator[dict[int, torch.Tensor]]:
    """Capture residual ``[B, d]`` at ``positions`` for each layer during forward."""
    blocks = get_blocks(model)
    cache: dict[int, torch.Tensor] = {}
    handles: list[torch.utils.hooks.RemovableHandle] = []

    def make_hook(layer_idx: int):
        def hook(_module, _inp, out):
            h = out[0] if isinstance(out, tuple) else out  # [B, T, d]
            b = torch.arange(h.size(0), device=h.device)
            pos = positions.to(h.device)
            cache[layer_idx] = h[b, pos].detach()
            return out

        return hook

    try:
        for layer_idx in layers:
            handles.append(blocks[layer_idx].register_forward_hook(make_hook(layer_idx)))
        yield cache
    finally:
        for handle in handles:
            handle.remove()


@torch.no_grad()
def collect_activations(
    model: LoadedModel | torch.nn.Module,
    prompts: str | Sequence[str],
    *,
    layers: Sequence[int] | None = None,
    pos_idx: int | None = -1,
    device: str | torch.device | None = None,
) -> torch.Tensor:
    """Collect residual activations shaped ``[n_layers, batch, d]``.

    ``pos_idx`` is a relative index (``None`` / ``-1`` = last real token).
    """
    if isinstance(prompts, str):
        prompts = [prompts]

    if isinstance(model, LoadedModel):
        hf_model = model.model
        tokenizer = model.tokenizer
        run_device = torch.device(device or model.device)
        n_layers = model.n_layers
    else:
        hf_model = model
        tokenizer = getattr(model, "tokenizer", None)
        if tokenizer is None:
            raise ValueError("model must be LoadedModel or expose .tokenizer")
        run_device = torch.device(device or next(hf_model.parameters()).device)
        n_layers = len(get_blocks(hf_model))

    if layers is None:
        layers = list(range(n_layers))

    tok = tokenizer(list(prompts), return_tensors="pt", padding=True)
    tok = {k: v.to(run_device) for k, v in tok.items()}
    attention_mask = tok.get("attention_mask")
    if attention_mask is None:
        attention_mask = torch.ones_like(tok["input_ids"])
    positions = resolve_positions(attention_mask, pos_idx)

    with capture_residuals(hf_model, layers, positions) as cache:
        if isinstance(model, LoadedModel):
            hf_model(**tok)
        else:
            # Tiny / custom models may take input_ids only
            try:
                hf_model(**tok)
            except TypeError:
                hf_model(tok["input_ids"])

    stacked = torch.stack([cache[L].cpu().float() for L in layers], dim=0)
    return stacked


# Back-compat name used by earlier scaffolding
def cache_activations(
    model: LoadedModel,
    prompt: str,
    *,
    layers: list[int] | None = None,
    device: str | torch.device | None = None,
):
    """Deprecated wrapper: prefer :func:`collect_activations`."""
    from multilingual_mechinterp.models.hooks import ActivationCache

    acts = collect_activations(model, prompt, layers=layers, device=device)
    layer_ids = layers if layers is not None else list(range(acts.shape[0]))
    store = {f"blocks.{L}": acts[i].unsqueeze(0) for i, L in enumerate(layer_ids)}
    # restore a fake [1, 1, d] last-token view for old callers expecting full seq
    return ActivationCache(store={k: v for k, v in store.items()})
