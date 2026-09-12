"""Forward-hook utilities for caching residual-stream activations."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator

import torch


@dataclass
class ActivationCache:
    """Stores activations keyed by hook name."""

    store: dict[str, torch.Tensor] = field(default_factory=dict)

    def clear(self) -> None:
        self.store.clear()

    def get(self, name: str) -> torch.Tensor:
        if name not in self.store:
            raise KeyError(f"No activation cached for '{name}'")
        return self.store[name]


def _default_layer_modules(model: Any) -> list[tuple[str, torch.nn.Module]]:
    """Best-effort discovery of transformer blocks.

    Supports LLaMA/Gemma-style (``model.layers``), GPT-2 (``transformer.h``),
    GPT-NeoX / Pythia (``gpt_neox.layers``), and toy models with top-level
    ``.layers``.
    """
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return [(f"blocks.{i}", layer) for i, layer in enumerate(model.model.layers)]
    if hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "layers"):
        return [(f"blocks.{i}", layer) for i, layer in enumerate(model.gpt_neox.layers)]
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return [(f"blocks.{i}", layer) for i, layer in enumerate(model.transformer.h)]
    if hasattr(model, "layers"):
        layers = model.layers
        if isinstance(layers, (torch.nn.ModuleList, list)) and len(layers) > 0:
            return [(f"blocks.{i}", layer) for i, layer in enumerate(layers)]
    raise ValueError("Unsupported model architecture for automatic hook attachment")


def attach_hooks(
    model: Any,
    cache: ActivationCache,
    *,
    layers: list[int] | None = None,
    hook_fn: Callable[[str], Callable[..., None]] | None = None,
) -> list[torch.utils.hooks.RemovableHandle]:
    """Attach forward hooks that write residual outputs into ``cache``."""
    modules = _default_layer_modules(model)
    if layers is not None:
        modules = [modules[i] for i in layers]

    def make_hook(name: str) -> Callable[..., None]:
        if hook_fn is not None:
            return hook_fn(name)

        def _hook(_module: torch.nn.Module, _inp: Any, out: Any) -> None:
            tensor = out[0] if isinstance(out, tuple) else out
            cache.store[name] = tensor.detach()

        return _hook

    handles = []
    for name, module in modules:
        handles.append(module.register_forward_hook(make_hook(name)))
    return handles


def remove_hooks(handles: list[torch.utils.hooks.RemovableHandle]) -> None:
    for handle in handles:
        handle.remove()


@contextmanager
def hooked_forward(
    model: Any,
    *,
    layers: list[int] | None = None,
) -> Iterator[ActivationCache]:
    """Context manager that caches activations for a forward pass."""
    cache = ActivationCache()
    handles = attach_hooks(model, cache, layers=layers)
    try:
        yield cache
    finally:
        remove_hooks(handles)
