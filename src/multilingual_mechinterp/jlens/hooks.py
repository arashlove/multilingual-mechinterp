"""Forward hooks that capture residual activations for Jacobian fitting / apply."""

from __future__ import annotations

from typing import Callable, Iterable, Sequence

import torch
import torch.nn as nn


class ActivationRecorder:
    """Context manager that records residual-stream tensors at block indices.

    Captured tensors are **not** detached — they must remain usable as
    ``torch.autograd.grad`` inputs. When ``start_graph_at`` is set, that layer's
    residual is marked ``requires_grad_(True)`` so it becomes the graph leaf
    (with model parameters frozen).
    """

    def __init__(
        self,
        blocks: Sequence[nn.Module],
        at: Iterable[int],
        *,
        start_graph_at: int | None = None,
    ) -> None:
        indices = sorted(set(int(i) for i in at))
        if start_graph_at is not None and start_graph_at not in indices:
            indices = sorted({*indices, int(start_graph_at)})
        self._blocks = blocks
        self._indices = indices
        self._start_graph_at = start_graph_at
        self._handles: list[torch.utils.hooks.RemovableHandle] = []
        self.activations: dict[int, torch.Tensor] = {}

    def _make_hook(self, index: int) -> Callable[..., None]:
        is_graph_root = index == self._start_graph_at

        def hook(_module: nn.Module, _inputs: object, output: object) -> None:
            tensor = output if torch.is_tensor(output) else output[0]  # type: ignore[index]
            if is_graph_root:
                tensor.requires_grad_(True)
            self.activations[index] = tensor

        return hook

    def __enter__(self) -> "ActivationRecorder":
        self.activations = {}
        try:
            for index in self._indices:
                self._handles.append(
                    self._blocks[index].register_forward_hook(self._make_hook(index))
                )
        except Exception:
            for handle in self._handles:
                handle.remove()
            self._handles = []
            raise
        return self

    def __exit__(self, *exc: object) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles = []
