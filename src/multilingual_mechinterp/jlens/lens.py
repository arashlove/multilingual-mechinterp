"""JacobianLens: transport, apply, persist, merge."""

from __future__ import annotations

import os
from typing import Sequence

import torch

from multilingual_mechinterp.jlens.hooks import ActivationRecorder
from multilingual_mechinterp.jlens.protocol import LensModel


class JacobianLens:
    """Fitted layer-wise Jacobians for residual → final-space transport."""

    def __init__(
        self,
        jacobians: dict[int, torch.Tensor],
        *,
        n_prompts: int,
        d_model: int,
    ) -> None:
        self.jacobians = {int(layer): J.float() for layer, J in jacobians.items()}
        self.source_layers = sorted(self.jacobians)
        self.n_prompts = int(n_prompts)
        self.d_model = int(d_model)

    def transport(self, residual: torch.Tensor, layer: int) -> torch.Tensor:
        """Apply ``h @ J.T`` (rows of ``J`` are ∂out/∂in)."""
        j_bar = self.jacobians[layer].to(device=residual.device, dtype=residual.dtype)
        return residual @ j_bar.T

    @torch.no_grad()
    def apply(
        self,
        model: LensModel,
        prompt: str,
        *,
        layers: Sequence[int] | None = None,
        positions: Sequence[int] | None = None,
        max_seq_len: int = 512,
        use_jacobian: bool = True,
    ) -> tuple[dict[int, torch.Tensor], torch.Tensor, torch.Tensor]:
        """Return ``(lens_logits, model_logits, input_ids)``.

        ``lens_logits[layer]`` and ``model_logits`` are CPU float tensors with
        shape ``[n_positions, vocab]``.
        """
        if layers is None:
            layers = list(self.source_layers)
        else:
            layers = list(layers)

        for layer in layers:
            if not (0 <= layer < model.n_layers):
                raise ValueError(f"layer {layer} out of range for n_layers={model.n_layers}")
            if use_jacobian and layer not in self.jacobians:
                raise ValueError(
                    f"layer {layer} not in fitted source_layers {self.source_layers}"
                )

        final_layer = model.n_layers - 1
        record_at = sorted(set(layers) | {final_layer})

        input_ids = model.encode(prompt, max_length=max_seq_len)
        with ActivationRecorder(model.layers, at=record_at) as recorder:
            model.forward(input_ids)
            activations = {i: recorder.activations[i].detach() for i in record_at}

        def select(layer: int) -> torch.Tensor:
            full = activations[layer][0]  # [seq, d]
            if positions is None:
                return full.float()
            idx = [p if p >= 0 else full.shape[0] + p for p in positions]
            return full[idx].float()

        lens_logits: dict[int, torch.Tensor] = {}
        for layer in layers:
            residual = select(layer)
            if use_jacobian:
                residual = self.transport(residual, layer)
            lens_logits[layer] = model.unembed(residual).float().cpu()

        model_logits = model.unembed(select(final_layer)).float().cpu()
        return lens_logits, model_logits, input_ids.cpu()

    def save(self, path: str, *, dtype: torch.dtype = torch.float16) -> None:
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        torch.save(
            {
                "J": {layer: J.to(dtype) for layer, J in self.jacobians.items()},
                "n_prompts": self.n_prompts,
                "source_layers": self.source_layers,
                "d_model": self.d_model,
            },
            path,
        )

    @classmethod
    def load(cls, path: str) -> "JacobianLens":
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if "J" not in checkpoint:
            raise ValueError(
                "checkpoint missing key 'J' — this looks like a fit() running "
                "checkpoint (jacobian_sum), not a saved lens"
            )
        return cls(
            jacobians=checkpoint["J"],
            n_prompts=int(checkpoint["n_prompts"]),
            d_model=int(checkpoint["d_model"]),
        )

    @classmethod
    def from_pretrained(
        cls,
        name_or_path: str,
        *,
        filename: str = "lens.pt",
        revision: str | None = None,
    ) -> "JacobianLens":
        if os.path.isfile(name_or_path):
            return cls.load(name_or_path)
        if not os.path.isdir(name_or_path):
            from huggingface_hub import snapshot_download

            name_or_path = snapshot_download(
                name_or_path, allow_patterns=[filename], revision=revision
            )
        return cls.load(os.path.join(name_or_path, filename))

    @classmethod
    def merge(cls, lenses: Sequence["JacobianLens"]) -> "JacobianLens":
        if not lenses:
            raise ValueError("merge() requires at least one lens")
        first = lenses[0]
        for lens in lenses[1:]:
            if lens.source_layers != first.source_layers:
                raise ValueError("all lenses must share the same source_layers")
            if lens.d_model != first.d_model:
                raise ValueError("all lenses must share the same d_model")
        n_total = sum(lens.n_prompts for lens in lenses)
        if n_total <= 0:
            raise ValueError("total n_prompts must be positive")
        merged: dict[int, torch.Tensor] = {}
        for layer in first.source_layers:
            weighted = sum(
                (lens.jacobians[layer] * lens.n_prompts for lens in lenses),
                start=torch.zeros_like(first.jacobians[layer]),
            )
            merged[layer] = weighted / n_total
        return cls(jacobians=merged, n_prompts=n_total, d_model=first.d_model)

    @classmethod
    def identity(
        cls,
        *,
        n_layers: int,
        d_model: int,
        layers: Sequence[int] | None = None,
    ) -> "JacobianLens":
        """Utility lens with ``I`` at each layer (logit-lens / smoke tests)."""
        if layers is None:
            layers = list(range(max(n_layers - 1, 1)))
        eye = torch.eye(d_model, dtype=torch.float32)
        return cls(
            jacobians={int(layer): eye.clone() for layer in layers},
            n_prompts=0,
            d_model=d_model,
        )
