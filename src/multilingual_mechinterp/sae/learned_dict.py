"""Shared dictionary-learning interface (paper / sparse_coding style)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


class LearnedDict(ABC):
    """Common encode / decode / predict interface for SAE and baselines."""

    n_feats: int
    activation_size: int

    @abstractmethod
    def get_learned_dict(self) -> torch.Tensor:
        """Return dictionary matrix shaped ``[n_feats, activation_size]``."""

    @abstractmethod
    def encode(self, batch: torch.Tensor) -> torch.Tensor:
        """Encode centered (or raw) activations to feature codes."""

    @abstractmethod
    def to_device(self, device: str | torch.device) -> "LearnedDict":
        ...

    def decode(self, code: torch.Tensor) -> torch.Tensor:
        learned_dict = self.get_learned_dict()
        return torch.einsum("nd,bn->bd", learned_dict, code)

    def center(self, batch: torch.Tensor) -> torch.Tensor:
        return batch

    def uncenter(self, batch: torch.Tensor) -> torch.Tensor:
        return batch

    def predict(self, batch: torch.Tensor) -> torch.Tensor:
        batch_centered = self.center(batch)
        code = self.encode(batch_centered)
        x_hat_centered = self.decode(code)
        return self.uncenter(x_hat_centered)

    def n_dict_components(self) -> int:
        return int(self.get_learned_dict().shape[0])


class TiedSAE(LearnedDict, nn.Module):
    """Tied-weight SAE from Cunningham et al. (ICLR 2024).

    Forward (paper Eqs. 1–4)::

        c = ReLU(M x + b)
        x_hat = M^T c

    with L2-normalized dictionary rows of ``M``.
    """

    def __init__(
        self,
        d_model: int,
        d_sae: int,
        *,
        alpha: float = 1e-3,
        encoder: torch.Tensor | None = None,
        encoder_bias: torch.Tensor | None = None,
        norm_encoder: bool = True,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_sae = d_sae
        self.activation_size = d_model
        self.n_feats = d_sae
        self.alpha = float(alpha)
        self.norm_encoder = norm_encoder

        if encoder is None:
            enc = torch.empty(d_sae, d_model)
            nn.init.xavier_uniform_(enc)
        else:
            enc = encoder.detach().clone()
        self.encoder = nn.Parameter(enc)

        if encoder_bias is None:
            bias = torch.zeros(d_sae)
        else:
            bias = encoder_bias.detach().clone()
        self.encoder_bias = nn.Parameter(bias)

    def get_learned_dict(self) -> torch.Tensor:
        if not self.norm_encoder:
            return self.encoder
        norms = torch.norm(self.encoder, 2, dim=-1)
        return self.encoder / torch.clamp(norms, min=1e-8)[:, None]

    def encode(self, batch: torch.Tensor) -> torch.Tensor:
        if batch.ndim == 1:
            batch = batch.unsqueeze(0)
        learned = self.get_learned_dict()
        code = torch.einsum("nd,bd->bn", learned, batch)
        code = code + self.encoder_bias
        return torch.clamp(code, min=0.0)

    def decode(self, code: torch.Tensor) -> torch.Tensor:
        if code.ndim == 1:
            code = code.unsqueeze(0)
        return torch.einsum("nd,bn->bd", self.get_learned_dict(), code)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        squeeze = x.ndim == 1
        if squeeze:
            x = x.unsqueeze(0)
        code = self.encode(x)
        recon = self.decode(code)
        if squeeze:
            return code.squeeze(0), recon.squeeze(0)
        return code, recon

    def loss(self, batch: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Paper loss: mean squared error + α ‖c‖₁ (batch-averaged)."""
        code = self.encode(batch)
        recon = self.decode(code)
        l_recon = (recon - batch).pow(2).mean()
        l_l1 = self.alpha * torch.norm(code, 1, dim=-1).mean()
        total = l_recon + l_l1
        return total, {"loss": total, "l_reconstruction": l_recon, "l_l1": l_l1, "c": code}

    def to_device(self, device: str | torch.device) -> "TiedSAE":
        return self.to(device)

    def state_dict_sae(self) -> dict[str, Any]:
        return {
            "d_model": self.d_model,
            "d_sae": self.d_sae,
            "alpha": self.alpha,
            "norm_encoder": self.norm_encoder,
            "encoder": self.encoder.detach().cpu(),
            "encoder_bias": self.encoder_bias.detach().cpu(),
        }

    @classmethod
    def from_state_dict_sae(cls, state: dict[str, Any], *, device: str | torch.device = "cpu") -> "TiedSAE":
        sae = cls(
            d_model=int(state["d_model"]),
            d_sae=int(state["d_sae"]),
            alpha=float(state.get("alpha", 1e-3)),
            encoder=state["encoder"],
            encoder_bias=state["encoder_bias"],
            norm_encoder=bool(state.get("norm_encoder", True)),
        )
        return sae.to(device)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict_sae(), path)
        return path

    @classmethod
    def load(cls, path: str | Path, *, device: str | torch.device = "cpu") -> "TiedSAE":
        state = torch.load(Path(path), map_location=device, weights_only=True)
        return cls.from_state_dict_sae(state, device=device)
