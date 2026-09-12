"""LensModel protocol and TinyDecoder for closed-form Jacobian tests."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Protocol, Sequence, runtime_checkable

import torch
import torch.nn as nn


@runtime_checkable
class LensModel(Protocol):
    """Contract required by fitting and ``JacobianLens.apply``."""

    n_layers: int
    d_model: int
    layers: Sequence[nn.Module]
    tokenizer: Any

    def encode(self, text: str, *, max_length: int = 128) -> torch.Tensor:
        """Return ``input_ids`` shaped ``[1, seq_len]`` on the model device."""
        ...

    def forward(self, input_ids: torch.Tensor) -> Any:
        """Run the residual stack (no LM head). Must be batch-deterministic."""
        ...

    def unembed(self, residual: torch.Tensor) -> torch.Tensor:
        """Map residual ``[..., d_model]`` → logits ``[..., vocab]``."""
        ...


@dataclass
class _Encoded:
    input_ids: torch.Tensor


class _ByteTokenizer:
    """Minimal tokenizer for TinyDecoder unit tests."""

    def __init__(self, vocab_size: int = 32) -> None:
        self.vocab_size = vocab_size

    def __call__(self, text: str, *, max_length: int = 128) -> _Encoded:
        ids = [((ord(c) % (self.vocab_size - 1)) + 1) for c in text][:max_length]
        if not ids:
            ids = [1]
        return _Encoded(input_ids=torch.tensor([ids], dtype=torch.long))

    def decode(self, token_ids: list[int] | torch.Tensor) -> str:
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        if token_ids and isinstance(token_ids[0], list):
            token_ids = token_ids[0]
        return "".join(chr(32 + (int(t) % 95)) for t in token_ids)


class _ResidualBlock(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.linear = nn.Linear(d_model, d_model, bias=False)
        with torch.no_grad():
            self.linear.weight.mul_(0.1)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return hidden + self.linear(hidden)


class TinyDecoder(nn.Module):
    """Toy residual decoder with closed-form late-layer Jacobian ``I + W``."""

    def __init__(
        self,
        n_layers: int = 4,
        d_model: int = 8,
        vocab_size: int = 32,
        seed: int = 0,
    ) -> None:
        super().__init__()
        torch.manual_seed(seed)
        self.n_layers = n_layers
        self.d_model = d_model
        self.tokenizer = _ByteTokenizer(vocab_size)
        self.embed_tokens = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([_ResidualBlock(d_model) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.eval()
        for p in self.parameters():
            p.requires_grad_(False)

    def encode(self, text: str, *, max_length: int = 128) -> torch.Tensor:
        return self.tokenizer(text, max_length=max_length).input_ids.to(
            self.embed_tokens.weight.device
        )

    def forward(self, input_ids: torch.Tensor) -> SimpleNamespace:
        hidden = self.embed_tokens(input_ids)
        for block in self.layers:
            hidden = block(hidden)
        return SimpleNamespace(last_hidden_state=hidden)

    def unembed(self, residual: torch.Tensor) -> torch.Tensor:
        return self.lm_head(self.norm(residual.float()))
