"""Tiny causal LM for offline notebook / unit-test demos."""

from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn as nn


class _Block(nn.Module):
    def __init__(self, d: int) -> None:
        super().__init__()
        self.linear = nn.Linear(d, d, bias=False)
        with torch.no_grad():
            self.linear.weight.mul_(0.05)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return hidden + self.linear(hidden)


class _ByteTok:
    def __init__(self, vocab_size: int = 64) -> None:
        self.vocab_size = vocab_size
        self.pad_token_id = 0
        self.eos_token_id = 1

    def __call__(self, texts, return_tensors="pt", padding=True, truncation=True, max_length=128, **kwargs):
        if isinstance(texts, str):
            texts = [texts]
        rows = []
        for t in texts:
            ids = [((ord(c) % (self.vocab_size - 2)) + 2) for c in t][:max_length] or [2]
            rows.append(ids)
        max_len = max(len(r) for r in rows)
        input_ids = torch.zeros(len(rows), max_len, dtype=torch.long)
        attention_mask = torch.zeros(len(rows), max_len, dtype=torch.long)
        for i, r in enumerate(rows):
            input_ids[i, : len(r)] = torch.tensor(r)
            attention_mask[i, : len(r)] = 1
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    def encode(self, text: str, add_special_tokens: bool = False):
        return [((ord(c) % (self.vocab_size - 2)) + 2) for c in text] or [2]

    def decode(self, ids) -> str:
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        if ids and isinstance(ids[0], list):
            ids = ids[0]
        return "".join(chr(32 + (int(i) % 95)) for i in ids)


class TinyCausalLM(nn.Module):
    """Small residual LM with top-level ``.layers`` for Colab demos without HF downloads."""

    def __init__(
        self,
        n_layers: int = 6,
        d_model: int = 32,
        vocab_size: int = 64,
        seed: int = 0,
    ) -> None:
        super().__init__()
        torch.manual_seed(seed)
        self.name = "TinyCausalLM"
        self.n_layers = n_layers
        self.d_model = d_model
        self.tokenizer = _ByteTok(vocab_size)
        self.embed = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([_Block(d_model) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.device = torch.device("cpu")
        self.eval()
        for p in self.parameters():
            p.requires_grad_(False)

    def to(self, device):  # type: ignore[override]
        super().to(device)
        self.device = torch.device(device)
        return self

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        if input_ids is None:
            raise ValueError("input_ids required")
        hidden = self.embed(input_ids)
        for block in self.layers:
            hidden = block(hidden)
        logits = self.lm_head(self.norm(hidden))
        return SimpleNamespace(logits=logits)
