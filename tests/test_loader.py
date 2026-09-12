"""Tests for model loader helpers (no HF download)."""

from __future__ import annotations

import types

import pytest
import torch

from multilingual_mechinterp.models.loader import LoadedModel, _infer_n_layers, _resolve_dtype


def test_resolve_dtype_aliases():
    assert _resolve_dtype("fp16") is torch.float16
    assert _resolve_dtype("bf16") is torch.bfloat16
    assert _resolve_dtype(torch.float32) is torch.float32


def test_resolve_dtype_invalid():
    with pytest.raises(ValueError):
        _resolve_dtype("float64")


def test_infer_n_layers_from_config():
    model = types.SimpleNamespace(config=types.SimpleNamespace(num_hidden_layers=42))
    assert _infer_n_layers(model) == 42


def test_loaded_model_dataclass():
    lm = LoadedModel(
        name="dummy",
        model=None,
        tokenizer=None,
        device=torch.device("cpu"),
        n_layers=4,
    )
    assert lm.n_layers == 4
    assert lm.name == "dummy"
