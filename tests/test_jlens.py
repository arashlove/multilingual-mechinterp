"""Jacobian Lens property tests (docs/jlensreimplement/06-checklist.md)."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from multilingual_mechinterp.jlens import (
    ActivationRecorder,
    JacobianLens,
    TinyDecoder,
    fit,
    jacobian_for_prompt,
    run_jlens,
    valid_position_mask,
)
from multilingual_mechinterp.jlens.intervention import intervene_jlens
from multilingual_mechinterp.jlens.readout import decode_residual, project_to_vocab
import torch.nn as nn


def test_valid_position_mask():
    mask = valid_position_mask(20, skip_first=16)
    assert mask.shape == (20,)
    assert not mask[:16].any()
    assert not mask[-1]
    assert mask[16:-1].all()


def test_valid_position_mask_too_short():
    with pytest.raises(ValueError, match="too short"):
        valid_position_mask(17, skip_first=16)


def test_activation_recorder_grad_leaf():
    model = TinyDecoder(n_layers=3, d_model=4, seed=0)
    ids = model.encode("abcdef", max_length=32)
    ids = ids.expand(2, -1)
    with ActivationRecorder(model.layers, at=[0, 1, 2], start_graph_at=0) as rec:
        model.forward(ids)
        src = rec.activations[0]
        tgt = rec.activations[2]
        assert src.requires_grad
        g = torch.autograd.grad(tgt.sum(), src, retain_graph=False)[0]
        assert g.shape == src.shape


def test_exact_late_jacobian_tiny_decoder():
    """On TinyDecoder, J_{L-2→L-1} should match I + W_last."""
    model = TinyDecoder(n_layers=4, d_model=8, seed=1)
    # Long enough for skip_first=0
    prompt = "abcdefghijklmnop"  # 16 chars
    Js, seq_len, n_valid = jacobian_for_prompt(
        model,
        prompt,
        source_layers=[2],
        target_layer=3,
        dim_batch=8,
        max_seq_len=64,
        skip_first=0,
    )
    assert seq_len >= 2
    assert n_valid >= 1
    W = model.layers[3].linear.weight.detach().cpu().float()
    expected = torch.eye(8) + W
    assert torch.allclose(Js[2], expected, atol=1e-4, rtol=1e-4)


def test_earlier_layers_farther_from_identity():
    model = TinyDecoder(n_layers=4, d_model=8, seed=2)
    prompt = "abcdefghijklmnopqrstuv"
    Js, _, _ = jacobian_for_prompt(
        model,
        prompt,
        source_layers=[0, 2],
        target_layer=3,
        dim_batch=8,
        skip_first=0,
    )
    eye = torch.eye(8)
    dist_early = (Js[0] - eye).pow(2).mean()
    dist_late = (Js[2] - eye).pow(2).mean()
    assert dist_early > dist_late


def test_fit_save_load_apply(tmp_path: Path):
    model = TinyDecoder(n_layers=3, d_model=6, seed=3)
    prompts = ["abcdefghij", "klmnopqrst"]
    lens = fit(
        model,
        prompts,
        source_layers=[0, 1],
        target_layer=2,
        dim_batch=6,
        skip_first=0,
        max_seq_len=64,
    )
    path = tmp_path / "lens.pt"
    lens.save(str(path), dtype=torch.float16)
    loaded = JacobianLens.load(str(path))
    assert loaded.d_model == 6
    assert loaded.source_layers == [0, 1]

    lens_logits, model_logits, input_ids = loaded.apply(
        model, "abcdefghij", positions=[-1], use_jacobian=True
    )
    assert set(lens_logits) == {0, 1}
    assert lens_logits[0].shape[-1] == model_logits.shape[-1]
    assert input_ids.ndim == 2


def test_logit_lens_baseline_matches_unembed():
    model = TinyDecoder(n_layers=3, d_model=6, seed=4)
    lens = JacobianLens.identity(n_layers=3, d_model=6, layers=[0, 1])
    prompt = "abcdef"
    with_j, model_logits, _ = lens.apply(model, prompt, layers=[1], positions=[-1], use_jacobian=True)
    no_j, _, _ = lens.apply(model, prompt, layers=[1], positions=[-1], use_jacobian=False)
    # identity transport ⇒ jacobian and logit-lens paths agree
    assert torch.allclose(with_j[1], no_j[1], atol=1e-5)


def test_merge_weighted():
    eye = torch.eye(4)
    a = JacobianLens(jacobians={0: eye * 2}, n_prompts=1, d_model=4)
    b = JacobianLens(jacobians={0: eye * 4}, n_prompts=3, d_model=4)
    merged = JacobianLens.merge([a, b])
    assert merged.n_prompts == 4
    expected = (eye * 2 * 1 + eye * 4 * 3) / 4
    assert torch.allclose(merged.jacobians[0], expected)


def test_checkpoint_resume_skips(tmp_path: Path):
    model = TinyDecoder(n_layers=3, d_model=4, seed=5)
    # first prompt too short for skip_first=2; second OK
    prompts = ["a", "abcdefghij"]
    ckpt = tmp_path / "fit.ckpt"
    lens = fit(
        model,
        prompts,
        source_layers=[0],
        target_layer=2,
        dim_batch=4,
        skip_first=2,
        max_seq_len=64,
        checkpoint_path=str(ckpt),
        checkpoint_every=1,
        resume=True,
    )
    assert lens.n_prompts == 1
    # resume should not double-count
    lens2 = fit(
        model,
        prompts,
        source_layers=[0],
        target_layer=2,
        dim_batch=4,
        skip_first=2,
        max_seq_len=64,
        checkpoint_path=str(ckpt),
        checkpoint_every=1,
        resume=True,
    )
    assert lens2.n_prompts == 1


def test_run_jlens_without_fitted_lens():
    model = TinyDecoder(n_layers=3, d_model=6, seed=6)
    result = run_jlens(model, "hello world!!", layers=[0, 1], top_k=3, positions=[-1])
    assert result.layers == [0, 1]
    assert len(result.top_tokens[0]) == 3
    assert result.metadata["use_jacobian"] is False


# --- legacy readout helpers ---


class TinyUnembed(nn.Module):
    def __init__(self, d_model: int = 8, vocab: int = 20) -> None:
        super().__init__()
        self.linear = nn.Linear(d_model, vocab, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class TinyTokenizer:
    def decode(self, ids: list[int]) -> str:
        return f"tok{ids[0]}"


def test_project_to_vocab():
    unembed = TinyUnembed()
    residual = torch.randn(8)
    logits = project_to_vocab(residual, unembed)
    assert logits.shape == (20,)


def test_decode_residual():
    unembed = TinyUnembed()
    tops = decode_residual(torch.randn(8), unembed, TinyTokenizer(), top_k=3)
    assert len(tops) == 3


def test_intervene_jlens_direction():
    residual = torch.zeros(4)
    out = intervene_jlens(residual, direction=torch.ones(4), scale=2.0)
    assert torch.allclose(out, torch.full((4,), 2.0))
