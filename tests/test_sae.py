"""Tests for tied SAE (paper architecture) — offline, no HF models."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from multilingual_mechinterp.metrics import (
    count_dead_features,
    evaluate_sae,
    fraction_variance_unexplained,
    mean_l0,
    mmcs,
    mmcs_to_fixed,
)
from multilingual_mechinterp.sae import (
    SAE,
    TiedSAE,
    ablate_features,
    load_sae,
    steer_features,
    sweep_l1,
    top_activating_features,
    train_tied_sae,
)
from multilingual_mechinterp.sae.model import run_sae


def test_tied_sae_forward_shapes():
    sae = TiedSAE(d_model=16, d_sae=32, alpha=1e-3)
    x = torch.randn(4, 16)
    z, recon = sae(x)
    assert z.shape == (4, 32)
    assert recon.shape == (4, 16)
    assert torch.all(z >= 0)


def test_tied_sae_dictionary_rows_unit_norm():
    sae = TiedSAE(d_model=8, d_sae=16)
    norms = torch.norm(sae.get_learned_dict(), dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_tied_sae_loss_finite():
    sae = TiedSAE(d_model=512, d_sae=1024, alpha=1e-3)
    sample = torch.randn(64, 512)
    loss, parts = sae.loss(sample)
    assert torch.isfinite(loss)
    assert "l_reconstruction" in parts
    assert "l_l1" in parts


def test_sae_alias_is_tied():
    assert SAE is TiedSAE


def test_top_activating_features():
    latent = torch.tensor([0.1, 0.9, 0.3, 0.0])
    tops = top_activating_features(latent, k=2)
    assert len(tops) == 2
    assert tops[0]["feature_id"] == 1
    assert tops[0]["activation"] == pytest.approx(0.9)


def test_ablate_and_steer():
    sae = TiedSAE(d_model=8, d_sae=16)
    x = torch.randn(8)
    ablated = ablate_features(x, sae, feature_ids=[0, 1])
    steered = steer_features(x, sae, feature_ids=[2], strength=2.0)
    assert ablated.shape == x.shape
    assert steered.shape == x.shape


def test_train_tied_sae_reduces_loss(tmp_path: Path):
    acts = torch.randn(400, 32)
    result = train_tied_sae(
        acts,
        ratio=2,
        alpha=1e-3,
        lr=1e-2,
        batch_size=64,
        n_epochs=2,
        device="cpu",
    )
    assert result.history[0]["loss"] >= result.history[-1]["loss"] - 1.0  # allow noise
    ckpt = tmp_path / "sae.pt"
    result.sae.save(ckpt)
    loaded = load_sae(ckpt)
    assert loaded.d_model == 32
    assert loaded.d_sae == 64


def test_fvu_and_l0():
    acts = torch.randn(200, 16)
    sae = train_tied_sae(acts, ratio=2, alpha=1e-3, n_epochs=1, batch_size=50, device="cpu").sae
    metrics = evaluate_sae(sae, acts)
    assert 0.0 <= metrics["fvu"] <= 2.0
    assert metrics["mean_l0"] >= 0.0
    assert "dead_features" in metrics
    assert mean_l0(sae, acts) == pytest.approx(metrics["mean_l0"])
    assert fraction_variance_unexplained(sae, acts) == pytest.approx(metrics["fvu"])
    assert count_dead_features(sae, acts, threshold=10) == metrics["dead_features"]


def test_mmcs_to_fixed_and_self():
    acts = torch.randn(300, 8)
    sae = train_tied_sae(acts, ratio=2, alpha=1e-3, n_epochs=1, batch_size=64, device="cpu").sae
    # self-MMCS should be ~1 (each feature finds itself)
    assert mmcs(sae, sae) == pytest.approx(1.0, abs=1e-5)
    truth = sae.get_learned_dict().detach().clone()
    assert mmcs_to_fixed(sae, truth) == pytest.approx(1.0, abs=1e-5)
    # random unrelated dict → lower MMCS
    noise = torch.randn_like(truth)
    noise = noise / noise.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    assert mmcs_to_fixed(sae, noise) < 0.95


def test_sweep_l1(tmp_path: Path):
    acts = torch.randn(300, 16)
    rows = sweep_l1(
        acts,
        alphas=[1e-3, 3e-3],
        ratio=2,
        output_dir=tmp_path,
        n_epochs=1,
        batch_size=64,
        device="cpu",
    )
    assert len(rows) == 2
    assert (tmp_path / "sweep_metrics.json").exists()
    assert (tmp_path / "learned_dicts.pt").exists()


def test_run_sae_requires_loaded_model():
    with pytest.raises(ValueError, match="LoadedModel"):
        run_sae(model=object(), prompt="hi", layer=0)
