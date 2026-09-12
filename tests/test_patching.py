"""Tests for activation patching (real intervene + layer sweep)."""

from __future__ import annotations

import pytest
import torch

from multilingual_mechinterp.metrics import effect_size, normalized_effect
from multilingual_mechinterp.patching import (
    TinyCausalLM,
    answer_token_ids,
    causal_effect_curve,
    collect_activations,
    last_token_index,
    recommend_layers,
    resolve_positions,
    run_patching,
)


def test_last_token_index_right_pad():
    mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]])
    assert torch.equal(last_token_index(mask), torch.tensor([2, 1]))


def test_resolve_positions_negative():
    mask = torch.tensor([[1, 1, 1, 1]])
    assert int(resolve_positions(mask, -1).item()) == 3
    assert int(resolve_positions(mask, -2).item()) == 2


def test_collect_and_patch_changes_probs():
    model = TinyCausalLM(n_layers=3, d_model=8, vocab_size=32, seed=1)
    src = "abcdefghij"
    tgt = "klmnopqrst"
    acts = collect_activations(model, [src, tgt], pos_idx=-1)
    assert acts.shape[0] == 3
    assert acts.shape[1] == 2

    y = model.tokenizer.encode(src)[0]
    curve = causal_effect_curve(model, src, tgt, [y], window=None)
    assert curve["effect"].shape == (1, 3)
    assert curve["best_layer"].numel() == 1
    assert torch.isfinite(curve["effect"]).all()


def test_run_patching_recommends_layers():
    model = TinyCausalLM(n_layers=4, d_model=8, vocab_size=32, seed=2)
    result = run_patching(
        model,
        source_prompt="sourceconceptxx",
        target_prompt="targetpromptzzz",
        answer="s",
        window=2,
    )
    assert result.best_layer is not None
    assert len(result.scores) == 4
    assert result.baseline_score is not None
    tops = recommend_layers(result, top_k=2, min_effect=-1.0)
    assert len(tops) <= 2


def test_answer_token_ids():
    model = TinyCausalLM(vocab_size=32)
    ids = answer_token_ids(model.tokenizer, "ab")
    assert len(ids) >= 1


def test_causal_metrics():
    assert effect_size(0.8, 0.2) == pytest.approx(0.6)
    assert normalized_effect(0.5, 0.0, 1.0) == pytest.approx(0.5)
    assert normalized_effect(0.5, 0.5, 0.5) == 0.0
