"""Activation patching — causal residual interventions + layer sweeps."""

from multilingual_mechinterp.patching.collect import (
    cache_activations,
    collect_activations,
    get_blocks,
)
from multilingual_mechinterp.patching.indexing import (
    answer_token_ids,
    last_token_index,
    resolve_positions,
)
from multilingual_mechinterp.patching.intervene import (
    answer_prob,
    next_token_probs_from_logits,
    patch_residuals,
)
from multilingual_mechinterp.patching.layer_sweep import sweep_layers
from multilingual_mechinterp.patching.lens import causal_effect_curve, patch_lens
from multilingual_mechinterp.patching.patch import PatchResult, recommend_layers, run_patching
from multilingual_mechinterp.patching.tiny import TinyCausalLM

__all__ = [
    "PatchResult",
    "TinyCausalLM",
    "answer_prob",
    "answer_token_ids",
    "cache_activations",
    "causal_effect_curve",
    "collect_activations",
    "get_blocks",
    "last_token_index",
    "next_token_probs_from_logits",
    "patch_lens",
    "patch_residuals",
    "recommend_layers",
    "resolve_positions",
    "run_patching",
    "sweep_layers",
]
