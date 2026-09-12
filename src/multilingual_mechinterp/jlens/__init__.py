"""Jacobian Lens (JLens) — fit / apply residual → verbalizable readouts."""

from multilingual_mechinterp.jlens.apply import JLensResult, as_lens_model, run_jlens
from multilingual_mechinterp.jlens.fitting import (
    SKIP_FIRST_N_POSITIONS,
    fit,
    jacobian_for_prompt,
    valid_position_mask,
)
from multilingual_mechinterp.jlens.hf import HFLensModel, Layout, from_hf
from multilingual_mechinterp.jlens.hooks import ActivationRecorder
from multilingual_mechinterp.jlens.intervention import intervene_jlens
from multilingual_mechinterp.jlens.lens import JacobianLens
from multilingual_mechinterp.jlens.protocol import LensModel, TinyDecoder
from multilingual_mechinterp.jlens.readout import decode_residual, project_to_vocab

__all__ = [
    "ActivationRecorder",
    "HFLensModel",
    "JLensResult",
    "JacobianLens",
    "Layout",
    "LensModel",
    "SKIP_FIRST_N_POSITIONS",
    "TinyDecoder",
    "as_lens_model",
    "decode_residual",
    "fit",
    "from_hf",
    "intervene_jlens",
    "jacobian_for_prompt",
    "project_to_vocab",
    "run_jlens",
    "valid_position_mask",
]
