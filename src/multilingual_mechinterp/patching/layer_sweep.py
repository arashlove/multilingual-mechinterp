"""Convenience wrappers for layer sweeps."""

from __future__ import annotations

from typing import Any, Sequence

from multilingual_mechinterp.models.loader import LoadedModel
from multilingual_mechinterp.patching.patch import PatchResult, run_patching


def sweep_layers(
    model: LoadedModel,
    source_prompt: str,
    target_prompt: str,
    *,
    answer: str | Sequence[str] | Sequence[int] | None = None,
    window: int | None = None,
    **kwargs: Any,
) -> PatchResult:
    """Run a full start-layer sweep (alias of :func:`run_patching`)."""
    return run_patching(
        model=model,
        source_prompt=source_prompt,
        target_prompt=target_prompt,
        answer=answer,
        window=window,
        **kwargs,
    )
