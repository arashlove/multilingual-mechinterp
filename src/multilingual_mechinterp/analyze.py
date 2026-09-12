"""Unified multi-method analysis entry point."""

from __future__ import annotations

from typing import Any, Iterable, Mapping, MutableMapping, Sequence

from multilingual_mechinterp.jlens import run_jlens
from multilingual_mechinterp.patching import run_patching
from multilingual_mechinterp.sae import run_sae


def analyze(
    model: Any,
    prompts: str | Sequence[str],
    methods: Sequence[str] = ("sae", "jlens", "patching"),
    *,
    layer: int | None = None,
    source_prompts: Sequence[str] | None = None,
    target_prompts: Sequence[str] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Run one or more interpretability methods on the same prompts.

    Parameters
    ----------
    model:
        Model handle returned by :func:`multilingual_mechinterp.models.load_model`.
    prompts:
        Prompt string or list of prompts (used by SAE / JLens; also default
        source prompts for patching when ``source_prompts`` is omitted).
    methods:
        Subset of ``{"sae", "jlens", "patching"}``.
    layer:
        Optional layer index for SAE / patching sweeps.
    source_prompts / target_prompts:
        Optional parallel prompt pairs for activation patching.
    **kwargs:
        Forwarded to the underlying ``run_*`` helpers where applicable.
    """
    if isinstance(prompts, str):
        prompt_list: list[str] = [prompts]
    else:
        prompt_list = list(prompts)

    if not prompt_list:
        raise ValueError("prompts must be non-empty")

    method_set = {m.lower() for m in methods}
    allowed = {"sae", "jlens", "patching"}
    unknown = method_set - allowed
    if unknown:
        raise ValueError(f"Unknown methods: {sorted(unknown)}. Allowed: {sorted(allowed)}")

    results: dict[str, Any] = {"prompts": prompt_list, "methods": sorted(method_set)}

    if "sae" in method_set:
        sae_kwargs = _subset_kwargs(kwargs, ("sae_path", "top_k", "device"))
        results["sae"] = [
            run_sae(model=model, prompt=p, layer=layer, **sae_kwargs) for p in prompt_list
        ]

    if "jlens" in method_set:
        jlens_kwargs = _subset_kwargs(kwargs, ("layers", "device"))
        results["jlens"] = [run_jlens(model=model, prompt=p, **jlens_kwargs) for p in prompt_list]

    if "patching" in method_set:
        sources = list(source_prompts) if source_prompts is not None else prompt_list
        if target_prompts is None:
            raise ValueError(
                "patching requires target_prompts (or pass them via analyze(..., target_prompts=...))"
            )
        targets = list(target_prompts)
        if len(sources) != len(targets):
            raise ValueError("source_prompts and target_prompts must have the same length")
        patch_kwargs = _subset_kwargs(kwargs, ("layers", "metric", "device"))
        if layer is not None and "layers" not in patch_kwargs:
            patch_kwargs["layers"] = [layer]
        results["patching"] = [
            run_patching(
                model=model,
                source_prompt=src,
                target_prompt=tgt,
                **patch_kwargs,
            )
            for src, tgt in zip(sources, targets)
        ]

    return results


def _subset_kwargs(kwargs: Mapping[str, Any], keys: Iterable[str]) -> MutableMapping[str, Any]:
    return {k: kwargs[k] for k in keys if k in kwargs}
