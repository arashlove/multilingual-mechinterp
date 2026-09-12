"""Fit Jacobian matrices that map intermediate residuals toward the final layer."""

from __future__ import annotations

import math
import os
from typing import Sequence

import torch

from multilingual_mechinterp.jlens.hooks import ActivationRecorder
from multilingual_mechinterp.jlens.lens import JacobianLens
from multilingual_mechinterp.jlens.protocol import LensModel

SKIP_FIRST_N_POSITIONS = 16


def valid_position_mask(
    seq_len: int, *, skip_first: int = SKIP_FIRST_N_POSITIONS
) -> torch.Tensor:
    """Boolean mask over sequence positions usable for the Jacobian estimator."""
    if skip_first < 0:
        raise ValueError(f"skip_first must be >= 0, got {skip_first}")
    mask = torch.zeros(seq_len, dtype=torch.bool)
    if seq_len > skip_first + 1:
        mask[skip_first : seq_len - 1] = True
    if int(mask.sum()) == 0:
        raise ValueError(
            f"prompt too short: seq_len={seq_len}, need > {skip_first + 1} tokens"
        )
    return mask


def _check_layer_indices(
    source_layers: Sequence[int] | None,
    target_layer: int | None,
    n_layers: int,
) -> tuple[list[int], int]:
    target = n_layers - 1 if target_layer is None else int(target_layer)
    if target < 0:
        target += n_layers
    if not (0 <= target < n_layers):
        raise ValueError(f"target_layer out of range: {target_layer} (n_layers={n_layers})")

    if source_layers is None:
        sources = list(range(target))
    else:
        sources = sorted({(l + n_layers if l < 0 else l) for l in source_layers})
        for layer in sources:
            if not (0 <= layer < n_layers):
                raise ValueError(f"source layer out of range: {layer}")
        if not sources:
            raise ValueError("source_layers must be non-empty")
        if sources[-1] >= target:
            raise ValueError(
                f"all source_layers must be < target_layer ({target}); got {sources}"
            )
    return sources, target


def jacobian_for_prompt(
    model: LensModel,
    prompt: str,
    source_layers: Sequence[int],
    *,
    target_layer: int | None = None,
    dim_batch: int = 8,
    max_seq_len: int = 128,
    skip_first: int = SKIP_FIRST_N_POSITIONS,
) -> tuple[dict[int, torch.Tensor], int, int]:
    """Estimate per-layer Jacobians for one prompt.

    Returns ``(jacobians, seq_len, n_valid_positions)`` with each Jacobian
    ``[d_model, d_model]`` fp32 on CPU. Rows are
    ``∂ output_dim / ∂ source_vector``.
    """
    n_layers, d_model = model.n_layers, model.d_model
    source_layers, target_layer = _check_layer_indices(source_layers, target_layer, n_layers)

    input_ids = model.encode(prompt, max_length=max_seq_len)
    seq_len = int(input_ids.shape[1])
    position_mask = valid_position_mask(seq_len, skip_first=skip_first)
    n_valid_positions = int(position_mask.sum())

    jacobians = {
        layer: torch.zeros(d_model, d_model, dtype=torch.float32) for layer in source_layers
    }
    n_passes = math.ceil(d_model / dim_batch)

    with (
        ActivationRecorder(
            model.layers,
            at=[*source_layers, target_layer],
            start_graph_at=min(source_layers),
        ) as recorder,
        torch.enable_grad(),
    ):
        replicated_ids = input_ids.expand(dim_batch, -1)
        model.forward(replicated_ids)

        target_activation = recorder.activations[target_layer]
        source_activations = [recorder.activations[layer] for layer in source_layers]

        valid_positions = position_mask.nonzero(as_tuple=True)[0].to(target_activation.device)
        batch_indices = torch.arange(dim_batch, device=target_activation.device)
        cotangent = torch.zeros_like(target_activation)

        for pass_idx, dim_start in enumerate(range(0, d_model, dim_batch)):
            n_dims_this_pass = min(dim_batch, d_model - dim_start)

            cotangent.zero_()
            cotangent[
                batch_indices[:n_dims_this_pass, None],
                valid_positions[None, :],
                dim_start + batch_indices[:n_dims_this_pass, None],
            ] = 1.0

            grads = torch.autograd.grad(
                outputs=target_activation,
                inputs=source_activations,
                grad_outputs=cotangent,
                retain_graph=(pass_idx < n_passes - 1),
            )

            for layer, grad in zip(source_layers, grads, strict=True):
                positions_on_device = valid_positions.to(grad.device, non_blocking=True)
                rows = grad[:n_dims_this_pass, positions_on_device, :].float().mean(dim=1)
                jacobians[layer][dim_start : dim_start + n_dims_this_pass, :] = rows.cpu()
            del grads

    return jacobians, seq_len, n_valid_positions


def _atomic_save(obj: object, path: str) -> None:
    tmp_path = f"{path}.tmp.{os.getpid()}"
    torch.save(obj, tmp_path)
    os.replace(tmp_path, path)


def fit(
    model: LensModel,
    prompts: Sequence[str],
    *,
    source_layers: Sequence[int] | None = None,
    target_layer: int | None = None,
    dim_batch: int = 8,
    max_seq_len: int = 128,
    skip_first: int = SKIP_FIRST_N_POSITIONS,
    checkpoint_path: str | None = None,
    checkpoint_every: int | None = 1,
    resume: bool = True,
) -> JacobianLens:
    """Average per-prompt Jacobians into a :class:`JacobianLens`."""
    sources, target = _check_layer_indices(source_layers, target_layer, model.n_layers)
    d_model = model.d_model

    jacobian_sum = {
        layer: torch.zeros(d_model, d_model, dtype=torch.float32) for layer in sources
    }
    n_done = 0
    next_idx = 0

    if checkpoint_path and resume and os.path.isfile(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if (
            list(ckpt["source_layers"]) != list(sources)
            or int(ckpt["target_layer"]) != target
            or int(ckpt["skip_first"]) != skip_first
        ):
            raise ValueError(
                "checkpoint disagrees with current fit() arguments "
                "(source_layers / target_layer / skip_first)"
            )
        jacobian_sum = {int(k): v.float() for k, v in ckpt["jacobian_sum"].items()}
        n_done = int(ckpt["n_done"])
        next_idx = int(ckpt["next_idx"])

    for prompt_idx, prompt in enumerate(prompts):
        if prompt_idx < next_idx:
            continue
        try:
            per_prompt_j, _seq_len, _n_valid = jacobian_for_prompt(
                model,
                prompt,
                sources,
                target_layer=target,
                dim_batch=dim_batch,
                max_seq_len=max_seq_len,
                skip_first=skip_first,
            )
        except ValueError:
            next_idx = prompt_idx + 1
            if checkpoint_path and checkpoint_every and (prompt_idx + 1) % checkpoint_every == 0:
                _atomic_save(
                    {
                        "jacobian_sum": jacobian_sum,
                        "n_done": n_done,
                        "next_idx": next_idx,
                        "source_layers": sources,
                        "target_layer": target,
                        "skip_first": skip_first,
                    },
                    checkpoint_path,
                )
            continue

        for layer in sources:
            jacobian_sum[layer] += per_prompt_j[layer]
        n_done += 1
        next_idx = prompt_idx + 1

        if checkpoint_path and checkpoint_every and next_idx % checkpoint_every == 0:
            _atomic_save(
                {
                    "jacobian_sum": jacobian_sum,
                    "n_done": n_done,
                    "next_idx": next_idx,
                    "source_layers": sources,
                    "target_layer": target,
                    "skip_first": skip_first,
                },
                checkpoint_path,
            )

    if n_done == 0:
        raise ValueError("fit() completed with zero successful prompts")

    jacobian_mean = {layer: jacobian_sum[layer] / n_done for layer in sources}
    return JacobianLens(jacobians=jacobian_mean, n_prompts=n_done, d_model=d_model)
