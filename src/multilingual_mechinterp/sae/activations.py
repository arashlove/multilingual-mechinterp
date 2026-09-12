"""Extract and cache residual-stream activations for SAE training."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence

import torch
from tqdm import tqdm

from multilingual_mechinterp.models.hooks import hooked_forward
from multilingual_mechinterp.models.loader import LoadedModel, load_model
from multilingual_mechinterp.utils.io import ensure_dir


@dataclass
class ActivationChunkMeta:
    path: Path
    n_tokens: int
    d_model: int
    layer: int


def extract_residual_activations(
    model: LoadedModel,
    texts: Sequence[str],
    *,
    layer: int = 1,
    max_length: int = 128,
    batch_size: int = 8,
    device: str | torch.device | None = None,
) -> torch.Tensor:
    """Return residual activations shaped ``[n_tokens, d_model]`` for ``layer``.

    Uses HF forward hooks (``blocks.{layer}`` residual output), analogous to
    TransformerLens ``blocks.{layer}.hook_resid_post``.
    """
    run_device = torch.device(device or model.device)
    tokenizer = model.tokenizer
    hf_model = model.model
    chunks: list[torch.Tensor] = []

    for start in tqdm(range(0, len(texts), batch_size), desc=f"extract L{layer}"):
        batch_texts = list(texts[start : start + batch_size])
        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        inputs = {k: v.to(run_device) for k, v in inputs.items()}
        attention = inputs.get("attention_mask")

        with torch.no_grad(), hooked_forward(hf_model, layers=[layer]) as cache:
            hf_model(**inputs)
            acts = cache.get(f"blocks.{layer}")  # [B, T, C]

        if attention is not None:
            mask = attention.bool()
            flat = acts[mask].float().cpu()
        else:
            flat = acts.reshape(-1, acts.shape[-1]).float().cpu()
        chunks.append(flat)

    if not chunks:
        raise ValueError("No texts provided for activation extraction")
    return torch.cat(chunks, dim=0)


def save_activation_chunk(acts: torch.Tensor, path: str | Path) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    torch.save(acts, path)
    return path


def load_activation_chunk(path: str | Path, *, device: str | torch.device = "cpu") -> torch.Tensor:
    return torch.load(Path(path), map_location=device, weights_only=True)


def iter_activation_dir(
    dataset_dir: str | Path,
    *,
    device: str | torch.device = "cpu",
) -> Iterator[torch.Tensor]:
    """Yield ``*.pt`` activation tensors from a directory (sorted by name)."""
    dataset_dir = Path(dataset_dir)
    paths = sorted(dataset_dir.glob("*.pt"))
    if not paths:
        raise FileNotFoundError(f"No .pt activation files in {dataset_dir}")
    for path in paths:
        yield load_activation_chunk(path, device=device)


def extract_and_save(
    *,
    model_name: str = "EleutherAI/pythia-70m-deduped",
    texts: Sequence[str] | None = None,
    dataset_name: str | None = "NeelNanda/pile-10k",
    dataset_text_column: str = "text",
    n_texts: int = 256,
    layer: int = 1,
    output_dir: str | Path = "data/processed/activations/layer_1",
    chunk_size: int = 50_000,
    max_length: int = 128,
    batch_size: int = 8,
    device: str | None = None,
) -> list[ActivationChunkMeta]:
    """Extract residual activations and write chunked ``.pt`` files.

    Provide either ``texts`` or a Hugging Face ``dataset_name``.
    """
    if texts is None:
        texts = _load_hf_texts(dataset_name, dataset_text_column, n_texts)

    model = load_model(model_name, device=device)
    acts = extract_residual_activations(
        model,
        texts,
        layer=layer,
        max_length=max_length,
        batch_size=batch_size,
        device=device,
    )

    output_dir = ensure_dir(output_dir)
    metas: list[ActivationChunkMeta] = []
    for i, start in enumerate(range(0, acts.shape[0], chunk_size)):
        chunk = acts[start : start + chunk_size]
        path = output_dir / f"{i}.pt"
        save_activation_chunk(chunk, path)
        metas.append(
            ActivationChunkMeta(
                path=path,
                n_tokens=int(chunk.shape[0]),
                d_model=int(chunk.shape[1]),
                layer=layer,
            )
        )
    return metas


def _load_hf_texts(dataset_name: str | None, text_column: str, n_texts: int) -> list[str]:
    if dataset_name is None:
        raise ValueError("Provide texts=... or dataset_name=...")
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "datasets is required to load HF text corpora. Install with: pip install datasets"
        ) from exc

    ds = load_dataset(dataset_name, split="train")
    n = min(n_texts, len(ds))
    return [str(ds[i][text_column]) for i in range(n) if ds[i][text_column]]
