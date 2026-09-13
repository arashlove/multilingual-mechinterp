"""Load Hugging Face causal LMs for interpretability experiments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


@dataclass
class LoadedModel:
    """Thin wrapper around a HF model + tokenizer."""

    name: str
    model: Any
    tokenizer: Any
    device: torch.device
    n_layers: int

    def to(self, device: str | torch.device) -> "LoadedModel":
        device = torch.device(device)
        self.model.to(device)
        self.device = device
        return self


def load_model(
    model_name: str = "Qwen/Qwen2.5-1.5B",
    *,
    device: str | None = None,
    dtype: str | torch.dtype | None = "auto",
    trust_remote_code: bool = True,
    token: str | None = None,
    **kwargs: Any,
) -> LoadedModel:
    """Load a causal LM and tokenizer.

    Parameters
    ----------
    model_name:
        Hugging Face model id. Defaults to Qwen2.5-1.5B; swap to Gemma etc.
        e.g. ``google/gemma-2-2b``, ``EleutherAI/pythia-70m-deduped``.
    device:
        ``"cuda"``, ``"cpu"``, or ``"mps"``. Defaults to CUDA if available.
    dtype:
        Torch dtype or ``"auto"`` / ``None`` to let Transformers decide.
    trust_remote_code:
        Required for some Qwen builds; usually ``False`` for Gemma/Pythia.
    token:
        Optional Hugging Face token for gated models.
    """
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "transformers is required to load models. Install with: pip install transformers"
        ) from exc

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_device = torch.device(device)

    # Absolute / existing paths must contain config.json; otherwise HF treats the
    # string as a repo id and raises "Repo id must be in the form...".
    local = Path(model_name)
    is_local = local.is_absolute() or local.exists()
    if is_local:
        local = local.expanduser().resolve()
        if not local.is_dir():
            raise FileNotFoundError(
                f"Local model path does not exist or is not a directory: {local}"
            )
        if not (local / "config.json").is_file():
            raise FileNotFoundError(
                f"No config.json in {local}. Point MODEL_NAME at the folder that "
                "contains config.json (check find /root -name config.json)."
            )
        model_name = str(local)

    tok_kwargs: dict[str, Any] = {"trust_remote_code": trust_remote_code}
    model_kwargs: dict[str, Any] = {"trust_remote_code": trust_remote_code, **kwargs}
    if is_local:
        tok_kwargs.setdefault("local_files_only", True)
        model_kwargs.setdefault("local_files_only", True)
    if token is not None:
        tok_kwargs["token"] = token
        model_kwargs["token"] = token

    tokenizer = AutoTokenizer.from_pretrained(model_name, **tok_kwargs)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if dtype is not None and dtype != "auto":
        model_kwargs["torch_dtype"] = _resolve_dtype(dtype)
    elif dtype == "auto":
        model_kwargs["torch_dtype"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    model.to(torch_device)
    model.eval()

    n_layers = _infer_n_layers(model)

    return LoadedModel(
        name=model_name,
        model=model,
        tokenizer=tokenizer,
        device=torch_device,
        n_layers=n_layers,
    )


def _resolve_dtype(dtype: str | torch.dtype) -> torch.dtype:
    if isinstance(dtype, torch.dtype):
        return dtype
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    key = dtype.lower()
    if key not in mapping:
        raise ValueError(f"Unsupported dtype: {dtype}")
    return mapping[key]


def _infer_n_layers(model: Any) -> int:
    cfg = getattr(model, "config", None)
    for attr in ("num_hidden_layers", "n_layer", "num_layers"):
        if cfg is not None and hasattr(cfg, attr):
            return int(getattr(cfg, attr))
    if hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "layers"):
        return len(model.gpt_neox.layers)
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return len(model.transformer.h)
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return len(model.model.layers)
    raise ValueError("Could not infer number of layers from model")
