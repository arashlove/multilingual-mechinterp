"""HuggingFace adapter: locate residual stack and wrap as LensModel."""

from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn


@dataclass(frozen=True)
class Layout:
    path: str
    layers: str = "layers"
    norm: str = "norm"
    embed: str = "embed_tokens"
    lm_head: str = "lm_head"


_LAYOUTS = (
    Layout("model"),
    Layout("model.language_model"),
    Layout("language_model"),
    Layout("model", norm="final_layernorm"),
    Layout("transformer", layers="h", norm="ln_f", embed="wte"),
    Layout("gpt_neox", norm="final_layer_norm", embed="embed_in", lm_head="embed_out"),
)


def _resolve_attr_path(obj: Any, dotted_path: str) -> Any:
    return functools.reduce(getattr, dotted_path.split("."), obj)


def _find_layout(hf_model: nn.Module) -> Layout:
    for layout in _LAYOUTS:
        try:
            candidate = _resolve_attr_path(hf_model, layout.path)
        except AttributeError:
            continue
        if all(
            hasattr(candidate, attr) for attr in (layout.layers, layout.norm, layout.embed)
        ) and hasattr(hf_model, layout.lm_head):
            return layout
    raise ValueError(
        f"could not locate the text decoder inside {type(hf_model).__name__}; "
        f"pass layout= explicitly"
    )


def _text_config(hf_model: nn.Module) -> Any:
    cfg = hf_model.config
    getter = getattr(cfg, "get_text_config", None)
    if callable(getter):
        return getter()
    return cfg


class HFLensModel:
    """Wrap a HF causal LM so fitting / apply can use the LensModel protocol."""

    def __init__(
        self,
        hf_model: nn.Module,
        tokenizer: Any,
        *,
        layout: Layout | None = None,
        compile: bool = False,
        force_bos: bool = True,
    ) -> None:
        self._hf_model = hf_model
        self.tokenizer = tokenizer
        self._layout = layout or _find_layout(hf_model)

        hf_model.eval()
        for param in hf_model.parameters():
            param.requires_grad_(False)

        if force_bos and getattr(tokenizer, "bos_token_id", None) is not None:
            if hasattr(tokenizer, "add_bos_token"):
                tokenizer.add_bos_token = True

        self._text_module = _resolve_attr_path(hf_model, self._layout.path)
        self.layers = getattr(self._text_module, self._layout.layers)
        self._final_norm = getattr(self._text_module, self._layout.norm)
        self._embed_tokens = getattr(self._text_module, self._layout.embed)
        self._lm_head = getattr(hf_model, self._layout.lm_head)

        text_cfg = _text_config(hf_model)
        self.n_layers = int(text_cfg.num_hidden_layers)
        self.d_model = int(text_cfg.hidden_size)
        self._logit_softcap = getattr(text_cfg, "final_logit_softcapping", None)

        if len(self.layers) != self.n_layers:
            raise ValueError(
                f"layout layers length {len(self.layers)} != num_hidden_layers {self.n_layers}"
            )

        if compile:
            for i in range(len(self.layers)):
                self.layers[i] = torch.compile(self.layers[i], mode="default", dynamic=False)

    @property
    def input_device(self) -> torch.device:
        return self._embed_tokens.weight.device

    def encode(self, text: str, *, max_length: int = 512) -> torch.Tensor:
        encoded = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=max_length
        )
        return encoded.input_ids.to(self.input_device)

    def forward(self, input_ids: torch.Tensor) -> Any:
        return self._text_module(input_ids=input_ids, use_cache=False)

    def unembed(self, residual: torch.Tensor) -> torch.Tensor:
        target_device = self._lm_head.weight.device
        target_dtype = self._lm_head.weight.dtype
        logits = self._lm_head(self._final_norm(residual.to(dtype=target_dtype, device=target_device)))
        if self._logit_softcap is not None:
            cap = float(self._logit_softcap)
            logits = cap * torch.tanh(logits / cap)
        return logits


def from_hf(
    hf_model: nn.Module,
    tokenizer: Any,
    *,
    layout: Layout | None = None,
    text_module: str | None = None,
    compile: bool = False,
    force_bos: bool = True,
) -> HFLensModel:
    """Wrap a caller-loaded HF model + tokenizer as a :class:`HFLensModel`."""
    if layout is None and text_module is not None:
        layout = Layout(path=text_module)
    return HFLensModel(
        hf_model,
        tokenizer,
        layout=layout,
        compile=compile,
        force_bos=force_bos,
    )
