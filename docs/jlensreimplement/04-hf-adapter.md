# 04 — HuggingFace Adapter

**File to mirror:** [`jlens/hf.py`](../../jlens/hf.py)

Model loading (`from_pretrained`, device, dtype) stays the **caller's** job.
This module only locates the residual stack and wraps it as a `LensModel`.

---

## `Layout`

```python
@dataclass(frozen=True)
class Layout:
    path: str          # dotted path from *ForCausalLM → text decoder
    layers: str = "layers"
    norm: str = "norm"
    embed: str = "embed_tokens"
    lm_head: str = "lm_head"
```

### Known layouts (try in order; first match wins)

```python
_LAYOUTS = (
    Layout("model"),                                      # Llama/Qwen/Mistral/Gemma/OLMo/...
    Layout("model.language_model"),                       # multimodal wrappers
    Layout("language_model"),
    Layout("model", norm="final_layernorm"),              # Phi
    Layout("transformer", layers="h", norm="ln_f", embed="wte"),  # GPT-2
    Layout("gpt_neox", norm="final_layer_norm",
           embed="embed_in", lm_head="embed_out"),        # Pythia / NeoX
)
```

Match rule: `path` resolves **and** the text decoder has `layers`/`norm`/`embed`
**and** the outer model has `lm_head`.

```python
def _find_layout(hf_model: nn.Module) -> Layout:
    for layout in _LAYOUTS:
        try:
            candidate = _resolve_attr_path(hf_model, layout.path)
        except AttributeError:
            continue
        if all(
            hasattr(candidate, a) for a in (layout.layers, layout.norm, layout.embed)
        ) and hasattr(hf_model, layout.lm_head):
            return layout
    raise ValueError(
        f"could not locate the text decoder inside {type(hf_model).__name__}; "
        f"pass layout= explicitly"
    )
```

Helper:

```python
def _resolve_attr_path(obj, dotted_path: str):
    return functools.reduce(getattr, dotted_path.split("."), obj)
```

---

## `HFLensModel`

Constructor **mutates** the caller's model in place:

1. `hf_model.eval()`
2. Every parameter → `requires_grad_(False)` (grads only w.r.t. activations)
3. Optional: `tokenizer.add_bos_token = True` when BOS exists (`force_bos`)
4. Optional: per-block `torch.compile` (`compile=True`)

```python
class HFLensModel:
    def __init__(
        self,
        hf_model: nn.Module,
        tokenizer,
        *,
        layout: Layout | None = None,
        compile: bool = False,
        force_bos: bool = True,
    ) -> None:
        ...
        text_config = hf_model.config.get_text_config()
        self.n_layers = text_config.num_hidden_layers
        self.d_model = text_config.hidden_size
        self._logit_softcap = getattr(
            text_config, "final_logit_softcapping", None
        )  # Gemma-style softcap

        # Assert len(layers) == n_layers

        if compile:
            # Per-layer only — preserves hook boundaries
            for i in range(len(self.layers)):
                self.layers[i] = torch.compile(
                    self.layers[i], mode="default", dynamic=False
                )
```

### Methods

```python
@property
def input_device(self) -> torch.device:
    return self._embed_tokens.weight.device

def encode(self, text: str, *, max_length: int = 512) -> torch.Tensor:
    encoded = self.tokenizer(
        text, return_tensors="pt", truncation=True, max_length=max_length
    )
    return encoded.input_ids.to(self.input_device)

def forward(self, input_ids: torch.Tensor):
    # Bare text decoder — NOT the full CausalLM forward (no LM head)
    return self._text_module(input_ids=input_ids, use_cache=False)

def unembed(self, residual: torch.Tensor) -> torch.Tensor:
    target_device = self._lm_head.weight.device
    target_dtype = self._lm_head.weight.dtype
    logits = self._lm_head(
        self._final_norm(residual.to(target_dtype).to(target_device))
    )
    if self._logit_softcap is not None:
        logits = self._logit_softcap * torch.tanh(logits / self._logit_softcap)
    return logits
```

### Why `use_cache=False`

Fitting/apply never need KV cache; keeping it off avoids extra outputs and
keeps hooks simpler.

### Why `force_bos=True`

Some instruction-tuned tokenizers ship with `add_bos_token=False`. Raw-text
prompts without an attention-sink BOS degrade early residual statistics (and
interact badly with `skip_first=16`).

---

## Factory

```python
def from_hf(
    hf_model: nn.Module,
    tokenizer,
    *,
    layout: Layout | None = None,
    text_module: str | None = None,  # deprecated alias → Layout(path=...)
    compile: bool = False,
    force_bos: bool = True,
) -> HFLensModel:
    ...
```

Do **not** combine `compile=True` with `device_map="auto"` (multi-device maps
break per-block compile / hooks).

---

## Non-HF models

Implement `LensModel` directly (see `tests/tiny.py`). The rest of the package
does not know or care about HuggingFace.

---

## Acceptance criteria for this stage

- [ ] Autodetect Llama/Qwen, GPT-2, Phi, NeoX layouts (see `tests/test_hf_layout.py`)
- [ ] Unknown layouts raise with a clear error; `layout=` override works
- [ ] `unembed` returns `[..., vocab]` and applies softcap when present
- [ ] Params frozen; `forward` uses text module with `use_cache=False`
- [ ] Per-block compile still fires `ActivationRecorder` hooks
