# 01 — Protocol and Hooks

Implement these first. Everything else depends on them.

---

## `LensModel` protocol

**File to mirror:** [`jlens/protocol.py`](../../jlens/protocol.py)

Any model (HF, custom, toy) plugs in by implementing these members. Fitting and
`apply` never touch the tokenizer; visualization does.

### Required attributes

| Attribute | Type | Meaning |
|-----------|------|---------|
| `n_layers` | `int` | Number of residual blocks |
| `d_model` | `int` | Residual-stream width |
| `layers` | `Sequence[nn.Module]` | Indexable residual blocks (hook targets) |
| `tokenizer` | any | Must provide `decode(token_ids) -> str` |

### Required methods

```python
def encode(self, text: str, *, max_length: int = ...) -> torch.Tensor:
    """Tokenize to input_ids of shape [1, seq_len] on the model's input device."""
    ...

def forward(self, input_ids: torch.Tensor) -> Any:
    """Run the residual stack (no LM head).

    Must:
    - Build an autograd graph through `layers` when grad is enabled
    - Be deterministic across batch elements (eval mode, dropout off)
      — the fitting estimator replicates the prompt along the batch axis
    """
    ...

def unembed(self, residual: torch.Tensor) -> torch.Tensor:
    """Map residual [..., d_model] → logits [..., vocab_size]
    (final norm + LM head)."""
    ...
```

### Reference toy implementation

**File:** [`tests/tiny.py`](../../tests/tiny.py)

Use this pattern to unit-test fitting before touching HF:

```python
class _ResidualBlock(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.linear = nn.Linear(d_model, d_model, bias=False)
        with torch.no_grad():
            self.linear.weight.mul_(0.1)  # keep J well-conditioned

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return hidden + self.linear(hidden)


class TinyDecoder(nn.Module):
    def __init__(self, n_layers=4, d_model=8, vocab_size=32, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.n_layers = n_layers
        self.d_model = d_model
        self.tokenizer = _ByteTokenizer()  # encode + decode surface
        self.embed_tokens = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([_ResidualBlock(d_model) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def encode(self, text, *, max_length=128):
        return self.tokenizer(text, max_length=max_length).input_ids.to(
            self.embed_tokens.weight.device
        )

    def forward(self, input_ids):
        hidden = self.embed_tokens(input_ids)
        for block in self.layers:
            hidden = block(hidden)
        return SimpleNamespace(last_hidden_state=hidden)

    def unembed(self, residual):
        return self.lm_head(self.norm(residual.float()))
```

Why `h + 0.1 * Wh`: late-layer Jacobians stay near \(I + W_{\text{later}}\), so
tests can assert exact closed-form equality.

---

## `ActivationRecorder`

**File to mirror:** [`jlens/hooks.py`](../../jlens/hooks.py)

Forward-hook context manager that captures residual-stream tensors at given
block indices. Stored tensors are **not** detached — they must be usable as
inputs to `torch.autograd.grad`.

### Constructor contract

```python
class ActivationRecorder:
    def __init__(
        self,
        blocks: Sequence[nn.Module],
        at: Iterable[int],
        *,
        start_graph_at: int | None = None,
    ) -> None:
        ...
        self.activations: dict[int, torch.Tensor] = {}
```

- `blocks`: e.g. `model.layers`
- `at`: block indices to record
- `start_graph_at`: if set, the captured tensor at that index is marked
  `requires_grad_(True)` before downstream blocks see it. With all model
  params frozen, this residual becomes the autograd leaf, so the retained
  graph spans only this block onward.

If `start_graph_at` is not already in `at`, include it automatically.

### Hook body (exact behavior)

```python
def _make_hook(self, index: int) -> Callable[..., None]:
    is_graph_root = index == self._start_graph_at

    def hook(module: nn.Module, inputs, output) -> None:
        # Some HF blocks return (hidden, present_kv, ...)
        tensor = output if torch.is_tensor(output) else output[0]
        if is_graph_root:
            tensor.requires_grad_(True)
        self.activations[index] = tensor

    return hook
```

### Context manager

```python
def __enter__(self) -> ActivationRecorder:
    try:
        for index in self._indices:
            self._handles.append(
                self._blocks[index].register_forward_hook(self._make_hook(index))
            )
    except Exception:
        for handle in self._handles:
            handle.remove()
        self._handles = []
        raise
    return self

def __exit__(self, *exc) -> None:
    for handle in self._handles:
        handle.remove()
    self._handles = []
```

### Usage patterns

**Fitting** (needs graph rooted at earliest source layer):

```python
with ActivationRecorder(
    model.layers,
    at=[*source_layers, target_layer],
    start_graph_at=min(source_layers),
) as recorder:
    model.forward(replicated_ids)
    # recorder.activations[layer] → [batch, seq, d_model]
```

**Apply / viz** (no grads needed):

```python
with ActivationRecorder(model.layers, at=record_at) as recorder:
    model.forward(input_ids)
    activations = {i: recorder.activations[i].detach() for i in record_at}
```

### Pitfalls

| Mistake | Effect |
|---------|--------|
| Detaching in the hook | `autograd.grad` fails or returns None |
| Whole-module `torch.compile` | Hooks never fire / graph bypassed |
| Leaving params with `requires_grad=True` | Huge graphs through weights; OOM / wrong leaf |
| Not handling tuple outputs | Breaks on most HF decoder blocks |
| Forgetting to remove hooks | Leaks / double-capture on next forward |

---

## Acceptance criteria for this stage

- [ ] Toy `TinyDecoder` implements the protocol
- [ ] `ActivationRecorder` captures `[batch, seq, d_model]` at requested indices
- [ ] With params frozen + `start_graph_at`, `torch.autograd.grad(target, source)` works
- [ ] Hooks are always removed on exit, including on registration failure
