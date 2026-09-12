# 05 — Slice Visualization

**File to mirror:** [`jlens/vis.py`](../../jlens/vis.py)

Optional product surface: interactive position × layer heatmap of top-1 lens
tokens with rank overlays. Ship package data `jlens/data/slice_vis.html`.

---

## Pipeline overview

```text
compute_slice(model, lens, prompt)
    → SliceData (numpy arrays + vocab fragment)
build_page(slice_data, ...) / write_slice_files(...)
    → HTML (embed) or HTML + meta.json + slice.bin + ranks/*.bin (fetch)
```

---

## `_ranks_of` — memory-safe full-vocab ranks

```python
def _ranks_of(
    logits: torch.Tensor,   # [seq_len, vocab]
    target_ids: torch.Tensor,  # [n_targets] or [seq_len, n_targets]
    *,
    chunk_size: int = 256,
) -> torch.Tensor:          # [seq_len, n_targets], 0 = top
    seq_len, vocab = logits.shape
    out = torch.empty(seq_len, target_ids.shape[-1], dtype=torch.long,
                      device=logits.device)
    arange = torch.arange(vocab, device=logits.device)
    for start in range(0, seq_len, chunk_size):
        sl = slice(start, start + chunk_size)
        sorted_idx = logits[sl].argsort(dim=-1, descending=True)
        full_rank = torch.empty_like(sorted_idx)
        full_rank.scatter_(1, sorted_idx, arange.expand_as(sorted_idx))
        idx = target_ids if target_ids.ndim == 1 else target_ids[sl]
        out[sl] = full_rank.gather(1, idx.expand(full_rank.shape[0], -1))
        del sorted_idx, full_rank
    return out
```

Peak memory ≈ one `[chunk_size, vocab]` sort buffer — required for large vocabs.

---

## `SliceData`

```python
@dataclass
class SliceData:
    seq_len: int
    layers: list[int]                 # always includes final layer
    context_token_ids: list[int]      # full prompt
    context_token_strs: list[str]
    top_ids: np.ndarray               # [seq, n_layers, top_n] int32
    top_ranks: np.ndarray             # [seq, n_layers, top_n] int32
    tracked_token_ids: list[int]
    rank_tensor: np.ndarray           # [seq, n_layers, n_tracked] int32
    vocab_fragment: dict[int, str]
    vocab_size: int = 0
    pinned_token_ids: list[int] = field(default_factory=list)
    ctx_offset: int = 0               # >0 when last_n_tokens windowed
```

**Final layer always included with \(J = I\)** — that row is the model's true
output, so divergences from earlier lens rows are visible.

---

## `compute_slice`

```python
@torch.no_grad()
def compute_slice(
    model: LensModel,
    lens: JacobianLens,
    prompt: str,
    *,
    top_n: int = 10,
    max_tracked: int | None = None,
    pinned_token_ids: set[int] | None = None,
    layer_stride: int = 1,
    last_n_tokens: int | None = None,
    max_seq_len: int = 512,
    mask_display: bool = False,
) -> SliceData:
```

### Layer selection

```python
fitted_layers = lens.source_layers
layers = fitted_layers[::layer_stride]
if fitted_layers[-1] not in layers:
    layers.append(fitted_layers[-1])
if final_layer not in layers:
    layers.append(final_layer)
layers = sorted(set(layers))
```

### Forward once, unembed per layer

```python
with ActivationRecorder(model.layers, at=layers) as recorder:
    model.forward(input_ids)
    activations = {layer: recorder.activations[layer].detach() for layer in layers}

def lens_logits(layer: int) -> torch.Tensor:
    residual = activations[layer][0, start:].float()
    if layer in lens.jacobians:
        residual = lens.transport(residual, layer)
    # else: final_layer, J = I
    return model.unembed(residual).float().detach()
```

`start = 0` unless `last_n_tokens` windows the slice grid (forward still uses
the full prompt; `ctx_offset` records the offset).

### Pass 1 — top-K per cell

Do **not** retain all-layer logits (memory blow-up). Per layer: `topk` (or
masked topk if `mask_display`) → fill `top_ids` / `top_ranks`.

With `mask_display=True`, restrict *displayed* tokens to word-like vocab
entries; ranks stay full-vocab via `_ranks_of`.

### Tracked token selection

Score each token that appears in the top-K grid:

```
score[token] += 1 / (rank + 1)
```

Take top `max_tracked` by score, union with `pinned_token_ids`.
`max_tracked=None` → every token that appeared in any top-K cell.

### Pass 2 — full ranks for tracked IDs

Recompute `lens_logits(layer)` and `_ranks_of(logits, tracked_tensor)` per
layer → `rank_tensor`.

### Vocab fragment

Decode unique IDs from top-K ∪ tracked ∪ context for the page.

---

## Page packaging

### Binary formats

| File | Format |
|------|--------|
| `slice.bin` | gzip of `top_ids<i4>` then `top_ranks<i4>`, each `[T, L, top_n]` row-major LE |
| `ranks/{tid}.bin` | gzip of `[T, L]` int32 LE for that tracked token |
| `meta.json` | title, prompt, layers, ctx_strs, tracked, vocab, pinned, … |

### Modes

| Mode | Behavior |
|------|----------|
| `"embed"` | Base64-inline files + inline d3 → single self-contained HTML |
| `"fetch"` | Write sidecars; page fetches `./meta.json`, `./slice.bin`, lazy `ranks/` |

d3@7.9.0 with SRI. Embed mode fetches d3 once, verifies integrity, inlines it.
Template placeholder `__D3__`; bootstrap JSON replaces `__BOOTSTRAP__`
(escape `</` → `<\/` so vocab can't close `<script>`).

### API

```python
write_slice_files(slice_data, out_dir, *, prompt, title, description, ...)
build_page(slice_data, prompt, *, title, description, mode="embed", out_dir=None, ...)
    -> (html, raw_bytes, payload_bytes)
notebook_iframe(page, *, height=620)  # srcdoc iframe for notebooks
```

For long prompts prefer `"fetch"` — rank files dominate payload
(~`len(tracked) * seq * n_layers * 4` pre-compression).

---

## Examples helper (optional)

**File:** [`jlens/examples.py`](../../jlens/examples.py)

- `EXAMPLES`: demo prompts (raw + chat-template)
- `resolve_prompt(example, tokenizer)`: apply chat template when needed
- `load_wikitext_prompts(n, min_chars=600)`: streaming WikiText-103 for fitting

Package data required: `jlens/data/slice_vis.html`, `jlens/data/blackmail.json`.

---

## Acceptance criteria for this stage

- [ ] `_ranks_of` matches naive full argsort (1D and per-position targets)
- [ ] Final layer row = model argmax / true output
- [ ] `last_n_tokens` sets `ctx_offset` correctly
- [ ] embed/fetch round-trips binary payloads
- [ ] Package ships HTML template via setuptools `package-data`
