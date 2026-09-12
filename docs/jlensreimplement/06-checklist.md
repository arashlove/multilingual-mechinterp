# 06 — Rebuild Checklist, Pitfalls, Verification

---

## Rebuild order (copy into your project)

| Step | Module | Depends on | Done when |
|------|--------|------------|-----------|
| 1 | `protocol` + toy decoder | — | Tiny model encodes/forwards/unembeds |
| 2 | `hooks.ActivationRecorder` | protocol | Grad through activation leaf works |
| 3 | `fitting.valid_position_mask` | — | Mask excludes first 16 + last |
| 4 | `fitting.jacobian_for_prompt` | 1–3 | Exact \(J\) on TinyDecoder |
| 5 | `fitting.fit` + checkpoints | 4 | Resume / skip accounting correct |
| 6 | `lens.JacobianLens` | 5 | transport `@ J.T`, apply, save/load/merge |
| 7 | `hf.from_hf` | 1–2 | Layout autodetection + unembed |
| 8 | `vis` | 6 | slice + ranks + HTML |
| 9 | (optional) paper runners | 6 + data JSON | See [07](07-paper-experiments.md) |

---

## Must-not-get-wrong list

1. **`residual @ J.T`**, not `J @ residual` (unless you reshape to columns).
2. Cotangents on **all valid targets at once**, then **mean over sources**.
3. Params frozen; **`start_graph_at=min(source_layers)`** roots the graph.
4. `forward` **deterministic** across batch replicas (eval, no dropout).
5. **`torch.compile` per block**, never whole module.
6. Fit checkpoint ≠ lens file (`jacobian_sum` vs `J` key).
7. Resume: advance `next_idx` on skips; only increment `n_done` on success.
8. Unembed = **final norm + lm_head** (+ softcap if present).
9. Viz final layer uses **identity** transport (true model logits).
10. Package data: ship `slice_vis.html` if you ship viz.

---

## Property tests to port

Mirror these from [`tests/`](../../tests/) — they define correctness:

### Fitting (`test_fitting.py`)

| Test idea | Assertion |
|-----------|-----------|
| Position mask | Length `seq_len`; False on `[0..skip)` and last index |
| Too short | `seq_len ≤ skip_first+1` → `ValueError` |
| Exact late Jacobian | On TinyDecoder, \(J_2 = I + W_3\) (residual `h+0.1Wh` stack) |
| Earlier layers | Farther from \(I\) than late layers |
| Fit → save → load | fp16 round-trip; apply shapes match |
| Negative layer indices | Resolve correctly |
| Logit lens | `use_jacobian=False` skips transport |
| Merge | Weighted by `n_prompts` |
| Checkpoint resume | Continues; skip-then-resume does **not** double-count |
| Mismatched ckpt | Rejects disagreeing `source_layers` / `target_layer` / `skip_first` |

### HF layout (`test_hf_layout.py`)

Mock module trees for Llama, multimodal, Phi, GPT-2, NeoX; unknown raises;
`from_hf` unembed shape.

### Vis

| File | Assertion |
|------|-----------|
| `test_ranks_of.py` | Chunked ranks ≡ naive argsort |
| `test_compute_slice.py` | Final row = model; pinned ranks; windowing; stride |
| `test_vis_modes.py` | embed/fetch packaging roundtrips |

### Minimal closed-form check (port this)

For TinyDecoder residual \(h \mapsto h + W h\) with \(W = 0.1 \cdot \text{Linear}\):

\[
\frac{\partial h_{L}}{\partial h_{L-1}} = I + W_L
\]

So `jacobian_for_prompt(..., source_layers=[n_layers-2], target_layer=n_layers-1)`
should match `I + layers[-1].linear.weight` (accounting for the estimator's
position average — for a position-independent linear residual map, this is
exact).

---

## Logging (optional)

**File:** [`jlens/_logging.py`](../../jlens/_logging.py)

```python
configure_logging(level=logging.INFO)
# stderr: [elapsed +delta] message on logger "jlens"
```

Idempotent; library callers that already configure logging should skip it.

---

## Project layout to recreate

```text
your_pkg/
  __init__.py          # public exports
  protocol.py          # LensModel
  hooks.py             # ActivationRecorder
  fitting.py           # mask, jacobian_for_prompt, fit
  lens.py              # JacobianLens
  hf.py                # Layout, HFLensModel, from_hf
  vis.py               # optional
  examples.py          # optional
  _logging.py          # optional
  data/
    slice_vis.html     # if shipping vis
tests/
  tiny.py
  test_fitting.py
  ...
```

`pyproject.toml` essentials:

```toml
requires-python = ">=3.10"
dependencies = ["torch", "huggingface_hub", "transformers>=5.5", "numpy"]

[tool.setuptools.package-data]
your_pkg = ["data/*"]
```

---

## Smoke script after port

```python
from tests.tiny import TinyDecoder
from your_pkg import fit, JacobianLens

model = TinyDecoder()
# Need prompts long enough: skip_first=16 ⇒ >17 tokens
prompts = ["hello world " * 20] * 3
lens = fit(model, prompts, dim_batch=4, skip_first=2, max_seq_len=64)
logits, model_logits, ids = lens.apply(model, prompts[0], positions=[-1])
assert set(logits) == set(lens.source_layers)
assert model_logits.shape[-1] == logits[lens.source_layers[0]].shape[-1]
lens.save("/tmp/lens.pt")
assert JacobianLens.load("/tmp/lens.pt").d_model == model.d_model
```

Then graduate to a real HF model + Hub or self-fitted lens.
