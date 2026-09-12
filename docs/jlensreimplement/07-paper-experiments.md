# 07 — Paper Experiments & Evaluations (Data Only)

The Python package does **not** implement experiment runners. Prompt sets live
under [`data/`](../../data/) with schemas documented in the READMEs. Reimplement
runners in your project if you need paper replication.

---

## Shared conventions

From [`data/experiments/README.md`](../../data/experiments/README.md) and
[`data/evaluations/README.md`](../../data/evaluations/README.md):

| Term | Meaning |
|------|---------|
| **Lens readout** | At each (layer, position), ranked vocab from the Jacobian lens |
| **Workspace band** | Contiguous mid-network layers where workspace content is read; metrics aggregate over the band |
| **Hit** | Target token appears at lens **rank 1** at any (layer, position) in the band over the scored span |
| **Swap** | Clamp a lens coordinate: replace one token's direction with another's at every band layer × specified positions, then sample the continuation |
| Multi-turn prompts | `[{"role": "user"\|"assistant", "content": ...}]` |

### Steering direction (verbal-introspection)

Jacobian-lens steering direction for a surface token:

- Take the **unit-normalized transpose row** of \(J_l\) for that token
- Scale by the layer's **mean residual norm** × a strength scalar
- Add to the residual stream at every band layer × every token of the user's
  question turn

This is described in the experiment README, not implemented as a library API.

---

## Experiment prompt sets (`data/experiments/`)

| File | What to implement |
|------|-------------------|
| `probe-swap.json` | Two-hop facts; swap intermediate→swap_to across band; score next-token |
| `verbal-introspection.json` | Inject steered thought; rank of reported word at open quote |
| `verbal-report.json` | "Think of a {category}"; swap answer→candidate; rank at final `:` |
| `directed-modulation.json` | Instruction about X + carrier sentence; hit rate of X in lens |
| `top-down-summoning.json` | Passage + Q1/Q2; expected vs foil hits; causal swaps |
| `flexible-generalization.json` | Arg/function templates; swap args; score mapped answers |
| `selectivity-language.json` | Multilingual passages; explicit vs automatic label hits |
| `selectivity-linecount.json` | Line-length counting; number-token hits by condition |
| `ignition.json` | Embedding mixtures α·A+(1−α)·B; reciprocal-rank share heatmaps |
| `capacity.json` | 80-word blocked lists; how many prior words stay ≤ rank k |
| `dual-task.json` | Covert concept/math while copying carrier; interference |

---

## Lens-quality evals (`data/evaluations/`)

Common metric pattern (**pass@k**): mean over items of the fraction of
`intermediates` whose **min-over-layers** lens rank ≤ k.

| File | Readout position |
|------|------------------|
| `lens-eval-multihop.json` | Token immediately preceding `target` |
| `lens-eval-multilingual.json` | Token immediately preceding `target` |
| `lens-eval-poetry.json` | Last newline (end of couplet line 1) |
| `lens-eval-order-ops.json` | Preceding `target`; rank = min over synonym tokens |
| `lens-eval-association.json` | Final prompt token (closing period) |
| `lens-eval-typo.json` | Final prompt token (last fragment of misspelling) |

---

## Minimal eval harness sketch

Using only the library you reimplemented:

```python
lens_logits, _, input_ids = lens.apply(model, prompt, layers=band_layers)

def min_rank(token_id: int) -> int:
    ranks = []
    for layer, logits in lens_logits.items():
        # logits: [n_pos, vocab] — select scored positions
        order = logits[pos].argsort(descending=True)
        ranks.append(int((order == token_id).nonzero()[0]))
    return min(ranks)

# pass@k: fraction of intermediates with min_rank(t) <= k
```

For **swap** interventions you need a forward hook that replaces residual
components along lens directions at band layers — that machinery is not in
`jlens`; build it on top of `ActivationRecorder` + `JacobianLens.jacobians`.

---

## License note

Code and synthetic prompt JSON: Apache-2.0. No model weights or text corpora
are bundled; runtime downloads follow their own licenses.
