# Slim `output/sae` and `output/jlens`

For analysis you usually only need **figures + numbers** (JSON/CSV). Most of what landed in `sae/` and `jlens/` was **weight / activation tensors**, not results.

## Are those folders’ contents necessary?

**No — not for figures and numeric analysis.**

### `output/sae/`

| Content | Needed for figures/numbers? | Notes |
|---------|----------------------------|--------|
| `l1_sweep/sweep_metrics.json` | **Yes** | FVU, L0, dead features per α |
| Pareto / feature PNGs under `output/figures/` | **Yes** | Prefer `figures/`, not duplicates |
| `acts_L*.pt` | **No** | Full residuals; re-extract if retraining |
| `l1_sweep/tied_sae_alpha_*.pt` | Only if reloading SAE | ~200MB each on 32B; keep **one** or none after plots |
| `l1_sweep/learned_dicts.pt` | **No** | Duplicate of all checkpoints |

### `output/jlens/`

| Content | Needed for figures/numbers? | Notes |
|---------|----------------------------|--------|
| `en_fa_verbalizable.json` (+ other JSON tables) | **Yes** | Ranks, top-k, summaries |
| PNGs under `output/figures/` | **Yes** | Prefer `figures/` |
| Duplicate PNGs inside `jlens/` | **No** | Same plots again |
| `qwen_jlens_fit.ckpt` | **No** after fit finishes | fp32 Jacobian sums; multi‑GB on 32B |
| `qwen_lens.pt` | Only to re-run Jacobian readout | Dense `d×d` × layers; multi‑GB |

Also keep elsewhere: `output/comparison/*.json`, `*.csv`, and `output/figures/*`.

## Cleanup cell (preferred)

`Full_mechanistic_interp.ipynb` ends with a **Cleanup** cell that deletes the same heavies (flags: `DELETE_*`, `DRY_RUN`, `KEEP_SAE_CHECKPOINT`). Run it after Part 6/7 exports.

## Safe cleanup on an existing run (pod shell)

```bash
cd /workspace/multilingual-mechinterp/output

# SAE heavies
rm -f sae/acts_L*.pt
rm -f sae/l1_sweep/learned_dicts.pt
# optional: keep one SAE, delete the rest
# ls sae/l1_sweep/*.pt

# JLens heavies
rm -f jlens/qwen_jlens_fit.ckpt
rm -f jlens/qwen_lens.pt
# optional: remove duplicate plots living under jlens/ if figures/ already has them
```

Then sync home only:

```text
output/figures/
output/comparison/
output/sae/l1_sweep/sweep_metrics.json
output/jlens/*.json
```

## Defaults in the main notebook (updated)

`notebooks/Full_mechanistic_interp.ipynb`:

- `SAVE_ACTS = False`
- `SAVE_JLENS_FIT_CKPT = False`
- `SAVE_JLENS_WEIGHTS = False`
- `save_learned_dicts=False` in `sweep_l1`
- `SAVE_SAE_CHECKPOINTS = True` still (for the top-feature cell); set `False` if you only want `sweep_metrics.json`

Package: `sweep_l1(..., save_learned_dicts=False)` by default so it no longer writes the duplicate `learned_dicts.pt`.
