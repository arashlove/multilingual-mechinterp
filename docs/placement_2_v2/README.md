# Placement 2, v2 run

`notebooks/placement_2_v2.ipynb` implements the reviewer's recommendations on top of the v1 run (`docs/placement_2_v1/`). Settings are in `configs/placement_2_v2.yaml`, and outputs go to `output/placement_2_v2/`. v1 files are unchanged.

## What is new

| Priority | Change | Files |
| --- | --- | --- |
| 1 | Gemma 2 9B Instruct (float32) with the chat template; neutral A/B labels (2 mappings × 2 orders × 2 languages × 40 = 320 variants); greedy generated answer, A/B probability and invalid-response rate; the base model rerun on the same prompts as the baseline | `placement2/controls.py`, `prompts.neutral_prompt`, `build_neutral_variants` |
| 2 | `counterbalance_pair_id` on UTILITARIAN_005/006 and 007/008 | `inputs/moral_machine_{en,fa}_v6_2.jsonl` (v6_1 text unchanged) |
| 2 | User-clustered intervals and the repeated-respondent check | `human_data.respondent_level_uncertainty` |
| 2 | English–Persian review sheet, reviewer columns empty | `data/placement_2/review/translation_review_v6_2.csv` |
| 3 | JLens calibration on count-substituted study prompts, none of which uses a study configuration; approved concepts with a coverage table; half-A/half-B fit merged into the main lens, with stability metrics | `placement2/calibration.py`, `inputs/jlens_calibration.jsonl`, `inputs/concepts.json` |
| 4 | Interventions on UTILITARIAN_005, UTILITARIAN_006, AGE_006 and FITNESS_007 in both languages and both orders; 2 layers, 7 directions (3 controls), strengths ±0.2/±0.4; probability change, order-averaged decision change and output validity | notebook B9 |
| 5 | Model commit hashes (read from the download metadata), TF32 flag, comparison with v1 scores, per-stage times; stale statements fixed in `placement_2.ipynb` | `experiment.local_commit`, notebook B5/B10 |

## Results already produced on CPU (Part A2)

- **Mirrored pairs (humans, All clusters):**
  - 2 pedestrians vs 5 passengers: the larger group is spared 70.6% of the time, averaged over the two orientations;
  - 5 pedestrians vs 4 passengers: 75.6%;
  - the STAY share averaged over each pair is 51.4% and 51.8%, so there is almost no preference for not intervening;
  - the Eastern cluster is lower on the first pair (61.5%).
- **Repeated respondents:**
  - 3,016 of 350,938 matched users answered one configuration twice, and none more than twice;
  - the design effect is at most 1.022;
  - no human label changes under the user-clustered interval, so the v1 reference labels stand.

## GPU time

Expected to take about 20–25 minutes on an A100 80 GB (not yet measured). The changes from v1:

- TF32 matrix multiplies (weights stay float32);
- one forward per scored prompt instead of one per label;
- only the needed logits are requested from the model.

There are no time limits. Every stage runs at the size set in the config, so to shorten a run reduce `jlens.fit_prompts_max`, the cases, or the intervention layers, directions and strengths. Each stage prints its time.

## Running on the pod

1. Push the repo (including `dist/multilingual_mechinterp-0.1.6-*.whl`, `configs/placement_2_v2.yaml`, the v6_2 question files and `jlens_calibration.jsonl`), then pull it on the pod.
2. Download the instruct model once. It is gated, so accept the licence on its Hugging Face page and set `HF_TOKEN` in the pod environment:
   ```bash
   huggingface-cli download google/gemma-2-9b-it --local-dir /models/gemma-2-9b-it
   ```
   It needs about 37 GB of disk; the base model stays at `/models/gemma-2-9b`.
3. Copy `data/placement_2/processed/moral_machine_40_human_references_by_cluster.csv` to the pod if it is not in the repo.
4. Open `notebooks/placement_2_v2.ipynb` on a fresh kernel. Start at "Part B2" and run to the end. Part A2 has already been run locally.
5. Download `output/placement_2_v2/` (records, figures and `jlens/*.pt`).

If a stage fails, rerun it after fixing: scored rows, lens checkpoints and finished stages are reloaded rather than recomputed.
