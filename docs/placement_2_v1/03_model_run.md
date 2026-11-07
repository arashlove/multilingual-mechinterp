# 3. Model run (Part B, stages 3–9)

## 3.1 Environment

| Item | Value |
| --- | --- |
| Model | Base Gemma 2 9B, `/models/gemma-2-9b` (downloaded from Hugging Face to the pod), float32, 42 layers, d_model 3584 |
| Hardware | RunPod pod, 80 GB GPU |
| Package | `multilingual_mechinterp` 0.1.5, installed from `dist/` by `notebooks/runpod_setup.py` |
| Seed | from `configs/placement_2.yaml` |
| Records | `output/placement_2_v1/records/environment.json` (model snapshot hashes, library versions, git info) |

## 3.2 What each stage does

**Stage 3: bilingual benchmark.** Loads the English and Persian v6_1 files and validates the Persian file against the English one (valid). It builds 160 prompt variants: 40 questions × {en, fa} × {STAY first, SWERVE first}.

**Stage 4: behaviour.** For each variant:
- **Score both labels.** The code adds up the log-probabilities of the continuation tokens of ` STAY` (one token) and ` SWERVE` (two tokens, ` SW` + `ERVE`) after the prompt and `\n\nAnswer:`. The model's normal final-layer output is used, not a lens.
- **Renormalise over the two labels:** p(SWERVE) = e^{log p(SWERVE)} / (e^{log p(STAY)} + e^{log p(SWERVE)}).
- **Average over the two orders** per question and language, so a preference for the first-listed option cancels. `position_bias` = p(SWERVE | SWERVE first) − p(SWERVE | STAY first).
- **Decide:** SWERVE if the averaged p(SWERVE) > 0.5.
- **Also saved:** `first_token_mass`, the next-token probability on the two labels' first tokens.
- **Compare languages:** English is compared with Persian per question (Δp_lang, Jensen–Shannon divergence) and each question is assigned a group:

| Group | Meaning |
| --- | --- |
| D1 | English STAY, Persian SWERVE |
| D2 | English SWERVE, Persian STAY |
| B1 | Same decision, different probabilities (JSD or \|Δp\| above threshold) |
| B2 | Same decision, similar probabilities |

**Stage 5: alignment with humans.** For each question, language and cluster:
- |p_model(SWERVE) − p_human(SWERVE)|, its square, and the cross-entropy;
- agreement of the model's decision with the human majority label, with UNDETERMINED references excluded.

Intervals are bootstrap 95% over questions.

**Stage 6: case selection.** Up to 4 cases per group, chosen from model behaviour only. D1/D2/B1 are ranked by the largest |Δp_lang|, and B2 by the smallest JSD (stable controls). The list is frozen in `records/cases.json`.

**Stage 7: Jacobian Lens.**
- **Fit:** layers 22, 26, 30 and 38, toward layer 41.
- **Calibration prompts:** 20 fit and 4 holdout, using the Placement 1 culture multiple-choice questions alternating English and Persian. This is the fallback in the plan; no dilemma-domain calibration file was provided.
- **Readouts:** each selected case is read at the end of the STAY line, the end of the SWERVE line and the answer position. Saved per layer: the raw top 50 tokens, the top 8 content tokens, and concept-family scores.

**Stage 8: interventions.**
- **Edit:** at the answer position, the residual is replaced by residual + s · ‖residual‖ · direction, at each lens layer, for s in {±0.01, ±0.03, ±0.05}, in both languages.
- **Directions:**
  - the embedding baseline and the Jacobian back-projected direction for the family STAY would kill and the family SWERVE would kill;
  - their difference ("swap");
  - controls: random, an unrelated concept, template words, and zero strength.
- **Size:** 8 cases × 2 languages × 4 layers × 55 edits = 3,520 edits.

**Stage 9: report notes.** Writes `records/report_notes.md` from the saved records.

## 3.3 Settings used in v1, and deviations from the plan

| Setting | Plan / config | v1 run | Effect |
| --- | --- | --- | --- |
| `RUN_JLENS_REPEAT_FIT` | true (two half-fits, 10 + 10 prompts) | **False** | No stability measure across calibration subsets; the half-vs-half fields in `jlens_stability.json` are null |
| `ALLOW_DRAFT_CONCEPTS` | False (needs `concepts.json` status "approved") | **True** | Concept scores and concept-based directions use the draft word lists; results are draft |
| JLens calibration | Domain prompts disjoint from the study questions, preferred | Placement 1 culture questions (fallback) | Domain mismatch; needs researcher approval and disclosure |
| `INTERVENTION_N` | 8 | 8 | As planned |
| Pending approvals | — | `human.primary_reference_aggregation`, `human.min_matched_responses_per_culture`, `model.revision` | Model weights are traced by snapshot hashes while `model.revision` is unset |

Draft concept words that are not a single Gemma token are dropped. In Persian these were:
- مردان (Male);
- چاق, تنومند, درشت (Fat);
- ورزشکار, ورزشکاران, ورزیده (Fit);
- سالمند, سالمندان, مسن (Old);
- گربه, حیوان, حیوانات (Pets);
- رودخانه (unrelated control).

Where a Persian family had no single-token words left, the interventions used the English words (recorded in `concept_word_language`).

## 3.4 Timings and memory

| Stage | Seconds | Peak GPU (GB) |
| --- | --- | --- |
| Baseline (stage 4) | 124.5 | 35.0 |
| JLens fit | 262.5 (final, resumed run) | 43.8 |
| JLens readouts | 29.1 | 34.5 |
| Interventions | 3,167.7 (about 53 min, about 6.6 min per case) | 35.0 |

## 3.5 Issues on the pod

- **CUDA out of memory in the first JLens fit attempt.** The float32 weights (about 37 GB) plus the backward graph from layer 22 to 41, for a batch of 32 copies × up to 128 tokens, exceeded 80 GB. Restarting the kernel freed the leftover memory. The fit then resumed from its per-prompt checkpoint on the same prompts and settings, and finished in about 4 minutes. Because the fit sums over prompts, the resumed lens matches an uninterrupted fit.
- **The first fit attempt was slow** (more than 25 minutes without finishing) because the GPU was near full.
- **Leftover files:** `jlens/jacobian_lens_fit_checkpoint.pt` (about 205 MB) is a leftover fit checkpoint. `jlens/jacobian_lens.pt` (about 103 MB) is the fitted lens and is kept.
