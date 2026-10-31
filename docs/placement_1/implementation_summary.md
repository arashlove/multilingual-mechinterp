# Placement 1 — What Has Been Implemented

This document records what exists in the repository at the end of Placement 1: the code, the two experimental runs, their results, where every artefact lives, and what is still open. It is the hand-off point for Placement 2.

`guide_1.md` is the breakdown of the original research plan. `guide_2.md` is the list of changes made after the first run. This file describes what was actually built and measured.

## 1. Research question and scope

> How do sparse autoencoders, the Jacobian Lens, and activation patching differ in the internal representations and causal mechanisms they reveal from the same controlled LLM prompts?

Placement 1 implements and validates the three methods on one model and one shared question set, then compares them. It does not attempt broad claims about multilingual or cultural behaviour; that is Placement 2.

| Method | Baseline paper | What it answers here |
| --- | --- | --- |
| Sparse autoencoder (SAE) | Cunningham, Ewart, Riggs, Huben, Sharkey (ICLR 2024), [arXiv 2309.08600](https://arxiv.org/abs/2309.08600) | Which sparse features are active in the residual stream |
| Jacobian Lens (JLens) | Gurnee et al. (2026), [Verbalizable Representations Form a Global Workspace](https://transformer-circuits.pub/2026/workspace/index.html) | Which vocabulary tokens an intermediate residual is disposed to produce |
| Activation patching | Dumas et al. (2025), *Separating Tongue from Thought* | Whether replacing one language's residual changes the other language's answer |

## 2. Where everything is

| Item | Location |
| --- | --- |
| Code repository | https://github.com/arashlove/multilingual-mechinterp (working copy in OneDrive) |
| Python package | `src/multilingual_mechinterp/`, version 0.1.3, built to `dist/*.whl` |
| Question set | `data/all200questions_persianMiddleEastCulture.json` |
| First run | `notebooks/placement_1_v1.ipynb` (executed, outputs kept) |
| Second run, executed | `notebooks/placement_1_v2.ipynb` (outputs kept) |
| Second run, clean copy | `notebooks/placement_1.ipynb` (same stages, outputs cleared) |
| Pod setup | `notebooks/runpod_setup.py`, `docs/others/runpod_setup.md` |
| Model weights | `/models/gemma-2-9b` on the pod; mirror at `s3://arash-llm-models-790189127408-ap-southeast-2-an/gemma-2-9b/` |
| Second-run results | `output/placement_1_v2/` (gitignored; downloaded locally) |
| First-run results | `output/placement_1/` on the pod; not present in the local copy |
| Method reimplementation notes | `docs/sae_reimpl/`, `docs/jlensreimplement/`, `docs/activation_patching_reimpl/` |
| Unit tests | `tests/` (`test_culture.py`, `test_loader.py`, `test_sae.py`, `test_jlens.py`, `test_patching.py`) |

Earlier exploratory notebooks (`Full_mechanistic_interp*.ipynb`, including a Qwen 2.5 32B variant) predate Placement 1 and are not part of its results.

## 3. Shared experimental setup

Both runs use the same model, prompt format, scoring, and hook.

| Setting | Value |
| --- | --- |
| Model | Base Gemma 2 9B (`google/gemma-2-9b`), not the instruction-tuned model. Hub revision not pinned. |
| Layers | 42 blocks, residual width 3584 |
| Precision | float32 |
| Hooks | Plain PyTorch forward hooks on each block output, the residual stream after the block (`resid_post`). No NNsight or TransformerLens. |
| Token position | Last non-padding token for scoring, SAE analysis, JLens readouts, and patching |
| Seed | 0 |

**Prompt format** (`CultureItem.mcq_prompt` in `data/culture.py`). Question, blank line, four lettered choices, blank line, `Answer:`. Gemma's tokenizer prepends its beginning-of-sequence token. Example, question 116, correct letter A:

```text
Which traditional Middle Eastern water pipe is used for smoking flavoured tobacco in social settings?

A. Shisha
B. Saj
C. Sahn
D. Sidr

Answer:
```

The Persian prompt has the same layout, with the Persian question and choices and the same English `A.`–`D.` labels and `Answer:`.

**Scoring** (`score_mcq_prompt` in `data/mcq.py`). Take next-token logits at the last position, keep the logits of the first token of `A`, `B`, `C`, `D`, and softmax over those four. Recorded fields: predicted letter, correct or not, `p_correct`, the four probabilities, entropy, and the **margin** (correct-letter logit minus the largest wrong-letter logit). The margin is the metric used by every intervention.

**Dataset.** 200 multiple-choice questions on Persian and Middle Eastern culture, each in English, Persian, Bengali, and Urdu, with a difficulty label and a correct index. English–Persian is the comparison pair. Bengali and Urdu are scored behaviourally only.

## 4. The package

Every stage in the notebooks calls this package. The SAE and JLens code was reimplemented from the papers and their reference code, not installed from upstream libraries.

| Module | Contents |
| --- | --- |
| `models/loader.py` | `load_model` → `LoadedModel` (model, tokenizer, device, layer count). Validates a local folder has `config.json`. |
| `models/hooks.py` | `hooked_forward`, `attach_hooks`, `ActivationCache` for residual capture |
| `data/culture.py` | `load_culture_questions`, `CultureItem.mcq_prompt` |
| `data/mcq.py` | `score_mcq_prompt`, `option_token_ids`, `margin_from_probs`, `correct_letter` |
| `sae/learned_dict.py` | `TiedSAE`: codes \(c = \mathrm{ReLU}(Mx + b)\), reconstruction \(\hat{x} = M^\top c\), rows of \(M\) L2-normalised on every forward, Xavier-uniform init |
| `sae/train.py` | `train_tied_sae`, `sweep_l1`. Loss is reconstruction MSE plus \(\alpha \lVert c \rVert_1\). Adam, learning rate `1e-3`. |
| `sae/activations.py` | `extract_residual_activations` (all non-padding tokens at one layer) |
| `sae/features.py`, `sae/intervention.py` | `top_activating_features`, `ablate_features`, `steer_features` |
| `metrics/sae.py` | `evaluate_sae`: FVU, R², mean L0, dead-feature fraction |
| `jlens/fitting.py` | `fit`, `jacobian_for_prompt`. Estimates \(J_l = \mathbb{E}[\partial h_\text{final} / \partial h_l]\) with batched one-hot cotangents, skipping the first 16 positions and the last token. Checkpoints during fitting. |
| `jlens/lens.py`, `jlens/apply.py` | `JacobianLens` (save, load, apply), `run_jlens` (transport residual, unembed, top-k tokens) |
| `jlens/hf.py` | `from_hf`: wraps a Hugging Face causal LM as a lens model |
| `jlens/intervention.py` | `intervene_jlens`: add or subtract a token's output-embedding direction |
| `patching/patch.py` | `run_patching`: per-layer residual replacement, returns the margin change at every layer |
| `patching/intervene.py`, `patching/collect.py` | `patch_residuals` context manager, `get_blocks`, activation collection |

## 5. First run — `placement_1_v1.ipynb`

**Design.** The first eight questions of the dataset (Q1–Q8) were shared across all three methods. Run on an A100 80 GB.

| Setting | Value |
| --- | --- |
| SAE | One layer (20), ratio 2 (7168 features), 4 epochs, alphas `3e-4`, `8.6e-4`, `3e-3`, trained on residual tokens from the eight English and Persian prompts only |
| JLens | Fitted on 2 English prompts, source layers 37–40, dimension batch 16 |
| Patching | English residual into the Persian run, one layer at a time, layers 0–41 |
| Interventions | One showcase item (Q1, Persian) for the SAE and for the JLens |

**Results.**

- Baseline: English correct on 7 of 8 and Persian on 8 of 8. Q4 was the only disagreement (English A, Persian D, key D).
- SAE: held-out FVU about 1.14 for every alpha, meaning reconstruction was worse than predicting the mean. Dead features were 4120–4451 of 7168 (57–62 %). Mean English–Persian top-feature Jaccard was 0.53, and the same handful of features (516, 1659, 4306, 5113, 5737) appeared in almost every item.
- SAE intervention on Q1: reconstruction alone dropped `p_correct` from 0.903 to 0.423. Ablation and steering did not change the letter.
- JLens: late-layer top tokens were dominated by answer letters and the word "Option". Mean late-layer Jaccard 0.37. Suppressing the top token flipped Q1 from C to A (`p_correct` 0.903 → 0.038).
- Patching: mean largest absolute margin change 2.85. Peak layers ranged from 27 to 41. One letter changed (Q4, D → A). The replacement check matched the source residual exactly on every item.
- The SAE layer (20) was outside the JLens window (37–40), so no layer was shared by all three methods.

**What this run showed was wrong.** These problems are the basis of `guide_2.md`:

1. Eight consecutive questions with almost no disagreement cannot show how the methods differ.
2. The SAE had too little data (hundreds of tokens) and did not reconstruct.
3. The JLens fit on two prompts was not representative.
4. No layer was shared by all three methods, so they could not be compared directly.
5. Interventions on a single item cannot show repeatability.

## 6. Second run — `placement_1_v2.ipynb`

Run on an NVIDIA A40 (48 GB), float32, package 0.1.3, Torch 2.8.0+cu128, Transformers 5.19.0, CUDA 12.8. Every stage completed (`records/status.json` is all `ok`). Total runtime was about 15.5 minutes; peak GPU memory was 36.6 GB, during Jacobian fitting.

Each stage writes JSON to `records/` and reloads it on a rerun unless its `FORCE_*` flag is set.

### Stage 1 — Behavioural screening

All 200 questions scored in all four languages.

| Language | Accuracy |
| --- | --- |
| English | 0.945 |
| Persian | 0.855 |
| Urdu | 0.690 |
| Bengali | 0.655 |

English–Persian Jensen–Shannon divergence (JSD, base 2, over the four option probabilities): mean 0.142, median 0.053.

Each question is assigned to one English–Persian group:

| Group | Rule | Count |
| --- | --- | --- |
| A1 | English correct, Persian wrong | 23 |
| A2 | Persian correct, English wrong | 5 |
| A3 | Both wrong, different letters | 2 |
| B1 | Same outcome and letter, but JSD ≥ 0.05 or `p_correct` differs by ≥ 0.10 | 85 |
| B2 | Same outcome and letter, and below both thresholds (control) | 85 |

Output: `records/baseline_all.json`, `figures/01_baseline_screen.png`.

### Stage 2 — Question selection

Selection used behaviour only, never SAE, JLens, or patching results.

- Within each group, questions are sorted by English–Persian JSD. A1, A2, A3, and B1 take the highest JSD. B2 takes the lowest, so the controls are the most alike pairs.
- Target mix for 24 questions: 6 A1, 6 A2, 4 A3, 4 B1, 4 B2. Only 5 A2 and 2 A3 existed, and the 3 open slots went to the next-highest-JSD remaining A1 items.
- Final set: 9 A1, 5 A2, 2 A3, 4 B1, 4 B2.

Evaluation ids: 116, 104, 192, 102, 137, 117, 115, 136, 124 (A1); 13, 146, 4, 9, 32 (A2); 41, 23 (A3); 125, 148, 122, 113 (B1); 62, 14, 30, 165 (B2).

The remaining 176 questions were split, shuffled with seed 0 within each difficulty level, into disjoint sets: 130 SAE training questions, 22 SAE validation questions, 20 JLens fitting questions, and 4 JLens holdout questions.

Output: `records/splits.json`, `records/review_sheet.csv` (both prompts and scores for each selected question, for manual translation review).

### Stage 3 — SAE training

| Setting | Value |
| --- | --- |
| Layers | 20 and 38, one SAE each |
| Training text | 160 prompts sampled from the SAE training questions across all four languages; 40 validation prompts from the validation questions |
| Tokens | 10,005 training and 2,416 validation residual vectors per layer (all non-padding positions, max length 256) |
| Architecture | Tied SAE, ratio 2 (7168 features), ReLU |
| Training | Adam, learning rate `1e-3`, 3 epochs, batch 256. Residual activations were not centered or rescaled. |
| Sweep | alpha `3e-4`, `8.6e-4`, `3e-3` |
| Selection | Lowest held-out mean L0 among runs within 0.05 FVU of the best held-out FVU |

Both layers selected alpha `8.6e-4`:

| Layer | Held-out FVU | R² | Mean L0 | Dead features |
| --- | --- | --- | --- | --- |
| 20 | 0.086 | 0.914 | 662.6 | 63.4 % |
| 38 | 0.192 | 0.808 | 969.9 | 57.8 % |

FVU improved from about 1.14 in the first run to 0.09 and 0.19. Writing the reconstruction back into the model did not change the English answer on any of 6 validation probes.

Output: `records/sae_training.json`, `records/sae_reconstruction_effect.json`, `sae/layer_*/sweep_metrics.json`, `figures/02_sae_pareto.png`.

### Stage 4 — SAE analysis and interventions

**Feature analysis.** For each selected question and each SAE layer, the last-token residual of the English and Persian prompts is encoded. The top 10 features per language are compared with Jaccard overlap. The feature with the largest English–Persian activation gap is the focus feature. Its dictionary row is projected onto the unembedding to find its top vocabulary token, and features whose top token is a letter, punctuation, a beginning-of-sequence marker, or "Answer" are flagged as format-like.

- Mean top-feature Jaccard: 0.867 at layer 20 (9 of 24 pairs identical) and 0.613 at layer 38.
- 1 of 24 focus features was flagged as format-like (Q62, a newline).

**Interventions.** On the Persian prompt at layer 38, for the first selected question of each of four groups. Each edit is compared with the SAE reconstruction, not with the clean model. A control feature (the least active feature outside both top-10 lists) is edited the same way.

| Question | Group | Focus feature (top token) | Δ`p_correct`, ablate focus | Letter changed |
| --- | --- | --- | --- | --- |
| 116 | A1 | 741 (`con`) | −0.005 | No |
| 13 | A2 | 1 (` (`) | −0.032 | No |
| 41 | A3 | 4637 (`Pen`) | −0.013 | No |
| 125 | B1 | 1554 (` period`) | −0.031 | No |

Steering at strengths 1 and 2 changed `p_correct` by less than 0.001. Ablating the control feature changed nothing. Reconstruction alone moved `p_correct` much more than any edit (Q13: 0.913 → 0.567; Q116: 0.052 → 0.111).

Output: `records/sae_features.json`, `records/sae_interventions.json`, `figures/03_sae_jaccard_by_group.png`.

### Stage 5 — JLens fitting

| Setting | Value |
| --- | --- |
| Source layers | 38 and 40 (target is the final block, 41) |
| Fitting prompts | 20, alternating English and Persian, from the JLens fitting questions |
| Estimator | Dimension batches of 32, first 16 tokens skipped, sequences truncated at 128 tokens, last token dropped |
| Stability check | 4 holdout prompts read out with the fitted lens. The repeat fit on half the prompts was off (`RUN_JLENS_REPEAT_FIT = False`). Layer 29 was off. |

Output: `records/jlens_stability.json`, `jlens/jacobian_lens.pt` (deleted by the cleanup cell after analysis).

### Stage 6 — JLens analysis and interventions

**Readouts.** Top 8 tokens at the last position, for both languages, at layers 38 and 40. Overlap is computed twice: on all tokens, and on content tokens only (excluding letters and template tokens). Layer 38 mean content-token Jaccard: 0.312.

**Interventions.** On the Persian prompt at layer 38, for the first question of four groups. The edits add or subtract a content token's output-embedding direction at 0.25 and 0.5 times the residual norm: strengthen the Persian content token, suppress it, swap (add Persian, subtract English), and a control direction for the letter `A`.

- The selected "content" token was ` Option` in most cases, because the exclusion list does not contain "option". The swap edit was then a no-op whenever both languages chose ` Option`.
- Content-direction edits changed `p_correct` by at most 0.04 and changed no letters.
- The letter-`A` control changed the letter on Q116 (B → A, which is correct) and Q125 (B → A, which is wrong). That shows the edit mechanism can move the answer when the direction is an answer letter.

Output: `records/jlens_readouts.json`, `records/jlens_interventions.json`, `figures/04_jlens_overlap_by_group.png`.

### Stage 7 — Bidirectional activation patching

Every selected pair is patched one layer at a time (window 1, layers 0–41), English into Persian and Persian into English. The metric is the change in the target prompt's correct-letter margin. At the layer with the largest absolute change, one more forward pass records the new letter, whether the source letter transferred, whether a wrong answer was recovered or a right one lost, and whether the written residual matches the source.

| Direction | Mean Δmargin at layer 38 | Source letter transferred | Recovered | Worsened | Replacement matched |
| --- | --- | --- | --- | --- | --- |
| English → Persian | +2.93 | 24 of 24 | 9 | 5 | 24 of 24 |
| Persian → English | −2.81 | 22 of 24 | 5 | 10 | 24 of 24 |

Controls, on two questions each:

- Same-language (an English residual written into its own run): mean absolute extreme change 0.000, as expected.
- Unrelated (a different question's English residual written into the Persian run): mean absolute extreme change 5.82.

The unrelated control is large. At the last token of late layers, the residual carries much of the answer by itself, so a large matched patch effect alone does not show the transfer of question-specific content. The control statistic is the extreme over layers and the matched statistic is at layer 38, so they are not directly comparable.

Matched prompts differed in length by 11.5 tokens on average.

Output: `records/patching.json`, `figures/05_patch_mean_delta_margin.png`.

### Stage 8 — Comparison at layer 38

One row per question: SAE top-feature Jaccard, JLens content-token Jaccard, and the patch effect in both directions, all at layer 38. Spearman rank correlations over the 24 questions:

| Pair | ρ |
| --- | --- |
| SAE difference vs. patch effect | −0.18 |
| JLens difference vs. patch effect | 0.01 |
| SAE difference vs. JLens difference | 0.49 |

The two descriptive methods agree moderately with each other, and neither predicts the causal patch effect.

Five case studies are chosen as the question with the largest absolute English → Persian patch effect in each group: Q116 (A1), Q146 (A2), Q41 (A3), Q113 (B1), Q62 (B2). These are chosen separately from the SAE edit questions, which is why the B1 edit used Q125 and the B1 case study is Q113.

Output: `records/comparison.json`, `figures/06_layer38_comparison.png`.

### Stage 9 — Report notes and cleanup

`records/report_notes.md` collects setup, behaviour, SAE, JLens, patching, the layer-38 comparison, the five case studies, the six-criterion comparison, and limitations. It reads only saved records.

The final cell deletes every `.pt` file (SAE checkpoints, cached activations, the lens, and the fitting checkpoint, about 1.3 GB) once SAE analysis and JLens analysis are both `ok`. If a stage failed, its checkpoints are kept so it can resume.

## 7. Six-criterion comparison

| Criterion | SAE | JLens | Activation patching |
| --- | --- | --- | --- |
| Internal representation | Sparse code of the residual | Tokens after a linear map toward the final residual | Causal effect of replacing the residual; names no feature or token |
| Important layers | 20 and 38 | 38 and 40 | 0–41, compared at 38 |
| Interpretability | Needs examples and a format/BOS check | Readable, but dominated by letters and "Option" | Readable signed effects |
| Causal evidence | Edits relative to reconstruction, on 4 items; effects were small | Direction edits on 4 items; only the letter control flipped answers | Systematic: 24 pairs, both directions, two controls |
| Compute (A40) | 100 s training, 37 s analysis | 307 s fitting, 15 s analysis | 343 s |
| Multilingual use | Descriptive overlap across tokenisers | Descriptive overlap across tokenisers | Directly tests whether one language's residual changes the other's decision |

## 8. Limitations and open items

These are known gaps that Placement 2 should either fix or report:

1. **Model revision not pinned.** The Gemma 2 9B commit hash is not recorded. NumPy and Matplotlib versions were not recorded on the pod. `environment.json` should record them in future runs.
2. **Small, disagreement-enriched subset.** 24 questions chosen for disagreement; A2 and A3 had only 5 and 2 candidates. Results do not generalise to all 200 questions.
3. **SAE edits were weak.** Ablation moved `p_correct` by at most 0.03, and the reconstruction error moved it more than any edit. Dead features remain 58–63 %. The SAEs were trained on about 10,000 tokens of question-style text only.
4. **JLens content tokens were mostly template words.** "Option" is not in the exclusion list, so the content overlap and content-direction edits mostly measured a template token. The repeat-fit stability check and layer 29 were not run.
5. **Patching controls are thin and not like-for-like.** Two questions per control type, and the control statistic (extreme over layers) differs from the matched statistic (layer 38). The large unrelated-control effect means the matched effect is not shown to be question-specific.
6. **Only the last token.** Every method reads or edits the final position only.
7. **No research log file.** The run record is `records/environment.json`, `records/report_notes.md`, `stage_times.json`, and `status.json`. A separate log of decisions and failed attempts does not yet exist.
8. **First-run outputs not kept locally.** Its numbers survive only in the executed notebook's printed outputs.

## 9. Reproducing the second run

1. Bump the version in `pyproject.toml` and `src/multilingual_mechinterp/__init__.py` if `src/` changed, run `poetry build`, and push the wheel (`docs/others/runpod_setup.md`).
2. On the pod, download Gemma 2 9B to `/models/gemma-2-9b` (Hugging Face first, S3 after a rate limit), clone the repo, and install `dist/*.whl`.
3. Open `notebooks/placement_1.ipynb` and run all cells. Results go to `output/placement_1_v2/`.
4. Download `output/placement_1_v2/`. After the cleanup cell it holds only JSON, CSV, Markdown, and PNG files.
