# Placement 2, run v1: what has been done

Placement 2 asks whether base Gemma 2 9B answers Moral Machine dilemmas differently in English and Persian, how its answers compare with human majorities in the three Moral Machine cultural clusters (Western, Eastern, Southern), and whether a Jacobian Lens can find and steer concept directions behind the decision.

Run v1 is the first complete end-to-end run. The executed notebook is [`notebooks/placement_2_v1.ipynb`](../../notebooks/placement_2_v1.ipynb).

| Part | Where it ran | Outputs |
| --- | --- | --- |
| A: human data (stages 1–2) | Local CPU | `output/placement_2/stage_1_human_sample/`, `output/placement_2/stage_2_human_references/`, `data/placement_2/processed/moral_machine_40_human_references_by_cluster.csv` |
| B: model (stages 3–9) | RunPod, 80 GB GPU | `output/placement_2_v1/` (`records/`, `figures/`, `jlens/`) |

## Status

All seven Part B stages finished (`records/status.json`: prompts ok, baseline loaded, alignment ok, select loaded, jlens_fit ok, jlens_readouts ok, interventions ok). The zero-strength patch check reproduces the clean scores to 2.7e-6.

## Headline results

1. **Gemma answers SWERVE on every question.** All 160 prompts (40 questions × 2 languages × 2 option orders) give SWERVE. Order-averaged p(SWERVE) is 0.667–0.728 in English and 0.554–0.656 in Persian, so the scenario barely moves the answer.
2. **Agreement with humans is at chance, for a mechanical reason.** Agreement with the labelled human majority is 50–54% in every language × cluster cell. Because Gemma always says SWERVE, this equals the share of questions whose human majority is SWERVE (53.8%). Model and human p(SWERVE) are uncorrelated (r = 0.01 English, −0.22 Persian).
3. **Persian shifts every question the same way.** Persian lowers p(SWERVE) on all 40 questions (mean −0.087), with no decision flips. So there are no D1/D2 cases; the case groups are 14 B1 and 26 B2.
4. **The mirrored pairs show the bias directly.** In U5/U6 and U7/U8, the same groups appear with the action reversed. Humans favour the larger group in both orientations (about 72–78% Western). Gemma's p(SWERVE) stays about 0.71–0.72 in both.
5. **Interventions have no effect beyond the controls.** 0 decision flips in 3,520 edits; the largest |Δp(SWERVE)| is 0.028. Lens-derived directions move p(SWERVE) by no more than random, unrelated-word and template-word directions, and the "STAY-killed" and "SWERVE-killed" directions push the same way.

## Documents

| File | Contents |
| --- | --- |
| [01_human_data.md](01_human_data.md) | Raw Moral Machine data, the balanced 300k sample, stage 1 descriptives, stage 2 per-question human references |
| [02_question_set.md](02_question_set.md) | The 40 study questions (v6_1): design, layouts, matching to human data, mirrored pairs, revision history |
| [03_model_run.md](03_model_run.md) | Part B methods, run settings, deviations from the plan, timings, RunPod issues |
| [04_results.md](04_results.md) | Behaviour, alignment, language comparison, case selection, Jacobian Lens, interventions, target-sparing chart |
| [05_issues_and_next_steps.md](05_issues_and_next_steps.md) | Limitations, pending approvals, known errors in notebook text and slides, recommended next steps |
