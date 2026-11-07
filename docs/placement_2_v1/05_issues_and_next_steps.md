# 5. Limitations, known errors, and next steps

## 5.1 Limitations of run v1

- **The behaviour readout has almost no variation.** Base Gemma 2 9B prefers SWERVE on every question. With no variation in the decision, the alignment, case-selection and intervention stages have little to explain.
- **Human references are configuration-level matches,** pooled over respondents in the balanced sample. Country-level references (including Iran) are too sparse to label per question.
- **Agreement counts only labelled human majorities.** UNDETERMINED references are excluded, never scored either way.
- **Cases were chosen from model behaviour only.** There are no D1/D2 cases, and 5 of the 8 cases are Fitness questions.
- **The JLens results are exploratory:**
  - no repeat-fit stability;
  - draft concept word lists, with multi-token words dropped;
  - calibration on Placement 1 culture questions instead of dilemma-domain prompts.
- **Interventions are small single-position edits** (at most 5% of the residual norm). A null result at these strengths does not rule out effects at larger strengths.

## 5.2 Deviations to record in the write-up

1. Repeat fit off (`RUN_JLENS_REPEAT_FIT = False`): no half-vs-half stability.
2. Draft concepts used (`ALLOW_DRAFT_CONCEPTS = True`): `concepts.json` status is still "draft".
3. JLens calibration used the Placement 1 fallback without a dilemma-domain calibration file.
4. The first JLens fit attempt ran out of GPU memory and was resumed from its checkpoint on the same prompts and settings.
5. Pending approvals:
   - `human.primary_reference_aggregation`;
   - `human.min_matched_responses_per_culture`;
   - `model.revision` (weights are traced by snapshot hashes instead).

## 5.3 Known errors in notebook text

These are stale sentences in `notebooks/placement_2_v1.ipynb` and `notebooks/placement_2.ipynb`. The numbers are unaffected.

| Section | Text | Fact |
| --- | --- | --- |
| 1.9 | "The 40 study questions all use the pedestrians, no-signal layout." | v6_1 uses three layouts: 20 passengers vs pedestrians, 10 pedestrians with no signal, 10 pedestrians at a crossing signal. |
| 2.3 | "The v6_1 questions have no exact mirrors." | Two exact mirrored pairs exist: UTILITARIAN_005/006 and UTILITARIAN_007/008. The mirrored-pair table is empty only because the file has no `counterbalance_pair_id`. |
| 2.5 | "Every pedestrian-vs-pedestrian, no-barrier, no-signal configuration" | The catalog now covers all three layouts (687 configurations). |

## 5.4 Corrections for the slides

- **The D1/D2 definitions are wrong on the slide.** D1 and D2 are not "larger" and "smaller" probability differences: **D1** = English STAY and Persian SWERVE; **D2** = English SWERVE and Persian STAY. Both mean the decision differs between languages. B1 and B2 are correct.
- **Country counts:** the raw-data slide gives 59 Western, 31 Southern and 27 Eastern countries (117 in total) alongside the in-scope user counts, but the balanced sample contains 67, 35 and 28 countries. The 117 appears to be the countries with more than 100 users, the rule in the proposal. Label it that way so it doesn't conflict with the sample plot ("top 10 of 67 countries").
- **Social Status** is listed as a dimension, but it was dropped from the sample (most of its rows were removed from the shared file). Mark it as excluded.
- **Gemma percentages** on the target-sparing chart are label probabilities, not counts of choices. Add a caption, for example: "Gemma prefers SWERVE on every question (p ≈ 0.69 English, 0.61 Persian); differences between bars reflect which action spares the target."
- **Raw p(SWERVE) vs "% sparing the target":** when quoting raw p(SWERVE) (for example U5 = 71.4%), say so. On the chart, the same output appears as 28.6% sparing the larger group.

## 5.5 Recommended next steps

1. **Fix the behaviour readout first,** since everything downstream depends on it:
   - run the instruction-tuned model (`gemma-2-9b-it`) with its chat template;
   - use neutral option labels (A/B, with the mapping to STAY/SWERVE counterbalanced), so the ` SW` token's head start goes away and an action bias can be told apart from a label bias;
   - optionally, calibrate against a content-free prompt (subtract the model's p(SWERVE) on a neutral scenario).
2. **Re-run stages 4–6 and check for D1/D2 cases** before investing GPU time in stages 7–8.
3. **Approve or revise `concepts.json`,** and provide dilemma-domain calibration prompts (disjoint from the study questions) for the lens.
4. **Turn the repeat fit back on** (about 4 extra minutes on a clean 80 GB GPU), so the lens results have a stability measure.
5. **Widen the intervention strengths** (for example 0.1–0.5 of the residual norm) to test whether anything moves at all.
6. **Add `counterbalance_pair_id`** to the U5/U6 and U7/U8 questions, so the mirrored-pair analysis in section 2.3 runs.
7. **Resolve the pending approvals** and set `model.revision`.
