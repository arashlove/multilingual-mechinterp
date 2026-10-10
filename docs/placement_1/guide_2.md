# Complete List of Changes for Placement 1 — Version 2

Based on your existing `placement_1_v1` implementation and our discussion, these are all the changes I recommend making.

The main objective is to improve the validity of SAE and JLens, select meaningful questions from your 200-question dataset, and compare all three methods on the same model, questions, and selected layers.

## 1. Dataset and behavioural screening

- Keep the existing 200-question multilingual dataset.
- Run baseline inference on all 200 questions in English, Persian, Bengali, and Urdu.
- Record the predicted option (A/B/C/D) for every question in each language.
- Record the correct option, accuracy, `p_correct`, option probabilities, margin, and entropy.
- Calculate Jensen–Shannon divergence (JSD) between the option-probability distributions of equivalent questions across languages.
- Use the same prompt format and option-scoring method for all languages.
- Save the complete baseline results before running any mechanistic experiments.
- Make English–Persian the primary comparison for Placement 1. Retain Bengali and Urdu behavioural results for supplementary analysis.

## 2. Filter and categorise questions

Instead of analysing the first eight questions, filter the 200 questions based on behavioural differences.

- Group A1: English correct, Persian incorrect.
- Group A2: English incorrect, Persian correct.
- Group A3: Both incorrect, but predict different options.
- Group B1: Both predict the same option, but the option distributions or confidence levels differ substantially.
- Group B2: Both predict the same option with similar confidence (control group).
- Calculate the number of available questions in each group.
- Rank disagreement questions by probability divergence and confidence differences.
- Manually review the translations and correct answers of candidate questions.
- Select approximately 20–30 questions for mechanistic experiments, depending on available cases and computational resources.
- Include examples from all available groups rather than only selecting the largest disagreements.
- Save selected question IDs and their selection criteria in a separate configuration file.
- Keep the final evaluation question set fixed for the remaining experiments.

Important: Do not select questions using results from SAE, JLens, or activation patching. Select them using the behavioural screening criteria so the later experiments can be evaluated more independently.

## 3. Standardise the model and experimental configuration

Use one consistent model and experimental environment for the three methods.

- Continue using Gemma 2 9B.
- Use the same model checkpoint and numerical precision wherever practical.
- Record the exact model revision, PyTorch version, Transformers version, and CUDA environment.
- Use a fixed random seed for reproducible experiments where supported.
- Standardise prompt formatting, answer scoring, and generation settings.
- Standardise the activation hook, preferably the residual stream after each transformer block (`resid_post`) where supported.
- Record exact token positions used for extracting activations.
- Verify that SAE, JLens, and patching refer to compatible activation representations before comparing them.
- Keep separate independent datasets for SAE training, JLens fitting, and final evaluation.
- Save all experimental configurations alongside results.

## 4. Change the layer selection strategy

Previously, SAE analysed layer 20, JLens focused on layers 37–40, and activation patching examined all 42 layers.

The revised configuration should be:

| Method              | Initial layers |
| ------------------- | -------------- |
| SAE                 | 20, 38         |
| JLens               | 38, 40         |
| Activation Patching | 0–41           |

- Retain SAE layer 20 as a middle-layer baseline.
- Add SAE layer 38 to allow direct comparison with JLens.
- Continue JLens analysis at layers 38 and 40.
- Explore JLens layer 29 if fitting and computational requirements permit.
- Continue activation patching across all 42 layers.
- Compare all three methods at layer 38 using the same questions.
- Consider adding SAE at layer 29 or 40 if the initial results justify additional experiments.
- Train or load a separate compatible SAE for every selected SAE layer.
- Avoid assuming that the layer with the greatest patching effect is necessarily the most semantically interpretable.

Priority: Start with one fully validated shared layer, layer 38, rather than immediately expanding to many layers.

## 5. Improve Sparse Autoencoder training

Your current SAE results indicate that the implementation runs, but the reconstruction quality needs improvement.

- Increase SAE training data beyond the previously used 759 tokens.
- Use a more diverse training corpus independent of the selected evaluation questions.
- Include multilingual data in the SAE training corpus.
- Use separate training and validation splits.
- Train separate SAEs for layers 20 and 38.
- Increase training steps or epochs as necessary.
- Evaluate different L1 sparsity coefficients.
- Experiment with feature dictionary size or expansion ratio if reconstruction remains poor.
- Monitor reconstruction loss during training.
- Calculate held-out FVU.
- Calculate reconstruction R².
- Calculate mean L0 (number of active features).
- Calculate the dead-feature percentage.
- Evaluate the reconstruction-versus-sparsity trade-off.
- Check whether substituting reconstructed activations changes the model's original predictions or probabilities.
- Select the SAE configuration based on held-out reconstruction and sparsity performance.
- Freeze the selected SAE weights before analysing the final questions.

Your previous results included FVU of approximately 1.14 and a dead-feature rate of approximately 62%.

The objective is to substantially improve reconstruction while maintaining useful sparsity. There is no single universally valid target FVU or L0; judge improvements relative to held-out baselines and the downstream effect of reconstruction.

## 6. Improve SAE feature analysis

After improving SAE training:

- Extract activations for all selected English–Persian question pairs at layers 20 and 38.
- Encode those activations using the corresponding trained SAE.
- Identify the strongest activated features for each question and language.
- Compare features shared between English and Persian.
- Identify features with substantial activation differences.
- Examine whether patterns differ between A1, A2, A3, B1, and B2.
- Inspect multiple high-activation examples for each interesting feature.
- Use feature autointerpretation only as supporting evidence, not as proof of semantic meaning.
- Check for features dominated by beginning-of-sequence tokens, prompt templates, or answer formatting.
- Avoid assigning cultural interpretations without supporting examples.

### SAE interventions

- Select a small number of well-supported features.
- Run original-model inference.
- Run the SAE reconstruction baseline.
- Ablate selected features.
- Strengthen selected features.
- Compare changes in `p_correct`, margin, and predicted option.
- Measure intervention effects relative to the reconstruction baseline.
- Include control features and, where appropriate, multiple intervention strengths.
- Record whether the changes are repeatable across related questions.

## 7. Improve Jacobian Lens fitting

Your previous JLens fitting used only two English prompts. The revised experiment should use a more representative fitting dataset.

- Create an independent JLens fitting set.
- Start with approximately 20–30 diverse fitting prompts and expand if needed to assess stability.
- Avoid using the selected evaluation questions for fitting.
- Include different topics, question structures, and relevant language variants.
- Confirm how the existing JLens implementation handles multilingual inputs.
- Fit or prepare JLens for layers 38 and 40.
- Extend to layer 29 only if technically justified.
- Evaluate JLens stability using held-out prompts or repeated fits.
- Record fitted parameters, approximation settings, and computational requirements.
- Save the fitted JLens artifacts for reuse.

Increasing the fitting sample does not automatically guarantee better readouts, so evaluate stability rather than relying on a fixed number.

## 8. Improve JLens concept analysis

- Apply the fitted JLens to the selected English–Persian question pairs.
- Extract the top-ranked tokens or concept-related readouts at each selected layer.
- Save the corresponding readout scores.
- Compare readouts across English and Persian.
- Identify shared and language-specific token patterns.
- Calculate exact-token overlap where meaningful.
- Perform a separate semantic comparison that accounts for translation and tokenisation differences.
- Compare JLens results between disagreement and agreement groups.
- Identify cases where intermediate readouts differ despite identical final answers.
- Separate answer-letter and prompt-format tokens from potentially meaningful content-related concepts.
- Avoid interpreting every high-ranked token as a validated internal reasoning step.

### JLens interventions

- Identify candidate content-related concept directions.
- Avoid relying exclusively on answer-letter directions such as `A`, `B`, or `C`.
- Strengthen selected directions.
- Suppress selected directions.
- Attempt direction swapping where technically supported.
- Include matched control directions and intervention-strength checks.
- Record changes in predicted option, probability, and margin.
- Compare intervention effects across selected questions.
- Document whether the results support causal influence, or only a descriptive association.

## 9. Expand Activation Patching

Your activation-patching implementation already works, so this stage mainly requires extending its design and strengthening its controls.

### Layer screening

- Run activation patching across layers 0–41.
- Apply it to the selected English–Persian question pairs.
- Keep the same target activation hook and token-position convention.
- Verify that the replacement is executed successfully.
- Save original source and target predictions and probabilities.

### Bidirectional patching

- Run English → Persian patching.
- Run Persian → English patching.
- Compare effects in both directions.
- Analyse whether the target adopts the source's predicted option.
- Determine whether each intervention improves or reduces answer correctness.
- Compare the results for A1, A2, A3, B1, and B2.

### Patching metrics

- Calculate Δp(correct).
- Calculate Δmargin.
- Record prediction flips.
- Calculate the source-prediction transfer rate.
- Calculate correct-answer recovery where applicable.
- Calculate how often interventions worsen correctness.
- Identify layers with the largest positive and negative effects.
- Summarise signed effects across questions rather than only reporting absolute maxima.
- Compare influential layers between the two patching directions.

### Patching controls

- Include unmodified target passes.
- Include compatible same-language patching controls.
- Include carefully selected unrelated-question controls.
- Verify token-position alignment and record possible confounds.
- Distinguish a causal intervention effect from evidence of a specific semantic concept.

## 10. Compare all three methods at common layers

This is the central part of your research question.

Start with layer 38.

- Use exactly the same selected English–Persian question pairs.
- Analyse the same activation hook and token position wherever the methods permit.
- Compare SAE feature activation patterns.
- Compare JLens concept-related readouts.
- Compare activation-patching effects.
- Investigate whether cases with large SAE differences also exhibit JLens differences.
- Investigate whether either method's differences correspond to large patching effects.
- Examine cases where the methods appear to agree.
- Examine cases where their results differ.
- Avoid claiming that different methods identified the same concept solely because they highlight the same layer.
- Select approximately five informative questions for detailed integrated case studies.

For each case study, explain:

1. What did the model predict in English and Persian?
2. What did SAE reveal?
3. What did JLens reveal?
4. What happened during activation patching?
5. What can you reasonably conclude from the combined evidence?

## 11. Compare the methods formally

Your original research plan requires a comparison across six dimensions.

- Internal representation: What exactly does each method expose?
- Important layers: Which model layers are informative for each method?
- Interpretability: How easy is it to explain the outputs?
- Causal evidence: What causal claims are supported by each intervention?
- Computational requirements: What time, VRAM, and processing requirements does each method have?
- Multilingual suitability: How useful is each method for studying cross-language differences?

Also document where the methods complement each other and where they remain incompatible or difficult to integrate.

## 12. Improve statistical analysis and reporting

- Report full-dataset behavioural accuracy by language.
- Report the proportions of agreement and disagreement questions.
- Report the distributions of probability differences and JSD.
- Summarise SAE reconstruction and sparsity metrics across training configurations.
- Summarise JLens readout overlap and stability.
- Summarise patching effects by layer, language direction, and question category.
- Distinguish descriptive results from intervention-based findings.
- Include uncertainty estimates where appropriate.
- Report the size and selection process of the mechanistic subset.
- Clearly state that a disagreement-enriched subset is not representative of all questions.
- Record failed experiments and methodological limitations.
- Avoid making broad claims about multilingual performance from small selected samples.

## 13. Update your project code and notebook

I recommend keeping the original version untouched.

Organise Version 2 into these stages:

| Stage | Notebook / module          | Purpose                              |
| ----- | -------------------------- | ------------------------------------ |
| 1     | `01_behavioural_screening` | Inference on all 200 questions       |
| 2     | `02_question_selection`    | Categorisation and subset selection  |
| 3     | `03_sae_training`          | SAE training and validation          |
| 4     | `04_sae_analysis`          | Feature extraction and interventions |
| 5     | `05_jlens_fitting`         | JLens fitting and validation         |
| 6     | `06_jlens_analysis`        | Readouts and interventions           |
| 7     | `07_activation_patching`   | Bidirectional layer screening        |
| 8     | `08_method_comparison`     | Cross-method analysis                |
| 9     | `09_final_report_results`  | Consolidated tables and outputs      |

- Preserve `placement_1_v1` as the initial implementation.
- Reuse existing functions wherever possible.
- Separate expensive model execution from lightweight analysis.
- Save intermediate outputs so experiments can be resumed without rerunning everything.
- Make model settings, layers, prompt sets, and experiment settings configurable.
- Log GPU memory and experiment runtime.
- Save all result files with sufficient metadata to reproduce them.
- Avoid keeping multiple full model copies in GPU memory.
- Save important results to persistent storage before terminating the RunPod Pod.

## 14. Final deliverables

- Complete behavioural baseline dataset.
- Fixed and documented mechanistic question subset.
- Validated SAE models for selected layers.
- SAE feature analysis and intervention results.
- JLens fitting artifacts, readouts, and interventions.
- Bidirectional activation-patching results.
- Integrated cross-method case studies.
- Six-criterion method comparison.
- Research log and reproducibility documentation.
- Final research report.
- Final research presentation.
- Confirmed examiner and submission arrangements.

## Priority order

Not every item above needs to be completed simultaneously. I would organise the work into five phases:

| Priority | Task                                                               | Importance                             |
| -------- | ------------------------------------------------------------------ | -------------------------------------- |
| 1        | Screen 200 questions and select informative pairs                  | Essential                              |
| 2        | Improve SAE reconstruction and JLens fitting                       | Essential for stronger interpretations |
| 3        | Run bidirectional patching and SAE/JLens analysis at shared layers | High                                   |
| 4        | Integrate the findings and validate causal claims                  | Essential                              |
| 5        | Expand to additional layers, prompts, or languages                 | Optional if time permits               |

One final recommendation: Treat layer 38 as your primary shared comparison layer, layer 20 as your SAE middle-layer baseline, and all 42 layers as your activation-patching screening range. This keeps the revised experiment focused while allowing meaningful comparison across all three approaches.

Your most important upgrade from Version 1 is not simply increasing the sample size. It is moving from three independently successful implementations to a controlled, reproducible experiment that establishes what the three methods reveal about the same multilingual model behaviour.