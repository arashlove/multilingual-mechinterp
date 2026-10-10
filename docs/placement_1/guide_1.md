# FOSE7901 Research Placement 1 — Complete Plan Breakdown

I've reviewed your 9-page research plan, Implementing Mechanistic Interpretability Methods for Multilingual LLM Analysis.

Your project is designed to be the first phase of a larger research programme, rather than a standalone investigation into multilingual LLM behaviour.

The central idea is to implement three mechanistic interpretability methods, understand what each reveals about an LLM's internal computations, and evaluate how they can work together.

## 1. What are you trying to achieve?

Your main research question is:

> How do sparse autoencoders, the Jacobian Lens, and activation patching differ in the internal representations and causal mechanisms they reveal from the same controlled LLM prompts?

In simpler terms, you want to investigate three things:

1. What information is represented inside the model? Use Sparse Autoencoders (SAEs) to identify potentially meaningful internal features.
2. How does that information become visible across layers? Use the Jacobian Lens (JLens) to examine concepts associated with the model's computations at different depths.
3. Does that information actually influence the answer? Use Activation Patching to test the causal importance of internal activations.

By the end, you should understand the strengths, limitations, and relationships between all three methods.

Your research workflow

One open-source LLM

Gemma 2 9B (preferred)

Controlled multilingual prompts

English and Persian examples, plus other controlled pairs

SAE

Features

What is represented?

JLens

Concepts

Where does it appear?

Patching

Causality

Does it affect output?

Method comparison and integrated findings

Results, figures, reproducibility records, final report

An important distinction: Placement 1 is primarily about implementing and validating these methods, not yet establishing new findings about cultural differences using a large dataset. That broader research is proposed for Placement 2.

## 2. Stage 1 — Baseline method review

Proposal Section 1.1 · Week 1

Before implementing the methods, your proposal requires a focused review of three baseline papers:

| Method              | Baseline paper                                                                                 |
| ------------------- | ---------------------------------------------------------------------------------------------- |
| SAE                 | Huben et al. (2024), Sparse Autoencoders Find Highly Interpretable Features in Language Models |
| JLens               | Gurnee et al. (2026), Verbalizable Representations Form a Global Workspace in Language Models  |
| Activation Patching | Dumas et al. (2025), Separating Tongue from Thought                                            |

For each paper, you need to establish the model components being investigated, what inputs the method requires, what mathematical or computational procedure it performs, what results it produces, and any limitations that could prevent the method from working on your chosen model.

This is not intended to be a second extensive literature review.

Expected outcome: A technical foundation that allows you to explain and justify each implementation, including any departures from the original experiments.

## 3. Stage 2 — Implement Sparse Autoencoders

Proposal Section 1.2 · Week 2

### What are you investigating?

Transformers contain high-dimensional activation vectors. An individual activation dimension may participate in representing several different patterns, making individual neurons difficult to interpret.

An SAE attempts to represent these dense activations through a larger feature space where relatively few features are active for a given input.

Conceptual illustration: many dense values are encoded into a feature vector with relatively few strong activations.

### Your planned procedure

You begin by selecting an LLM and extracting its internal activations from specific transformer layers.

You then pass these activations through an SAE using published code or a compatible implementation. The proposal does not require training an entirely new SAE if a suitable implementation can be used.

Once feature activations have been obtained, you examine:

- Which features activate most strongly for a particular prompt.
- Whether similar prompts activate similar features.
- Whether features consistently respond to a particular concept, topic, or behaviour.
- Whether the identified features appear semantically interpretable.

For example, suppose a Persian cultural question strongly activates feature 516. You would inspect the contexts in which that feature becomes active and determine whether a meaningful interpretation is supported.

A high feature activation alone does not prove that the feature represents a particular concept.

### Optional causal experiment

Your proposal also states that, where technically feasible, you will modify or suppress individual SAE features and examine whether the final output changes.

This is important because identifying a feature is observational. Modifying it and observing a repeatable effect can provide causal evidence.

### Expected SAE results

| Output                         | Purpose                                                |
| ------------------------------ | ------------------------------------------------------ |
| Selected layer activations     | Input to the SAE                                       |
| Feature activation strengths   | Identify which features are most active                |
| Feature rankings               | Identify the strongest features for particular prompts |
| Prompt–feature comparisons     | Evaluate consistency and possible meaning              |
| Optional feature interventions | Test whether selected features influence behaviour     |

What counts as completion? You have a working SAE implementation, interpretable visualisations of selected feature activations, and an evaluation of what those features reveal and what remains uncertain.

## 4. Stage 3 — Implement the Jacobian Lens

Proposal Section 1.3 · Week 3

### What are you investigating?

The Jacobian Lens aims to identify information within intermediate model representations that is associated with concepts the model may later verbalise.

Unlike a standard logit lens, which directly projects intermediate activations into vocabulary space, JLens uses an approximation of how changes to intermediate activations propagate toward the output.

The purpose is to investigate concepts that may be relevant to the computation even when they do not appear in the final response.

### Your planned procedure

First, use a fitted Jacobian Lens compatible with your chosen model.

Next, pass your selected prompts through the model and collect JLens readouts at different layers.

For each layer, examine the highest-ranked tokens or concept-related readouts.

Consider a hypothetical example:

Illustrative JLens concept trajectory

| Early layers  | Iran · name · country         |
| ------------- | ----------------------------- |
| Middle layers | Persian · literature · poetry |
| Late layers   | poet · Hafez · Shiraz         |

These are invented example tokens to illustrate the expected kind of analysis, not recorded results.

You would investigate whether certain concepts become more prominent through later layers, whether intermediate concepts differ between equivalent English and Persian questions, and whether relevant concepts are visible even when absent from the final answer.

The proposal also includes experiments involving concept directions: strengthening, suppressing, or swapping directions identified by JLens, then observing the model's output.

These interventions require additional implementation beyond simply generating readouts.

### Expected JLens results

| Output                            | Purpose                                                |
| --------------------------------- | ------------------------------------------------------ |
| Layer-wise token/concept rankings | Identify information accessible at each layer          |
| Concept trajectories              | Examine how readouts change across layers              |
| Prompt comparisons                | Identify similarities and differences across inputs    |
| Concept-direction interventions   | Test the effects of manipulating identified directions |

What counts as completion? You can generate and explain layer-wise JLens readouts for selected prompts, compare their trajectories, and report the results of feasible concept-direction interventions.

An important limitation to discuss is that a JLens readout does not necessarily reveal the model's literal step-by-step reasoning. It identifies information associated with intermediate representations.

## 5. Stage 4 — Implement Activation Patching

Proposal Section 1.4 · Week 4

### What are you investigating?

Activation patching provides the clearest causal intervention among your three baseline methods.

You take an internal activation produced by one input and insert it into the model's execution for another input.

You then observe whether the output changes.

### Your planned procedure

Source run

English prompt

Activation at layer L

Target run

Persian prompt

Replace activation

Run patched target model

Compare original and patched outputs

Predicted answer · Probability · Confidence

In detail, you:

1. Construct two related prompts, such as equivalent English and Persian questions.
2. Run the source prompt and save activations at selected layers.
3. Run the target prompt without intervention to establish its baseline output.
4. Repeat the target run while replacing one selected activation with the source activation.
5. Measure changes in predicted answers, output probabilities, or confidence.
6. Repeat across layers to identify where interventions have the greatest effect.

The proposal specifies patching selected layers one at a time, rather than replacing every layer simultaneously.

### Expected results

You should be able to identify whether particular patched layers are associated with larger behavioural changes.

For example, an intervention might increase the probability of the correct answer, change the predicted option, or have almost no observable effect.

The important distinction is that a successful patching experiment establishes the effect of a particular intervention, not necessarily the complete causal mechanism responsible for the original behaviour.

### What counts as completion?

You need source and target runs, successful activation replacement, baseline and patched results, and layer-wise comparisons demonstrating the effect of the intervention.

This becomes your main causal reference point when comparing SAE and JLens findings.

## 6. Stage 5 — Build the controlled prompt dataset

Proposal Section 1.6 · Week 5

Once the three baseline methods work, your next step is to create a shared dataset that can be used across the methods.

Planned dataset size

# 30–100

Prompt pairs, depending on computational requirements

Main model

# Gemma 2 9B

Preferred, subject to compatibility

Your proposal specifies four categories of prompts:

| Category                       | What it tests                                                                  | Illustrative example                                |
| ------------------------------ | ------------------------------------------------------------------------------ | --------------------------------------------------- |
| Semantically matched questions | Whether equivalent meanings produce similar representations                    | Same cultural question in English and Persian       |
| Multilingual variants          | Differences associated with language                                           | Equivalent prompts using different languages        |
| Intermediate inference         | Representations associated with information needed before the answer           | Answering a question using an inferred relationship |
| Controlled causal pairs        | Whether changing a specific part of the input allows a meaningful intervention | Two prompts differing in a controlled way           |

An LLM may help generate candidate questions, but the final prompt set must be manually reviewed and edited.

### What you need to collect

For every selected prompt, you should have a reproducible record of the input, language, experimental condition, model output, and relevant output probabilities.

The dataset then serves as the shared experimental basis for all three methods.

Your proposal's moustache example on page 4 illustrates this idea: equivalent English and Persian cultural questions may be associated with partly shared and partly different internal concepts. The figure is hypothetical, not an experimental finding.&#x20;

Part1_Arash_Karimadini.pdf



## 7. Stage 6 — Apply all three methods to common prompts

Proposal Sections 1.7–1.8 · Week 6

This is arguably the most important stage for answering your research question.

Up to this point, you have demonstrated that SAE, JLens, and Activation Patching work independently.

Now you need to investigate what happens when they examine the same model behaviour.

For example, consider an English and Persian question where the model produces the same correct answer but exhibits different output probabilities.

| Method              | Your investigation                                                                                               |
| ------------------- | ---------------------------------------------------------------------------------------------------------------- |
| SAE                 | Do the two inputs activate different sparse features? Are any shared?                                            |
| JLens               | Do concept-related readouts differ across layers?                                                                |
| Activation Patching | Does replacing selected Persian activations with English activations change output probabilities or predictions? |

You are trying to understand whether the results complement one another.

For example, if SAE identifies an interesting feature in a particular layer, JLens produces different concept readouts around that layer, and activation patching has a substantial effect there, the three findings may provide mutually useful evidence.

However, a shared layer number alone does not prove that all three methods have identified the same mechanism. That interpretation would require further tests.

### Your formal comparison criteria

The proposal identifies six comparison dimensions:

| Criterion                | Question you need to answer                               |
| ------------------------ | --------------------------------------------------------- |
| Internal representation  | What type of information does each method expose?         |
| Important layers         | Which layers appear particularly relevant?                |
| Interpretability         | How understandable are the results?                       |
| Causal evidence          | Is the method descriptive, causal, or capable of both?    |
| Computation              | What GPU resources, memory, and runtime does it require?  |
| Multilingual suitability | How useful might it be for comparing different languages? |

Your proposal explicitly states that you do not need to choose a single winning method.

Instead, you want to determine which methods are complementary and whether they could form one integrated research workflow.

Expected output: A method comparison table, visualisations, a discussion of similarities and differences, and a recommendation for future research.

## 8. Stage 7 — Reporting, documentation, and reproducibility

Proposal Sections 3–5 · Week 7

There are three formal deliverables, proposed as subdivisions of the placement's 40% assessment component.

Proposed assessment weighting

| Research presentation              | 10% |
| ---------------------------------- | --- |
| Research records / reproducibility | 5%  |
| Final research report              | 25% |
| Total                              | 40% |

These are proposed assessment allocations in your research plan, rather than independently verified final unit weightings.

### A. Research presentation — 10%

Your presentation should explain the research motivation, the three methods, their implementation, and the main preliminary findings.

It should show enough experimental evidence to establish what you achieved, without attempting to present every implementation detail.

### B. Research records — 5%

This deliverable demonstrates that another researcher could understand how you performed the experiments.

For each experiment, the plan says to record the model version, code version, prompt set, method, relevant layer or token position, parameters, execution date, and location of outputs.

The research log should also document failed or modified experiments and the reasons for important decisions.

You should retain raw results separately from processed results and figures where practical. Large model weights do not normally need to be committed to Git.

### C. Final research report — 25%

The final report needs to describe the implementation and evaluation of the three methods, their application to controlled prompts, the comparison between their findings, major limitations, and recommendations for future work.

A useful report structure, based on the plan, would be:

| Section                       | Main content                                         |
| ----------------------------- | ---------------------------------------------------- |
| 1. Introduction               | Motivation, problem, research questions              |
| 2. Background                 | SAE, JLens, activation patching and related work     |
| 3. Experimental setup         | Model, hardware, prompts, libraries, reproducibility |
| 4. SAE implementation         | Procedure, feature analysis, results                 |
| 5. JLens implementation       | Fitting/application, concept readouts, results       |
| 6. Activation patching        | Experimental setup, interventions, results           |
| 7. Comparative analysis       | Common prompts and cross-method findings             |
| 8. Discussion                 | Strengths, limitations, computational challenges     |
| 9. Conclusion and future work | Main lessons and Placement 2 direction               |

This is a suggested report structure, not a mandatory set of headings from the proposal.

The plan also contains two examiner placeholders. Those need to be replaced with confirmed examiners through the process agreed with your mentor.&#x20;

Part1_Arash_Karimadini.pdf



## 9. Your planned seven-week timeline

Your plan allocates 35 hours of research activity, 7 hours of mentor time, and 20 hours of deliverable preparation.

| Week  | Research work                                            | Research hours | Deliverable hours |
| ----- | -------------------------------------------------------- | -------------- | ----------------- |
| 1     | Finalise scope, model, environment, baseline experiments | 4              | 1                 |
| 2     | Implement SAE and analyse features                       | 6              | 2                 |
| 3     | Implement JLens and analyse layer-wise readouts          | 6              | 2                 |
| 4     | Implement activation patching and test interventions     | 6              | 2                 |
| 5     | Create prompts and perform baseline inference            | 5              | 3                 |
| 6     | Run common experiments and compare outputs               | 5              | 4                 |
| 7     | Compare methods; finalise figures, report, presentation  | 3              | 6                 |
| Total |                                                          | 35             | 20                |

The plan additionally allocates one mentor hour per week.

The priority is successful implementation and meaningful comparison, not the number of prompts processed.

If computation becomes too expensive, the proposal permits reducing the prompt set. Extensive hyperparameter searches, additional models, and large-scale datasets are outside the intended Placement 1 scope.&#x20;

Part1_Arash_Karimadini.pdf



## 10. What is planned for Placement 2?

Your final research direction is deliberately broader than Placement 1.

Placement 1 — Build and validate

Implement the three techniques and assess whether they can provide a useful combined approach.

Placement 2 — Apply and investigate

Identify public datasets concerned with multilingual, cultural, moral, or regional behaviour. Review, clean, adapt, and categorise them, then apply the combined workflow to a broader research question.

In your proposal, Placement 2 is a possible future direction, rather than a required deliverable of Placement 1.

This separation matters because otherwise the Placement 1 report could expand into a much larger project than originally approved.

# 11. Final completion checklist

The following is an actionable checklist derived from your plan. I have left the items unchecked because the proposal itself documents intentions, not proof of completion. Given the implementation work you have previously described, you can treat this as a final verification and reporting checklist, rather than a requirement to rerun everything.

Placement completion tracker

Tick items as you verify the evidence.

## 0/37

A. Project setup and baseline

0/4

Confirm the main research question and the selected experiments from all three papers.

Record the chosen model, model version, GPU environment, libraries, and source-code references.

Document method compatibility issues and any changes from the published implementations.

Confirm applicable induction, research-integrity, licensing, storage and computing-access requirements.

B. Sparse Autoencoders

0/6

Demonstrate successful extraction of internal model activations.

Run the SAE implementation and save feature activation results.

Identify and visualise strongly activated features for selected prompts.

Compare features across prompts and evaluate semantic interpretations.

Attempt and document SAE feature interventions where technically feasible, or explain why omitted.

Report SAE limitations, including uncertainty in feature labels.

C. Jacobian Lens

0/5

Verify the fitted JLens implementation and document its configuration.

Generate and save layer-wise concept/token readouts.

Visualise and compare concept trajectories for selected prompts.

Test JLens concept-direction interventions and record their effects, or document technical limitations.

Explain what JLens readouts support and what they cannot establish.

D. Activation Patching

0/6

Construct suitable source and target prompt pairs.

Record clean baseline outputs and source activations.

Verify that selected activations are actually replaced during the target run.

Run patching across selected layers.

Measure changes in predictions, probabilities, or confidence.

Produce layer-wise patching results and explain their causal interpretation.

E. Common prompt dataset

0/4

Prepare approximately 30–100 controlled prompt pairs, or document a justified smaller sample.

Include matched questions, multilingual variants, inference cases and causal pairs.

Manually review and edit the generated candidate prompts.

Save prompt identifiers, languages, exact text, baseline outputs and relevant probabilities.

F. Combined method evaluation

0/5

Run all three methods on selected common prompts using the main model where compatible.

Compare SAE features, JLens readouts and patching results for the same examples.

Create a comparison covering representations, layers, interpretability, causal evidence, computation and multilingual usefulness.

Identify findings supported by multiple methods without overstating causal conclusions.

Summarise how the three methods could be integrated in future research.

G. Research records and final submissions

0/7

Organise code, prompts, notebooks, raw outputs, processed results and figures.

Record model/code versions, parameters, layer/token positions, execution dates and output paths.

Document failed experiments, modifications and research decisions.

Verify that the important experiments are reproducible from documented instructions.

Finalise and deliver the research presentation.

Write the final report with methods, results, comparison, limitations and future recommendations.

Confirm examiner arrangements and final submission requirements with the mentor.

Reset checklist Copy checklist

## 12. What I would prioritise now

Since you have previously discussed completing the implementations and obtaining results from all three methods, I would focus your remaining effort on demonstrating that the three implementations meet the original plan and form a coherent research outcome.

My suggested order is:

1. Verify the experiment records and reproducibility documentation.
2. Select the strongest common English–Persian examples for your comparison.
3. Check whether the SAE and JLens intervention experiments were actually completed, since these are distinct from ordinary SAE feature extraction and JLens readouts.
4. Prepare the cross-method comparison, including computational requirements and limitations.
5. Finalise the report, figures, and examiner arrangements.

The biggest potential gap to watch is the integrated comparison. Three successful implementations satisfy a major part of the plan, but your central research question is specifically about how the three methods differ and complement each other. That synthesis needs to be prominent in the final report.