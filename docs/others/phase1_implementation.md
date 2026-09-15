# FOSE7901 Placement — Minimum Required Implementation

## Project Goal

Implement and compare three mechanistic interpretability methods on a common
open-source LLM:

1. Sparse Autoencoders (SAE)
2. Jacobian Lens (JLens)
3. Activation Patching

The purpose is to determine:

- what internal representation each method reveals;
- where in the model that information appears;
- whether the evidence is descriptive or causal;
- and how the three methods could be combined into a later multilingual
interpretability workflow.

The project should prioritise correct implementation and interpretable results
over dataset size.

---



# 1. Common Experimental Setup



## 1.1 Model

Use one primary open-source LLM where possible.

Preferred:

- Gemma 2 9B

If implementation compatibility prevents all methods from using the same model,
a compatible alternative may be used and documented.

For every experiment record:

- model name;
- exact model/version identifier;
- number of layers;
- hidden-state dimension;
- tokenizer;
- dtype;
- device/GPU;
- code version;
- experiment date.

---



## 1.2 Prompt Set

Construct a deliberately small controlled prompt set.

Target:

- approximately 30–100 prompt pairs;
- reduce further if computational requirements are high.

Include:

### A. Semantically matched prompts

Same meaning expressed in different languages.

Example:

English:

> What does a moustache represent in your culture?

Persian:

> سبیل در فرهنگ شما چه مفهومی دارد؟



### B. Simple multilingual variants

Prompts where translation should ideally preserve the same semantic content.

### C. Intermediate-inference prompts

Questions where the model must perform some reasoning before producing the
answer.

### D. Controlled source/target pairs

Prompt pairs specifically designed for activation patching.

Each prompt should have:

- prompt ID;
- language;
- prompt text;
- matched-prompt ID;
- expected relationship;
- prompt category.

---



# 2. Baseline Model Behaviour

Before interpretability analysis, run normal inference.

For every prompt record:

- generated answer;
- predicted token / answer;
- output logits;
- output probabilities;
- confidence where appropriate.

For multilingual matched prompts, compare whether the model gives:

- the same answer;
- different answers;
- different confidence;
- different probability distributions.

This establishes the behaviour that the interpretability methods are trying
to explain.

---



# 3. Sparse Autoencoder Implementation



## 3.1 Objective

Use an SAE to decompose dense transformer activations into a larger sparse
feature representation.

Conceptually:

Transformer activation:

```
x ∈ R^d_model
```

SAE encoder:

```
z = activation(W_enc x + b_enc)
```

where:

```
z ∈ R^d_sae
```

and normally:

```
d_sae > d_model
```

Most entries of z should be approximately zero.

The decoder reconstructs the original activation:

```
x_hat = W_dec z + b_dec
```

---



## 3.2 Activation Collection

Choose one or more model layers.

For each selected layer:

1. Run prompts through the transformer.
2. Hook the required activation.
3. Save activations for the selected token positions.
4. Build the activation dataset used by the SAE.

Record:

- layer;
- token;
- token position;
- activation vector;
- prompt ID.

---



## 3.3 SAE Training / Loading

Either:

- train a compatible SAE;
- or load an appropriate pretrained SAE if available.

The implementation must include:

### Reconstruction objective

Measure how accurately the SAE reconstructs the original activation.

Example:

```
reconstruction loss = ||x - x_hat||²
```



### Sparsity objective

Encourage only a small number of features to activate.

Example:

```
L1 = ||z||₁
```

Typical total objective:

```
Loss = reconstruction_loss + λ * sparsity_loss
```

---



## 3.4 Required SAE Outputs

For selected prompts obtain:

- active SAE feature IDs;
- activation magnitude of each feature;
- top-k features;
- sparsity level;
- reconstruction quality.

At minimum produce:

### Feature activation table


| Prompt | Layer | Token | Feature | Activation |
| ------ | ----- | ----- | ------- | ---------- |




### Sparsity measurement

Examples:

- number of active features;
- fraction of active features;
- L0-like sparsity measurement.



### Reconstruction measurement

Examples:

- MSE;
- explained variance;
- reconstruction cosine similarity.

---



## 3.5 Feature Interpretation

For an interesting SAE feature:

1. identify prompts/tokens where it activates most strongly;
2. inspect those examples;
3. determine whether they share a semantic pattern;
4. assign an informal interpretation only when supported by examples.

Example:

```
Feature 1843
```

Top activating contexts:

```
"Persian tradition ..."
"cultural custom ..."
"historical practice ..."
```

Possible interpretation:

```
tradition / cultural custom
```

Do not claim that the label is definitely what the neuron "means".
It is an interpretation based on activating examples.

---



## 3.6 Optional SAE Intervention

If computationally feasible:

- suppress an SAE feature;
- increase an SAE feature;
- reconstruct the modified activation;
- insert it back into the model;
- measure output change.

Compare:

```
original output
      ↓
SAE intervention
      ↓
modified output
```

This turns SAE analysis from purely descriptive evidence toward causal
evidence.

This is useful but should not prevent completion of the core SAE analysis if
time is limited.

---



# 4. Jacobian Lens Implementation



## 4.1 Objective

Use JLens to identify concepts/tokens that intermediate model states are
positioned to influence later in the computation.

Unlike a standard logit lens, JLens accounts for how changes to an intermediate
representation propagate through the remaining transformer layers.

---



## 4.2 Required Pipeline

For each selected prompt:

1. tokenize the input;
2. perform the normal forward pass;
3. collect intermediate activations;
4. apply the fitted JLens;
5. obtain vocabulary/concept scores;
6. rank the highest-scoring verbalizable tokens;
7. repeat across layers.

---



## 4.3 Required JLens Output

For each layer record:

- layer number;
- token position;
- top concept/token;
- JLens score;
- top-k concepts/tokens.

Example:


| Layer | Top JLens concept | Score |
| ----- | ----------------- | ----- |
| 10    | culture           | ...   |
| 20    | identity          | ...   |
| 30    | tradition         | ...   |
| 40    | moustache         | ...   |


---



## 4.4 Concept Trajectory

The important JLens result is not simply one token.

Track how the readout changes through depth:

```
Early layers
    ↓
intermediate concepts
    ↓
later concepts
    ↓
final answer
```

For matched multilingual prompts compare trajectories such as:

English:

```
culture → tradition → identity → answer
```

Persian:

```
culture → masculinity → honour → answer
```

The actual results must come from the model;
these labels must not be assumed beforehand.

---



## 4.5 JLens Visualisation

Create a layer × concept representation.

Preferred visualisations:

### Option A — Concept trajectory

```
Layer 5      culture
   ↓
Layer 15     tradition
   ↓
Layer 25     identity
   ↓
Layer 35     answer concept
```



### Option B — Heatmap

Rows:

```
concepts / tokens
```

Columns:

```
model layers
```

Values:

```
JLens scores
```

This provides a clear view of when particular concepts become visible.

---



## 4.6 JLens Intervention

If supported by the implementation, test:

- strengthening a concept direction;
- suppressing a concept direction;
- swapping a concept direction.

Then measure whether the output changes.

Record:

```
Original probability
Intervention probability
Δ probability
```

This provides evidence about whether the verbalizable representation is
actually involved in computation.

---



# 5. Activation Patching Implementation



## 5.1 Objective

Activation patching provides the strongest direct causal experiment in the
project.

Take an activation from a source prompt and insert it into the corresponding
location of a target prompt.

---



## 5.2 Source and Target Runs

For each controlled pair:

### Source prompt

Run the source prompt and cache internal activations.

### Target prompt

Run the target prompt normally to obtain the baseline output.

### Patched target

Run the target prompt again while replacing a selected activation with the
source activation.

---



## 5.3 Layer Sweep

Patch one layer at a time:

```
Layer 0
Layer 1
Layer 2
...
Layer N
```

For each layer measure the output effect.

The simplest experimental structure is:

```
Source activation at layer L
              ↓
Target model at layer L
              ↓
      continue forward pass
              ↓
         new output
```

---



## 5.4 Required Patching Metrics

At minimum measure one probability/logit-based metric.

Recommended:

### Correct-answer logit

```
logit(correct)
```



### Logit margin

```
logit(correct) - max(logit(wrong))
```



### Probability change

```
Δp = p_patched(correct) - p_target(correct)
```



### Recovery

If a source/clean prompt gives the desired behaviour:

```
Recovery =
(patched_metric - target_metric)
--------------------------------
(source_metric - target_metric)
```

Interpretation:

```
0%   = no recovery
100% = target behaviour restored to source level
```

---



## 5.5 Required Activation-Patching Output

Produce a layer-effect curve.

Example:

```
Patch effect
    ^
    |
    |             ███
    |          ███████
    |        █████████
    |____████________________> layer
         20  25  30  35
```

This immediately shows the layers where transferring the source representation
most strongly affects behaviour.

This should be one of the main final figures.

---



# 6. Common Example Across the Three Methods

Select at least one especially clear prompt pair.

For that SAME example show:

## Behaviour

```
EN prompt → Output A
FA prompt → Output B
```

or

```
EN confidence ≠ FA confidence
```

Then apply all three interpretability methods.

---



## SAE

Show:

```
Which sparse features activate?
```

Example:

```
EN → F183, F924, F1510
FA → F183, F441, F2208
```

Look for:

- shared features;
- language-specific features;
- semantically interpretable features.

---



## JLens

Show:

```
What concepts become verbalizable through the layers?
```

Example structure:

```
EN:
tradition → identity → final answer

FA:
tradition → honour → final answer
```

---



## Activation Patching

Ask:

```
If the relevant EN internal representation is inserted into FA,
does the FA output move toward the EN behaviour?
```

or vice versa.

This gives the causal test.

---



# 7. Method Comparison

Create one final comparison table.


| Method              | Representation          | Main Output            | Layer Information | Causal?                                   | Interpretation                   |
| ------------------- | ----------------------- | ---------------------- | ----------------- | ----------------------------------------- | -------------------------------- |
| SAE                 | Sparse features         | Feature activations    | Yes               | Mostly descriptive; intervention possible | What features are active?        |
| JLens               | Verbalizable directions | Concept/token readouts | Yes               | Intervention possible                     | What concepts are represented?   |
| Activation Patching | Transformer activations | Change in behaviour    | Yes               | Yes                                       | Which activations/layers matter? |


The key conceptual distinction should be:

```
SAE
↓
WHAT FEATURES exist in the activation?

JLens
↓
WHAT CONCEPTS can be read from internal computation?

Activation Patching
↓
WHICH INTERNAL STATES CAUSE the behavioural difference?
```

---



# 8. Final Combined Workflow

The strongest outcome of Placement 1 would be evidence that the methods can
form a pipeline rather than three unrelated experiments.

Potential workflow:

```
Prompt pair
     ↓
Model behaviour differs
     ↓
SAE
Identify candidate internal features
     ↓
JLens
Identify candidate concept representations
and when they emerge
     ↓
Activation Patching
Test which representation/layer causally
changes behaviour
     ↓
Mechanistic explanation
```

In simple language:

```
SAE tells us WHAT is active.

JLens tells us WHAT THE MODEL APPEARS TO BE REPRESENTING.

Activation patching tests WHAT ACTUALLY MATTERS FOR THE OUTPUT.
```

---



# 9. Minimum Results Required for Successful Placement

The project should be considered successfully implemented if the following
outputs exist.

## SAE

- [ ] Transformer activations collected
- [ ] SAE trained or successfully loaded
- [ ] Sparse feature activations obtained
- [ ] Reconstruction metric calculated
- [ ] Sparsity metric calculated
- [ ] Top activating examples inspected
- [ ] At least one interpretable feature example produced



## JLens

- [ ] JLens successfully fitted/loaded
- [ ] Applied to intermediate layers
- [ ] Top-k verbalizable tokens/concepts obtained
- [ ] Layer-wise trajectory generated
- [ ] At least one meaningful example analysed



## Activation Patching

- [ ] Source/target prompt pairs created
- [ ] Source activations cached
- [ ] Target activations replaced
- [ ] Layer-by-layer patching performed
- [ ] Output probability/logit effect calculated
- [ ] Causal layer-effect plot produced



## Combined Experiment

- [ ] At least one common prompt analysed using all three methods
- [ ] Results compared
- [ ] Agreement/disagreement between methods discussed

---



# 10. Research Records

For every experiment record:

- experiment ID;
- date;
- model;
- model revision;
- prompt ID;
- language;
- method;
- layer;
- token position;
- hyperparameters;
- software/code version;
- raw result path;
- processed result path;
- notes;
- failures or modifications.

Keep separate folders such as:

```
prompts/
src/
notebooks/
results/raw/
results/processed/
figures/
logs/
```

Do not store model weights in the Git repository.

---



# 11. What Is NOT Necessary for Placement 1

The following should not become requirements:

- hundreds or thousands of multilingual questions;
- many LLMs;
- extensive benchmark comparison;
- extensive SAE hyperparameter sweeps;
- complete multilingual cultural analysis;
- proving that every SAE feature has a semantic interpretation;
- full circuit discovery;
- neuron-level causal tracing;
- exhaustive analysis of all model layers and tokens.

These belong to later research.

Placement 1 should answer:

> Can I successfully implement these three methods, understand their outputs,
> compare them on the same examples, and determine how they could work
> together for later multilingual research?

