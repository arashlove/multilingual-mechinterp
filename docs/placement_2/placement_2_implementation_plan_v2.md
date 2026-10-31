# Placement 2: Complete Implementation Specification and Agent Handoff (v2)

**Major update:** Human-majority reference labels + raw human preprocessing **first** + A1/A2 reformulation + four-layer JLens plan.

**Project:** A Multilingual Mechanistic Benchmark for Moral Decision-Making in LLMs  
**Unit:** FOSE7901 - STEM Research Placement 2  
**Researcher:** Arash Karimadini  
**Mentor:** Dr. Usman Naseem  
**Document status:** Revised implementation specification v2 (18-19 October 2026); not completed Placement 2 experiments  
**Prepared:** 19 October 2026  
**Primary sources:** `part2_Arash_Karimadini(1).pdf` (Placement 2 proposal) and `implementation_summary.md` (Placement 1 handoff)

> **Instruction to the implementation agent:** Implement the Phase 0 raw human-response preprocessing first, using only the approved source/version and published data codebook; generate auditable country/culture preferences and a scenario-coverage catalog. **Do not invent the 13 real study dilemmas, translations, cultures, or missing human answers.** The researcher will create and approve those later. Build schemas/templates, validators, tested software, a CPU smoke test, the model/JLens/intervention pipeline and reproducible reports while preserving Placement 1 notebooks and results. No final experiment may run on fabricated inputs.

---

## 0. Executive specification (v2: human-majority reference labels)

### What changed, and why the research pipeline must be updated

**Researcher decision (18 October 2026):** For this benchmark, the operational *reference answer* for a Moral Machine dilemma is the **majority human choice within a specified cultural reference group**. This may be called the *human-majority benchmark label* or *human-consensus label*; it is **not an objectively correct moral choice**. A model answer matching the label is *human-reference aligned*, not necessarily morally correct.

This change means **human-response preprocessing and coverage analysis must come before the researcher authors the 13 benchmark scenarios**. It also changes behaviour metrics, A1/A2-style groups, and the selection criteria for the mechanistic experiments. Preserve raw STAY/SWERVE probabilities and language disagreement as separate outcomes.

**Critical distinction:** The previous proposal's country-level preference direction, a direct majority over responses to a *specific matched scenario*, and an equal-country aggregate majority are **three different estimates**. The software must calculate and label these separately and cannot silently substitute one for another. For a final human-majority *scenario* label, the strongest evidence is observed responses to an exact structured Moral Machine scenario match with sufficient sample size. A direction derived only from a dimension such as Age is a **dimension-level proxy**, not a direct majority for the entire scenario. If sufficient matched human responses do not exist, explicitly mark that reference as unavailable or proxy-based.

### Research goal

Examine whether equivalent moral dilemmas in different languages (i) cause changes in Gemma 2 9B STAY/SWERVE decisions and conditional probabilities, (ii) show different verbalizable internal concepts according to JLens, (iii) can be influenced by controlled concept-level interventions, and (iv) agree or disagree with observed, culture-associated human-majority reference decisions. The study is about empirical preference alignment and mechanisms, not objective ethics.

**Primary question:** How do multilingual LLMs differ in behavioural decisions and internal concept representations of moral trade-offs, and how do these differ from culture-associated human response patterns?

**Secondary question:** Can targeted interventions on validated concept directions influence model decisions and their agreement with human reference labels?

### Revised phase order: research dependencies

```
PHASE 0 - AGENT: Load and preprocess original Moral Machine human-response data
          -> valid paired choices, completed sessions, eligible countries,
             country/culture profiles, scenario-support catalog
               |
               v
PHASE 1 - RESEARCHER: Use the support catalog to write/approve 13 canonical
          scenarios, their outcome metadata, culture mapping and translations
               |
               v
PHASE 2 - AGENT: Match those scenarios back to human response evidence;
          generate culture-specific STAY/SWERVE human-reference labels
               |
               v
PHASE 3 - AGENT: Run multilingual forced-choice model inference and compute
          human-reference agreement + language-to-language divergence
               |
               v
PHASE 4 - AGENT: Select A1/A2/B1/B2 and decision-flip/control cases;
          fit and validate four-layer JLens, compare concept trajectories
               |
               v
PHASE 5 - AGENT: Test targeted concept steering, suppression, swaps and controls
               |
               v
PHASE 6 - AGENT: Produce figures, evidence traceability, notebook and report data
```

Some infrastructure, schemas and unit tests can be coded in parallel with Phase 0. The **first real-data scientific output** should be the human preprocessing report and coverage catalog. The agent must **not** write the 13 real scenarios, translate the prompts, fabricate cultural mappings or label missing evidence by assumption.

### Minimum viable study

- Use the existing **base** `google/gemma-2-9b` pipeline, subject to exact model revision approval.
- Preprocess a user-approved, public original Moral Machine human dataset; retain the 2018 design provenance and codebook.
- Use the original proposal's completed-session and **more than 100 users per country** rule; reproduce or document any deviation from the previously reported 117 eligible countries.
- Treat **country**, **culture/reference group**, and **prompt language** as three explicit identifiers, never interchangeable by default.
- Deliver a pre-scenario **coverage and preference catalog** so the researcher knows which dilemmas can have empirically supported labels.
- After researcher review, accept exactly 13 scenarios (two per each of six character dimensions, one random) translated into at least two approved languages.
- Produce **per-scenario, per-culture** human reference records: observed STAY and SWERVE counts/rates, winning label, support/ambiguity status, matching rule, aggregation policy and source provenance.
- Evaluate all 13 x L model prompts; report model choice/probabilities, *human-majority alignment*, pairwise distributional differences, and confidence.
- Analyse informative cases with JLens at **provisionally 22, 26, 30, 38**. An alternative **18, 22, 26, 38** adds an early pre-rise control; researcher selects before the fit. Retain adequate calibration/holdout data and fit-stability tests.
- Perform a small number of controlled concept interventions; retain negative controls and report null outcomes.
- Produce tested code, resumable stages, versioned data, charts, tables and a reproducibility record.

### Ownership split

**The agent implements:** raw human-source schema audit, preprocessing, pairing of human outcomes, respondent/session eligibility, country and culture aggregation, scenario support catalog, human-majority labeling once scenarios are provided, software contracts, choice scoring, model evaluation, case selection, JLens and intervention work, tests and reporting.

**The researcher supplies or approves:** the authentic public data source/version and licence; country-to-culture grouping rules and the language-to-reference-group comparison; the **13 actual scenarios** and their structured outcome descriptions; all translations and semantic checks; target concepts and calibration material if required; final protocol and interpretation.

**Do not require the researcher to hand-calculate human majority labels or clean millions of raw responses manually.** The pipeline should do that. If the public data are only supplied as precomputed country summaries, the agent must clearly disclose that *exact scenario-level majority* is unavailable unless the source includes those counts.

### Prerequisite acknowledgement

This specification is built from the researcher-supplied Placement 2 proposal and `implementation_summary.md`. The current repository has not been audited in this document. An implementation agent must inspect real API signatures and actual input file headers, and must not treat schema suggestions as already verified implementation details. Reference codebooks cited here are cross-checks, not replacements for reading the chosen dataset release.

---

## 1. What exists in Placement 1, and what to do with it

The repository is reported at `https://github.com/arashlove/multilingual-mechinterp`; the package is `src/multilingual_mechinterp/`, version `0.1.3` in the handoff. Keep Placement 2 as additional submodules, tests, configuration and a new notebook/CLI, rather than rewriting the working Placement 1 experiment.

| Existing item | Placement 1 implementation | Placement 2 action |
|---|---|---|
| `models/loader.py` | `load_model`, `LoadedModel` | **Reuse**, validate model revision, tokenizer, dtype, and device; record exact identifiers. |
| `models/hooks.py` | `hooked_forward`, `attach_hooks`, `ActivationCache` | **Reuse/extend** for robust selected-layer selected-position hooks and automatic cleanup. |
| `data/culture.py` | Four-choice Persian/Middle Eastern culture question loader | **Keep untouched**; add a separate Moral Machine schema and loader. |
| `data/mcq.py` | A/B/C/D first-token scoring, `p_correct`, correctness margin | **Do not use as-is**; create new STAY/SWERVE sequence-scoring functions. |
| `jlens/fitting.py` | Jacobian fit via one-hot cotangents, checkpoints | **Reuse/extend** with calibrated fitting, held-out checks, repeated-fit stability and durable checkpoints. |
| `jlens/lens.py`, `jlens/apply.py`, `jlens/hf.py` | Saved lens, model adapter, top-k vocabulary readouts | **Reuse/extend** for multi-language readouts, scores, filtering and concept trajectories. |
| `jlens/intervention.py` | Adds/subtracts a token output-embedding direction | **Reuse as baseline only**; introduce and validate a Jacobian-informed layer-appropriate concept direction. |
| `patching/` | Full-residual substitution across 42 blocks | **Keep optional** as a diagnostic; not the main causal method. |
| `sae/` | Tied SAE, training, activation, edit code | **Do not rerun by default** in Placement 2. |
| `notebooks/placement_1*.ipynb` | Executed/canonical research notebooks | **Preserve**; create `notebooks/placement_2.ipynb` for the new project. |
| `tests/` | Existing tests for loader/SAE/JLens/patching/culture | **Retain** and add Placement 2 focused unit/integration tests. |
| `notebooks/runpod_setup.py`, `docs/others/runpod_setup.md` | Remote GPU bootstrap | **Reuse/update documentation**; no new infrastructure required by default. |
| `/models/gemma-2-9b` and S3 mirror | Local model weights during runs | **Prefer existing local cache**, downloading only through approved, configured procedures. |

### Placement 1 facts relevant to the design

- Base Gemma 2 9B, **42 transformer blocks**, hidden dimension **3584**, float32, seed 0, PyTorch forward hooks at residual block output, final non-padding prompt token.
- The second run evaluated 200 factual/cultural questions in English, Persian, Bengali and Urdu, then selected 24 English/Persian cases across disagreement/agreement categories. These **are Placement 1 results, not Placement 2 moral benchmark results**.
- The second run on an A40 48 GB reported about **36.6 GB peak GPU memory**, mostly during JLens fitting. Treat this as an observed starting point, not a guarantee of Placement 2 requirements.
- The JLens fit used 20 prompts for source layers 38 and 40 (final block 41), four holdout checks, and **did not run the planned repeat-fit stability check**.
- Many supposedly content-oriented JLens readouts were actually ` Option`; the filtering did not exclude it. A letter-direction control could flip decisions, while content edits mostly did not.
- First-token MCQ scoring assumes the option label is one token; that assumption is **not valid to reuse blindly** for the new STAY/SWERVE targets.
- The Placement 1 cleanup cell **deleted JLens and SAE `.pt` checkpoints**. Placement 2 must keep the fitted lens and intervention artifacts required for repeatable experiments.
- Model Hub revision and some software versions were not pinned. Placement 2 must correct this.
- The original activation-patching unrelated-question control was large; a strong intervention effect does not alone establish concept- or question-specific causality.

**Design implication:** Reuse tested model/JLens infrastructure, but rebuild the task-specific data model, forced-choice scoring, interpretation filters, reproducibility, and concept-intervention evaluation.

---

## 2. Repository architecture and ownership

Add a separate Placement 2 namespace; preserve all Placement 1 modules, notebooks, tests and published results. Proposed structure (adjust to current repository conventions):

```text
configs/
  placement_2.yaml

data/placement_2/
  README.md
  raw/                              # Approved raw release; gitignored, read-only
  input_templates/                  # Created by agent; clear placeholders
  inputs/                           # Researcher-owned, approved inputs
    human_source.yaml               # Dataset source, version, path, mapping, licence
    country_culture_map.csv         # Country -> explicitly defined cultural group
    language_reference_map.csv      # Prompt language -> chosen reference group
    languages.json
    scenarios.json                  # 13 author-written dilemma specifications
    translations.jsonl              # Researcher-reviewed text
    concepts.json                   # Curated concept aliases if used
    jlens_calibration.jsonl         # Approved calibration material

src/multilingual_mechinterp/placement2/
  __init__.py
  schemas.py
  data_io.py
  validation.py
  human_source.py                   # Inspect source/header/codebook; stream large files
  human_preprocessing.py            # Pair response outcome rows; derive human action
  human_eligibility.py              # Completed-session/user filters, country counts
  human_profiles.py                 # Respondent and equal-country summary modes
  human_coverage.py                 # Pre-scenario catalog and matchability report
  human_reference.py                # Scenario-specific majority STAY/SWERVE labels
  scenario_signatures.py            # Reproducible two-outcome signature matching
  prompts.py
  choices.py                        # Full-sequence STAY/SWERVE choice scoring
  behavior.py                       # Baseline and human-reference alignment metrics
  selection.py                      # Orthogonal A1/A2 + flip/B1/B2 groupings
  jlens_analysis.py                 # Fit, held-out and repeat fit, layer trajectories
  concept_alignment.py              # Non-format semantic readout filtering
  interventions.py                  # Targeted edits and controls
  experiment.py                     # Cache/run IDs/logs/resume
  metrics.py                        # JSD, rates, effects, uncertainty
  figures.py
  reporting.py
  cli.py

notebooks/
  placement_2.ipynb

tests/placement2/
  fixtures/                       # Clearly synthetic, never research outcomes
  test_human_pairing.py
  test_human_eligibility.py
  test_human_majority.py
  test_aggregation_weights.py
  test_scenario_matching.py
  test_reference_groups.py
  test_schemas.py
  test_choices.py
  test_behavior.py
  test_selection.py
  test_jlens_analysis.py
  test_interventions.py
  test_cli_smoke.py

docs/placement_2/
  DATA_CONTRACT.md
  HUMAN_PREPROCESSING.md
  SCENARIO_AUTHORING_GUIDE.md
  METHODOLOGY.md
  RESEARCH_DECISIONS.md
  RUNBOOK.md
  LIMITATIONS.md

output/placement_2/<run_id>/
  human/raw_audit/
  human/paired_responses/          # Partitioned Parquet, access controlled as needed
  human/eligible_countries.csv
  human/country_dimension_profiles.csv
  human/culture_dimension_profiles.csv
  human/scenario_support_catalog.csv
  human/scenario_reference_labels.csv
  human/reference_quality_report.md
  behavior/
  selection/
  jlens/
  interventions/
  figures/
  records/
```

Use Parquet/Arrow/Polars or a documented streaming equivalent when raw files are too large for ordinary in-memory pandas reads. Avoid collecting direct identifiers in published outputs or logs; use pseudonymous IDs only for necessary eligibility checks. Respect applicable dataset licensing and university research-data handling.

---

## 3. Data contracts: what the researcher creates versus what code derives

**IMPORTANT: create empty templates and validators before awaiting study data.** Except for sample code snippets below, all real scenarios, cultural group assignments and translations are the researcher's responsibility. Human reference counts/labels should be **computed** from the approved public human data, not manually authored.

### 3.1 Human input: `human_source.yaml` (researcher approves; agent reads)

```yaml
source_id: moral_machine_original_release
source_url: '[APPROVED ORIGINAL DATA URL]'
source_version: '[RESEARCHER-VERIFIED RELEASE ID]'
source_sha256: '[COMPUTE AFTER FILE IS SUPPLIED]'
local_path: data/placement_2/raw/SharedResponses.csv
licence_or_terms: '[RESEARCHER-VERIFIED TERMS]'
codebook_url: '[APPROVED SOURCE DOCUMENTATION]'
format: shared_responses_two_outcome_rows
schema_adapter: moral_machine_2018_sharedresponses
country_field: UserCountry3
response_id_field: ResponseID
session_id_field: ExtendedSessionID
user_id_field: UserID
choice_saved_field: Saved
casualty_action_field: Intervention
scenario_order_field: ScenarioOrder
strict_scenario_type_field: ScenarioTypeStrict
fallback_scenario_type_field: ScenarioType
```

Do not rely blindly on exact capitalization/spelling in the sample: inspect the chosen release on disk. Fail with a schema-diff report when fields differ. Do not download an alternative dataset or use a derived LLM dataset as if it were original human responses without explicit approval.

**Public codebook background, subject to checking against the chosen release:** Original-style `SharedResponses` data encode one decision with two rows having the same `ResponseID`. `Intervention` identifies which action would kill the characters represented by an outcome (0 for staying, 1 for swerving); `Saved` marks the outcome spared by the human. `ExtendedSessionID`, `UserID`, `ScenarioOrder`, `UserCountry3`, `ScenarioType`, `ScenarioTypeStrict` and character counts support eligibility and matching. Details and source links are documented in section 6. These columns do **not** mean that `Intervention` is already the human's choice.

### 3.2 Culture/reference mappings (researcher defines; agent validates)

`country_culture_map.csv`:

```csv
country_code,culture_id,mapping_source,mapping_notes
```

`language_reference_map.csv`:

```csv
language,culture_id,comparison_rationale
```

- `country_code` must agree with the chosen human response country code format (commonly 3-letter `UserCountry3` codes).
- `culture_id` is an **explicitly defined grouping**, not an inferred biological or inherent personal property. It may represent a geographic/country cohort, a declared region, or a researcher's language-associated set of countries. Document its membership and limitations.
- `language` is the **language of the prompt**, not the person's actual spoken language. The `language_reference_map` is a researcher-approved *evaluation mapping*, not a claim that all speakers of that language share a culture.
- One country should not be counted twice within one group unless a versioned policy explicitly permits it. If a country belongs to multiple groups, record and flag overlapping membership in comparisons.
- The original proposed mapping from countries to language-associated groups can still be implemented via these two explicit tables.

### 3.3 Benchmark languages: `languages.json`

```json
{
  "benchmark_version": "moral-v1",
  "languages": [
    {"code":"en", "name":"English"},
    {"code":"fa", "name":"Persian"}
  ],
  "baseline_language":"en"
}
```

The displayed codes illustrate a likely initial pair from Placement 1; final scope is approved by the researcher. At least two languages are needed for multilingual comparison. Inference always runs independently of whether a cultural reference is available; missing human labels must not erase behavioural rows.

### 3.4 Preprocessing outputs (generated by agent, BEFORE scenario creation)

After reading the public raw release, produce:

- `eligible_countries.csv`: completed-session user count, included/excluded, reason. Expectation from original proposal: 117 countries, but this must be **recomputed**, never forced.
- `country_dimension_profiles.csv`: preference summaries for six character dimensions (optionally all nine), denominators, survey choice counts/uncertainty, aggregation assumptions.
- `culture_dimension_profiles.csv`: both equal-country and respondent-weighted aggregations; ties and absent evidence retained.
- `scenario_support_catalog.csv`: available scenario signatures or signature families, valid response counts, countries, culture coverage and the status of possible exact or conditional human-majority labels.
- `SCENARIO_AUTHORING_GUIDE.md`: derived dimension/trait vocabulary and available structural configurations, highlighting which combinations have enough human coverage to make labels feasible.

These outputs guide the researcher's actual scenario choice. They must never contain fabricated example labels indistinguishable from real observations.

### 3.5 Researcher-created canonical scenarios: `scenarios.json`

Create exactly 13 unique scenarios: two each for Age, Gender, Fitness, Social Status, Number of Characters and Species, plus one Random case. Example *placeholder/schema illustration*:

```json
{
  "benchmark_version": "moral-v1",
  "scenarios": [
    {
      "scenario_id": "age_01",
      "dimension": "age",
      "repetition": 1,
      "is_random": false,
      "english_prompt": "[RESEARCHER WILL WRITE EXACT TEXT]",
      "stay": {"harmed": [{"trait":"elderly", "count":3}], "favours":"young"},
      "swerve": {"harmed": [{"trait":"young", "count":3}], "favours":"elderly"},
      "structural_factors": {
        "relation_to_vehicle": "pedestrians_vs_pedestrians",
        "crossing_legality": "not_applicable"
      },
      "human_matching": {
        "mode": "exact_structured_match",
        "signature": "[CALCULATED OR RESEARCHER-VERIFIED SIGNATURE]"
      },
      "content_review_status": "pending"
    }
  ]
}
```

Only **researcher-supplied scenarios** become study prompts. The agent derives canonical matching signatures from structured character/outcome metadata and cross-checks them against raw Moral Machine entries. It must not infer whether STAY or SWERVE spares a group from free text alone. Require reviewed demographic categories, character totals, vehicle relation, legality, harmed versus spared groups and both action outcomes. Matching must be **outcome-order invariant**, while still retaining which outcome corresponds to STAY versus SWERVE. If exact human responses for this signature are scarce, show a coverage error/warning to the researcher and let them revise or approve a declared proxy mode.

Avoid using labels `correct`/`incorrect` in the scenarios file: **reference labels vary with `culture_id`** and are produced later by code.

### 3.6 Translations: `translations.jsonl` (researcher creates/reviews)

```json
{"scenario_id":"age_01","language":"en","prompt":"[REVIEWED TEXT]","review_status":"approved"}
{"scenario_id":"age_01","language":"fa","prompt":"[REVIEWED TEXT]","review_status":"approved"}
```

Require all **13 x L** unique scenario/language combinations. A validator checks completeness, approved review status, nonempty strings and consistent answer-format instructions; it cannot certify semantic equivalence. Literal `STAY` and `SWERVE` are the primary scored answer labels across languages unless an explicitly approved protocol changes them. Preserve all language-specific Unicode.

### 3.7 Final human label table (generated by agent AFTER 13 scenarios exist)

`human/scenario_reference_labels.csv`: one row per `(scenario_id, culture_id, aggregation_policy, evidence_type)`.

| Column | Meaning |
|---|---|
| `scenario_id`, `culture_id` | Explicit linked scenario and cultural cohort |
| `matching_policy` | `exact_structured_match`, `conditional_match`, or `dimension_proxy` |
| `evidence_type` | `observed_human_responses`, `derived_preference_direction`, or `model_estimate` |
| `aggregation_policy` | `equal_country_vote` or `pooled_respondents` (+ explicit variants) |
| `n_stay`, `n_swerve`, `n_valid` | Observed matched respondent choices where available |
| `n_eligible_countries`, `n_supporting_countries` | Country coverage after filtering/support checks |
| `p_human_stay`, `p_human_swerve` | Descriptive human share (valid for observed responses) |
| `reference_label` | `STAY`, `SWERVE`, or `UNDETERMINED` |
| `majority_margin` | Lead under declared aggregation rule |
| `reference_status` | `supported`, `tie`, `low_support`, `no_match`, `proxy_only` or `unreviewed` |
| `reference_quality_flag`, `source_sha256` | Evidence/uncertainty/provenance flags |

Where country votes are aggregated equally, `p_human_stay` must **not** be misrepresented as a respondent-weighted population probability. Save the country-vote share and respondent-pooled share in distinct columns, with the appropriate denominator for each. Unobserved/proxy-only references must never be reported as a measured per-scenario human majority.

### 3.8 Concept vocabulary: `concepts.json` (researcher reviews)

```json
{
  "concepts_version":"concept-v1",
  "concepts":[
    {"concept_id":"young", "dimension":"age", "terms":{"en":["young","child"], "fa":["[RESEARCHER-VALIDATED TERMS]"]}}
  ]
}
```

The code must filter formatting tokens and validate tokenization/coverage before calling JLens readouts concept-aligned. Do not treat different language strings as evidence of different meaning. Do not quietly use the first token of a multi-token concept phrase.

### 3.9 JLens calibration corpus: `jlens_calibration.jsonl`

Prefer independent reviewed prompts from the relevant domain, with `prompt_id`, `language`, `prompt`, `split` (`fit` or `holdout`). An explicit researcher-approved fallback to existing Placement 1 texts is possible but must disclose the domain mismatch. Keep study selection disjoint from calibration wherever feasible. Do not reuse the 13 benchmark prompts as training calibration by default.

### 3.10 Contracts to deliver BEFORE the researcher begins data authoring

- Versioned YAML, CSV and JSON templates with placeholder markers and a data dictionary.
- CPU-only fixture data **clearly marked synthetic** for tests, not for publication.
- `inspect-human-source` and `preprocess-human` tools, plus a guide describing completed-session eligibility and how respondent choices are reconstructed.
- A queryable support catalog listing available human comparison types and sample sizes.
- `init-templates`, `validate-data --allow-incomplete`, `check-human-coverage`, and a review sheet to guide researcher input.
- A `reference-status` gate that blocks unsubstantiated measured-majority claims while allowing language-only behavioural measurements for uncovered scenarios.

---

## 4. Study protocol: decisions to freeze before inference

All scientifically consequential decisions must be versioned in `docs/placement_2/RESEARCH_DECISIONS.md` and the run manifest. The agent can implement a configurable solution before all decisions are signed, but a final human-alignment result must identify its approved reference policy.

| Decision | Recommended development approach | Approval requirement |
|---|---|---|
| Human data provenance | Approved original 2018-style human release, inspected headers and codebook | Required |
| Country eligibility | More than 100 **distinct users with a completed 13-decision session** per country | Required interpretation of user/session multiplicity |
| Culture groups | Explicit researcher-defined country-to-`culture_id` map | Required; do not equate culture and language |
| Human reference evidence | **Observed exact scenario-matched responses** preferred; proxy kept separate | Required on low-support scenarios |
| Primary aggregation | Implement **both** equal-country-majority (matches proposal) and pooled-respondent majority | Researcher must choose the primary estimand |
| Majority threshold | Strict unique majority (> 50% of counted votes), otherwise `UNDETERMINED` | Confirm minimum support and near-tie policy |
| Confidence/support | Counts, coverage, Wilson/binomial-style intervals for pooled votes and suitable country-level uncertainty if required | Approve use and interpret limitations |
| Human labels | Culture-specific `reference_label`, never globally attached to the scenario | Required |
| Human alignment | Model decision equals human label, only when label is supported | Required |
| A1/A2 grouping | Agreement with *each language's mapped cultural reference*, independently of language flips | Required; never conflate reference with model language |
| Language comparison | Decision-change category and JSD independent of cultural reference | Required |
| Main model | Base `google/gemma-2-9b`, exact revision pinned | Required |
| Choice scoring | Full `STAY`/`SWERVE` continuation log-probabilities | Confirm prefix/termination |
| JLens layers | User candidate `[22, 26, 30, 38]`; optional early baseline `[18, 22, 26, 38]` | Freeze before JLens fitting |
| JLens position | Last non-padding prompt token; optional content-position diagnostics | Confirm |
| JLens evaluation | Independent fit/holdout, actual repeat-fit check, content filter | Mandatory |
| Interventions | Concept/contrast steering with fixed directions and same-norm negative controls | Feasibility check + approval |
| Study scale | 13 scenarios per language, exploratory interpretation | Required |

**Decision on the word 'correct':** It is acceptable internally to compute `is_reference_match` or a reference-relative margin for analyses analogous to Placement 1, but never to report `moral_correct`, `ethical_accuracy`, or an unqualified *accuracy*. Prefer `human_majority_agreement_rate`, with matched-scenario denominators, coverage, and evidence quality.

**Example of a non-equivalent majority:** If Country X has 1,000 matching respondent choices (70% STAY) and Country Y has 100 (20% STAY), respondent pooling produces a very different estimand from giving Country X and Y one vote each for their within-country majority. Both are implementable; they must be reported as different analyses.

---

## 5. Configuration and execution contract

`configs/placement_2.yaml` should provide all policies centrally. Example below; placeholder fields that require sign-off must block a scientific final run.

```yaml
run:
  name: placement_2_majority_v2
  seed: 0
  output_root: output/placement_2
  require_reviewed_data: true
  require_approved_reference_policy: true
  cache: true

human:
  source_manifest: data/placement_2/inputs/human_source.yaml
  country_culture_map: data/placement_2/inputs/country_culture_map.csv
  language_reference_map: data/placement_2/inputs/language_reference_map.csv
  raw_processing:
    paired_outcome_format: moral_machine_2018_sharedresponses
    stream_source: true
    output_format: parquet
    require_two_valid_outcomes: true
    completed_session_unique_orders: 13
    eligible_country_completed_users_gt: 100
  primary_reference_aggregation: approval_required  # equal_country_vote | pooled_respondents
  secondary_aggregation: [equal_country_vote, pooled_respondents]
  scenario_match_policy: exact_structured_match
  allow_proxy_for_primary_majority: false
  min_matched_responses_per_culture: approval_required
  near_tie_warning_fraction: 0.05    # not automatically a no-label threshold

inputs:
  languages: data/placement_2/inputs/languages.json
  scenarios: data/placement_2/inputs/scenarios.json
  translations: data/placement_2/inputs/translations.jsonl
  concepts: data/placement_2/inputs/concepts.json
  jlens_calibration: data/placement_2/inputs/jlens_calibration.jsonl

model:
  model_id: google/gemma-2-9b
  revision: null                     # Freeze resolved full commit hash
  local_path: /models/gemma-2-9b
  dtype: float32
  device: cuda
  batch_size: 1
  max_prompt_tokens: 256

choices:
  labels: [STAY, SWERVE]
  method: full_sequence_conditional_logprob
  prefix_policy: explicit_and_identical_across_languages
  include_termination: false
  length_normalization: none

behavior:
  jsd_log_base: 2
  baseline_language: en
  human_alignment_column: is_reference_match
  high_jsd_threshold: 0.05
  high_p_difference_threshold: 0.10
  include_all_language_pairs: true

selection:
  max_scenarios: 6
  prioritize_human_misalignment: true
  prioritize_decision_disagreement: true
  include_same_decision_high_jsd: true
  include_stable_controls: true
  require_reference_quality: supported

jlens:
  source_layers: [22, 26, 30, 38]     # Researcher-suggested; confirm before fitting
  alternative_early_control_layers: [18, 22, 26, 38]
  target_layer: 41
  fit_prompts_min: 20
  holdout_prompts_min: 4
  repeat_fit: true
  dimension_batch: 32
  max_fit_tokens: 128
  skip_first_tokens: 16
  top_k_raw: 50
  top_k_filtered: 8
  position: last_prompt_non_padding_token
  save_checkpoints: true

interventions:
  method: jacobian_backprojected_concept
  baseline_method: placement1_token_embedding
  relative_residual_norm_strengths: [-0.05, -0.03, -0.01, 0.0, 0.01, 0.03, 0.05]
  controls: [zero_strength, random_same_norm, unrelated_concept, template_token]
  min_cases: 3
  only_approved_concepts: true

reporting:
  output_format: [json, csv, png, md]
  retain_lens_checkpoints: true
```

### CLI: the first five commands work before the 13 prompts exist

```bash
# Phase 0: inspect and preprocess the original human source FIRST.
python -m multilingual_mechinterp.placement2.cli inspect-human-source --config configs/placement_2.yaml
python -m multilingual_mechinterp.placement2.cli preprocess-human --config configs/placement_2.yaml
python -m multilingual_mechinterp.placement2.cli summarize-human --config configs/placement_2.yaml
python -m multilingual_mechinterp.placement2.cli export-scenario-guide --config configs/placement_2.yaml
python -m multilingual_mechinterp.placement2.cli init-templates --config configs/placement_2.yaml

# When the researcher has written 13 scenarios and translations:
python -m multilingual_mechinterp.placement2.cli validate-data --config configs/placement_2.yaml
python -m multilingual_mechinterp.placement2.cli check-human-coverage --config configs/placement_2.yaml
python -m multilingual_mechinterp.placement2.cli finalize-human-labels --config configs/placement_2.yaml

# Model phases:
python -m multilingual_mechinterp.placement2.cli run-baseline --config configs/placement_2.yaml
python -m multilingual_mechinterp.placement2.cli evaluate-human-agreement --config configs/placement_2.yaml
python -m multilingual_mechinterp.placement2.cli select-cases --config configs/placement_2.yaml
python -m multilingual_mechinterp.placement2.cli fit-jlens --config configs/placement_2.yaml
python -m multilingual_mechinterp.placement2.cli analyze-jlens --config configs/placement_2.yaml
python -m multilingual_mechinterp.placement2.cli intervene --config configs/placement_2.yaml
python -m multilingual_mechinterp.placement2.cli report --config configs/placement_2.yaml
```

An optional `run-all` command must respect the preconditions and phase order; it must not silently create missing human labels or user-written prompts. Freeze run IDs, input hashes, model revision, code SHA and policy version. Resuming must not reuse outputs if source data, matching policy or cultural map changes.

---

## 6. Stage 0: Raw Moral Machine preprocessing (FIRST research stage)

**This is the first deliverable for the implementation agent. It does not require the researcher to write any of the 13 scenario texts yet.** Audit the original public human data before defining scenario-level labels. The aim is a dependable pool of human STAY/SWERVE choices, eligibility decisions and a catalog of well-supported scenario types.

### 6.1 Acquire/verify input without substituting a different dataset

1. Require an approved `human_source.yaml`; record source URL, publication, licence/terms, file checksum, dimensions and exact data release. Do not infer licensing or accept an unchecked file named `responses.csv` as authoritative.
2. Inspect **actual** fields and codebook, make a schema mapping, estimate rows/file size/missingness before running expensive transformations.
3. For original-style `SharedResponses`, expect **two rows per decision sharing `ResponseID`**. Design for roughly tens of millions of rows with streaming/Parquet intermediate partitions; do not load the entire file into memory as one large pandas DataFrame.
4. Retain a source audit: raw records, malformed IDs, duplicated/orphaned outcome rows, type inconsistencies and discarded rows by reason. Preserve source data read-only.

### 6.2 Pair outcomes correctly: human choice is NOT the `Intervention` field

According to the publicly documented original-style coding:

- `Intervention = 0` on an outcome indicates *these characters would die if the AV STAYS*.
- `Intervention = 1` indicates *these characters would die if the AV SWERVES*.
- `Saved = 1` identifies the characters the person chose to **spare**, and `Saved = 0` the characters the person chose to **kill**. This is not the on-screen click convention; it is a recoded variable in the released-style files.

**Derive `human_action` from the paired saved outcome:**

```
if Saved==1 on outcome with Intervention==1 -> human_action = STAY
if Saved==1 on outcome with Intervention==0 -> human_action = SWERVE
otherwise -> INVALID_PAIR / manual audit, never guess
```

This follows the defined casualty actions: if the respondent saves the group that would die on SWERVE, they chose STAY. Confirm the rule on independently worked examples from the chosen source, test both choices and action reversal, and cross-check `Saved` complementarity. Never use `Intervention == 1` by itself as the respondent's actual choice.

For each valid pair ensure: exactly two outcome rows; same `ResponseID`, compatible session/user/country; one `Intervention=0`, one `Intervention=1`; exactly one `Saved=1`; consistent structural scenario metadata and character counts. Reject or quarantine invalid pairs with an explicit reason and audit counts. Use `Saved` and character count fields as documented by the selected dataset. Store `outcome_stay` and `outcome_swerve` canonical structured descriptions separately.

### 6.3 Determine completed-session users and eligible countries

The proposal says **countries with more than 100 users who completed a full session**. Implement an explicit rule, audit multiple attempts, and document whether the source's session/order fields support it:

1. Reconstruct valid 13-decision sessions by `ExtendedSessionID`, checking 13 unique `ResponseID`s and the appropriate order values `1..13` where present.
2. Establish a deduplication rule for multiple completed sessions by the same `UserID`. Count **distinct eligible users** per country for the `>100` filter. If users change countries, document/flag the rule rather than double-counting silently.
3. Exclude countries with 100 or fewer such users; 101 must pass. Report eligible full-session user counts, valid response counts and excluded cases separately.
4. Compare the resulting eligible-country count to **117**, the number reported in the user's current plan. Treat disagreement as a provenance/methodology diagnostic; **never manipulate filtering to force 117**.

A partial/invalid session should not accidentally qualify as a completed one. For descriptive response analyses, the exact policy for whether partial-session responses are excluded or separately analysed must be predeclared and logged.

### 6.4 Normalize the scenario design and nine preferences

Create a canonical, orientation-aware but presentation-order-invariant **scenario signature** from BOTH outcomes, recording:

- Strict/fallback scenario type, including `Utilitarian` (= number of characters) and `Social Value` (= social status) naming normalization; six character dimensions plus Random.
- Full character-type count vectors for STAY and SWERVE casualty groups; total characters and group composition.
- Passenger/pedestrian relation (`PedPed`, `Barrier`), crossing legality (`CrossingSignal`), vehicle action that produces each outcome (`Intervention`), and any additional codebook-required attributes.
- `DefaultChoice`, `AttributeLevel` and `DefaultChoiceIsOmission` only after confirming semantics; raw left/right display position must not be mistaken for the STAY/SWERVE action.
- The canonical normalized dimension/directional preference `young` versus `old`, etc. for dimension-level summaries.

Do not treat an Age question involving 3 versus 3 as interchangeable with one involving 1 versus 5, or change passenger-vs-pedestrian/crossing legality. The study's 13 prompts must later be linkable to matching signatures if exact majority labels are desired.

### 6.5 Produce HUMAN summary tables for researcher scenario design

Create the first substantial outputs **before any prompt dataset is authored**:

1. `country_dimension_profiles.csv`: country preference counts/directions for the six main dimensions; optionally all nine dimensions for provenance and comparison with the published framework.
2. `culture_dimension_profiles.csv`: aggregated groups from approved `country_culture_map.csv` with **both** (a) respondent-pooled preference shares and (b) country-equal or one-country-one-direction aggregate references.
3. `scenario_support_catalog.csv`: available canonical signatures/signature families, number of valid responses per country and per culture, eligible-country support, STAY/SWERVE frequencies, and exact/conditional matching availability.
4. `human_preprocessing_report.md`: source provenance; number of outcome rows and valid paired decisions; lost/unmatched rows; complete-session filters; eligible countries; dimension names; cultural mapping coverage; example reconstruction checks; sensitivity differences.
5. `SCENARIO_AUTHORING_GUIDE.md`: a plain-English indexed set of supported dimension contrasts, structural variations and minimum-response counts to guide the researcher. This **does not generate the user's 13 scenario wordings**.

These summaries may contain empirical observed human choices, but until a final researcher-authored benchmark scenario matches a recorded signature, do not call any entry the "correct" answer to that author's future scenario.

### 6.6 Scenario-aware MAJORITY labeling after researcher creates scenarios

For each user-supplied `scenario_id`, construct its canonical signature and join to evidence:

**Level 1 (primary when available):** Exact structured scenario-match: match both casualty outcomes and relevant law/passenger/other conditions, preserving the STAY/SWERVE mapping. Compute empirical `n_stay`, `n_swerve` for each culture and country from valid eligible human decisions. A supported unique majority produces a candidate reference label.

**Level 2 (flagged sensitivity only unless approved):** Conditional match for dimension, chosen alternatives and structural factors, relaxing explicitly named nonessential constraints. Define which constraints were dropped and report coverage; do not silently call this an exact-match majority.

**Level 3 (proxy, not observed scenario majority):** Infer the favoured alternative from country/dimension preference directions and map that preferred trait onto the scenario's STAY/SWERVE outcome tags. This follows the original proposal's dimension approach, but the label MUST be tagged `dimension_proxy`, **not** an observed majority of the exact 13 dilemma text.

If no valid supported reference is available: `reference_label=UNDETERMINED`, `reference_status=no_match/low_support/tie`. The behavioural and JLens experiments may still run; only the corresponding human alignment values become missing.

### 6.7 Two competing aggregation policies (compute both, approve one as primary)

**Pooled-respondent majority** (within a named culture group):

```
n_stay = sum of human STAY responses across matched eligible countries
n_swerve = sum of human SWERVE responses across matched eligible countries
p_human_stay_pooled = n_stay / (n_stay + n_swerve)
reference_pooled = STAY if n_stay > n_swerve, else SWERVE if lower, else UNDETERMINED
```

This measures the majority of **sampled human choices** in the cohort, subject to coverage and respondent/session weighting. It lets large-country samples dominate.

**Equal-country majority**, corresponding more closely to the *planned country-level preference direction strategy*:

```
for each eligible country with adequate matching evidence:
    country_vote = STAY if n_stay_country > n_swerve_country
                   else SWERVE if lower, else UNDETERMINED
aggregate_country_votes = count(STAY votes), count(SWERVE votes)
reference_equal_country = unique winner or UNDETERMINED
```

Countries get **one vote** each. This measures the majority of contributing countries' within-country preferred directions; it is **not** the majority of all surveyed individuals. Record nonvoting/tied countries separately. The proposal originally uses this country-equal idea; the researcher should choose whether it remains the primary analysis given their clarified desire to use cultural-majority answers.

Optionally record the mean of each country's within-country STAY share as a **third** equally weighted estimand; do not confuse it with one-country-one-vote. Report denominator, sample sizes, country count, vote strength and differences between aggregation modes.

**Tie/quality policy:** `50/50` -> `UNDETERMINED`; missing country/culture -> missing; low sample size -> `low_support` according to an approved threshold. A narrow but non-tied 51/49 may yield a *candidate* majority, but must be flagged as near-tie and its uncertainty displayed. Do not arbitrarily convert it into a strong human consensus. Predefine a label eligibility threshold and interval methodology that respects respondent clustering; naïve independent-binomial error bars can misstate uncertainty.

**Deliverable quality gate:** every label used in A1/A2 and headline human agreement has a traceable `(source hash -> raw paired decisions -> completed-user rule -> culture map -> scenario match -> aggregation policy -> label)` lineage.

### 6.8 Unit tests (must pass with synthetic fixtures before real-data runs)

- Pair two outcome rows correctly and derive STAY and SWERVE from complementary `Saved`/`Intervention` fields; reverse action labels to test direction consistency.
- Reject three-row duplicates, single orphan rows, inconsistent country/session identifiers and invalid Saved flags.
- Count 13 valid choices in a completed session; reject 12 or duplicate order values; deduplicate multiple sessions per user by approved policy.
- Country with 100 eligible users is excluded, 101 included.
- Exact signature matching distinguishes age 3-vs-3 from 2-vs-3, passengers versus pedestrians, and legal versus illegal crossing.
- Verify pooled and equal-country majorities **can disagree** and the output clearly identifies the policy.
- Handle 50/50 ties, 51/49 near-ties, low counts, unknown culture, unmatchable and random scenarios without fabricated results.
- If input has only direction summaries and no scenario-level counts, exact-match-majority creation is blocked as `proxy_only`.
- All test inputs marked `SYNTHETIC_TEST_DATA`; no test outputs in report figures or scientific results.

### Sources and validation guidance

- Original study: Edmond Awad et al., *The Moral Machine experiment* (Nature, 2018), DOI `10.1038/s41586-018-0637-6` (`https://www.nature.com/articles/s41586-018-0637-6`), with study data availability linked from the paper.
- Public description of the original-style `SharedResponses` variables and 2-row representation in a later LLM comparison data package: `https://doi.org/10.5061/dryad.d7wm37q6v`. **This is explanatory codebook evidence, not permission to substitute the later study's LLM-generated data for the original human dataset.**
- Verify fields and value codings against the exact original dataset release and codebook before execution. Source data may contain missing/out-of-order outcomes; handle through the audit and validation reports.


---

## 7. Stage A: Environment, model identity, and reproducibility

**Goal:** make all later outcomes traceable to an exact model, package, input, configuration, and intervention.

**Implementation work:**

1. Review `pyproject.toml`, the current Python package, and the actual installed environment before adding dependencies; keep backward compatibility.
2. Resolve the **exact** model Hub commit/revision (or a verifiable local snapshot hash) and persist it. Pin tokenizer and model source, not just `google/gemma-2-9b`.
3. Record torch, transformers, CUDA, GPU name, total VRAM, numpy, matplotlib, Python, package version, OS/platform, Git commit, configuration SHA256, and every supplied data-file SHA256.
4. Set `model.eval()` and deterministic seeds. JLens Jacobian fitting requires gradients; ordinary scoring and readouts should use appropriate no-grad inference. Do not globally disable gradients in the fit stage.
5. Implement reliable residual hooks with context-managed lifetime and verification that hooks are fully removed even after exceptions. Preserve layer indexes 0-41 and verify output tensor shapes before editing.
6. Implement stage-level `status.json` with `pending/running/ok/failed/skipped/block_missing_data`, started and finished times, error message, input/config hashes, and output locations.
7. Preserve `.pt` lens checkpoints and experiment settings in Placement 2. Add an **explicit optional** cleanup command limited to disposable caches. Do not inherit Placement 1's unconditional `.pt` deletion.
8. Log resource usage (wall time, peak GPU memory) per stage; gracefully recover or provide instructions after out-of-memory failure.

**Acceptance criteria:** the run manifest uniquely identifies the experiment; changing input text or model revision invalidates the relevant stage cache; no residual hooks survive a stage; rerunning an unchanged stage reuses verified artifacts; Placement 1 tests still pass.

---

## 8. Stage B: Input validation and prompt preparation

**Goal:** transform validated researcher data into immutable model-ready prompt instances.

**Implementation work:**

1. Parse all required input files and perform schema and cross-record checks described in Section 3.
2. Build one canonical prompt instance per `(scenario_id, language)`, including scenario dimension, STAY/SWERVE structured mapping, human-reference eligibility and original prompt text.
3. Keep prompt wording **exactly as researcher supplied** except for a documented, invariant answer suffix applied consistently to all languages if explicitly approved.
4. Store the actual input IDs, prompt text hash, prompt token IDs/length, answer-candidate token IDs/length, truncation flags and reviewer status.
5. Reject any prompt that would be truncated (unless researcher approves an explicit, logged truncation policy). Reject prompt text with unfilled template placeholders.
6. Validate the set against the requirement `13 * number_of_languages` unique instances.
7. Provide a review-sheet CSV with EN and each non-EN counterpart side by side and all structured scenario metadata, for human translation review.
8. Detect basic answer-label leakage and variation in STAY/SWERVE instruction conventions; warn that static checks cannot prove translation equivalence.

**Outputs:** `validated_data/prompt_instances.jsonl`, `validated_data/validation_report.json`, `validated_data/review_sheet.csv`, `validated_data/tokenization_report.csv`.

**Acceptance criteria:** no duplicate/missing scenario-language pair, no unapproved prompt in a strict final run, all 13 structured scenarios valid, and fixed prompt-instance hashes for use in later stages.

---

## 9. Stage C: Binary-choice scoring and multilingual behavioural evaluation

### Why a new scorer is required

Placement 1's `score_mcq_prompt` compares the next-token logits of A/B/C/D with a universal factual key. Placement 2 now has a **culture-specific human-majority reference label**, generated by preprocessing, **not a universal moral truth**. STAY/SWERVE may have different token lengths; therefore use full-sequence scoring. Compute model behaviour independently of the human label, then add `is_reference_match` and a reference-relative margin only for `supported` labels.

### Required scoring algorithm

For every scenario-language prompt `x`, use two fixed candidate continuations `y_STAY` and `y_SWERVE`. Compute each candidate's conditional full-sequence log-probability:

```
score(y | x) = sum_{j=1..len(y)} log P(y_j | x, y_1, ..., y_(j-1))
```

Then obtain a **two-candidate normalized distribution**:

```
p_STAY   = softmax([score(STAY), score(SWERVE)])[0]
p_SWERVE = softmax([score(STAY), score(SWERVE)])[1]
margin   = score(STAY) - score(SWERVE)
choice   = STAY if margin > 0 else SWERVE if margin < 0 else TIE
```

This is the model's **normalised preference conditional on the two offered responses**, not necessarily its unconstrained probability of emitting one of these strings. Keep original scores and token counts. A tie must be preserved as a tie rather than silently picking an option. Explicitly test whether leading spaces/newlines, answer suffixes and optional termination tokens affect candidate scoring. Choose and document a primary scoring convention and run a limited sensitivity check; multi-token length bias is possible with raw sequence likelihoods, so report candidate tokenization and do not quietly switch to length-normalized scores.

**Operational benchmark agreement IS allowed:** compare predicted choices with the separately calculated `human_majority_reference_label` within the explicitly mapped culture group. Do not call agreement morally correct, and never invent labels for ties, small samples or proxy-only evidence. Report `human_majority_agreement_rate` and, optionally, `p_reference_option` / `reference_margin` alongside label-free STAY/SWERVE behavioural metrics.

### Required baseline output fields

One row per model/scenario/language:

- `run_id`, `model_id`, `model_revision`, `benchmark_version`, `scenario_id`, `dimension`, `language`, `prompt_sha256`.
- `prompt_token_count`, `stay_candidate_token_ids`, `swerve_candidate_token_ids`, `stay_token_count`, `swerve_token_count`, `choice_scoring_policy`.
- `score_stay`, `score_swerve`, `margin_stay_minus_swerve`.
- `p_stay`, `p_swerve`, `decision` (`STAY`/`SWERVE`/`TIE`).
- `entropy_bits` of the two-option distribution.
- `raw_generated_answer` in an **optional diagnostic** run, `parse_status` and `choice_matches_free_generation`; do not rely on free generation for primary forced-choice analysis.
- `culture_id` used for evaluation (if approved), `human_reference_label`, `human_reference_status`, `human_reference_aggregation`, `human_reference_evidence_type`, `is_reference_match`, `p_reference_option`, `reference_margin`; null if no supported reference.
- `seed`, `dtype`, `prompt_position`, `status`, runtime/memory measurements.

### Behavioural comparisons

Implement and save:

1. Overall STAY and SWERVE rates, broken down by language and dimension (descriptive counts).
2. Scenario-level cross-language decision agreement; treat ties explicitly.
3. English-anchored pairwise agreement **and** full pairwise language matrix, if at least two languages exist.
4. Per-scenario `delta_p_stay`, `delta_p_swerve`, `delta_margin` between language pairs.
5. Jensen-Shannon divergence (JSD, logarithm base 2) between the **same STAY/SWERVE probability support** for each language pair; do not compare differently labelled decisions.
6. Optional entropy and mean absolute probability gap per dimension and language.
7. Stable descriptive statistics with `n`, missingness and exact scenario IDs; no unsupported significance claims from 13 scenarios.
8. Keep label-neutral contrasts `different_decision`, `same_decision_high_distributional_difference` and `same_decision_low_difference`.
9. Add *culture-specific human-majority agreement* for only supported labels: agreement rate with clear numerator, denominator, matching method, aggregation choice and unsupported fraction; split by culture/language/dimension.
10. Add `delta_p_reference` and reference-relative margins only where both comparisons use an explicitly stated reference policy. Preserve label-neutral probability differences as the primary cross-language comparison.

**Outputs:** `behavior/decisions.jsonl`, `behavior/decisions.csv`, `behavior/pairwise_scenarios.csv`, `behavior/language_summary.csv`, `behavior/pairwise_summary.csv`, `figures/01_decisions_by_language.png`, `figures/02_pairwise_jsd.png`, `figures/03_probabilities_by_scenario.png`.

**Acceptance criteria:** exactly 13 x L baseline rows with valid conditional probabilities when all inputs are supplied; probabilities sum to one within tolerance; decisions and margins agree; both one-token and multi-token candidates pass tests; pairwise JSD is zero for identical distributions and symmetric under language order.

---

## 10. Stage D: Final culture-specific human references + model agreement

This stage runs after (a) the Phase 0 raw preprocessing, (b) researcher-supplied scenarios and translations, and (c) exact or explicitly approved scenario coverage validation. Do **not** recreate human profiles from manually entered country winner labels if response-level data have been preprocessed.

### Join strategy and label generation

1. Match each of the 13 scenarios' structured STAY/SWERVE casualty/action signatures against the canonical signatures in the preprocessed human-response data. Record `exact`, `conditional`, or `dimension_proxy` explicitly.
2. For every candidate culture group, aggregate matched human choices using **both** respondent-pooled and equal-country-vote policies; emit `n_stay`, `n_swerve`, country count, group preference share and tie/uncertainty flags where meaningful.
3. Apply the **approved** primary reference policy; a supported label can be `STAY` or `SWERVE`, otherwise `UNDETERMINED`. A random scenario may receive a label **only if** it has sufficient direct matched-response support; never infer it from a single character dimension.
4. Use `language_reference_map.csv` to map each prompt language to an explicitly declared `culture_id` for alignment analysis. If mapping is absent/ambiguous, **still run model behaviour**, but leave that human-comparison field missing rather than assuming country/language/culture equivalence.
5. Join each model decision to its `(scenario_id, culture_id, approved policy)` label and compute `is_reference_match`, `p_reference_option`, signed `reference_margin`, and per-group agreement with denominators and evidence coverage. No labels should leak into the prompt or influence which response probability is calculated.
6. Produce sensitivity tables comparing both aggregation policies and any approved conditional/dimension-proxy reference separately; a proxy-only label must not appear in the observed-human-majority headline table.

### Formal definition

For a supported `(scenario=s, culture=c)` reference:

```
y_human(s,c) = STAY    if majority under approved policy favours STAY
             = SWERVE  if majority favours SWERVE
             = UNDETERMINED if tied, missing, unverified or underpowered

is_reference_match(s,language,c) =
    1 if model_choice(s,language) == y_human(s,c)
    0 if model_choice(s,language) != y_human(s,c)
    missing if y_human == UNDETERMINED or model_choice == TIE

human_majority_agreement_rate = sum(valid is_reference_match) / count(valid rows)
```

That proportion is **benchmark reference agreement**, not model moral accuracy. If two languages are compared against *different* cultural human references, their apparent match/mismatch differences can result partly from **different human labels**, even if the model returns the **same** STAY/SWERVE decision. Both phenomena must be displayed independently.

### Required outputs

- `human/scenario_reference_labels.csv`, `human/reference_quality_report.md` and `human/reference_lineage.jsonl`.
- `human/model_human_agreement.csv` by language/culture/dimension, including supported/unsupported counts and source and policy hashes.
- `human/aggregation_sensitivity.csv`: flag scenarios where changing from pooled to equal-country votes changes the winning label.
- `figures/04_human_model_agreement.png`, labelled unambiguously with its denominator and evidence type.

**Acceptance criteria:** no cultural group silently inferred from the prompt language, no `correct` attached to the scenario independent of culture, no forced label for a tie/low support or 13th Random case without evidence, and no conflation of an exact matched majority with a dimension-level proxy. Agreement and reference-relative margins must be traceable back to individual human choice counts and the predeclared grouping policy.

---

## 11. Stage E: Pre-registered case selection: A1/A2 AND decision-flip controls

Placement 1 grouped 200 factual MCQs into A1 English-correct/Persian-wrong, A2 reverse, A3 both-wrong/different, B1 same answer with confidence shift and B2 stable controls. We can retain the **research purpose** of these groups but must redefine 'correct' as **matches the majority human reference for a declared culture**, not objective ethical truth.

### Keep THREE separate axes; do not force them into one exclusive label

**Axis H -- culture-human reference alignment (only when both labels are supported):**

| Human-alignment group | English vs English-mapped culture | Persian vs Persian-mapped culture |
|---|---|---|
| `A1_human` | Matches EN cultural majority | Does NOT match FA cultural majority |
| `A2_human` | Does NOT match EN cultural majority | Matches FA cultural majority |
| `A3_human` | Both fail to match their own declared cultural majorities | Both mismatched |
| `A4_human` | Both match their own cultural majorities | Both matched |
| `H_unknown` | One or both unsupported/tied human labels | Not classifiable |

If the English-mapped and Persian-mapped human majorities differ, identical model answers **can** produce A1/A2. Therefore A1/A2 do not necessarily mean a language-caused model decision flip. Store `human_reference_labels_equal: true/false/unknown` and always show the actual human reference labels for each side.

**Axis D -- raw cross-language model decision:**

- `D1_flip`: English STAY, Persian SWERVE.
- `D2_flip`: English SWERVE, Persian STAY.
- `D_same`: same STAY/SWERVE choice.
- `D_unknown`: either model tie/error.

**Axis P -- probability-distribution divergence:**

- `B1_high_jsd`: same model decision, with `JSD >= 0.05 bits` **or** absolute `delta_p_stay >= 0.10` (the previous thresholds, provisional for this binary-choice task).
- `B2_low_jsd`: same model decision and below both thresholds.
- `P_not_applicable`: decision differs or probabilities unavailable.

Keep labels independent; `A1_human` and `B1_high_jsd` may coexist because they describe different phenomena. Never assume A1/A2 forms an exhaustive mutually exclusive list with B1/B2. Report full counts including unknown groups.

### Primary case-selection strategy (with only 13 scenarios)

1. Run all languages' baseline behaviour and join supported human references **first**.
2. Prioritise verified A1/A2 human-mismatch pairs *and* D1/D2 decision flips if available.
3. Include **same-decision/high-JSD** B1 cases because internal concepts may differ despite the same answer.
4. Include **stable same-choice/low-JSD** B2 controls to interpret JLens readout stability.
5. Prefer matched scenario signatures with high human evidence and clearly reported denominator; never choose cases solely because a JLens plot or intervention already looked striking.
6. If no A1/A2 or D1/D2 cases exist, do **not fabricate** them. Fall back to the strongest valid confidence divergence and stable controls; document observed availability.
7. Selection quotas (e.g. six scenarios), JSD cutoffs, layer set, selection seed and quality gates are frozen before reading JLens/causal outcomes.

**Outputs:** `selection/case_groups.csv` (all axes, choices, references, human evidence, JSD), `selection/selected_cases.json`, `selection/missing_reference_cases.csv`, and `selection/selection_protocol.json`.

**Acceptance criteria:** A1/A2 are always defined **relative to their specific reference groups**; decision flips are tracked separately; selection is deterministic; no strong inference based on unverified human labels; stable/agreeing controls included where available.

---

## 12. Stage F: Jacobian Lens fitting, stability, and mechanistic readouts

**Goal:** observe language-associated intermediate verbalizable representations for the same moral dilemma while ensuring that readouts are not dominated by answer-template artifacts.

### 12.1 Fit the lens robustly

Start from existing `jlens/fitting.py`, `jlens/hf.py`, and `jlens/lens.py`. The Placement 1 JLens fit approximates a mapping from the residual stream at an intermediate layer to the final residual. Avoid inventing a replacement JLens algorithm without documenting the deviation.

1. Require a suitable independent calibration set, ideally distinct from the 13 benchmark scenarios; ensure deterministic fit/holdout split and disjoint IDs.
2. Reproduce the current Gemma 2 9B setup, inspecting actual fitted Jacobian orientation and tensor shapes. **Provisional four-layer plan: 22, 26, 30 and 38** (researcher's candidates); final block target **41**. The prior patching curve suggests ~21-28 is a transition and later layers approach a plateau. An alternative **18, 22, 26, 38** includes a pre-rise baseline, but must be explicitly approved before fitting. Placement 1's fitted late layers 38/40 do **not** constitute validation of new layers 22/26/30.
3. Calibrate dimension batching, max length, precision and checkpoint frequency to the available GPU. The Placement 1 handoff used a dimension batch of 32, skipped the first 16 positions and omitted the last position during **fitting**; this does not mean the application position must be skipped. Document the distinction.
4. Run a real repeated-fit stability check on independent calibration subsets or different fit seeds. Record similarity of the fitted mappings and/or readouts on identical held-out prompts; never mark stability as validated because a disabled option exists.
5. Evaluate held-out readouts with a documented metric and selected comparison examples. If output is unstable or overwhelmingly format-dominated, flag it and do not proceed to a strong semantic claim.
6. Save fit calibration IDs, model/tokenizer revision, layers, fitted matrices, checkpoints and validation artifacts. **Do not delete fitted lens files** before interventions.
7. The approved four-layer design covers a limited transition/late profile, not all stages of computation. If additional compute is available, a fifth layer can be included only as a predeclared extension. Earlier JLens layers may have weaker or less stable concept readouts; report fit and coverage at every selected layer.

### 12.2 JLens readouts

For each selected prompt and source layer, at the same well-defined **last non-padding prompt token** position:

- Capture `resid_post` using the existing PyTorch hook path.
- Apply the fitted lens and the appropriate model unembedding / final normalization as the existing implementation defines.
- Save at least the top **50 raw vocabulary candidates** with token IDs, decoded strings, readout scores/ranks, layer, language, and prompt ID.
- Apply content filtering and retain top **8 eligible tokens** or a configurable equivalent, preserving the unfiltered originals as evidence.
- Measure the fraction of raw top-k entries removed as format/answer/template tokens. Empty content readouts must be represented as **no usable content**, not replaced by arbitrary filler tokens.

### 12.3 Fix the Placement 1 content-filter error

Add case-insensitive, Unicode-aware, whitespace-aware and punctuation-aware filtering for at least:

- `Option`, `Answer`, `Question` and relevant translations/templates actually used;
- `STAY`, `SWERVE`, single-choice letters and explicitly listed answer labels;
- BOS/EOS/PAD/special tokens, standalone whitespace, punctuation and newline tokens;
- prompt wrappers, response markers and tokens shown by pilot diagnostics to be generic formatting artifacts.

Retain a transparent `excluded_reason` for every removed token, and permit an override list for real study-specific meaningful vocabulary. Do not hide the fact that `Option` was present in Placement 1; report how the new filter changes interpretation. A generic keyword may sometimes be substantive, so make final inclusion/exclusion decisions inspectable.

### 12.4 Align concepts across languages, not just string forms

Two different strings may verbalize the same concept. Produce two distinct outputs:

**A. Surface/token diagnostic:** Exact decoded-token Jaccard, primarily for engineering and comparison with Placement 1; not a semantic similarity metric.

**B. Concept-aligned primary comparison:** Use researcher-reviewed `concepts.json` to map eligible readout terms into canonical concept IDs. Match with careful tokenizer validation and defined lexical normalization; record unmatched terms and coverage. If feasible, form concept scores from approved single-token verbalizers using a declared aggregation (e.g. log-sum-exp of lexical candidates or a stated contrast score). Do not silently aggregate raw logits with incomparable scales or mix single-token and multi-token phrase measures.

For selected concept pairs (e.g. `young` vs `elderly`), report **relative contrast within each prompt/layer**, rank/prominence, and calibrated or normalized scores with caveats. Comparing absolute vocabulary logits of different language words may reflect lexical frequency/tokenization effects. Record both language-specific term scores and the canonical concept mapping so readers can inspect the correspondence.

### 12.5 Mechanistic comparisons and visualisations

Implement, where concept coverage allows:

1. For each case, layer-wise top eligible verbalizations per language, including raw and filtered views.
2. Concept-ID Jaccard or another transparently defined overlap metric per layer; include missing/low-coverage flags.
3. Canonical concept score/rank trajectories across studied layers for equivalent prompts.
4. A behavioural-versus-mechanistic quadrant: same/different decision against similar/different internal concept readouts, with carefully stated thresholds.
5. Relationships between behavioural JSD and descriptive concept divergence; correlations are exploratory with such small samples.
6. Case study reports for selected high-disagreement, same-decision/high-JSD and stable-control examples.
7. Fit and readout stability/coverage plots; no causal wording at this stage.

**Outputs:** `jlens/calibration_manifest.json`, `jlens/fitted_lens.pt`, `jlens/fitting_metrics.json`, `jlens/repeat_fit_metrics.json`, `jlens/holdout_readouts.jsonl`, `jlens/readouts_raw.jsonl`, `jlens/readouts_filtered.jsonl`, `jlens/concept_scores.csv`, `jlens/cross_language_comparisons.csv`, `figures/05_concept_overlap_by_layer.png`, `figures/06_concept_trajectories.png`, `figures/07_case_studies.png`.

**Acceptance criteria:** loaded lens agrees with saved lens on a fixed held-out fixture; repeat-fit stage actually executes and reports reproducible diagnostics; ` Option` is not silently counted as a meaningful concept; multilingual comparisons clearly separate lexical overlap from semantic concept alignment; concept-aligned analysis blocks or flags insufficient lexicon coverage; the fit checkpoint survives the pipeline.

---

## 13. Stage G: Targeted concept interventions with controls

**Goal:** test whether changing selected internal directions is associated with a change in STAY/SWERVE model behaviour, rather than just observing descriptive JLens projections.

### 13.1 Important conceptual distinction

The existing `jlens/intervention.py` reportedly adds or subtracts a **token output-embedding direction at an intermediate layer**. This is a functioning perturbation procedure, but **it is not automatically a validated layer-local semantic concept intervention**. Placement 1 showed that answer-letter edits could strongly move the model's output while content edits did little.

Placement 2 should therefore implement **both**:

1. A **Placement 1-compatible embedding-direction baseline** (well documented as such).
2. A **Jacobian-informed layer-specific direction** for the selected verbalizer/contrast, validated for dimensions and local effects.

Do not call an edit "JLens causal concept manipulation" until the chosen direction is mechanistically and empirically justified. Failed or inconclusive interventions are legitimate reportable outcomes.

### 13.2 Candidate construction approaches

For a target verbalizable concept `c`, define a documented final-output readout direction `u_c` (for example a contrast between approved output readout directions for `young` and `elderly`). Given a fitted local Jacobian mapping `J_l: h_l -> h_final`, a **possible** source-layer steering direction is a backprojection:

```text
v_l,c = normalize(J_l^T u_c)
```

Here the orientation of `J_l` must be checked against **the actual implementation** and the exact final readout including normalization/unembedding. The formula is a design candidate, **not an assertion that the current repository already computes or validates this direction**. Where appropriate, a more faithful final-logit gradient can replace a naive unembedding vector. Validate with finite differences: small perturbations along `v_l,c` should produce a measurable predicted direction of change in the relevant readout, if the local approximation is useful.

Alternative direction methods (e.g. regularized inverse solutions, independently trained probes) are optional method extensions, not MVP dependencies. Never construct a multi-token concept direction by blindly selecting the first tokenizer token.

### 13.3 Intervention procedure

For each **preselected** scenario-language pair and approved concept/contrast:

1. Run clean baseline through the new STAY/SWERVE scorer and cache the reference probabilities/log-scores.
2. Capture the selected layer's residual at the **last prompt token** (or a preapproved token span sensitivity check).
3. Construct the proposed `v_l,c` using the fitted lens and approved concept terms, then check its dimension, norm, finiteness and expected readout direction.
4. Apply a single controlled PyTorch hook: `h'_l,t = h_l,t + delta`, with `delta` proportional to a configured fraction of the **original residual norm**. Make the normalization convention explicit in output records.
5. Sweep **positive, zero and negative** intervention strengths with a conservative small-norm default (see config), rather than immediately repeating Placement 1's large 0.25/0.5 residual-norm edits.
6. Re-evaluate both **full-sequence** STAY/SWERVE candidate scores under the same scoring convention, ensuring the intervention is positioned consistently for both candidate continuations.
7. Remove hooks; verify no contamination of later prompts or controls.
8. Save the clean and edited results for every strength, layer, concept, language, scenario and control.

The default intervention site is the final prompt token to maintain comparability with Placement 1, but the report must explicitly state that this does not localise effects to the dilemma's character tokens. If candidate continuation scoring is implemented by appending full options, the hook index must refer to the **original prompt last token**, not the final token of the entire prompt-plus-answer sequence.

### 13.4 Required intervention types

| Type | Implementation | Interpretation |
|---|---|---|
| Strengthen | Add a validated concept/contrast direction | Test effect of positive steering |
| Suppress | Subtract the same direction with same norm convention | Test effect of negative steering |
| Swap / contrast edit | Add target direction and subtract source direction, with a controlled combined norm | Test direction replacement in a defined concept pair |
| Zero-strength sham | Hook attached but `delta = 0` | Must reproduce clean baseline |
| Random matched-norm | Seeded random direction with exactly matched perturbation norm | Quantifies generic residual disturbance |
| Unrelated concept | Different approved semantic direction with matched norm | Tests target specificity |
| Format/answer marker diagnostic | E.g. `Option` / answer-symbol direction if technically valid | Detects answer-template steering, **not** a semantic negative control |

Optional extension: apply same intervention across equivalent languages and across counterbalanced scenario variants (if supplied). Never substitute an answer-symbol direction for the main semantic intervention.

### 13.5 Required causal results

For every intervention store:

- scenario, language, layer, token position, concept ID, direction-construction method and lexicon version;
- intervention sign, target relative norm, actual `||delta|| / ||h||`, random seed, direction norm and shape;
- original `p_stay`, `p_swerve`, `margin`, `decision`;
- edited `p_stay`, `p_swerve`, `margin`, `decision`;
- `delta_p_stay`, `delta_p_swerve`, `delta_margin`, flip boolean, and JSD(clean vs intervention);
- corresponding same-strength control outcomes and a clearly defined `target_minus_control` effect when meaningful;
- status (`ok`, `failed_invalid_direction`, `failed_hook`, `no_valid_lexicalization`, etc.), compute time and diagnostic warnings.

Interpretation rules:

- A large perturbation effect **alone** does not show a meaningful concept was isolated.
- A predicted directional effect which exceeds random/unrelated controls and replicates across matched scenarios/languages would provide stronger causal evidence **for the tested intervention procedure**.
- The same decision after an intervention is not necessarily a null result if the two-option probabilities move; report both continuous and discrete outcomes.
- An effect that is as large for a random or template direction as for a moral concept is **inconclusive for concept specificity**, even when the answer flips.
- If JLens fitting or concept validation fails, retain the descriptive result and report the intervention limitation; never force a positive intervention finding.

**Outputs:** `interventions/feasibility.json`, `interventions/trials.jsonl`, `interventions/trials.csv`, `interventions/control_comparisons.csv`, `interventions/diagnostics.md`, `figures/08_intervention_dose_response.png`, `figures/09_target_vs_control_effects.png`.

**Acceptance criteria:** zero edit reproduces clean scoring to defined numerical tolerance; all intended hooks are removed; random matched-norm controls are truly norm-matched; all edits are auditable; no unchecked answer-label direction is presented as proof of a semantic mechanism; full-candidate option scores are used consistently for clean and edited runs.

---

## 14. Stage H: Cross-method integration and research reporting

The report should directly answer the two research questions using the outputs actually obtained; it must not imply every planned stage produced a positive finding.

### Report-ready comparisons

1. **Study manifest and data summary:** 13 scenario structure, languages, eligibility rules, number of country groups and calibration items; source/version identifiers.
2. **Behavioural stability:** decision agreements, STAY percentages, JSD, probability gaps, distribution plots, dimension-wise descriptive tables.
3. **Human reference:** raw paired response provenance; source/country/culture coverage; scenario-level observed-majority vs dimension-proxy distinction; primary and alternate aggregation policies; valid numerator/denominator alignment, missing/tied/low-support references and country-language caveats.
4. **Mechanistic representations:** raw vs filtered JLens readouts, lexicon coverage, concept trajectories, comparisons where the answer agrees or disagrees.
5. **Intervention outcomes:** target concept direction, strength, score changes, flips, matched controls, failed or inconclusive trials.
6. **Cross-stage case studies:** show prompt text or ID, language pair, model decision/probability, human reference if mappable, top validated concepts per layer, and any intervention effects. Ensure each case can be traced to raw artifacts.
7. **Reproducibility:** exact model revision, dependencies, configs, RNG seed, inputs, run IDs, code commit, JLens fit checkpoints, method definitions.
8. **Limitations:** 13-scenario sample size, country-language proxy, translation equivalence, fixed model, restricted layers/token positions, JLens lexicalization issues, calibration domain, controls, computational constraints and possible label-length effects.

Generate **machine-written descriptive summaries**, not fabricated explanations of why the model chose something. Support copy/paste into the final report without requiring that Markdown report notes be the sole archive.

### Mandatory figures and tables (only with real input data)

- Dataset/session composition table with six dimensions, repeats and languages.
- Baseline STAY/SWERVE decision matrix by scenario x language.
- Option-probability comparison heatmap or per-scenario point plot, and full language-pair JSD matrix.
- Preprocessing funnel (paired decisions, completed users, eligible countries), scenario-support catalog, culture-specific STAY/SWERVE majority table with evidence strength, and human-model agreement figure with denominators and uncertainty.
- Selected mechanistic case comparison showing raw JLens readouts **and** filtered concept IDs.
- Layer-wise concept-contrast trajectories (only over layers actually fitted).
- Intervention dose-response versus matched controls, with zero-strength checks and uncertainty/replicate indication where available.
- Methods/runtime table and concise study flow schematic created programmatically (no invented data values).

### Suggested output tree

```text
output/placement_2/<run_id>/
  manifest.json
  status.json
  stage_times.json
  failures.jsonl
  research_log.md
  validated_data/
    prompt_instances.jsonl
    validation_report.json
    review_sheet.csv
    tokenization_report.csv
  behavior/
    decisions.jsonl
    decisions.csv
    pairwise_scenarios.csv
    language_summary.csv
    pairwise_summary.csv
  human/
    raw_audit/
      schema_report.json
      raw_pairing_audit.json
    paired_responses/                 # Partitioned Parquet, approved storage
    eligible_countries.csv
    country_dimension_profiles.csv
    culture_dimension_profiles.csv
    scenario_support_catalog.csv
    scenario_reference_labels.csv
    reference_lineage.jsonl
    reference_quality_report.md
    aggregation_sensitivity.csv
    model_human_agreement.csv
  selection/
    all_candidates.csv
    selected_cases.json
    selection_rationale.md
  jlens/
    calibration_manifest.json
    fitted_lens.pt
    fitting_metrics.json
    repeat_fit_metrics.json
    holdout_readouts.jsonl
    readouts_raw.jsonl
    readouts_filtered.jsonl
    concept_scores.csv
    cross_language_comparisons.csv
  interventions/
    feasibility.json
    trials.jsonl
    trials.csv
    control_comparisons.csv
    diagnostics.md
  tables/
  figures/
  report_notes.md
  limitations.md
```

There must be a single index or manifest linking every figure and table to the exact source files and methods. Do not delete intermediate scientific evidence by default. Follow university-approved storage and any data or model licensing restrictions; raw human response records and large model weights do not need to be copied into the Git repository.

---

## 15. Test plan and quality gates

The **agent can implement and test most software before actual research data exists**. Use tiny synthetic tests with obvious provenance; scientific charts and conclusions are only produced from approved real data.

### Gate 0: Repo compatibility (no study data needed)

- [ ] Existing Placement 1 `pytest` suite passes without changing old expected results.
- [ ] Wheel installation and editable installation expose the same Placement 2 commands.
- [ ] Base model loading, layer indexing, `resid_post` hook and hook teardown follow current conventions.
- [ ] Versions/model revision/config hashes are captured; no old `.pt` files are deleted.

### Gate 1: Data structure (CPU-only)

- [ ] Exactly 13 scenarios with six dimensions twice plus one random, enforced.
- [ ] Missing/duplicate language translations are reported with keys; empty and unreviewed strings blocked in strict mode.
- [ ] Raw `ResponseID` pairing and human-action coding from `Saved` + `Intervention` pass both STAY/SWERVE tests.
- [ ] Completed sessions are reconstructed from 13 valid scenario responses and user deduplication is correct.
- [ ] Country filter excludes `n_completed_users = 100`, includes `101`.
- [ ] Both respondent-pooled and equal-country aggregation are implemented; ambiguity and different resulting majorities are reported correctly.
- [ ] Culture and language mappings are separate and approved; missing/tied reference labels become `UNDETERMINED`.
- [ ] Human-majority reference matching requires exact scenario structure or an explicit, separately labelled proxy. 
- [ ] STAY/SWERVE human mapping follows outcome metadata in both directions.
- [ ] Random scenario is excluded from dimension-proxy agreement, but may have direct observed-majority coverage if exactly matched and sufficiently supported.
- [ ] Concepts and bilingual term aliases are versioned; unsupported multi-token lexicalizations flagged.
- [ ] Missing real datasets never trigger creation of mock 'research' data.

### Gate 2: Behavioural math (CPU-only)

- [ ] Compare scorer against a tiny fixed language-model fixture with known conditional probabilities.
- [ ] Multi-token labels are scored across **all** continuation tokens, not their first tokens.
- [ ] Candidate prompt boundary and padding index tests pass.
- [ ] Probabilities sum to 1, and `margin` sign agrees with the chosen option.
- [ ] JSD is symmetric, nonnegative, zero for identical distributions and base-2 bounded.
- [ ] A1/A2 human-reference categories, D1/D2 choice flips and B1/B2 divergence axes are independently correct; tests include differing EN/FA human-reference labels and *same* model choice.
- [ ] Decision disagreements and same-choice/high-JSD classifications work, including ties.
- [ ] Language-pair comparison logic remains correct for 2, 3 or more languages.

### Gate 3: JLens and model integration (requires model/GPU)

- [ ] Frozen model and correct last-prompt-token hooks confirmed with small probes.
- [ ] Saved and loaded JLens produce equivalent readouts on a held-out prompt.
- [ ] Repeat-fit stability stage runs (not merely configured); out-of-domain calibration caveat recorded.
- [ ] Raw and filtered top tokens saved; ` Option`, template labels and special tokens excluded from *content* metrics.
- [ ] Filter and concept lexicon retain traceable token IDs/scores and record coverage.
- [ ] Source-layer versus final-layer dimensions/orientations checked explicitly.

### Gate 4: Causal interventions (requires model/GPU)

- [ ] Zero-strength hook returns the same output distribution as clean inference.
- [ ] Non-zero intervention changes precisely the intended layer/token activation, within chosen numerical tolerance.
- [ ] All hooks are removed, and results are unaffected by intervention trial order.
- [ ] Same-norm random controls have identical prescribed perturbation norm.
- [ ] Both directions (strengthen/suppress) and a target contrast produce complete signed-effect records.
- [ ] Concept direction construction passes shape, finite-value and local sensitivity checks, or produces an explicit `inconclusive/invalid` record.
- [ ] Intervention scoring uses the same complete STAY/SWERVE candidate sequences as baseline.
- [ ] A format/answer control is labelled as a diagnostic, never as evidence of moral-concept specificity.

### Gate 5: Whole-pipeline reproducibility (requires completed research data)

- [ ] One command respects raw human preprocessing -> study scenario creation gate -> exact scenario matching/reference finalisation -> baseline -> alignment/groups -> JLens -> intervention -> reports.
- [ ] Runs can resume after a failed GPU stage without restarting finished stages.
- [ ] Every artifact identifies its run/config/input/code version; no stale-cache collisions.
- [ ] All figures are generated from real records, with axis labels, units and sample sizes.
- [ ] Study report contains all null outcomes and limitations; no unsupported significance or moral-correctness language.
- [ ] Exact paths and CLI examples in `RUNBOOK.md` reproduce the outputs.

---

## 16. Execution milestones, priorities, and stopping points

The proposed placement budget is 35 h research + 7 h mentor meetings + 20 h deliverables (62 h, seven weeks). Prioritize a reproducible comparison over a large new modelling framework. The first research block is **human preprocessing**, even while the agent can build shared software scaffolding in parallel.

| Stage | Approx. week | Owner | Deliverable | Dependency |
|---|---|---|---|---|
| 0. Human raw data audit + preprocessing | 1 | Agent | Paired human decisions, completed-user filter, eligible countries, country/culture profiles, signature catalog | Approved raw source and mapping |
| 1. Researcher designs benchmark | 2 | Researcher | 13 canonical structured dilemmas, translations, review/label mapping | Human support catalog |
| 2. Scenario-specific human majorities | 2-3 | Agent | Matched reference labels, support/uncertainty report, alternative aggregation policies | 13 scenarios and approved reference rule |
| 3. Baseline behaviour | 3 | Agent | 13 x L model predictions, reference agreement, JSD, group table | Researcher translations, Gemma GPU |
| 4. Selection/JLens | 4-5 | Agent | Frozen A1/A2/D1/D2/B1/B2 groups, verified JLens and four-layer readouts | Baseline and calibration |
| 5. Concept interventions | 6 | Agent | Approved edit direction, negative controls, signed probability/label effects | Mechanistic cases, GPU |
| 6. Report/reproducibility | 7 | Agent + researcher | Figures, tables, human reference lineage, notebook, report notes, presentation | All approved experiments |

### Implementation commits, in order

**PR 1 - Human source contract and repo audit:** preserve Placement 1; add source manifest, format/codebook inspector, run manifest and streaming requirements.

**PR 2 - Human preprocessing (P0):** reconstruct paired choices, full-session/user/country filter, signatures, respondent vs equal-country summaries, coverage catalog, synthetic tests. Agent sends catalog to researcher **before requesting the 13 prompts**.

**PR 3 - Researcher scenario interface:** provide scenario templates/data dictionary, validator, translation review report, exact human coverage query, label-finalisation CLI. User authors real scenarios and translations.

**PR 4 - Forced-choice baseline:** full-continuation STAY/SWERVE scorer, 13 x L batch runner, reference matching and metric exports.

**PR 5 - Case selection and JLens:** orthogonal A1/A2, D1/D2, B1/B2, four-layer candidate set, calibration stability, raw/filtered concept comparisons, preserve fitted lenses.

**PR 6 - Intervention controls:** validated concept/contrast directions, zero strength and random norm-matched controls, sensitivity and quality flags.

**PR 7 - Reproducibility:** end-to-end stage runner, slim notebook, figures, report notes, user documentation, final provenance check.

### Block and ask the researcher when:

- Raw release/version/licensing or row-pair `Saved`/`Intervention` semantics cannot be verified.
- The choice between **pooled individuals** versus **one vote per country** as the headline cultural majority has not been approved.
- `culture_id` membership or comparison to a prompt language is not explicitly supported by a supplied mapping.
- Exact matching yields too few human decisions to label an authored scenario; report coverage and propose alternatives, **do not silently expand matching**.
- A tied/near-tied or underpowered group cannot support a strong majority claim.
- Mechanistic concept aliases are unreviewed or JLens produces mainly template tokens.
- Additional models/languages/layers are proposed before the basic study and controls work.

The first researcher handoff should contain (a) human support catalog, (b) exact scenario template, (c) the required dimensions/structure, (d) expected language mappings, and (e) a list of unsupported combinations to avoid when authoring scenarios.

---

## 17. Initial scientific decisions requiring approval

Version the following fields (`status`, `value`, `reason`, `approved_by`, `approved_date`, `protocol_version`) and block final scientific runs if critical fields remain unresolved.

1. **Raw human dataset:** Which original public release, hash, source terms and codebook are approved? Is it the two-outcome `SharedResponses` format or an alternate export?
2. **Completed-session counting:** Exactly how to count users with >1 session; which 13-response completeness/ordering criteria; how to handle cross-country IDs and missing user IDs?
3. **Culture definitions:** Which countries form each `culture_id`? Is the benchmark comparing language-associated country groups, published cross-cultural clusters, single countries or other explicitly justified units?
4. **Language-to-culture reference:** Which approved cultural group is compared against each prompt language? Do not assume an English-speaking model inherits the culture of respondents grouped as English.
5. **What does "majority" mean?** Choose **respondent-pooled** majority or **equal-country-majority**, with the other as sensitivity. The original proposal leans toward equal-country direction voting; the later user clarification emphasizes majority human answers. These may disagree.
6. **Scenario-level evidence:** Do final labels require direct exact scenario matches? Recommended **yes**; if not, specify conditional relaxations and keep dimension-proxy evidence separately described.
7. **Support/ties:** Minimum matched human responses and contributing countries; 50/50, close-majority and uncertainty flags. No arbitrary strong claims at tiny n.
8. **Final 13 scenarios/translations:** Researcher authors them **after** reviewing human support catalog; must include structured STAY/SWERVE outcomes to permit exact signature matching.
9. **Scoring:** Approve model candidate prefixes, termination/length policy and sensitivity tests; all 13 x L prompts use same `STAY`/`SWERVE` candidate convention.
10. **Group comparisons:** A1/A2 defined by language-specific cultural-majority **agreement**; D1/D2 label-independent model decision flips; B1/B2 high/low probability divergence. Never merge axes without an explicit composite rule.
11. **JLens layers:** Researcher proposed `[22,26,30,38]`; earlier-stage comparison `[18,22,26,38]` is an alternative. Confirm exact four before fitting, plus approved calibration corpus and concept lexicon.
12. **Intervention proof threshold:** Predefine direction and controls, normalization, layer/position, minimum case count and when causal results are inconclusive.
13. **Compute budget:** Available GPU/VRAM and stop rules for adding more cases, concept types or models.

---

## 18. Definition of done

The implementation can be considered **software-complete** before the researcher writes the 13 scenarios if:

- The new package, config, raw-response pairing/preprocessor, completed-session country filter, human-majority aggregation alternatives, scenario-support catalog, templates, validators, choice scorer, behavioural/human-alignment metrics, case selection, JLens, interventions, controls, reports and CLI **exist and pass tests**.
- The pipeline runs end-to-end on **clearly labelled synthetic software fixtures** (CPU portions) and has separately verified model/GPU smoke probes for JLens/intervention mechanics.
- Missing researcher data results in an explicit gate, not made-up values.
- Placement 1 code/notebook functionality remains intact.
- The instructions allow a fresh approved GPU environment to reproduce the run using exact IDs, hashes and retained JLens artifacts.

The placement can be considered **research-complete** once the researcher provides/approves all required data and protocol choices, the raw human reference majority is measured and validated with source provenance, the actual Gemma 2 9B run passes the quality gates, human and cross-language analyses are produced, selected JLens cases have valid coverage and stability diagnostics, intervention attempts and controls are fully documented (whether positive, null or inconclusive), and final figures, report, presentation material and reproducibility records are ready for mentor review.

**Do not confuse a working intervention tool with evidence for a culturally distinct moral mechanism.** Report exactly what was manipulated, what moved, what remained unchanged, and what the controls can or cannot rule out.

---

## 19. One-page checklist for the coding agent

### Implement FIRST -- no user-authored 13-scenario benchmark required

- [ ] Audit repo and old tests; freeze model/environment/config identity.
- [ ] Build `human_source.yaml` manifest parser and actual raw input schema/codebook inspector.
- [ ] Stream and pair two outcome rows per `ResponseID`; derive actual human STAY/SWERVE using `Saved` AND `Intervention`.
- [ ] Check pair validity, create reproducible scenario signatures and rejection audit.
- [ ] Identify completed 13-decision sessions, distinct eligible users, countries with >100 completed users; report count vs 117.
- [ ] Implement versioned country-to-culture map and independent language-to-reference-group map.
- [ ] Compute country and culture dimension profiles with respondent-pooled and equal-country-vote results.
- [ ] Export scenario-support catalog and researcher authoring guide.
- [ ] Implement 13-scenario JSON templates, validators, translation templates, and scenario-human evidence matching.
- [ ] Implement scenario-level observed human-majority STAY/SWERVE labels, clear missing/tie/low-support flags, and policy sensitivity outputs.

### Waiting for researcher action AFTER human preprocessing catalog exists

- [ ] Approve original raw source, eligibility/country culture grouping, and **primary majority weighting rule**.
- [ ] Review scenario-support catalog and select 13 supported dilemma structures (two per six dimensions + random).
- [ ] Author canonical prompts with explicit casualty/outcome metadata.
- [ ] Translate/review the prompts across selected languages.
- [ ] Confirm four JLens layers `[22,26,30,38]` or `[18,22,26,38]`, model revision and concept vocabulary/calibration source.

### Implement/execute after reviewed research input exists

- [ ] Calculate exact scenario-matched human-majority references and verify sufficient support; otherwise report proxies separately.
- [ ] Implement/run full-sequence two-choice Gemma scoring for 13 x L prompts.
- [ ] Report p(STAY), p(SWERVE), JSD, human majority agreement, evidence coverage and optional reference margin.
- [ ] Independently tag A1/A2 human-alignment categories, D1/D2 decision flips, B1/B2 same-choice confidence divergence, stable controls.
- [ ] Fit/revalidate JLens at approved four layers; filter `Option`, answer markers and other template tokens, align concepts across languages.
- [ ] Run approved concept interventions with baseline/dose-response and norm-matched negative controls.
- [ ] Archive analysis-ready tables, figures, research log, all controls, model/JLens checkpoints and provenance.
- [ ] Prepare report-ready notebook and data/analysis methods for final FOSE7901 deliverables.

---

## 20. Short summary for the researcher

**Yes -- start with HUMAN DATA PREPROCESSING before creating the 13 prompts.** The agent first cleans and validates original Moral Machine human response records, reconstructs actual STAY/SWERVE decisions, applies completed-session and country filters, aggregates country/culture profiles under both weighting schemes, and produces a **scenario support catalog**. The catalog tells you which age/gender/fitness/status/number/species contrasts and structural situations have enough human evidence.

**Then YOU author and approve** the actual 13 moral scenarios, their STAY/SWERVE casualty/outcome structures, the chosen cultures/languages and reviewed translations. You do **not** need to manually compute millions of respondent majorities.

**Then the agent computes culture-specific majority labels** for each matched scenario and runs Gemma 2 9B. A model decision can be recorded as *human-majority aligned* or *not aligned*, relative to that culture's reference and with a clear denominator; it cannot be described as ethically correct/incorrect.

**Mechanistic selection:** A1/A2 now mean English/Persian agreement versus the **respective human reference groups**; D1/D2 independently mean actual language-dependent STAY/SWERVE choice flips; B1/B2 are same-answer confidence-divergence and stable controls. Prefer all three perspectives rather than only A1/A2. Run four JLens source layers, provisionally `[22,26,30,38]` (or `[18,22,26,38]` for an early pre-rise comparison), and test validated concept-level interventions with controls.

**Critical research decision before reporting human "majority":** Is it the majority of **individual respondents** pooled within a culture, or the majority of **eligible countries** with each country one vote? The original proposal uses country-equal preference directions. The agent should implement **both** and clearly label results; you choose the primary analysis after reviewing the documented estimands.

**Evidence rule:** a dimension-level preference for young over elderly is **not necessarily** a directly observed STAY/SWERVE majority for an authored scenario with its exact counts, passengers/pedestrians and crossing legality. Prefer exact matched-response labels and explicitly mark missing or proxy-based references.

**Main work order:** `human raw preprocessing -> coverage catalog -> your 13 authored prompts/translations -> exact human references -> LLM behaviour and reference agreement -> selected groups -> JLens -> interventions -> figures and report`.
