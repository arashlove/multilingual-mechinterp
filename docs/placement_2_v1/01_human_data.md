# 1. Human data (Part A, stages 1–2)

## 1.1 From raw data to the balanced sample

The source is `SharedResponsesFullFirstSessions.csv` from the OSF release of Awad et al. (2018), *The Moral Machine experiment*. The full construction is in [`docs/placement_2/placement_2_raw_data_preprocessed.md`](../placement_2/placement_2_raw_data_preprocessed.md). In short:

- **The raw file:** 38,078,300 rows and 1,622,019 users, one session per user. One row is one side of one dilemma, and the two rows of a dilemma share a `ResponseID`.
- **Users in scope:** about 1.61M, after removing users seen under more than one country (5,108) or from a country outside the cluster map (5,511). By cluster: Western 1,208,960; Southern 304,586; Eastern 103,186.
- **Social Status and Random dilemmas are dropped.** Most Social Status rows were removed from the shared file, and Random dilemmas don't test a single dimension. Five types are kept: Utilitarian, Gender, Fitness, Age and Species, two dilemmas each per user.
- **Sampling:** 100,000 users drawn at random from each cluster (seed 42), each with all 10 kept dilemmas complete. Eastern is the binding constraint: nearly all in-scope Eastern users are in the sample.
- **Result:** `data_raw_output/balanced_cluster_responses_300k.csv`, with 300,000 users, 3,000,000 responses and 6,000,000 rows.

## 1.2 Stage 1: checks and descriptives

Outputs are in `output/placement_2/stage_1_human_sample/`.

**Structure checks.** All 11 checks pass:
- 300,000 users in both files;
- 20 rows per user;
- 2 rows per response, with exactly one `Saved = 1`;
- one STAY side and one SWERVE side;
- the same type on both rows;
- only the 5 kept types.

**Clusters and countries.** Users are balanced across clusters, but countries within a cluster are not:

| Cluster | Countries in sample | Largest country | Top 3 share | Countries with ≤100 sampled users |
| --- | --- | --- | --- | --- |
| Western | 67 | United States, 28.0% | 45.1% | 29 |
| Eastern | 28 | Japan, 42.6% | 57.2% | 1 |
| Southern | 35 | France, 33.2% | 63.6% | 8 |

Iran, the reference country closest to the Persian prompts, is in the Eastern cluster with about 600 sampled users (604 users, 6,040 responses).

**Responses per type.** Some typed dilemmas show random characters on both sides and have no clean contrast. Those are excluded from preference shares: Fitness 69,565 (11.6%), Age 32,311, Gender 12,496, Species 12,391, Utilitarian 0.

**Human preferences per cluster** (% of clean responses sparing the first level):

| Dimension | Western | Eastern | Southern | All clusters |
| --- | --- | --- | --- | --- |
| Utilitarian: more over fewer | 78.0 | 73.1 | 78.3 | 76.5 |
| Gender: male over female | 43.3 | 44.1 | 40.3 | 42.6 |
| Fitness: fit over larger-bodied | 59.1 | 57.1 | 60.0 | 58.7 |
| Age: young over old | 76.5 | 71.6 | 78.8 | 75.6 |
| Species: humans over pets | 80.3 | 81.5 | 78.8 | 80.2 |

The largest cluster differences are on Age (Eastern 4.9 points below Western and 7.2 below Southern) and Utilitarian (Eastern 4.9 below Western and 5.3 below Southern). This matches the published pattern: the Eastern cluster is less pro-young and less focused on numbers. With tens of thousands of responses per cell, every difference "excludes zero", so the size of the difference is what matters.

**STAY share.** Humans choose STAY in 53.0–54.4% of responses per cluster. This mixes a preference for not intervening with which side the preferred group happened to be on. By layout, the STAY share runs from 33.6–35.6% (when STAY kills a group crossing on red) to 70.2–73.3% (when STAY kills a group crossing legally on green).

**Data quality.** There are no duplicated rows or duplicated (ResponseID, Intervention) pairs. Missing cells sit in descriptive columns (`DefaultChoice`, `Template`, etc.), not in the columns used for matching.

## 1.3 Stage 2: human reference per question and cluster

Outputs:
- `data/placement_2/processed/moral_machine_40_human_references_by_cluster.csv`;
- tables in `output/placement_2/stage_2_human_references/`.

**Matching.** Each of the 40 English questions is matched to human responses with the same configuration on both sides:
- scenario type;
- attribute level and head count per side;
- `Barrier` (passengers vs pedestrians) per side;
- `CrossingSignal` (0 none, 1 green, 2 red) per side;
- `PedPed`.

The row with `Intervention = 0` is the side STAY kills. A person chose STAY when the `Intervention = 1` row has `Saved = 1`. Responses are pooled within each cluster, and "All clusters" pools the whole balanced sample.

**Support.** Every question has matches. The smallest matched count is 1,345 (Western), 1,392 (Eastern), 1,414 (Southern) and 4,151 (All clusters), all for AGE_006.

**Labels.** Labels come from a 95% Wilson interval on the STAY share with a near-tie band of ±5 points:

| Strength | Rule | Label |
| --- | --- | --- |
| clear | interval excludes 50% and the share is at least 5 points from 50% | STAY / SWERVE |
| weak | interval excludes 50% but the share is within 5 points of 50% | STAY / SWERVE |
| tie | interval includes 50% | UNDETERMINED |

| Cluster | clear | weak | tie |
| --- | --- | --- | --- |
| Western | 36 | 2 | 2 |
| Eastern | 36 | 2 | 2 |
| Southern | 37 | 1 | 2 |
| All clusters | 37 | 2 | 1 |

`expected_action` (the design intent of each question) equals the human majority on 86.8% (Western), 81.6% (Eastern), 92.1% (Southern) and 87.2% (All clusters) of labelled questions. It is a design field, never treated as a correct answer.

**Where clusters disagree.** 6 of the 40 questions have different labels across clusters: GENDER_002, GENDER_005, FITNESS_001, FITNESS_005, FITNESS_008 and AGE_006. All of them are near 50% in at least one cluster.

**Cultural divergence per question.** `D_culture` is the largest minus the smallest human p(SWERVE) across clusters. The top questions:

| Question | D_culture | Highest cluster | Lowest cluster |
| --- | --- | --- | --- |
| AGE_006 | 0.160 | Southern | Eastern |
| UTILITARIAN_006 | 0.152 | Southern | Eastern |
| UTILITARIAN_005 | 0.133 | Eastern | Southern |
| AGE_005 | 0.130 | Eastern | Southern |
| GENDER_002 | 0.121 | Southern | Eastern |
| FITNESS_005 | 0.109 | Southern | Eastern |

**Configuration catalog.** The sample contains 687 distinct configurations, 40 of which the questions use:

| Layout | Configurations | Used by questions | Smallest per-cluster n among those used |
| --- | --- | --- | --- |
| passengers_vs_pedestrians | 457 | 20 | 1,345 |
| pedestrians_crossing_signal | 153 | 10 | 1,751 |
| pedestrians_no_signal | 77 | 10 | 3,364 |

**Iran sensitivity.** Iran has 6–38 matched responses per question (median 17): 22 questions are ties and 18 clear. That is too sparse to serve as a per-question reference. At the type level (about 1,100–1,200 clean responses per type), Iran is close to the Eastern cluster:

| Dimension | Iran | Eastern |
| --- | --- | --- |
| Utilitarian: more over fewer | 75.3 | 73.1 |
| Gender: male over female | 47.0 | 44.1 |
| Fitness: fit over larger-bodied | 54.7 | 57.1 |
| Age: young over old | 75.0 | 71.6 |
| Species: humans over pets | 85.6 | 81.5 |

**Not completed in v1:**
- **Mirrored-pair averaging (2.3):** this table is empty because the question file has no `counterbalance_pair_id`. Two exact mirrored pairs do exist; see [02_question_set.md](02_question_set.md).
- **Repeated answers by one user (2.6):** the cell has no recorded output in the v1 notebook.
