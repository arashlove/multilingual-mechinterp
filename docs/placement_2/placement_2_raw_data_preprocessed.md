# Placement 2: raw data → preprocessed balanced responses

How the balanced human response set (100,000 users from each of the three moral clusters) was built from the raw Moral Machine data, and what each raw column means.

| | |
|---|---|
| Notebook | [`notebooks/balanced_cluster_sample.ipynb`](../notebooks/balanced_cluster_sample.ipynb) |
| Raw input | `3hvt2-osfstorage-archive/Datasets/Moral Machine Data/SharedResponsesFullFirstSessions.csv` |
| Cluster labels | `3hvt2-osfstorage-archive/Datasets/Moral Machine Data/country_cluster_map.csv` |
| Output folder | `data_raw_output/` |
| Related docs | [`scenarios_and_dilemmas.md`](scenarios_and_dilemmas.md) · [`prompt_coverage.md`](prompt_coverage.md) · [`csv/SharedResponses.md`](csv/SharedResponses.md) · [`csv/country_cluster_map.md`](csv/country_cluster_map.md) |

---

## 1. The raw data

`SharedResponsesFullFirstSessions.csv` is part of the OSF release of Awad et al. (2018), *The Moral Machine experiment*, *Nature*.

| Fact | Value |
|---|---|
| Size | ~4.8 GB |
| Rows | 38,078,300 |
| Users | 1,622,019 |
| Sessions per user | 1 (each user's first full session; no user has more than one `ExtendedSessionID`) |

Key properties:

- **One row = one side of one question.** Every question (dilemma) shows the car with two possible outcomes. Each outcome is its own row, and both rows share a `ResponseID`.
- **A full session = 13 questions = 26 rows.** 2 questions for each of the 6 types (Utilitarian, Gender, Fitness, Age, Social Status, Species) plus 1 Random.
- **Rows are shuffled.** One user's rows are scattered across the whole file, so every step streams the full file in 1M-row chunks.
- **Rows were removed by the authors.** Only 50,003 users still have all 26 rows. The removals are almost entirely **Social Status** rows: about 97% of users have at least one Social Status question half-missing or gone, while the other 5 types are almost always intact.

### Example: one question = two rows

| ResponseID | ScenarioTypeStrict | AttributeLevel | NumberOfCharacters | Intervention | Saved |
|---|---|---|---|---|---|
| X | Gender | Male | 4 | 0 | 0 |
| X | Gender | Female | 4 | 1 | 1 |

The user spared the women's side (`Saved = 1`). Swerving would have killed the women (`Intervention = 1`), so the user chose to **stay**, killing the men. The type (Gender) is the same on both rows; `AttributeLevel` says which side each row is.

### The 21 raw columns

**Identifiers**

| Column | Meaning | Example |
|---|---|---|
| `ResponseID` | ID of one question; shared by its 2 rows | `2223CNmvTr2Coj4wp` |
| `ExtendedSessionID` | Session ID, `<session>_<user>` | `-1613944085_422160228641876.0` |
| `UserID` | User ID | `422160228641876` |
| `UserCountry3` | User's country (ISO3, from IP address) | `ISR` |

**The question (same on both rows)**

| Column | Meaning | Values |
|---|---|---|
| `ScenarioOrder` | Position in the session. Randomised per user, so it carries no meaning across users | 1–13 (24 rows have 14, from 3 users) |
| `ScenarioTypeStrict` | Dimension the question tests (strict classification) | Utilitarian, Gender, Fitness, Age, Social Status, Species, Random |
| `ScenarioType` | Looser version of the same label; used as a fallback when Strict is blank | same values |
| `DefaultChoice` / `NonDefaultChoice` | The two levels being compared | More/Less, Male/Female, Fit/Fat, Young/Old, High/Low, Hoomans/Pets (blank for Random) |
| `Template` | Device | Desktop, Mobile |
| `DescriptionShown` | Whether the user opened the text description | 0/1 |

**This side of the question (differs between the 2 rows)**

| Column | Meaning | Values |
|---|---|---|
| `AttributeLevel` | Which level this side has | e.g. `Female`, `Old`, `Pets`, `Rand` |
| `NumberOfCharacters` | Characters on this side | 1–5 |
| `DiffNumberOFCharacters` | Difference in head count between the two sides | 0–4 |
| `Intervention` | 1 = the characters on this row die if the car **swerves**; 0 = they die if the car **stays**. Human chose SWERVE ⇔ the `Intervention = 0` row has `Saved = 1` | 0/1 |
| `Barrier` | 1 = this side is the car's **passengers** (car hits a barrier) | 0/1 |
| `PedPed` | 1 = **both** sides are pedestrians; 0 = passengers vs pedestrians | 0/1 |
| `CrossingSignal` | Traffic light for these pedestrians | 0 none, 1 green (legal), 2 red (illegal) |
| `DefaultChoiceIsOmission` | 1 = the default level's side is the "stay in lane" side | 0/1 |
| `LeftHand` | 1 = this side was shown on the left | 0/1 |
| **`Saved`** | **The user's answer: 1 = this side was spared** | 0/1 |

The 9 preference dimensions of the paper are **not** columns. The character dimensions come from filtering by `ScenarioTypeStrict`; the structural ones (Intervention, Relation to vehicle, Law) come from `Intervention`, `Barrier`/`PedPed` and `CrossingSignal`. See [`scenarios_and_dilemmas.md`](scenarios_and_dilemmas.md).

### Cluster map

`country_cluster_map.csv` assigns 130 countries (ISO3) to the paper's three moral clusters: `0 = Western`, `1 = Eastern`, `2 = Southern`.

---

## 2. Design decisions

| Decision | Reason |
|---|---|
| **Drop Random** questions | They don't test a single dimension, so they can't serve as labelled items |
| **Drop Social Status** questions | Most Social Status rows were removed from the shared file; requiring them would leave only ~3k Eastern users |
| **Keep 5 types:** Utilitarian, Gender, Fitness, Age, Species | Almost always intact; 2 questions each = **10 questions / 20 rows per user** |
| **Only complete questions** (both rows, one `Saved = 1`) | With one row missing, the user's choice can't be read |
| **Simple random sampling within cluster** | Each country keeps its natural share of its cluster |
| **IDs read as text** | As floats, long IDs lose precision (e.g. `UserID` `…909470` in the raw file was already rounded from `…909467`) |

---

## 3. Pipeline

| Step | Notebook section | Reads | Does |
|---|---|---|---|
| 1 | Setup | — | Paths; `KEEP_TYPES`; 10 questions / 20 rows per user; 100,000 per cluster; seed 42; IDs as strings |
| 2 | Cluster map | `country_cluster_map.csv` | ISO3 → cluster lookup for 130 countries |
| 3 | Pass 1: user profiles | `UserID`, `UserCountry3` | Row count and country per user; flags users seen under >1 country. Blank country → empty string |
| 4 | Users in scope | — | Removes users with >1 country or a country outside the map |
| 5 | Pass 2: completeness grid (5b) | `UserID`, `ScenarioOrder`, `ScenarioTypeStrict`, `ScenarioType` | Users × 13 positions grid: rows present (2 / 1 / 0) and type per position; counts complete questions of the kept types |
| 6 | Threshold diagnostic | — | Distribution, pass/below per cluster, `users_below_threshold.csv`, `show_user()` |
| 7 | Eligibility (5c) | — | In scope **and** all 10 kept-type questions complete |
| 8 | Sampling | — | 100,000 random users per cluster (seed 42) → `balanced_cluster_users_300k.csv` |
| 9 | Pass 3: extract | all 21 columns | Rows of sampled users whose type (`ScenarioTypeStrict`, falling back to `ScenarioType`) is a kept type; adds `Cluster`, `ClusterName` → `balanced_cluster_responses_300k.csv` |
| 10 | Validation | output file | Rows/questions per user, one `Saved = 1` per question, no excluded types, types per user |

### Funnel

| Stage | Users |
|---|---|
| All users in raw file | 1,622,019 |
| Excluded: seen under >1 country | 5,108 |
| Excluded: country not in cluster map | 5,511 |
| In scope (Western ~1.21M, Southern ~304k, Eastern ~103k) | ~1.61M |
| Excluded: a kept-type question incomplete | 730 |
| **Sampled** | **300,000** (100,000 per cluster) |

Eastern is the binding constraint: about 103k in scope, so nearly all Eastern users are in the sample.

---

## 4. Output and validation results

Files in `data_raw_output/`:

| File | Contents |
|---|---|
| `balanced_cluster_responses_300k.csv` | 21 raw columns + `Cluster`, `ClusterName`; 20 rows per user |
| `balanced_cluster_users_300k.csv` | One row per sampled user: `UserID`, `UserCountry3`, `Cluster`, `ClusterName` |
| `users_below_threshold.csv` | Diagnostic: in-scope users missing a kept-type question, with the missing positions |

Validation of the run:

| Check | Result |
|---|---|
| Users | 300,000 (Western 100,000 · Southern 100,000 · Eastern 100,000) |
| Rows | 6,000,000 |
| Rows per user | 20 for every user |
| Questions (`ResponseID`s) | 3,000,000 |
| Questions with 2 rows and exactly one `Saved = 1` | 3,000,000 (all) |
| Questions per type | 600,000 each of Utilitarian, Gender, Fitness, Age, Species (2 per user) |
| Social Status / Random rows | 0 |
| Blank `ScenarioTypeStrict` | 0 |

---

## 5. Caveats for the ground-truth step

- **Typed questions with random characters.** 126,763 questions (4.2%) have `AttributeLevel = Rand` on **both** sides even though they are typed (Fitness 69,565; Age 32,311; Gender 12,496; Species 12,391; Utilitarian 0). They have no clean contrast on their dimension. When computing a dimension's preference, keep only questions whose two sides match `DefaultChoice` / `NonDefaultChoice`.
- **A few mixed Species rows.** About 400 Species rows carry non-species levels (e.g. `Female`, `Fat`, `High`). Filter them the same way.
- **Social Status is not covered.** This sample supports 5 character dimensions plus the 3 structural ones (Intervention, Relation to vehicle, Law).
- **Users are balanced, countries are not.** Within each cluster, large countries dominate (roughly 28% US in Western, 33% France in Southern, 43% Japan in Eastern).
- **`ScenarioOrder` is presentation order only.** Never group preferences by it; positions of the dropped Social Status and Random questions are simply absent.

## Next step

Compute human ground truth from `balanced_cluster_responses_300k.csv`: spare rate, majority side and AMCE per dimension, overall and per cluster, then build the parallel English/Persian prompt bank on the same question IDs.
