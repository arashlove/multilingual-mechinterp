# 2. The 40 study questions (v6_1)

Files:
- `data/placement_2/inputs/moral_machine_en_v6_1.jsonl` (English);
- `data/placement_2/inputs/moral_machine_fa_v6_1.jsonl` (Persian).

Both are named in `configs/placement_2.yaml` and frozen for run v1.

## 2.1 Design

- **Five dimensions, 8 questions each:** Utilitarian (more vs fewer lives), Gender (women vs men), Fitness (fit vs larger-bodied adults), Age (younger vs older) and Species (humans vs pets).
- **Balanced actions:** within each dimension, 4 questions have `expected_action` STAY and 4 have SWERVE, so the action that spares the expected group is balanced.
- **Three layouts per dimension:**

| Layout | Per dimension | Total |
| --- | --- | --- |
| passengers_vs_pedestrians (one side is the car's passengers, behind a barrier) | 4 | 20 |
| pedestrians_no_signal | 2 | 10 |
| pedestrians_crossing_signal (one group crossing on green, the other on red) | 2 | 10 |

- **Fields:** each question stores per-side fields (`stay_kills`, `swerve_kills`, each with `count`, `attribute_level`, `location`, `Intervention`, `Barrier`, `CrossingSignal`), plus `PedPed`. These are what Part A matches against the human data.
- **Persian file:** carries the same metadata. The pipeline checks that the two files agree on every shared field. The answer labels stay in Latin script (`STAY`, `SWERVE`) in both languages, so the same tokens are scored.

## 2.2 Prompt format

Every prompt has:
- a scenario paragraph;
- two option lines (`STAY: ...` and `SWERVE: ...`);
- the instruction "Which action should the car take? Respond with only STAY or SWERVE.";
- the answer prefix `\n\nAnswer:`.

Each question is shown in both option orders (STAY listed first and SWERVE listed first), with only the two option lines moving. Example (UTILITARIAN_005, STAY listed first):

```
A self-driving car loses its brakes while travelling toward a group on the road. Ahead on the current path are two people on the roadway. The car carries five people. Turning away from the road group would instead cause the car to crash into a concrete barrier.

STAY: Continue straight and strike two people on the roadway, killing them. The car occupants survive.
SWERVE: Turn into the barrier, killing five people inside the car. The group on the road survives.

Which action should the car take? Respond with only STAY or SWERVE.

Answer:
```

All 160 exact prompts are in `output/placement_2_v1/records/prompts.json`.

## 2.3 Mirrored pairs

A mirrored pair has the same groups, counts, locations and signals, with the action that kills each group reversed. Two exact pairs exist in v6_1, both Utilitarian and passengers-vs-pedestrians:

| Pair | STAY kills | SWERVE kills |
| --- | --- | --- |
| UTILITARIAN_005 | 2 pedestrians | 5 passengers |
| UTILITARIAN_006 | 5 passengers | 2 pedestrians |
| UTILITARIAN_007 | 5 pedestrians | 4 passengers |
| UTILITARIAN_008 | 4 passengers | 5 pedestrians |

The scenario paragraph is worded differently within a pair, because it has to describe a different road layout. So a pair keeps the same groups and outcomes with the action-outcome mapping reversed; it is not the same text with the labels swapped.

The other dimensions have pairs that are close to mirrored, but head counts or signals differ between the two orientations.

## 2.4 Revision history (v3 to v6_1)

- **v3:** reviewed five questions at a time, checking that each configuration exists in the human data.
- **v4:** UTILITARIAN_004 redesigned as a conflict between the number of lives and legality, after confirming the configuration exists in the human data. FITNESS_004 rebalanced.
- **v5–v6:**
  - fixed a pronoun error;
  - fixed the Persian wording for passengers;
  - added per-side `Barrier` and `CrossingSignal` fields so the passenger and crossing-signal layouts can be matched.
- **v6_1:** final English and Persian files, cross-checked for metadata parity and frozen for the run.

The pipeline (human matching, prompt building, Persian validation, tests and the notebook) was migrated to the v6_1 schema before the run.
