# Placement 2 data

| Folder | Owner | Contents |
|---|---|---|
| `raw/` | Researcher | The approved original Moral Machine release. Read-only and gitignored. |
| `input_templates/` | Code | Templates for the researcher. Never used as study input. |
| `inputs/` | Researcher | Approved study inputs (below). |
| `processed/` | Code (notebook Part A) | `moral_machine_40_human_references_by_cluster.csv`, the human reference per question and cluster. |

Study inputs in `inputs/`:

| File | Status | Used by |
|---|---|---|
| `moral_machine_en_v6_1.jsonl` | provided, frozen; prompt in `question_en`, per-side configuration in `stay_kills` / `swerve_kills` | all stages |
| `moral_machine_fa_v6_1.jsonl` | provided, frozen; prompt in `question_fa`, same metadata as the English file, `STAY:` / `SWERVE:` labels in Latin script | notebook stages 3–8 |
| `moral_machine_40_*`, `moral_machine_*_v6.jsonl` | earlier drafts, not read by the pipeline | — |
| `concepts.json` | draft — a Persian speaker checks the `fa` word lists, then set `"status": "approved"` | notebook stages 7–8 |
| `jlens_calibration.jsonl` | optional — one `{"prompt": ...}` per line; without it the lens is fitted on the Placement 1 culture questions | notebook stage 7 |

Human reference labels are computed from the sample; they are never written by hand.
Nothing in this folder may be fabricated or substituted with a different dataset without approval.
Paths are configured in `configs/placement_2.yaml`.
