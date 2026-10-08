# Katana: remaining questions and actions

The answered paths and environment choices are implemented in `configs/katana.yaml`, `katana.pbs`, and `KATANA_IMPLEMENTATION.md`. I will keep only unresolved items here.

## 1. SAPC pathology mapping (needed before preparation/training)

The model expects six condition IDs: `0` for healthy speech and `1`–`5` for five pathology conditions. `SAPC_full` has `speaker`, `etiology`, and `Category`, but the column schema alone does not tell us which values correspond to those six IDs.

Please tell me the intended mapping from actual SAPC `speaker` or `Category` values to IDs `0`–`5`. If the six TORGO patient IDs are still intended, say whether all other speakers should be mapped to `0` or excluded. I will then fill `dataset.pathology_labels` in `configs/katana.yaml` and adjust filtering if needed. The code currently refuses unmapped rows, so it cannot silently mislabel them.

To write a report on Katana without reading or decoding audio:

```bash
source /srv/scratch/z5327748/venv/controll-dys-gen/bin/activate
python scripts/inspect_sapc_metadata.py --out sapc_dataset_report.txt
```

Please share `sapc_dataset_report.txt`. It contains `train`/`dev` counts by `speaker`, `Category`, and `etiology`, plus their combinations.

## 2. Seed-VC pairing policy (optional correction)

The current SAPC generator matches `norm_text_without_disfluency_team` across different pathology classes, chooses at most one donor class for each target utterance, and converts only `train` audio. Is that the pairing policy you want? If you want one conversion for **every** available donor class as in the TORGO recipe, I can change `generate.max_source_labels_per_target` from `1` to `5`.

No inference prompt or text is needed yet, as requested.
