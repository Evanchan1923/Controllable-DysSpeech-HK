# Original TORGO pipeline

This directory preserves the original TORGO data path. It is separate from the
SAPC_full Katana pipeline.

## High-level flow

1. `generate_data.py` scans the 15 TORGO speakers and matches recordings with
   the same prompt text.
2. Seed-VC creates two counterfactual sets:
   - `X_converted_from_Y`: speech content and pathology from one of five modeled
     patients (`Y`), rendered with the timbre of another speaker (`X`).
   - `X_converted`: healthy/control content rendered with a modeled patient's
     timbre. These examples help separate articulation from speaker identity.
3. `prepare_torgo.py` combines 15 real-speaker directories, 5 `X_converted`
   directories, and 50 `X_converted_from_Y` directories. It extracts mel
   features, DVAE codes, speaker conditioning, manifests, and initial condition
   embeddings.
4. `train.py` fine-tunes IndexTTS with LoRA and learns a pathology prefix for
   each condition. At inference, the prompt supplies timbre and the chosen
   prefix supplies the articulation condition.

## Six conditions versus binary classification

TORGO has **six synthesis conditions**:

| Condition ID | Meaning |
|---:|---|
| 0 | Healthy/control articulation, including controls, F03, F04, M03, and `X_converted` |
| 1 | F01 articulation |
| 2 | M01 articulation |
| 3 | M02 articulation |
| 4 | M04 articulation |
| 5 | M05 articulation |

The model therefore learns six separate `mean_pathology_condition_*` prefixes.
The older TORGO training setup also uses a **binary auxiliary classifier**:
condition `0` becomes label `0`, while conditions `1`–`5` become label `1`.
This auxiliary loss distinguishes healthy versus modeled dysarthric
articulation. It does not reduce synthesis conditioning from six choices to
two choices.

The classifier applied to the combined conditioning encourages pathology
information to be present. The classifier applied to the speaker embedding is
behind a gradient reversal layer, encouraging the timbre representation to
exclude pathology information. Together with the counterfactual audio, this is
the mechanism used to separate timbre and articulation.

SAPC_full differs: it has no healthy class and has five disease etiologies, so
its Katana pipeline uses five condition prefixes and a five-way auxiliary
classifier.

## Commands

Run all original TORGO stages with environment variables:

```bash
TORGO_ROOT=/path/to/TORGO \
MODEL_DIR=/path/to/IndexTTS-Model \
bash scripts/torgo/run_pipeline.sh
```

The individual generation and preparation commands are documented in the root
`README.md`. Both scripts skip completed artifacts and support partial checks.
