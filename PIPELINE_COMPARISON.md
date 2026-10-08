# Original TORGO and SAPC pipeline comparison

This document distinguishes the repository's original TORGO workflow from the
SAPC_full workflow being developed for Katana. It also separates the current
SAPC implementation from the planned severity-controlled model.

## High-level model flow

Both workflows can use the same two model families, but they serve different
purposes:

```text
Seed-VC (optional/offline)
    source articulation + target speaker timbre
                      |
                      v
       additional converted training WAVs
                      |
                      v
IndexTTS feature preparation and fine-tuning
                      |
                      v
  text + prompt audio + control label -> output speech
```

- **Seed-VC** creates counterfactual training audio before TTS training. Its
  weights are not jointly trained with IndexTTS.
- **IndexTTS-1.5** is the final TTS model that is fine-tuned and used for
  synthesis.
- TTS can be trained using only real recordings. Seed-VC is needed only when
  converted counterfactual examples are included.

## Main differences

| Area | Original TORGO | SAPC etiology baseline | Full SAPC v1 |
|---|---|---|---|
| Dataset | TORGO directory and transcripts | SAPC_full Hugging Face DatasetDict | Same SAPC_full data plus speaker severity CSV |
| Cluster workflow | Local shell commands | Component scripts | `v1.pbs` with severity-aware generation, preparation, and training |
| Primary control | Six synthesis conditions | Five etiologies | Five etiologies and three severity levels |
| Condition meanings | Healthy-like condition plus five selected TORGO patients | ALS, Cerebral Palsy, Down Syndrome, Parkinson's Disease, Stroke | Same etiologies crossed with low, middle, and high severity |
| Healthy class | Condition 0 represents healthy/control articulation | No healthy class in SAPC | Still no healthy class; low severity is not healthy speech |
| Auxiliary classifier | Binary healthy-like versus dysarthric classifier | Five-way etiology classifier | 15-way joint etiology/severity classifier |
| Severity supervision | No explicit severity label | Speaker severity CSV is prepared | V1 uses a joint etiology/severity condition and training label |
| Severity source | Not applicable | Baseline Parakeet statistics retained in CSV | Speaker mean CER: low `<10%`, middle `10-30%`, high `>=30%` |
| Voice conversion | Required by the documented original data construction | Optional `generate` stage; disabled in the default `prepare, train` run | Optional, with pairing extended to account for etiology and severity |
| Training audio | Real TORGO plus Seed-VC counterfactual WAVs | Real SAPC audio by default; optional converted WAVs | Real SAPC plus any selected severity-aware converted WAVs |
| Validation | A 2% split is created inside each prepared group | Existing SAPC `dev` split | Existing SAPC `dev` split with unseen speakers |
| Inference controls | Text, prompt WAV, TORGO condition ID | Text, prompt WAV, etiology ID | Text, prompt WAV, joint etiology/severity condition ID |

## Original TORGO workflow

The original pipeline uses Seed-VC to separate and recombine articulation and
speaker timbre. It creates counterfactual WAVs such as a dysarthric patient's
articulation in another speaker's voice. These real and converted recordings
are then used to fine-tune IndexTTS.

TORGO exposes six synthesis conditions, while its auxiliary classifier is
binary. The six conditions still remain separate during synthesis; the binary
loss only encourages the model to distinguish healthy-like and dysarthric
articulation.

At inference, IndexTTS receives:

```text
text + prompt WAV + TORGO condition ID
```

Seed-VC is not run during normal IndexTTS inference.

## Current SAPC workflow

SAPC replaces TORGO's patient-based conditions with five disease etiologies.
The default Katana stages are:

```text
prepare -> train
```

This default path fine-tunes IndexTTS directly from real SAPC recordings and
does not run Seed-VC. The optional full path is:

```text
generate with Seed-VC -> prepare -> train
```

The current implemented inference interface receives:

```text
text + prompt WAV + etiology ID
```

The earlier SAPC baseline controls etiology only. Full v1 uses a joint
etiology/severity condition ID.

## SAPC severity extension

The speaker-level table is:

```text
configs/utterance_research_v1_noFTModel/severity-speaker-labelling.csv
```

It contains one row per `(split, speaker)`, the SAPC etiology, WER/CER
statistics, and a three-level severity label. The final severity label is based
on speaker mean CER; WER is retained for analysis and auditing.

Full v1 connects severity to preparation and represents each pair as
`etiology_id * 3 + severity_id`. Each training sample contains:

```text
text
speaker/prompt condition
etiology ID
severity ID
target acoustic tokens and mel features
```

The v1 inference interface uses the corresponding joint condition ID:

```text
text + prompt WAV + joint etiology/severity condition ID
```

For example, a low-severity speaker's prompt can be combined with the same
etiology and the `high` severity label. The model will attempt to retain the
prompt voice while applying high-severity characteristics learned from other
speakers with that etiology. SAPC is cross-sectional, so this represents a
population-level transformation rather than observed progression of the same
patient.

## Implementation status

Available now in the code and v1 configuration:

- SAPC DatasetDict loading and official train/dev handling;
- five etiology IDs and three speaker severity IDs;
- 15 joint condition prefixes and a 15-way auxiliary classifier;
- same-etiology, cross-severity Seed-VC pairing;
- manifests containing joint, etiology, and severity IDs;
- speaker-level severity CSV with low, middle, and high labels;
- full `generate -> prepare -> train` orchestration in `v1.pbs`.

Still required after the first v1 run:

- inspect Park_v1 audio and its debug report before large-scale conversion;
- rerun Parakeet on converted outputs to verify that source severity survives VC;
- add balanced sampling if the 15 joint classes remain too imbalanced;
- consider factorized etiology and severity embeddings after the joint baseline;
- reduce severity leakage from the prompt representation if evaluation shows it.
