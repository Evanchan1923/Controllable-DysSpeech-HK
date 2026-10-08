# Katana SAPC_full implementation notes

Both Katana jobs use `scripts/run_pipeline.py` as the only pipeline entry
point. The YAML `pipeline.type` selects `vc_validation` or `full_training`.
Stage implementations remain reusable (`generate_sapc.py`, `prepare_sapc.py`,
and `train.py`), while `debug_vc.py` is the separate debug utility.

The SAPC code reads the Hugging Face DatasetDict at
`/srv/scratch/speechdata/speech-corpora/dysarthric/SAPC_HF/SAPC_full`, using
`train` for training and `dev` for validation. The audio reader follows
SAPC-Qwen: `Audio(decode=False)` followed by `soundfile` decoding of bytes or
path (or direct use of an array). IndexTTS features are resampled to 24 kHz.

The chosen run directory is
`/srv/scratch/z5327748/dys-gen-runs/sapc_full_v1`. It contains
`converted_audio/`, `prepared_data/`, `training/`, and `outputs/`.

## SAPC pathology conditions

The metadata report shows that `Category` describes prompt types, while
`etiology` contains the five disease groups used as controllable conditions.
SAPC_full contains no healthy/control class. The configured IDs are:

| ID | Etiology |
|---:|---|
| 0 | ALS |
| 1 | Cerebral Palsy |
| 2 | Down Syndrome |
| 3 | Parkinson's Disease |
| 4 | Stroke |

No sixth healthy label is required when healthy speech is used only as the
target timbre: converted samples retain the dysarthric source's etiology ID.
However, SAPC_full also has no healthy speakers, so an external healthy corpus
is required to supervise “healthy timbre + dysarthric articulation” directly.
Without it, healthy prompts depend on the base IndexTTS model's zero-shot
generalization.

Both train and dev contain every etiology. Each speaker belongs to exactly one
etiology, and the 836 train speakers and 126 dev speakers do not overlap. The
preparation code therefore preserves dev-only speakers and creates a five-way
etiology classifier in the etiology-only baseline. Full v1 instead creates 15
joint etiology/severity conditions. To regenerate the metadata-only report without reading
audio:

```bash
source /srv/scratch/z5327748/venv/controll-dys-gen/bin/activate
python scripts/inspect_sapc_metadata.py --out sapc_dataset_report.txt
```

## Speaker severity labels

The baseline Parakeet utterance results are under
`configs/utterance_research_v1_noFTModel/`. The speaker table aggregates all
utterances for each `(split, speaker)` and joins the speaker's SAPC etiology:

`configs/utterance_research_v1_noFTModel/severity-speaker-labelling.csv`

Severity is based on speaker mean CER: `low < 0.10`,
`0.10 <= middle < 0.30`, and `high >= 0.30`. The CSV also retains mean and
median WER/CER, utterance counts, and the fraction of utterances in each CER
band. Rebuild it without loading or decoding audio using:

```bash
python scripts/build_sapc_speaker_severity.py
```

On Katana the script can read etiology directly from SAPC instead of the local
metadata report:

```bash
python scripts/build_sapc_speaker_severity.py \
  --dataset /srv/scratch/speechdata/speech-corpora/dysarthric/SAPC_HF/SAPC_full
```

The table can be joined on `(split, speaker)` and added to training manifests
without recalculating ASR metrics.

## Parkinson's low/high VC check

`Park_v1.pbs` and `configs/Park_v1.yaml` define a small Seed-VC experiment.
It finds every train-split Parkinson's low/high speaker pair that shares at
least one prompt, then uses seed `1234` to select one speaker pair, one shared
prompt, and one utterance from each speaker. A dry run reads metadata only:

```bash
qsub -v DRY_RUN=1 Park_v1.pbs
```

Run the two conversions with:

```bash
qsub Park_v1.pbs
```

The output directory is `/srv/scratch/z5327748/dys-gen-runs/Park_v1/` and
contains:

```text
selection.json
low_original.wav
high_original.wav
high_articulation_low_timbre.wav
low_articulation_high_timbre.wav
debug.json
```

The first conversion uses the high recording as the Seed-VC source and the low
recording as its timbre reference. The reverse conversion provides a symmetric
comparison. `selection.json` records speaker IDs, utterance IDs, transcripts,
speaker CER/WER, conversion directions, and expected output attributes.
`debug.json` checks audio readability, duration, RMS, peak/clipping, and CAMPPlus
speaker similarity against the expected timbre reference. It does not assign a
new severity label; output CER still needs to be measured with Parakeet.

The text for training is `norm_text_with_disfluency_team`, matching Qwen's
training config. Seed-VC pairs use `norm_text_without_disfluency_team` for
same-text matching and use the source's disfluency text as the generated
utterance transcript. Generation uses only `train` rows and selects at most
one different etiology per target utterance by default, keeping the already
large generation set bounded. Set `generate.max_source_labels_per_target` to
`4` to generate every other etiology for each target. The `dev` split is never
converted.

## Model files

Your Katana `ls` output confirms all four required files are now under
`/srv/scratch/speechdata/Jinghao_shared/IndexTTS-Model/`:
`bpe.model`, `dvae.pth`, `gpt.pth`, and `bigvgan_generator.pth`.

Seed-VC, Whisper-small, CAMPPlus, and BigVGAN use the shared cache under
`/srv/scratch/speechdata/Jinghao_shared/controll-dys-gen/`.

## Environment and jobs

On Katana, create the requested dedicated environment using Python 3.10:

```bash
module load python/3.10.8 ffmpeg/7.0.2 cuda/12.1.1
python -m venv /srv/scratch/z5327748/venv/controll-dys-gen
source /srv/scratch/z5327748/venv/controll-dys-gen/bin/activate
pip install -e .
pip install -r requirements-seed-vc.txt
```

The generic Katana PBS and YAML launcher were removed. `Park_v1.pbs` is the
focused VC validation task. `v1.pbs` and `configs/v1.yaml` define the full
training workflow.

## Full v1 training workflow

The full v1 condition is a joint `etiology x severity` class:

```text
condition_id = etiology_id * 3 + severity_id
```

This produces 15 condition prefixes. Low, middle, and high use IDs 0, 1, and 2
inside each etiology. For example, Parkinson's high is `3 * 3 + 2 = 11`.
Seed-VC pairing stays within one etiology and crosses severity levels. The
converted waveform keeps the target speaker's timbre and receives the source
speaker's joint condition. Preparation combines real and converted audio,
writes the joint label plus the separate etiology/severity IDs into manifests,
and initializes all 15 condition embeddings. IndexTTS then fine-tunes LoRA,
the Perceiver, condition prefixes, and condition classifiers.

Because conversion and feature extraction cover the full dataset, submit the
stages separately so each receives its own walltime:

```bash
qsub -v STAGE=generate v1.pbs
qsub -v STAGE=prepare v1.pbs
qsub -v STAGE=train v1.pbs
```

The YAML also lists all three stages in order, so `qsub v1.pbs` represents the
complete workflow in one job when its runtime fits the queue limit. Run a
path/config check without executing a stage using:

```bash
qsub -v CHECK_ONLY=1 v1.pbs
```
