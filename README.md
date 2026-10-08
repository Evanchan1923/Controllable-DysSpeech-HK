# Controllable Dysarthric Speech Synthesis with SAPC_full

This repository prepares SAPC_full, fine-tunes IndexTTS-1.5, and synthesizes
speech conditioned on one of five dysarthria etiologies. The main workflow is
configured for the UNSW Katana cluster.

The original TORGO workflow has been preserved separately in
[`scripts/torgo/README.md`](scripts/torgo/README.md).
See [`PIPELINE_COMPARISON.md`](PIPELINE_COMPARISON.md) for a side-by-side
comparison of the original TORGO, current SAPC, and planned severity-controlled
workflows.

## Requirements

Use Linux, Python 3.10, and an NVIDIA GPU. The Katana configuration expects:

- SAPC_full at
  `/srv/scratch/speechdata/speech-corpora/dysarthric/SAPC_HF/SAPC_full`;
- IndexTTS-1.5 files under
  `/srv/scratch/speechdata/Jinghao_shared/IndexTTS-Model`;
- output under `/srv/scratch/z5327748/dys-gen-runs/sapc_full_v1`.

The model directory must contain:

```text
bpe.model
dvae.pth
gpt.pth
bigvgan_generator.pth
```

Create the dedicated Katana virtual environment once:

```bash
module load python/3.10.8 ffmpeg/7.0.2 cuda/12.1.1
python -m venv /srv/scratch/z5327748/venv/controll-dys-gen
source /srv/scratch/z5327748/venv/controll-dys-gen/bin/activate
pip install -e .
pip install -r requirements-seed-vc.txt
```

All commands below are run from the repository root. The generic Katana PBS
and YAML launcher has been removed. The current runnable Katana experiment is
the Parkinson's low/high Seed-VC check in
[`configs/Park_v1.yaml`](configs/Park_v1.yaml) and [`Park_v1.pbs`](Park_v1.pbs).
The full severity-aware VC, preparation, and IndexTTS training workflow is
defined by [`configs/v1.yaml`](configs/v1.yaml) and [`v1.pbs`](v1.pbs).

## Etiology conditions

SAPC `Category` describes the prompt type. The synthesis condition comes from
`etiology`:

| ID | Etiology |
|---:|---|
| 0 | ALS |
| 1 | Cerebral Palsy |
| 2 | Down Syndrome |
| 3 | Parkinson's Disease |
| 4 | Stroke |

SAPC_full contains no healthy/control class. The etiology-only baseline uses
five condition prefixes. Full v1 crosses those etiologies with low, middle,
and high severity, producing 15 joint condition prefixes and a 15-way
auxiliary classifier. The original TORGO pipeline uses a different
six-condition scheme.

The current SAPC data also contains no healthy speakers. A healthy prompt may
still work through the pretrained IndexTTS zero-shot capability, but that
combination is not directly supervised by SAPC. For robust generation with a
healthy speaker's timbre and dysarthric articulation, add an external healthy
speech corpus as the Seed-VC timbre-reference pool. See
[`KATANA_QUESTIONS.md`](KATANA_QUESTIONS.md) for the required dataset details.

## Optional counterfactual audio with Seed-VC

The SAPC code can fine-tune IndexTTS directly from real recordings or add
offline Seed-VC counterfactual recordings. The focused `Park_v1` experiment
tests the latter on one reproducibly selected low/high Parkinson's pair. Check
its selection without decoding audio or loading Seed-VC:

```bash
qsub -v DRY_RUN=1 Park_v1.pbs
```

Generate both conversion directions with `qsub Park_v1.pbs`.

## Step 2: Prepare training features

This stage loads SAPC through its Hugging Face DatasetDict, decodes the `audio`
column only when extracting features, and preserves the existing `train` and
`dev` split. It writes mel features, DVAE codes, conditioning features,
manifests, and 15 initial joint etiology/severity embeddings for full v1.

Prepared data is written to:

```text
/srv/scratch/z5327748/dys-gen-runs/sapc_full_v1/prepared_data
```

## Step 3: Train

Training uses the 836 SAPC train speakers and validates on 126 separate dev
speakers. Checkpoints and logs are written beneath:

```text
/srv/scratch/z5327748/dys-gen-runs/sapc_full_v1/training
```

## Step 4: Run inference

Use the same prompt and text with different etiology IDs to retain the prompt
speaker's timbre while changing the requested dysarthric condition. Generated
audio is written beneath the selected experiment's output folder. A new
training/inference experiment config is required before the full SAPC model is
launched again.

## More detail

See [`KATANA_IMPLEMENTATION.md`](KATANA_IMPLEMENTATION.md) for dataset-loading,
cache, output, and PBS details. To inspect SAPC metadata without touching the
audio column, run:

```bash
python scripts/inspect_sapc_metadata.py --out sapc_dataset_report.txt
```

## License and disclaimer

The [license directory](license/) contains the English and Chinese model
licenses, the disclaimer, and a copy of the Seed-VC license. The original
Seed-VC license also remains with its third-party source code.
