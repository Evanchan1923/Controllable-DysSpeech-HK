# Original TORGO Controllable Dysarthric Speech Pipeline

This document preserves the repository's original TORGO data generation,
training, and inference workflow.

**Project page:** [Audio samples and system overview](https://mors20.github.io/Controllable-Dysarthric-Speech-Synthesis/)

The active SAPC_full/Katana workflow is documented in the repository root
[`README.md`](../../README.md).

## Requirements

Use Linux, Python 3.10, and an NVIDIA GPU. Prepare:

- the original TORGO dataset;
- the four IndexTTS-1.5 files shown below;
- enough disk space for generated audio and prepared features.

```text
artifacts/pretrained/
 bpe.model
 dvae.pth
 gpt.pth
 bigvgan_generator.pth
```

The original environment setup used conda:

```bash
conda create -n tts python=3.10 -y
conda activate tts
pip install -e .
pip install -r requirements-seed-vc.txt
```

All commands below are run from the repository root unless stated otherwise.

## Six conditions versus binary classification

TORGO has six synthesis conditions:

| Condition ID | Meaning |
|---:|---|
| 0 | Healthy/control articulation, including controls, F03, F04, M03, and `X_converted` |
| 1 | F01 articulation |
| 2 | M01 articulation |
| 3 | M02 articulation |
| 4 | M04 articulation |
| 5 | M05 articulation |

The model learns six separate `mean_pathology_condition_*` prefixes. The TORGO
training setup also uses a binary auxiliary classifier: condition `0` becomes
binary label `0`, while conditions `1`–`5` become binary label `1`. This loss
distinguishes healthy-like and modeled dysarthric articulation; it does not
reduce synthesis conditioning from six choices to two.

The classifier on combined conditioning encourages pathology information to be
present. A second classifier is applied to the speaker embedding through a
gradient reversal layer, encouraging the timbre representation to exclude
pathology information.

## Step 1: Generate counterfactual audio with Seed-VC

Input: the original TORGO folder. Output: voice-converted WAV files.

```bash
cd third_party/seed-vc
python ../../scripts/torgo/generate_data.py \
  --mode both \
  --torgo_root /path/to/TORGO \
  --output /path/to/converted_audio
cd ../..
```

Seed-VC creates two counterfactual sets:

- `X_converted_from_Y`: patient `Y` supplies content/pathology and speaker `X`
  supplies timbre. Ten timbre targets crossed with five modeled patients
  produce 50 directory groups.
- `X_converted`: a control supplies healthy content while one of the five
  modeled patients supplies timbre. These examples receive condition `0`.

Pairs must share the same prompt text. The script skips completed output files,
so it can resume. Add `--dry-run` to check pairs without loading models.

## Step 2: Prepare training features

Input: original TORGO audio, Step 1 output, and IndexTTS-1.5 files. Output: mel
features, codec tokens, conditioning features, manifests, and pathology
embeddings.

```bash
python scripts/torgo/prepare_torgo.py \
  --torgo_root /path/to/TORGO \
  --converted_root /path/to/converted_audio \
  --out_dir /path/to/prepared_data \
  --finetune_dir artifacts/pretrained \
  --config configs/controllable_dysarthric_speech_synthesis.yaml
```

The prepared set combines 15 real TORGO speakers, 5 `X_converted` groups, and
50 `X_converted_from_Y` groups. A 2% validation split is created per group. The
script skips completed features and can resume.

## Step 3: Train

```bash
python train.py \
  --config configs/controllable_dysarthric_speech_synthesis.yaml \
  --model-dir artifacts/pretrained \
  --data-dir /path/to/prepared_data \
  --embedding-dir /path/to/prepared_data/pathology_embedding \
  --epochs 20 \
  --batch-size 2 \
  --num-workers 4
```

For a quick training check, append:

```text
--epochs 1 --max-train-batches 1 --skip-validation --no-save
```

The convenience wrapper runs generation, preparation, and training:

```bash
TORGO_ROOT=/path/to/TORGO \
MODEL_DIR=/path/to/IndexTTS-Model \
bash scripts/torgo/run_pipeline.sh
```

## Step 4: Run inference

Choose a prompt WAV for the target voice and a TORGO condition ID:

```bash
python -m indextts.inference \
  --cfg configs/controllable_dysarthric_speech_synthesis.yaml \
  --model-dir artifacts/pretrained \
  --gpt-ckpt /path/to/gpt_best.pth \
  --prompt /path/to/prompt.wav \
  --text "Please call Stella." \
  --pathology 4 \
  --out outputs/example.wav
```

Use the same prompt with different condition IDs to change articulation while
retaining the prompt speaker's timbre.

## License and disclaimer

See the repository [license directory](../../license/) for the model licenses,
disclaimer, and Seed-VC license copy.
