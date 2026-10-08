# Katana SAPC_full pipeline

The Katana config is `configs/katana.yaml`. It reads the Hugging Face DatasetDict at
`/srv/scratch/speechdata/speech-corpora/dysarthric/SAPC_HF/SAPC_full`, using
`train` for training and `dev` for validation. The audio reader follows
SAPC-Qwen: `Audio(decode=False)` followed by `soundfile` decoding of bytes or
path (or direct use of an array). IndexTTS features are resampled to 24 kHz.

The chosen run directory is
`/srv/scratch/z5327748/dys-gen-runs/run_sapc_full_v1`. It contains
`converted_audio/`, `prepared_data/`, `training/`, and `outputs/`.

## One remaining data decision

The SAPC schema has `speaker`, `etiology`, and `Category`, but it does not
identify the six pathology conditions used by this model. Fill
`dataset.pathology_labels.by_speaker` or `by_category` in `configs/katana.yaml`
with integer values 0–5. The loader requires every included row to be mapped,
and preparation checks that all six classes have training samples. It will
raise a clear error rather than silently assign classes. To inspect actual
values on Katana, run this metadata-only report:

```bash
source /srv/scratch/z5327748/venv/controll-dys-gen/bin/activate
python scripts/inspect_sapc_metadata.py --out sapc_dataset_report.txt
```

The text for training is `norm_text_with_disfluency_team`, matching Qwen's
training config. Seed-VC pairs use `norm_text_without_disfluency_team` for
same-text matching and use the source's disfluency text as the generated
utterance transcript. Generation uses only `train` rows and selects at most
one different pathology class per target utterance by default. The `dev`
split is never converted.

## Model files

Your Katana `ls` output confirms all four required files are now under
`/srv/scratch/speechdata/Jinghao_shared/IndexTTS-Model/`:
`bpe.model`, `dvae.pth`, `gpt.pth`, and `bigvgan_generator.pth`.

Seed-VC, Whisper-small, CAMPPlus, and BigVGAN can be cached in
`/srv/scratch/speechdata/Jinghao_shared/controll-dys-gen/` by submitting
`qsub -v STAGE=cache_models katana.pbs`. Both the preloader and Seed-VC
inference use these shared cache locations.

## Environment and jobs

On Katana, create the requested dedicated environment using Python 3.10:

```bash
module load python/3.10.8 ffmpeg/7.0.2 cuda/12.1.1
python -m venv /srv/scratch/z5327748/venv/controll-dys-gen
source /srv/scratch/z5327748/venv/controll-dys-gen/bin/activate
pip install -e .
pip install -r requirements-seed-vc.txt
```

After the pathology mapping is filled in, check paths
with `python scripts/run_katana.py --check`, then run `qsub katana.pbs`.
The default stages are `prepare` and `train` on original SAPC audio. If
Seed-VC augmentation is wanted, set `run.stages` to
`[generate, prepare, train]`; for long generation jobs, submit stages
separately with `qsub -v STAGE=generate katana.pbs` and then
`qsub -v STAGE=prepare katana.pbs`, followed by training.

The PBS file requests one GPU, four CPUs, 64 GB RAM, and 48 hours per job.
These resource choices are a starting point because no full Katana run has
yet been measured. `generate_sapc.py --plan` reports the exact number of
same-text pairs before loading Seed-VC; conversion is resumable and supports
`--num-shards` and `--shard`.

Inference stays optional until a prompt WAV and text are chosen.
