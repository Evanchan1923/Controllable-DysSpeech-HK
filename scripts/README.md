# Script layout

## Entry point

`run_pipeline.py` is the only SAPC pipeline runner. It reads `pipeline.type`
from YAML and supports both configurations:

- `vc_validation`: generate one reproducible VC pair and run VC debug;
- `full_training`: generate augmentation, prepare features, train, and
  optionally run inference.

The PBS files only prepare the Katana environment and call this runner.

## Pipeline stages

- `generate_sapc.py`: full SAPC conversion or one validation pair;
- `prepare_sapc.py`: IndexTTS feature and manifest preparation;
- `cache_seedvc_models.py`: shared Seed-VC model cache;
- `debug_vc.py`: audio integrity and speaker-similarity debug;
- `feature_extractor.py`, `sapc_audio.py`: shared implementation modules.

## Metadata utilities

- `inspect_sapc_metadata.py`: inspect SAPC without decoding audio;
- `build_sapc_speaker_severity.py`: aggregate utterance CER/WER by speaker.

The original TORGO scripts and their runner remain isolated under `torgo/`.
