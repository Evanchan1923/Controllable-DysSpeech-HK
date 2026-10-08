#!/usr/bin/env python3
"""Validate VC audio files and compare converted speaker embeddings."""
from __future__ import annotations

import argparse
import json
import math
import sys
import types
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch


ROOT = Path(__file__).resolve().parents[1]


def audio_stats(path: Path) -> dict[str, Any]:
    samples, sample_rate = sf.read(path, dtype="float32", always_2d=False)
    if samples.ndim > 1:
        samples = samples.mean(axis=1)
    if samples.size == 0 or not np.isfinite(samples).all():
        raise ValueError(f"Invalid audio: {path}")
    return {
        "path": str(path),
        "sample_rate": int(sample_rate),
        "duration_seconds": round(float(samples.size / sample_rate), 4),
        "rms": round(float(np.sqrt(np.mean(np.square(samples)))), 6),
        "peak": round(float(np.max(np.abs(samples))), 6),
        "clipped_sample_ratio": round(float(np.mean(np.abs(samples) >= 0.999)), 6),
    }


@torch.no_grad()
def speaker_embedding(path: Path, campplus_model: Any, device: torch.device) -> torch.Tensor:
    import librosa
    import torchaudio

    samples = librosa.load(str(path), sr=16000, mono=True)[0]
    waveform = torch.from_numpy(samples).float().to(device)
    features = torchaudio.compliance.kaldi.fbank(
        waveform.unsqueeze(0), num_mel_bins=80, dither=0, sample_frequency=16000
    )
    features = features - features.mean(dim=0, keepdim=True)
    embedding = campplus_model(features.unsqueeze(0)).float().squeeze(0)
    return torch.nn.functional.normalize(embedding, dim=-1)


def cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    return round(float(torch.dot(left.flatten(), right.flatten()).item()), 6)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", type=Path, required=True)
    args = parser.parse_args()
    experiment_dir = args.experiment_dir.expanduser().resolve()
    selection_path = experiment_dir / "selection.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    paths = {
        "low_original": Path(selection["low"]["audio"]),
        "high_original": Path(selection["high"]["audio"]),
        **{name: Path(path) for name, path in selection["outputs"].items()},
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing VC debug inputs: {missing}")
    if not torch.cuda.is_available():
        raise RuntimeError("VC speaker-similarity debug requires a GPU")

    seed_vc_dir = ROOT / "third_party/seed-vc"
    sys.path.insert(0, str(seed_vc_dir))
    import inference as seed_inference

    device = torch.device("cuda")
    seed_inference.device = device
    seed_inference.fp16 = True
    model_args = types.SimpleNamespace(f0_condition=False, checkpoint=None, config=None, fp16=True)
    _model, _semantic_fn, _, _vocoder_fn, campplus_model, _mel_fn, _ = (
        seed_inference.load_models(model_args)
    )
    embeddings = {
        name: speaker_embedding(path, campplus_model, device)
        for name, path in paths.items()
    }
    high_to_low = embeddings["high_articulation_low_timbre"]
    low_to_high = embeddings["low_articulation_high_timbre"]
    high_to_low_expected = cosine(high_to_low, embeddings["low_original"])
    high_to_low_source = cosine(high_to_low, embeddings["high_original"])
    low_to_high_expected = cosine(low_to_high, embeddings["high_original"])
    low_to_high_source = cosine(low_to_high, embeddings["low_original"])
    debug = {
        "audio": {name: audio_stats(path) for name, path in paths.items()},
        "speaker_similarity": {
            "high_articulation_low_timbre": {
                "to_expected_low_timbre": high_to_low_expected,
                "to_high_source_timbre": high_to_low_source,
                "closer_to_expected_reference": high_to_low_expected > high_to_low_source,
            },
            "low_articulation_high_timbre": {
                "to_expected_high_timbre": low_to_high_expected,
                "to_low_source_timbre": low_to_high_source,
                "closer_to_expected_reference": low_to_high_expected > low_to_high_source,
            },
        },
        "interpretation": (
            "Speaker similarity checks timbre transfer only. Rerun Parakeet on the "
            "converted audio to measure CER and the severity proxy."
        ),
    }
    if any(
        not math.isfinite(value)
        for item in debug["speaker_similarity"].values()
        for value in item.values()
        if isinstance(value, float)
    ):
        raise ValueError("Non-finite speaker similarity")
    debug_path = experiment_dir / "debug.json"
    temporary = debug_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(debug, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(debug_path)
    print(json.dumps(debug, indent=2, ensure_ascii=False), flush=True)
    print(f"VC_DEBUG_DONE: {debug_path}", flush=True)


if __name__ == "__main__":
    main()
