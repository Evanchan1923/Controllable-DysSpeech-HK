#!/usr/bin/env python3
"""Run a two-speaker Parkinson's low/high Seed-VC validation experiment."""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import types
from collections import defaultdict
from pathlib import Path
from typing import Any

import soundfile as sf
import torch
import yaml

from sapc_audio import decode_audio, load_split


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/Park_v1.yaml")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Select and report the pair without decoding audio or loading Seed-VC.",
    )
    return parser.parse_args()


def resolve_repo_path(value: str) -> Path:
    path = Path(value).expanduser()
    return (ROOT / path).resolve() if not path.is_absolute() else path.resolve()


def read_speakers(
    csv_path: Path,
    *,
    split: str,
    etiology: str,
    low_label: str,
    high_label: str,
) -> tuple[dict[str, dict[str, str]], dict[str, dict[str, str]]]:
    with csv_path.open(encoding="utf-8", newline="") as input_file:
        reader = csv.DictReader(input_file)
        required = {
            "speaker",
            "split",
            "etiology",
            "speaker_avg_wer",
            "speaker_avg_cer",
            "speaker_severity",
        }
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise KeyError(f"{csv_path} is missing columns: {sorted(missing)}")
        selected = {
            str(row["speaker"]): dict(row)
            for row in reader
            if str(row["split"]) == split and str(row["etiology"]) == etiology
        }
    low = {
        speaker: row
        for speaker, row in selected.items()
        if row["speaker_severity"] == low_label
    }
    high = {
        speaker: row
        for speaker, row in selected.items()
        if row["speaker_severity"] == high_label
    }
    if not low or not high:
        raise ValueError(
            f"Need both {low_label!r} and {high_label!r} {etiology} speakers; "
            f"found {len(low)} and {len(high)}"
        )
    return low, high


def choose_pair(
    dataset: Any,
    data_cfg: dict[str, Any],
    selection_cfg: dict[str, Any],
    low_speakers: dict[str, dict[str, str]],
    high_speakers: dict[str, dict[str, str]],
    seed: int,
) -> tuple[dict[str, Any], dict[str, Any], int]:
    speaker_col = data_cfg["speaker_column"]
    etiology_col = data_cfg["etiology_column"]
    id_col = data_cfg["id_column"]
    text_col = data_cfg["text_column"]
    pairing_col = data_cfg["pairing_text_column"]
    required = [speaker_col, etiology_col, id_col, text_col, pairing_col]
    missing = [column for column in required if column not in dataset.column_names]
    if missing:
        raise KeyError(f"SAPC split is missing columns: {missing}")

    requested_low = selection_cfg.get("low_speaker")
    requested_high = selection_cfg.get("high_speaker")
    if requested_low and requested_low not in low_speakers:
        raise ValueError(f"Configured low_speaker is not an eligible low speaker: {requested_low}")
    if requested_high and requested_high not in high_speakers:
        raise ValueError(f"Configured high_speaker is not an eligible high speaker: {requested_high}")
    allowed_low = {str(requested_low)} if requested_low else set(low_speakers)
    allowed_high = {str(requested_high)} if requested_high else set(high_speakers)

    by_speaker_text: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    metadata = dataset.select_columns(required)
    for index, row in enumerate(metadata):
        if str(row[etiology_col]).strip() != data_cfg["etiology"]:
            continue
        speaker = str(row[speaker_col]).strip()
        if speaker not in allowed_low and speaker not in allowed_high:
            continue
        pairing_text = str(row[pairing_col] or "").strip().casefold()
        transcript = str(row[text_col] or "").strip()
        if not pairing_text or not transcript:
            continue
        by_speaker_text[speaker][pairing_text].append(
            {
                "dataset_index": index,
                "id": str(row[id_col]),
                "speaker": speaker,
                "text": transcript,
                "pairing_text": pairing_text,
            }
        )

    feasible: list[tuple[str, str, list[str]]] = []
    for low_speaker in sorted(allowed_low):
        low_texts = set(by_speaker_text[low_speaker])
        if not low_texts:
            continue
        for high_speaker in sorted(allowed_high):
            shared = sorted(low_texts & set(by_speaker_text[high_speaker]))
            if shared:
                feasible.append((low_speaker, high_speaker, shared))
    if not feasible:
        raise ValueError("No eligible low/high speaker pair has a shared nonempty prompt")

    rng = random.Random(seed)
    low_speaker, high_speaker, shared_texts = rng.choice(feasible)
    pairing_text = rng.choice(shared_texts)
    low_row = rng.choice(by_speaker_text[low_speaker][pairing_text])
    high_row = rng.choice(by_speaker_text[high_speaker][pairing_text])
    return low_row, high_row, len(feasible)


def save_original(dataset: Any, index: int, path: Path, audio_column: str) -> None:
    samples, sample_rate = decode_audio(dataset[index][audio_column])
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, samples, sample_rate)


def audio_stats(path: Path) -> dict[str, Any]:
    import numpy as np

    samples, sample_rate = sf.read(path, dtype="float32", always_2d=False)
    if samples.ndim > 1:
        samples = samples.mean(axis=1)
    if samples.size == 0 or not np.isfinite(samples).all():
        raise ValueError(f"Invalid audio produced: {path}")
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
        waveform.unsqueeze(0),
        num_mel_bins=80,
        dither=0,
        sample_frequency=16000,
    )
    features = features - features.mean(dim=0, keepdim=True)
    embedding = campplus_model(features.unsqueeze(0)).float().squeeze(0)
    return torch.nn.functional.normalize(embedding, dim=-1)


def cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    return round(float(torch.dot(left.flatten(), right.flatten()).item()), 6)


def main() -> None:
    args = parse_args()
    config_path = args.config.expanduser().resolve()
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    exp_cfg = cfg["experiment"]
    data_cfg = cfg["dataset"]
    selection_cfg = cfg["selection"]
    vc_cfg = cfg["seed_vc"]

    split = str(data_cfg["split"])
    etiology = str(data_cfg["etiology"])
    severity_csv = resolve_repo_path(data_cfg["speaker_severity_csv"])
    low_speakers, high_speakers = read_speakers(
        severity_csv,
        split=split,
        etiology=etiology,
        low_label=str(selection_cfg["low_label"]),
        high_label=str(selection_cfg["high_label"]),
    )
    dataset = load_split(data_cfg["path"], split)
    low_row, high_row, feasible_pairs = choose_pair(
        dataset,
        data_cfg,
        selection_cfg,
        low_speakers,
        high_speakers,
        int(exp_cfg["seed"]),
    )

    output_dir = resolve_repo_path(exp_cfg["output_dir"])
    paths = {
        "low_original": output_dir / "low_original.wav",
        "high_original": output_dir / "high_original.wav",
        "high_articulation_low_timbre": output_dir / "high_articulation_low_timbre.wav",
        "low_articulation_high_timbre": output_dir / "low_articulation_high_timbre.wav",
    }
    report = {
        "experiment": exp_cfg["name"],
        "seed": int(exp_cfg["seed"]),
        "etiology": etiology,
        "split": split,
        "feasible_speaker_pairs": feasible_pairs,
        "shared_pairing_text": low_row["pairing_text"],
        "low": {
            **low_row,
            "speaker_avg_cer": float(low_speakers[low_row["speaker"]]["speaker_avg_cer"]),
            "speaker_avg_wer": float(low_speakers[low_row["speaker"]]["speaker_avg_wer"]),
            "speaker_severity": low_speakers[low_row["speaker"]]["speaker_severity"],
            "audio": str(paths["low_original"]),
        },
        "high": {
            **high_row,
            "speaker_avg_cer": float(high_speakers[high_row["speaker"]]["speaker_avg_cer"]),
            "speaker_avg_wer": float(high_speakers[high_row["speaker"]]["speaker_avg_wer"]),
            "speaker_severity": high_speakers[high_row["speaker"]]["speaker_severity"],
            "audio": str(paths["high_original"]),
        },
        "outputs": {
            "high_articulation_low_timbre": str(paths["high_articulation_low_timbre"]),
            "low_articulation_high_timbre": str(paths["low_articulation_high_timbre"]),
        },
        "directions": {
            "high_articulation_low_timbre": {
                "source": "high_original",
                "reference": "low_original",
                "expected_timbre": "low speaker",
                "expected_severity": "high",
            },
            "low_articulation_high_timbre": {
                "source": "low_original",
                "reference": "high_original",
                "expected_timbre": "high speaker",
                "expected_severity": "low",
            },
        },
    }
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)
    if args.dry_run:
        print("DRY_RUN_DONE: audio was not decoded and Seed-VC was not loaded", flush=True)
        return

    if not torch.cuda.is_available():
        raise RuntimeError("Park_v1 requires the GPU requested by Park_v1.pbs")
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "selection.json"
    temporary_report = report_path.with_suffix(".json.tmp")
    temporary_report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    temporary_report.replace(report_path)

    audio_column = str(data_cfg["audio_column"])
    if audio_column not in dataset.column_names:
        raise KeyError(f"SAPC split has no audio column {audio_column!r}")
    save_original(dataset, int(low_row["dataset_index"]), paths["low_original"], audio_column)
    save_original(dataset, int(high_row["dataset_index"]), paths["high_original"], audio_column)

    seed_vc_dir = ROOT / "third_party/seed-vc"
    scripts_dir = ROOT / "scripts"
    sys.path.insert(0, str(seed_vc_dir))
    sys.path.insert(0, str(scripts_dir))
    import inference as seed_inference
    from torgo.generate_data import convert_one

    device = torch.device("cuda")
    fp16 = bool(vc_cfg["fp16"])
    seed_inference.device = device
    seed_inference.fp16 = fp16
    model_args = types.SimpleNamespace(
        f0_condition=False,
        checkpoint=None,
        config=None,
        fp16=fp16,
    )
    model, semantic_fn, _, vocoder_fn, campplus_model, mel_fn, _ = (
        seed_inference.load_models(model_args)
    )

    conversions = [
        (
            paths["high_original"],
            paths["low_original"],
            paths["high_articulation_low_timbre"],
        )
    ]
    if bool(vc_cfg.get("bidirectional", True)):
        conversions.append(
            (
                paths["low_original"],
                paths["high_original"],
                paths["low_articulation_high_timbre"],
            )
        )
    overwrite = bool(exp_cfg.get("overwrite", False))
    for source, reference, output in conversions:
        if output.is_file() and not overwrite:
            print(f"Keeping existing output: {output}", flush=True)
            continue
        print(f"Converting source={source} reference={reference} output={output}", flush=True)
        convert_one(
            model,
            semantic_fn,
            vocoder_fn,
            campplus_model,
            mel_fn,
            device,
            fp16,
            str(source),
            str(reference),
            str(output),
            int(vc_cfg["diffusion_steps"]),
            float(vc_cfg["length_adjust"]),
            float(vc_cfg["inference_cfg_rate"]),
        )

    required_outputs = [paths["low_original"], paths["high_original"]]
    required_outputs.extend(output for _source, _reference, output in conversions)
    missing_outputs = [str(path) for path in required_outputs if not path.is_file()]
    if missing_outputs:
        raise FileNotFoundError(f"Expected experiment outputs are missing: {missing_outputs}")

    embeddings = {
        name: speaker_embedding(path, campplus_model, device)
        for name, path in paths.items()
        if path.is_file()
    }
    debug = {
        "audio": {name: audio_stats(path) for name, path in paths.items() if path.is_file()},
        "speaker_similarity": {},
        "interpretation": (
            "A conversion should normally be closer to its timbre reference than to its "
            "articulation source. This checks timbre transfer only; rerun Parakeet to "
            "measure output CER and the severity proxy."
        ),
    }
    if "high_articulation_low_timbre" in embeddings:
        converted = embeddings["high_articulation_low_timbre"]
        to_low = cosine(converted, embeddings["low_original"])
        to_high = cosine(converted, embeddings["high_original"])
        debug["speaker_similarity"]["high_articulation_low_timbre"] = {
            "to_expected_low_timbre": to_low,
            "to_high_source_timbre": to_high,
            "closer_to_expected_reference": to_low > to_high,
        }
    if "low_articulation_high_timbre" in embeddings:
        converted = embeddings["low_articulation_high_timbre"]
        to_high = cosine(converted, embeddings["high_original"])
        to_low = cosine(converted, embeddings["low_original"])
        debug["speaker_similarity"]["low_articulation_high_timbre"] = {
            "to_expected_high_timbre": to_high,
            "to_low_source_timbre": to_low,
            "closer_to_expected_reference": to_high > to_low,
        }
    if any(not math.isfinite(value) for item in debug["speaker_similarity"].values()
           for value in item.values() if isinstance(value, float)):
        raise ValueError("Non-finite speaker similarity in debug output")
    debug_path = output_dir / "debug.json"
    temporary_debug = debug_path.with_suffix(".json.tmp")
    temporary_debug.write_text(json.dumps(debug, indent=2, ensure_ascii=False) + "\n")
    temporary_debug.replace(debug_path)
    print(json.dumps(debug, indent=2, ensure_ascii=False), flush=True)
    print(f"DEBUG_DONE: {debug_path}", flush=True)
    print(f"PARK_V1_DONE: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
