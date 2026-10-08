"""SAPC_full loading and audio decoding, following SAPC-Qwen's decode=False path."""
import io
import hashlib
import csv
import re
from pathlib import Path

import numpy as np
import soundfile as sf
from datasets import Audio, Dataset, DatasetDict, load_from_disk


def speaker_dir_name(speaker):
    safe = re.sub(r"[^A-Za-z0-9.-]+", "_", speaker).strip("._")[:80] or "speaker"
    return safe + "_" + hashlib.sha1(speaker.encode()).hexdigest()[:8]


def pathology_label(row, label_cfg):
    column = str(label_cfg.get("column") or "").strip()
    mapping = label_cfg.get("mapping") or {}
    if column and mapping:
        value = str(row.get(column) or "").strip()
        if value in mapping:
            return int(mapping[value])
        raise ValueError(f"No pathology label for {column}={value!r}")

    # Compatibility with the first Katana config format.
    speaker = str(row.get("speaker") or "").strip()
    category = str(row.get("Category") or "").strip()
    by_speaker = label_cfg.get("by_speaker") or {}
    by_category = label_cfg.get("by_category") or {}
    if speaker in by_speaker:
        return int(by_speaker[speaker])
    if category in by_category:
        return int(by_category[category])
    raise ValueError(f"No pathology label for speaker={speaker!r}, Category={category!r}")


def load_speaker_severity(path):
    """Load one severity record per (split, speaker)."""
    lookup = {}
    with Path(path).expanduser().resolve().open(encoding="utf-8", newline="") as input_file:
        reader = csv.DictReader(input_file)
        required = {"split", "speaker", "etiology", "speaker_avg_cer", "speaker_severity"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise KeyError(f"Speaker severity CSV is missing columns: {sorted(missing)}")
        for row in reader:
            key = (str(row["split"]).strip(), str(row["speaker"]).strip())
            if key in lookup:
                raise ValueError(f"Duplicate speaker severity key: {key}")
            lookup[key] = dict(row)
    return lookup


def condition_label(row, split, data_cfg, severity_lookup=None):
    """Return (joint condition ID, etiology ID, severity ID or None)."""
    etiology_id = pathology_label(row, data_cfg["pathology_labels"])
    condition_cfg = data_cfg.get("condition_labels") or {}
    if condition_cfg.get("mode") != "joint_etiology_severity":
        return etiology_id, etiology_id, None
    if severity_lookup is None:
        raise ValueError("joint_etiology_severity requires the speaker severity CSV")
    speaker = str(row.get("speaker") or "").strip()
    key = (str(split), speaker)
    if key not in severity_lookup:
        raise KeyError(f"No speaker severity row for {key}")
    severity_row = severity_lookup[key]
    etiology_column = data_cfg["pathology_labels"].get("column")
    if etiology_column and str(severity_row["etiology"]).strip() != str(row.get(etiology_column) or "").strip():
        raise ValueError(f"Etiology mismatch for {key}")
    severity_name = str(severity_row["speaker_severity"]).strip()
    severity_mapping = condition_cfg["severity_mapping"]
    if severity_name not in severity_mapping:
        raise ValueError(f"Unknown severity {severity_name!r} for {key}")
    severity_id = int(severity_mapping[severity_name])
    num_severities = len(severity_mapping)
    joint_id = etiology_id * num_severities + severity_id
    if not 0 <= joint_id < int(condition_cfg["num_classes"]):
        raise ValueError(f"Joint condition {joint_id} outside configured class range")
    return joint_id, etiology_id, severity_id


def load_split(dataset_path, split):
    path = Path(dataset_path).expanduser().resolve()
    # Qwen accepts either a DatasetDict root or a directory containing one split.
    load_path = path / split if (path / split / "dataset_info.json").exists() else path
    obj = load_from_disk(str(load_path))
    if isinstance(obj, DatasetDict):
        if split not in obj:
            raise KeyError(f"Split {split!r} missing; available={list(obj)}")
        ds = obj[split]
    elif isinstance(obj, Dataset):
        ds = obj
    else:
        raise TypeError(f"Unsupported dataset at {load_path}: {type(obj)}")
    if "audio" not in ds.column_names:
        raise KeyError(f"Missing audio column; available={ds.column_names}")
    feature = ds.features.get("audio")
    if isinstance(feature, Audio) and getattr(feature, "decode", True):
        ds = ds.cast_column("audio", Audio(decode=False))
    return ds


def decode_audio(raw_audio):
    """Return mono float32 samples and actual sampling rate from Audio(decode=False)."""
    if isinstance(raw_audio, dict):
        if raw_audio.get("bytes") is not None:
            wav, sr = sf.read(io.BytesIO(raw_audio["bytes"]), dtype="float32", always_2d=False)
        elif raw_audio.get("array") is not None:
            wav = np.asarray(raw_audio["array"], dtype=np.float32)
            sr = int(raw_audio.get("sampling_rate") or 16000)
        elif raw_audio.get("path"):
            wav, sr = sf.read(str(raw_audio["path"]), dtype="float32", always_2d=False)
        else:
            raise ValueError("Audio value has no bytes, array, or path")
    else:
        wav = np.asarray(raw_audio, dtype=np.float32)
        sr = 16000
    wav = np.asarray(wav, dtype=np.float32)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if wav.ndim != 1 or wav.size == 0 or not np.isfinite(wav).all():
        raise ValueError("Audio must be nonempty finite mono samples")
    return wav, int(sr)
