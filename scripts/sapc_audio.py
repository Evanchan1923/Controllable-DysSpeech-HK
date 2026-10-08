"""SAPC_full loading and audio decoding, following SAPC-Qwen's decode=False path."""
import io
import hashlib
import re
from pathlib import Path

import numpy as np
import soundfile as sf
from datasets import Audio, Dataset, DatasetDict, load_from_disk


def speaker_dir_name(speaker):
    safe = re.sub(r"[^A-Za-z0-9.-]+", "_", speaker).strip("._")[:80] or "speaker"
    return safe + "_" + hashlib.sha1(speaker.encode()).hexdigest()[:8]


def pathology_label(row, label_cfg):
    speaker = str(row["speaker"]).strip()
    category = str(row.get("Category") or "").strip()
    by_speaker = label_cfg.get("by_speaker") or {}
    by_category = label_cfg.get("by_category") or {}
    if speaker in by_speaker:
        return int(by_speaker[speaker])
    if category in by_category:
        return int(by_category[category])
    raise ValueError(f"No pathology label for speaker={speaker!r}, Category={category!r}")


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
