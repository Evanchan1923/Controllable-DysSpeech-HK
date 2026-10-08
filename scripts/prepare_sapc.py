"""Extract IndexTTS features from SAPC_full's Hugging Face train/dev splits."""
import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import yaml

from sapc_audio import decode_audio, load_split, pathology_label, speaker_dir_name


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--settings", required=True, help="Katana YAML with dataset mapping")
    ap.add_argument("--out-dir")
    ap.add_argument("--model-dir")
    ap.add_argument("--config", help="IndexTTS architecture YAML")
    ap.add_argument("--converted-dir", help="Seed-VC output containing pairs_shard_*.jsonl")
    ap.add_argument("--max-items-per-split", type=int, help="Small end-to-end check")
    ap.add_argument("--inspect", action="store_true", help="Print SAPC labels without loading the model")
    args = ap.parse_args()
    with open(args.settings, encoding="utf-8") as f:
        settings = yaml.safe_load(f)
    data_cfg = settings["dataset"]
    ds_by_split = {split: load_split(data_cfg["path"], split) for split in ("train", "dev")}
    required = ("id", "speaker", "Category", data_cfg["text_column"], "audio")
    for split, ds in ds_by_split.items():
        missing = [name for name in required if name not in ds.column_names]
        if missing:
            ap.error(f"{split} missing columns {missing}; available={ds.column_names}")
        print(f"SAPC {split}: {len(ds)} rows", flush=True)
    if args.inspect:
        for split, ds in ds_by_split.items():
            for column in ("speaker", "Category", "etiology"):
                if column in ds.column_names:
                    counts = Counter(str(v) for v in ds[column])
                    print(f"{split} {column}: {dict(counts.most_common())}")
        return

    if not args.out_dir or not args.model_dir or not args.config:
        ap.error("--out-dir, --model-dir and --config are required for feature extraction")
    import torch
    from loguru import logger
    from prepare_torgo import FeatureExtractor

    labels = data_cfg.get("pathology_labels") or {}
    if not labels.get("by_speaker") and not labels.get("by_category"):
        ap.error("Set dataset.pathology_labels.by_speaker or by_category after inspecting SAPC labels")
    arch = yaml.safe_load(Path(args.config).read_text())
    num_classes = int(arch["pathology"]["num_classes"])
    # Validate the full mapping before loading checkpoints or running GPU work.
    preflight_counts = Counter()
    for split, ds in ds_by_split.items():
        for row in ds.select_columns(["speaker", "Category", data_cfg["text_column"]]):
            if str(row.get(data_cfg["text_column"]) or "").strip():
                label = pathology_label(row, labels)
                if not 0 <= label < num_classes:
                    raise ValueError(f"Pathology label {label} outside 0..{num_classes - 1}")
                preflight_counts[(split, label)] += 1
    absent = [label for label in range(num_classes) if not preflight_counts[("train", label)]]
    if absent:
        raise ValueError(f"SAPC train has no rows for pathology classes {absent}; check the mapping")
    if not any(preflight_counts[("dev", label)] for label in range(num_classes)):
        raise ValueError("SAPC dev has no labeled rows with text")
    logger.info(f"Preflight rows by split/pathology class: {dict(preflight_counts)}")
    out = Path(args.out_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    fx = FeatureExtractor(args.model_dir, args.config)
    items_by_speaker = defaultdict(lambda: {"train": [], "dev": []})
    seen_keys = set()
    counts = Counter()
    for split, ds in ds_by_split.items():
        for index, row in enumerate(ds):
            if args.max_items_per_split is not None and index >= args.max_items_per_split:
                break
            speaker = str(row["speaker"]).strip()
            text = str(row.get(data_cfg["text_column"]) or "").strip()
            if not speaker or not text:
                counts[(split, "empty_speaker_or_text")] += 1
                continue
            label = pathology_label(row, labels)
            if not 0 <= label < num_classes:
                raise ValueError(f"Label {label} for speaker {speaker!r} is outside 0..{num_classes - 1}")
            speaker_key = speaker_dir_name(speaker)
            uid = str(row["id"] or index)
            key = hashlib.sha1(f"{split}\0{speaker}\0{uid}".encode()).hexdigest()[:20]
            if (split, key) in seen_keys:
                raise ValueError(f"Duplicate SAPC id {uid!r} for speaker {speaker!r} in {split}")
            seen_keys.add((split, key))
            feats = out / speaker_key / "feats"
            mel_p = feats / f"{split}_{key}_mel.npy"
            codes_p = feats / f"{split}_{key}_codes.npy"
            cond_p = feats / f"{split}_{key}_condition.npy"
            dur_p = feats / f"{split}_{key}_dur.json"
            if all(p.is_file() for p in (mel_p, codes_p, cond_p, dur_p)):
                duration = json.loads(dur_p.read_text())["duration"]
            else:
                try:
                    samples, sr = decode_audio(row["audio"])
                    result = fx.extract_audio(torch.from_numpy(samples).unsqueeze(0), sr)
                except Exception as exc:
                    logger.warning(f"Skipped {split} id={uid!r}: {exc}")
                    counts[(split, "bad_audio")] += 1
                    continue
                if result is None:
                    counts[(split, "invalid_duration_or_features")] += 1
                    continue
                mel, codes, cond, duration = result
                feats.mkdir(parents=True, exist_ok=True)
                np.save(mel_p, mel)
                np.save(codes_p, codes)
                np.save(cond_p, cond)
                dur_p.write_text(json.dumps({"duration": duration}))
            items_by_speaker[speaker_key][split].append({
                "id": uid, "text": text, "codes": str(codes_p), "mels": str(mel_p),
                "condition": str(cond_p), "duration": duration,
                "pathology_label": label, "patient_id": speaker,
                "audio_filepath": row.get("audio_filepath"),
            })
            counts[(split, "kept")] += 1

    if args.converted_dir:
        converted_dir = Path(args.converted_dir).expanduser().resolve()
        for manifest_path in sorted(converted_dir.glob("pairs_shard_*.jsonl")):
            with manifest_path.open(encoding="utf-8") as f:
                for line in f:
                    item = json.loads(line)
                    wav_path = Path(item["wav"])
                    if not wav_path.is_file():
                        logger.warning(f"Missing converted WAV: {wav_path}")
                        continue
                    speaker_key = item["speaker"]
                    label = int(item["pathology_label"])
                    feats = out / speaker_key / "feats"
                    key = item["id"]
                    mel_p = feats / f"train_{key}_mel.npy"
                    codes_p = feats / f"train_{key}_codes.npy"
                    cond_p = feats / f"train_{key}_condition.npy"
                    dur_p = feats / f"train_{key}_dur.json"
                    if all(p.is_file() for p in (mel_p, codes_p, cond_p, dur_p)):
                        duration = json.loads(dur_p.read_text())["duration"]
                    else:
                        try:
                            result = fx.extract(str(wav_path))
                        except Exception as exc:
                            logger.warning(f"Skipped converted {wav_path}: {exc}")
                            counts[("converted", "bad_audio")] += 1
                            continue
                        if result is None:
                            counts[("converted", "invalid_duration_or_features")] += 1
                            continue
                        mel, codes, cond, duration = result
                        feats.mkdir(parents=True, exist_ok=True)
                        np.save(mel_p, mel)
                        np.save(codes_p, codes)
                        np.save(cond_p, cond)
                        dur_p.write_text(json.dumps({"duration": duration}))
                    items_by_speaker[speaker_key]["train"].append({
                        "id": key, "text": item["text"], "codes": str(codes_p),
                        "mels": str(mel_p), "condition": str(cond_p),
                        "duration": duration, "pathology_label": label,
                        "patient_id": item["patient_id"], "wav": str(wav_path),
                    })
                    counts[("converted", "kept")] += 1

    speaker_info = []
    class_sums = {}
    class_counts = Counter()
    for speaker_key, splits in sorted(items_by_speaker.items()):
        if not splits["train"]:
            logger.warning(f"Speaker {speaker_key} has dev only; omitting from training manifests")
            continue
        speaker_out = out / speaker_key
        for split, manifest_name in (("train", "metadata_train.jsonl"), ("dev", "metadata_valid.jsonl")):
            with (speaker_out / manifest_name).open("w", encoding="utf-8") as f:
                for item in splits[split]:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
        # Same summary shape used by prepare_torgo.py.
        sample = splits["train"][::max(1, len(splits["train"]) // 200)]
        speaker_conditions = np.stack([np.load(it["condition"])[0] for it in sample])
        medoid = speaker_out / "medoid_condition.npy"
        np.save(medoid, speaker_conditions.mean(axis=0, keepdims=True).astype(np.float32))
        for item in splits["train"]:
            label = item["pathology_label"]
            cond = np.load(item["condition"])[0].astype(np.float64)
            class_sums[label] = cond if label not in class_sums else class_sums[label] + cond
            class_counts[label] += 1
        speaker_info.append({
            "speaker": speaker_key,
            "train_jsonl": str(speaker_out / "metadata_train.jsonl"),
            "valid_jsonl": str(speaker_out / "metadata_valid.jsonl"),
            "medoid_condition": str(medoid),
            "n_items": len(splits["train"]) + len(splits["dev"]),
        })
    (out / "speaker_info.json").write_text(json.dumps(speaker_info, indent=2))
    emb_dir = out / "pathology_embedding"
    emb_dir.mkdir(exist_ok=True)
    for label, value in class_sums.items():
        np.save(emb_dir / f"mean_pathology_condition_{label}.npy",
                (value / class_counts[label]).astype(np.float32)[None])
    logger.info(f"Preparation counts: {dict(counts)}; train class counts: {dict(class_counts)}")
    missing = sorted(set(range(num_classes)) - set(class_counts))
    if missing:
        raise ValueError(f"Training requires all {num_classes} pathology classes; missing {missing}. Check mapping and dataset coverage.")
    if not any(s["valid_jsonl"] and (out / s["speaker"] / "metadata_valid.jsonl").stat().st_size for s in speaker_info):
        raise ValueError("No dev items prepared; cannot run validation")


if __name__ == "__main__":
    main()
