"""Generate same-text, cross-condition Seed-VC samples from SAPC train only."""
import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf
import yaml

from sapc_audio import decode_audio, load_split, pathology_label, speaker_dir_name


def plan_pairs(dataset, data_cfg, max_labels, seed):
    if max_labels < 1:
        raise ValueError("generate.max_source_labels_per_target must be >= 1")
    label_column = data_cfg["pathology_labels"].get("column") or "Category"
    required = ("id", "speaker", label_column, data_cfg["text_column"], data_cfg["pairing_text_column"])
    missing = [col for col in required if col not in dataset.column_names]
    if missing:
        raise ValueError(f"SAPC train missing {missing}")
    rows = dataset.select_columns(list(required))
    records = []
    by_text_label = defaultdict(list)
    labels = data_cfg["pathology_labels"]
    for index, row in enumerate(rows):
        speaker = str(row["speaker"] or "").strip()
        text = str(row[data_cfg["text_column"]] or "").strip()
        pairing_text = str(row[data_cfg["pairing_text_column"]] or "").strip().casefold()
        if not speaker or not text or not pairing_text:
            continue
        label = pathology_label(row, labels)
        num_classes = int(labels["num_classes"])
        if not 0 <= label < num_classes:
            raise ValueError(f"Pathology label {label} for speaker {speaker!r} must be 0..{num_classes - 1}")
        rec = {"index": index, "id": str(row["id"] or index), "speaker": speaker,
               "text": text, "pairing_text": pairing_text, "pathology_label": label}
        records.append(rec)
        by_text_label[(pairing_text, label)].append(rec)
    classes = sorted({r["pathology_label"] for r in records})
    rng = np.random.default_rng(seed)
    pairs = []
    for target in records:
        candidates_by_class = []
        for label in classes:
            if label == target["pathology_label"]:
                continue
            donors = [r for r in by_text_label[(target["pairing_text"], label)]
                      if r["speaker"] != target["speaker"]]
            if donors:
                candidates_by_class.append((label, donors))
        rng.shuffle(candidates_by_class)
        for label, donors in candidates_by_class[:max_labels]:
            donor = donors[int(rng.integers(len(donors)))]
            pairs.append((donor, target))
    return pairs, len(records)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--settings", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--max-pairs", type=int, help="Limit conversions for a smoke check")
    ap.add_argument("--plan", action="store_true", help="Show pairing counts without loading Seed-VC")
    args = ap.parse_args()
    if args.num_shards < 1 or not 0 <= args.shard < args.num_shards:
        ap.error("require num-shards >= 1 and 0 <= shard < num-shards")
    settings = yaml.safe_load(Path(args.settings).read_text())
    cfg = settings["dataset"]
    gen = settings["generate"]
    ds = load_split(cfg["path"], "train")
    pairs, n_records = plan_pairs(ds, cfg, int(gen["max_source_labels_per_target"]), int(gen["seed"]))
    pairs = [p for i, p in enumerate(pairs) if i % args.num_shards == args.shard]
    if args.max_pairs is not None:
        pairs = pairs[:args.max_pairs]
    print(f"SAPC train rows with text: {n_records}; same-text conversion pairs: {len(pairs)} on shard {args.shard}/{args.num_shards}", flush=True)
    if not pairs:
        raise ValueError("No same-text cross-condition pairs. Check mapping and pairing_text_column.")
    if args.plan:
        for donor, target in pairs[:5]:
            print(f"donor={donor['speaker']}:{donor['id']} class={donor['pathology_label']} target={target['speaker']}:{target['id']}")
        return

    out = Path(args.out_dir).expanduser().resolve()
    audio_cache = out / "source_audio"
    audio_cache.mkdir(parents=True, exist_ok=True)
    seed_vc_dir = Path(__file__).resolve().parents[1] / "third_party/seed-vc"
    sys.path.insert(0, str(seed_vc_dir))
    import torch
    import types
    import inference as seed_inference
    from torgo.generate_data import convert_one

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_inference.device = device
    seed_inference.fp16 = device.type == "cuda"
    model_args = types.SimpleNamespace(f0_condition=False, checkpoint=None, config=None, fp16=seed_inference.fp16)
    model, semantic_fn, _, vocoder_fn, campplus_model, mel_fn, _ = seed_inference.load_models(model_args)

    def audio_file(record):
        path = audio_cache / f"{record['index']}.wav"
        if not path.is_file():
            samples, sr = decode_audio(ds[int(record["index"])]["audio"])
            sf.write(path, samples, sr)
        return path

    manifest = []
    failures = 0
    for i, (donor, target) in enumerate(pairs, 1):
        target_speaker = speaker_dir_name(target["speaker"])
        label = donor["pathology_label"]
        key = hashlib.sha1(f"{target['index']}\0{donor['index']}\0{label}".encode()).hexdigest()[:20]
        wav_path = out / f"{target_speaker}_converted_from_{label}" / f"{key}.wav"
        if not wav_path.is_file():
            try:
                convert_one(model, semantic_fn, vocoder_fn, campplus_model, mel_fn, device,
                            seed_inference.fp16, str(audio_file(donor)), str(audio_file(target)), str(wav_path))
            except Exception as exc:
                print(f"[{i}/{len(pairs)}] conversion failed: {exc}", flush=True)
                failures += 1
                continue
        manifest.append({"id": key, "speaker": f"{target_speaker}_converted_from_{label}",
                         "patient_id": target["speaker"], "pathology_label": label,
                         "text": donor["text"], "wav": str(wav_path),
                         "source_id": donor["id"], "target_id": target["id"]})
        if i % 100 == 0:
            print(f"[{i}/{len(pairs)}] converted; failures={failures}", flush=True)
    manifest_path = out / f"pairs_shard_{args.shard}.jsonl"
    temporary = manifest_path.with_suffix(".jsonl.tmp")
    with temporary.open("w", encoding="utf-8") as f:
        for item in manifest:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    temporary.replace(manifest_path)
    print(f"Saved {len(manifest)} converted rows to {manifest_path}; failures={failures}", flush=True)
    if failures:
        raise RuntimeError(f"{failures} conversions failed; inspect log and rerun the shard")


if __name__ == "__main__":
    main()
