"""Generate same-text, cross-condition Seed-VC samples from SAPC train only."""
import argparse
import hashlib
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf
import yaml

from sapc_audio import condition_label, decode_audio, load_speaker_severity, load_split, speaker_dir_name


def select_validation_pair(dataset, data_cfg, settings, severity_lookup):
    """Select one reproducible same-text low/high speaker pair."""
    selection = settings["selection"]
    split = str(data_cfg.get("split", "train"))
    etiology = str(data_cfg["etiology"])
    low_label = str(selection["low_label"])
    high_label = str(selection["high_label"])
    low_speakers = {
        speaker for (row_split, speaker), row in severity_lookup.items()
        if row_split == split and row["etiology"] == etiology
        and row["speaker_severity"] == low_label
    }
    high_speakers = {
        speaker for (row_split, speaker), row in severity_lookup.items()
        if row_split == split and row["etiology"] == etiology
        and row["speaker_severity"] == high_label
    }
    requested_low = selection.get("low_speaker")
    requested_high = selection.get("high_speaker")
    if requested_low:
        if requested_low not in low_speakers:
            raise ValueError(f"Configured low_speaker is not eligible: {requested_low}")
        low_speakers = {requested_low}
    if requested_high:
        if requested_high not in high_speakers:
            raise ValueError(f"Configured high_speaker is not eligible: {requested_high}")
        high_speakers = {requested_high}
    if not low_speakers or not high_speakers:
        raise ValueError(f"Need both {low_label} and {high_label} {etiology} speakers")

    speaker_col = data_cfg.get("speaker_column", "speaker")
    etiology_col = data_cfg.get("etiology_column", "etiology")
    id_col = data_cfg.get("id_column", "id")
    text_col = data_cfg["text_column"]
    pairing_col = data_cfg["pairing_text_column"]
    required = [speaker_col, etiology_col, id_col, text_col, pairing_col]
    missing = [column for column in required if column not in dataset.column_names]
    if missing:
        raise KeyError(f"SAPC split is missing columns: {missing}")
    by_speaker_text = defaultdict(lambda: defaultdict(list))
    for index, row in enumerate(dataset.select_columns(required)):
        if str(row[etiology_col]).strip() != etiology:
            continue
        speaker = str(row[speaker_col]).strip()
        if speaker not in low_speakers and speaker not in high_speakers:
            continue
        pairing_text = str(row[pairing_col] or "").strip().casefold()
        transcript = str(row[text_col] or "").strip()
        if pairing_text and transcript:
            by_speaker_text[speaker][pairing_text].append({
                "dataset_index": index,
                "id": str(row[id_col]),
                "speaker": speaker,
                "text": transcript,
                "pairing_text": pairing_text,
            })
    feasible = []
    for low_speaker in sorted(low_speakers):
        low_texts = set(by_speaker_text[low_speaker])
        for high_speaker in sorted(high_speakers):
            shared = sorted(low_texts & set(by_speaker_text[high_speaker]))
            if shared:
                feasible.append((low_speaker, high_speaker, shared))
    if not feasible:
        raise ValueError("No eligible low/high speaker pair has a shared prompt")
    rng = random.Random(int(settings["experiment"]["seed"]))
    low_speaker, high_speaker, shared = rng.choice(feasible)
    pairing_text = rng.choice(shared)
    return (
        rng.choice(by_speaker_text[low_speaker][pairing_text]),
        rng.choice(by_speaker_text[high_speaker][pairing_text]),
        len(feasible),
    )


def run_validation_pair(settings, out_dir, plan=False):
    """Generate both directions for one low/high pair; debug is a separate stage."""
    data_cfg = settings["dataset"]
    split = str(data_cfg.get("split", "train"))
    severity_csv = data_cfg["condition_labels"]["speaker_csv"]
    severity_lookup = load_speaker_severity(severity_csv)
    dataset = load_split(data_cfg["path"], split)
    low_row, high_row, feasible_pairs = select_validation_pair(
        dataset, data_cfg, settings, severity_lookup
    )
    output_dir = Path(out_dir).expanduser().resolve()
    paths = {
        "low_original": output_dir / "low_original.wav",
        "high_original": output_dir / "high_original.wav",
        "high_articulation_low_timbre": output_dir / "high_articulation_low_timbre.wav",
        "low_articulation_high_timbre": output_dir / "low_articulation_high_timbre.wav",
    }
    low_stats = severity_lookup[(split, low_row["speaker"])]
    high_stats = severity_lookup[(split, high_row["speaker"])]
    report = {
        "experiment": settings["experiment"]["name"],
        "seed": int(settings["experiment"]["seed"]),
        "etiology": data_cfg["etiology"],
        "split": split,
        "feasible_speaker_pairs": feasible_pairs,
        "shared_pairing_text": low_row["pairing_text"],
        "low": {**low_row, "speaker_avg_cer": float(low_stats["speaker_avg_cer"]),
                "speaker_avg_wer": float(low_stats["speaker_avg_wer"]),
                "speaker_severity": low_stats["speaker_severity"],
                "audio": str(paths["low_original"])},
        "high": {**high_row, "speaker_avg_cer": float(high_stats["speaker_avg_cer"]),
                 "speaker_avg_wer": float(high_stats["speaker_avg_wer"]),
                 "speaker_severity": high_stats["speaker_severity"],
                 "audio": str(paths["high_original"])},
        "outputs": {name: str(path) for name, path in paths.items()
                    if name not in {"low_original", "high_original"}},
        "directions": {
            "high_articulation_low_timbre": {
                "source": "high_original", "reference": "low_original",
                "expected_timbre": "low speaker", "expected_severity": "high"},
            "low_articulation_high_timbre": {
                "source": "low_original", "reference": "high_original",
                "expected_timbre": "high speaker", "expected_severity": "low"},
        },
    }
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)
    if plan:
        print("VALIDATION_PAIR_PLAN_DONE: audio was not decoded and Seed-VC was not loaded")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "selection.json"
    temporary = report_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(report_path)
    audio_column = str(data_cfg.get("audio_column", "audio"))
    for row, path in ((low_row, paths["low_original"]), (high_row, paths["high_original"])):
        samples, sample_rate = decode_audio(dataset[int(row["dataset_index"])][audio_column])
        sf.write(path, samples, sample_rate)

    seed_vc_dir = Path(__file__).resolve().parents[1] / "third_party/seed-vc"
    sys.path.insert(0, str(seed_vc_dir))
    import torch
    import types
    import inference as seed_inference
    from torgo.generate_data import convert_one

    if not torch.cuda.is_available():
        raise RuntimeError("VC validation requires a GPU")
    device = torch.device("cuda")
    vc_cfg = settings["seed_vc"]
    fp16 = bool(vc_cfg.get("fp16", True))
    seed_inference.device = device
    seed_inference.fp16 = fp16
    model_args = types.SimpleNamespace(f0_condition=False, checkpoint=None, config=None, fp16=fp16)
    model, semantic_fn, _, vocoder_fn, campplus_model, mel_fn, _ = seed_inference.load_models(model_args)
    conversions = [
        (paths["high_original"], paths["low_original"], paths["high_articulation_low_timbre"]),
        (paths["low_original"], paths["high_original"], paths["low_articulation_high_timbre"]),
    ]
    overwrite = bool(settings["experiment"].get("overwrite", False))
    for source, reference, output in conversions:
        if output.is_file() and not overwrite:
            print(f"Keeping existing output: {output}", flush=True)
            continue
        convert_one(model, semantic_fn, vocoder_fn, campplus_model, mel_fn, device, fp16,
                    str(source), str(reference), str(output),
                    int(vc_cfg["diffusion_steps"]), float(vc_cfg["length_adjust"]),
                    float(vc_cfg["inference_cfg_rate"]))
    print(f"VC_VALIDATION_GENERATE_DONE: {output_dir}", flush=True)


def plan_pairs(dataset, data_cfg, max_labels, seed, severity_lookup=None):
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
    condition_cfg = data_cfg.get("condition_labels") or {}
    pairing_mode = condition_cfg.get("pairing_mode", "cross_condition")
    for index, row in enumerate(rows):
        speaker = str(row["speaker"] or "").strip()
        text = str(row[data_cfg["text_column"]] or "").strip()
        pairing_text = str(row[data_cfg["pairing_text_column"]] or "").strip().casefold()
        if not speaker or not text or not pairing_text:
            continue
        label, etiology_label, severity_label = condition_label(
            row, "train", data_cfg, severity_lookup
        )
        num_classes = int(condition_cfg.get("num_classes", data_cfg["pathology_labels"]["num_classes"]))
        if not 0 <= label < num_classes:
            raise ValueError(f"Pathology label {label} for speaker {speaker!r} must be 0..{num_classes - 1}")
        rec = {"index": index, "id": str(row["id"] or index), "speaker": speaker,
               "text": text, "pairing_text": pairing_text, "pathology_label": label,
               "etiology_label": etiology_label, "severity_label": severity_label}
        records.append(rec)
        by_text_label[(pairing_text, label)].append(rec)
    classes = sorted({r["pathology_label"] for r in records})
    rng = np.random.default_rng(seed)
    pairs = []
    for target in records:
        candidates_by_class = []
        for label in classes:
            donors = by_text_label[(target["pairing_text"], label)]
            if pairing_mode == "same_etiology_cross_severity":
                donors = [r for r in donors
                          if r["etiology_label"] == target["etiology_label"]
                          and r["severity_label"] != target["severity_label"]]
            elif label == target["pathology_label"]:
                continue
            donors = [r for r in donors if r["speaker"] != target["speaker"]]
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
    ap.add_argument("--validation-pair", action="store_true", help="Generate one bidirectional low/high validation pair")
    args = ap.parse_args()
    if args.num_shards < 1 or not 0 <= args.shard < args.num_shards:
        ap.error("require num-shards >= 1 and 0 <= shard < num-shards")
    settings = yaml.safe_load(Path(args.settings).read_text())
    if args.validation_pair:
        run_validation_pair(settings, args.out_dir, plan=args.plan)
        return
    cfg = settings["dataset"]
    gen = settings["generate"]
    ds = load_split(cfg["path"], "train")
    condition_cfg = cfg.get("condition_labels") or {}
    severity_lookup = None
    if condition_cfg.get("mode") == "joint_etiology_severity":
        severity_lookup = load_speaker_severity(condition_cfg["speaker_csv"])
    pairs, n_records = plan_pairs(ds, cfg, int(gen["max_source_labels_per_target"]), int(gen["seed"]), severity_lookup)
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
                         "etiology_label": donor["etiology_label"],
                         "severity_label": donor["severity_label"],
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
