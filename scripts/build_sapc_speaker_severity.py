#!/usr/bin/env python3
"""Aggregate SAPC utterance WER/CER into one severity row per speaker."""
from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median


OUTPUT_COLUMNS = [
    "speaker",
    "split",
    "etiology",
    "num_utterances",
    "speaker_avg_wer",
    "speaker_median_wer",
    "speaker_avg_cer",
    "speaker_median_cer",
    "speaker_score_avg_cer_wer",
    "speaker_severity",
    "low_utterances",
    "middle_utterances",
    "high_utterances",
    "low_utterance_ratio",
    "middle_utterance_ratio",
    "high_utterance_ratio",
    "low_cer_threshold",
    "high_cer_threshold",
    "severity_source",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--utterance-csv",
        type=Path,
        default=Path(
            "configs/utterance_research_v1_noFTModel/"
            "severity-utterance-labelling.csv"
        ),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path(
            "configs/utterance_research_v1_noFTModel/"
            "severity-speaker-labelling.csv"
        ),
    )
    parser.add_argument(
        "--metadata-report",
        type=Path,
        default=Path("sapc_dataset_report.txt"),
        help="Metadata-only report produced by inspect_sapc_metadata.py.",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        help="Optional SAPC DatasetDict; takes precedence over --metadata-report.",
    )
    parser.add_argument("--low-cer", type=float, default=0.10)
    parser.add_argument("--high-cer", type=float, default=0.30)
    return parser.parse_args()


def severity(cer: float, low_cer: float, high_cer: float) -> str:
    if cer < low_cer:
        return "low"
    if cer < high_cer:
        return "middle"
    return "high"


def etiologies_from_report(path: Path) -> dict[tuple[str, str], str]:
    """Read (split, speaker) -> etiology without accessing SAPC audio."""
    result: dict[tuple[str, str], str] = {}
    split = ""
    in_per_speaker = False
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line.startswith("=== ") and line.endswith(" ==="):
            split = line[4:-4].strip()
            in_per_speaker = False
            continue
        if line == "Per speaker: Category | etiology | rows":
            in_per_speaker = True
            continue
        if not in_per_speaker or "|" not in line:
            continue
        parts = [part.strip() for part in line.split("|")]
        if len(parts) != 4:
            continue
        speaker, _category, etiology, _count = parts
        key = (split, speaker)
        previous = result.setdefault(key, etiology)
        if previous != etiology:
            raise ValueError(
                f"Speaker {speaker!r} has multiple etiologies in {split}: "
                f"{previous!r}, {etiology!r}"
            )
    if not result:
        raise ValueError(f"No per-speaker etiology rows found in {path}")
    return result


def etiologies_from_dataset(path: Path) -> dict[tuple[str, str], str]:
    """Read only speaker and etiology columns from a Hugging Face dataset."""
    from datasets import Dataset, DatasetDict, load_from_disk

    result: dict[tuple[str, str], str] = {}
    for split in ("train", "dev"):
        split_path = path / split
        source = split_path if (split_path / "dataset_info.json").is_file() else path
        dataset = load_from_disk(str(source))
        if isinstance(dataset, DatasetDict):
            dataset = dataset[split]
        if not isinstance(dataset, Dataset):
            raise TypeError(f"Unsupported dataset object for {split}: {type(dataset)}")
        missing = {"speaker", "etiology"} - set(dataset.column_names)
        if missing:
            raise KeyError(f"SAPC {split} is missing columns: {sorted(missing)}")
        for row in dataset.select_columns(["speaker", "etiology"]):
            speaker = str(row["speaker"]).strip()
            etiology = str(row["etiology"]).strip()
            key = (split, speaker)
            previous = result.setdefault(key, etiology)
            if previous != etiology:
                raise ValueError(
                    f"Speaker {speaker!r} has multiple etiologies in {split}: "
                    f"{previous!r}, {etiology!r}"
                )
    return result


def fmt(value: float) -> str:
    return f"{value:.4f}"


def main() -> None:
    args = parse_args()
    if not 0.0 < args.low_cer < args.high_cer:
        raise ValueError("Thresholds must satisfy 0 < low-cer < high-cer")

    if args.dataset is not None:
        speaker_etiology = etiologies_from_dataset(args.dataset.expanduser().resolve())
        metadata_source = f"dataset:{args.dataset}"
    else:
        report = args.metadata_report.expanduser().resolve()
        speaker_etiology = etiologies_from_report(report)
        metadata_source = f"report:{report}"

    grouped: dict[tuple[str, str], dict[str, object]] = defaultdict(
        lambda: {"wers": [], "cers": [], "ids": set()}
    )
    utterance_csv = args.utterance_csv.expanduser().resolve()
    with utterance_csv.open(encoding="utf-8", newline="") as input_file:
        reader = csv.DictReader(input_file)
        required = {"utt_id", "speaker", "split", "utt_wer", "utt_cer"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise KeyError(f"{utterance_csv} is missing columns: {sorted(missing)}")
        for line_number, row in enumerate(reader, start=2):
            split = str(row["split"]).strip()
            speaker = str(row["speaker"]).strip()
            utt_id = str(row["utt_id"]).strip()
            if not split or not speaker or not utt_id:
                raise ValueError(f"Empty split, speaker, or utt_id at line {line_number}")
            values = grouped[(split, speaker)]
            ids = values["ids"]
            if utt_id in ids:
                raise ValueError(f"Duplicate utt_id {utt_id!r} at line {line_number}")
            ids.add(utt_id)
            values["wers"].append(float(row["utt_wer"]))
            values["cers"].append(float(row["utt_cer"]))

    missing_metadata = sorted(set(grouped) - set(speaker_etiology))
    if missing_metadata:
        preview = ", ".join(f"{split}/{speaker}" for split, speaker in missing_metadata[:5])
        raise KeyError(
            f"Missing etiology metadata for {len(missing_metadata)} speakers: {preview}"
        )

    output_rows = []
    for (split, speaker), values in sorted(grouped.items()):
        wers = values["wers"]
        cers = values["cers"]
        count = len(cers)
        avg_wer = sum(wers) / count
        avg_cer = sum(cers) / count
        utterance_counts = Counter(
            severity(cer, args.low_cer, args.high_cer) for cer in cers
        )
        output_rows.append(
            {
                "speaker": speaker,
                "split": split,
                "etiology": speaker_etiology[(split, speaker)],
                "num_utterances": count,
                "speaker_avg_wer": fmt(avg_wer),
                "speaker_median_wer": fmt(median(wers)),
                "speaker_avg_cer": fmt(avg_cer),
                "speaker_median_cer": fmt(median(cers)),
                "speaker_score_avg_cer_wer": fmt((avg_wer + avg_cer) / 2.0),
                "speaker_severity": severity(avg_cer, args.low_cer, args.high_cer),
                "low_utterances": utterance_counts["low"],
                "middle_utterances": utterance_counts["middle"],
                "high_utterances": utterance_counts["high"],
                "low_utterance_ratio": fmt(utterance_counts["low"] / count),
                "middle_utterance_ratio": fmt(utterance_counts["middle"] / count),
                "high_utterance_ratio": fmt(utterance_counts["high"] / count),
                "low_cer_threshold": fmt(args.low_cer),
                "high_cer_threshold": fmt(args.high_cer),
                "severity_source": "parakeet_baseline_speaker_mean_cer",
            }
        )

    output_csv = args.output_csv.expanduser().resolve()
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(output_rows)

    speaker_counts = Counter(row["speaker_severity"] for row in output_rows)
    split_counts = Counter(row["split"] for row in output_rows)
    print(f"Wrote {len(output_rows)} speakers to {output_csv}")
    print(f"Splits: {dict(sorted(split_counts.items()))}")
    print(f"Severity: {dict(sorted(speaker_counts.items()))}")
    print(f"Etiology metadata: {metadata_source}")


if __name__ == "__main__":
    main()
