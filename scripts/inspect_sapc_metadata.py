"""Write SAPC speaker, Category, and etiology counts without touching audio."""
import argparse
from collections import Counter, defaultdict
from pathlib import Path

from datasets import Dataset, DatasetDict, load_from_disk


def load_split(root, split):
    split_dir = root / split
    source = split_dir if (split_dir / "dataset_info.json").is_file() else root
    data = load_from_disk(str(source))
    if isinstance(data, DatasetDict):
        return data[split]
    if isinstance(data, Dataset):
        return data
    raise TypeError(f"Unsupported dataset at {source}: {type(data)}")


def display(value):
    value = str(value or "").strip()
    return value if value else "<EMPTY>"


def add_counts(lines, title, counts):
    lines.append(f"\n{title} ({len(counts)} values)")
    for value, count in sorted(counts.items(), key=lambda pair: (-pair[1], pair[0])):
        lines.append(f"  {count:>8}  {value}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="/srv/scratch/speechdata/speech-corpora/dysarthric/SAPC_HF/SAPC_full")
    parser.add_argument("--out", default="sapc_dataset_report.txt")
    args = parser.parse_args()
    root = Path(args.dataset).expanduser().resolve()
    lines = [f"SAPC metadata report", f"Dataset: {root}", "Audio column is excluded from every row read."]

    for split in ("train", "dev"):
        ds = load_split(root, split)
        required = ("speaker", "Category", "etiology")
        missing = [column for column in required if column not in ds.column_names]
        if missing:
            parser.error(f"{split} is missing {missing}; columns={ds.column_names}")
        # Project out audio and every other unneeded column before iteration.
        metadata = ds.select_columns(list(required))
        speakers = Counter()
        categories = Counter()
        etiologies = Counter()
        per_speaker = defaultdict(Counter)
        for row in metadata:
            speaker = display(row["speaker"])
            category = display(row["Category"])
            etiology = display(row["etiology"])
            speakers[speaker] += 1
            categories[category] += 1
            etiologies[etiology] += 1
            per_speaker[speaker][(category, etiology)] += 1

        lines.extend(("", f"=== {split} ===", f"Rows: {len(ds)}", f"Columns: {', '.join(ds.column_names)}"))
        add_counts(lines, "Category", categories)
        add_counts(lines, "etiology", etiologies)
        add_counts(lines, "speaker", speakers)
        lines.append("\nPer speaker: Category | etiology | rows")
        for speaker in sorted(per_speaker):
            for (category, etiology), count in sorted(per_speaker[speaker].items()):
                lines.append(f"  {speaker} | {category} | {etiology} | {count}")

    out = Path(args.out).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
