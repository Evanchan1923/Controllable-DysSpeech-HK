#!/usr/bin/env python3
"""Universal config-driven entry point for SAPC VC checks and full training."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
FULL_STAGES = ("cache_models", "generate", "prepare", "train", "infer")
VC_VALIDATION_STAGES = ("generate", "debug")


def required_path(value: Any, name: str) -> Path:
    if not value:
        raise ValueError(f"Set {name} in the selected pipeline config")
    path = Path(str(value)).expanduser()
    return (ROOT / path).resolve() if not path.is_absolute() else path.resolve()


def selected_stages(configured: list[str], requested: str | None, allowed: tuple[str, ...]) -> list[str]:
    stages = [requested] if requested else list(configured)
    if not stages or any(stage not in allowed for stage in stages):
        raise ValueError(f"Stages must come from {allowed}; got {stages}")
    if len(set(stages)) != len(stages) or stages != sorted(stages, key=allowed.index):
        raise ValueError(f"Stages must be unique and follow this order: {allowed}")
    return stages


def validate_dataset(path: Path) -> None:
    if not (path / "dataset_dict.json").is_file() and not (path / "train/dataset_info.json").is_file():
        raise ValueError(f"SAPC DatasetDict/train split not found at {path}")


def full_training_commands(
    cfg: dict[str, Any], settings: Path, requested_stage: str | None
) -> tuple[Path, list[tuple[Path, list[str]]]]:
    stages = selected_stages(cfg["run"]["stages"], requested_stage, FULL_STAGES)
    run_root = required_path(cfg["run"]["root"], "run.root")
    name = str(cfg["run"]["name"])
    if not name or Path(name).name != name or name in (".", ".."):
        raise ValueError("run.name must be one directory name")
    run_dir = run_root / name
    converted = run_dir / "converted_audio"
    prepared = run_dir / "prepared_data"
    training = run_dir / "training"
    outputs = run_dir / "outputs"

    if any(stage in stages for stage in ("generate", "prepare")):
        dataset = required_path(cfg["dataset"]["path"], "dataset.path")
        validate_dataset(dataset)
        labels = cfg["dataset"].get("pathology_labels") or {}
        if not labels.get("mapping") and not labels.get("by_speaker") and not labels.get("by_category"):
            raise ValueError("Set dataset.pathology_labels mapping")
        condition_cfg = cfg["dataset"].get("condition_labels") or {}
        if condition_cfg.get("mode") == "joint_etiology_severity":
            severity_csv = required_path(
                condition_cfg.get("speaker_csv"), "dataset.condition_labels.speaker_csv"
            )
            if not severity_csv.is_file():
                raise ValueError(f"Speaker severity CSV missing: {severity_csv}")
            severity_mapping = condition_cfg.get("severity_mapping") or {}
            expected_classes = len(labels.get("mapping") or {}) * len(severity_mapping)
            if int(condition_cfg.get("num_classes", -1)) != expected_classes:
                raise ValueError(
                    "dataset.condition_labels.num_classes must equal "
                    f"etiologies x severities ({expected_classes})"
                )

    model_dir = None
    if any(stage in stages for stage in ("prepare", "train", "infer")):
        model_dir = required_path(cfg["model"]["pretrained_dir"], "model.pretrained_dir")
        needed = set()
        if "prepare" in stages:
            needed.update(("dvae.pth", "gpt.pth"))
        if "train" in stages:
            needed.update(("bpe.model", "gpt.pth"))
        if "infer" in stages:
            needed.update(("bpe.model", "dvae.pth", "bigvgan_generator.pth"))
        missing = [str(model_dir / item) for item in sorted(needed) if not (model_dir / item).is_file()]
        if missing:
            raise ValueError("Missing pretrained files: " + ", ".join(missing))
    if "train" in stages and "prepare" not in stages and not (prepared / "speaker_info.json").is_file():
        raise ValueError(f"Prepared data missing: {prepared / 'speaker_info.json'}")

    condition_cfg = cfg["dataset"].get("condition_labels") or {}
    num_classes = int(
        condition_cfg.get("num_classes", cfg["dataset"]["pathology_labels"]["num_classes"])
    )
    python = sys.executable
    commands: list[tuple[Path, list[str]]] = []
    if "cache_models" in stages:
        commands.append((ROOT, [python, "scripts/cache_seedvc_models.py"]))
    if "generate" in stages:
        generation = cfg["generate"]
        commands.append((ROOT, [
            python, "scripts/generate_sapc.py", "--settings", str(settings),
            "--out-dir", str(converted), "--num-shards", str(generation["num_shards"]),
            "--shard", str(generation["shard"]),
        ]))
    if "prepare" in stages:
        commands.append((ROOT, [
            python, "scripts/prepare_sapc.py", "--settings", str(settings),
            "--out-dir", str(prepared), "--model-dir", str(model_dir),
            "--config", str(ROOT / "configs/controllable_dysarthric_speech_synthesis.yaml"),
            "--converted-dir", str(converted),
        ]))
    if "train" in stages:
        training_cfg = cfg["train"]
        commands.append((ROOT, [
            python, "train.py", "--config",
            str(ROOT / "configs/controllable_dysarthric_speech_synthesis.yaml"),
            "--model-dir", str(model_dir), "--output-dir", str(training),
            "--data-dir", str(prepared), "--embedding-dir", str(prepared / "pathology_embedding"),
            "--num-pathology-classes", str(num_classes), "--epochs", str(training_cfg["epochs"]),
            "--batch-size", str(training_cfg["batch_size"]),
            "--num-workers", str(training_cfg["num_workers"]),
        ]))
    if "infer" in stages:
        prompt = required_path(cfg["run"].get("prompt_wav"), "run.prompt_wav")
        if not prompt.is_file():
            raise ValueError(f"Prompt WAV missing: {prompt}")
        text = cfg["run"].get("text")
        if not text:
            raise ValueError("Set run.text for inference")
        checkpoint = training / "checkpoints_dys_spk_grl_exp1/gpt_best.pth"
        if "train" not in stages and not checkpoint.is_file():
            raise ValueError(f"Trained checkpoint missing: {checkpoint}")
        condition_id = cfg["run"].get("condition_id", cfg["run"].get("pathology"))
        if condition_id is None:
            raise ValueError("Set run.condition_id for inference")
        commands.append((ROOT, [
            python, "-m", "indextts.inference", "--cfg",
            str(ROOT / "configs/controllable_dysarthric_speech_synthesis.yaml"),
            "--model-dir", str(model_dir), "--gpt-ckpt", str(checkpoint),
            "--num-pathology-classes", str(num_classes), "--prompt", str(prompt),
            "--text", str(text), "--pathology", str(condition_id),
            "--out", str(outputs / cfg["inference"]["output_name"]),
        ]))
    return run_dir, commands


def vc_validation_commands(
    cfg: dict[str, Any], settings: Path, requested_stage: str | None, dry_run: bool
) -> tuple[Path, list[tuple[Path, list[str]]]]:
    stages = selected_stages(cfg["pipeline"]["stages"], requested_stage, VC_VALIDATION_STAGES)
    dataset = required_path(cfg["dataset"]["path"], "dataset.path")
    validate_dataset(dataset)
    severity_csv = required_path(
        cfg["dataset"]["condition_labels"].get("speaker_csv"),
        "dataset.condition_labels.speaker_csv",
    )
    if not severity_csv.is_file():
        raise ValueError(f"Speaker severity CSV missing: {severity_csv}")
    output_dir = required_path(cfg["experiment"]["output_dir"], "experiment.output_dir")
    python = sys.executable
    commands: list[tuple[Path, list[str]]] = []
    if "generate" in stages:
        command = [
            python, "scripts/generate_sapc.py", "--settings", str(settings),
            "--out-dir", str(output_dir), "--validation-pair",
        ]
        if dry_run:
            command.append("--plan")
        commands.append((ROOT, command))
    if "debug" in stages and not dry_run:
        commands.append((ROOT, [
            python, "scripts/debug_vc.py", "--experiment-dir", str(output_dir),
        ]))
    return output_dir, commands


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stage", help="Run one stage from the selected pipeline type")
    parser.add_argument("--check", action="store_true", help="Validate paths and print commands")
    parser.add_argument("--dry-run", action="store_true", help="Select a VC validation pair without audio")
    args = parser.parse_args()
    settings = args.config.expanduser().resolve()
    cfg = yaml.safe_load(settings.read_text(encoding="utf-8"))
    pipeline_type = str((cfg.get("pipeline") or {}).get("type") or "").strip()
    if pipeline_type == "full_training":
        if args.dry_run:
            raise ValueError("--dry-run is only valid for vc_validation")
        output_dir, commands = full_training_commands(cfg, settings, args.stage)
    elif pipeline_type == "vc_validation":
        output_dir, commands = vc_validation_commands(cfg, settings, args.stage, args.dry_run)
    else:
        raise ValueError(
            "pipeline.type must be 'vc_validation' or 'full_training'"
        )
    print(f"Pipeline type: {pipeline_type}")
    print(f"Output directory: {output_dir}")
    for cwd, command in commands:
        print(f"[{cwd}] {' '.join(command)}", flush=True)
        if not args.check:
            output_dir.mkdir(parents=True, exist_ok=True)
            subprocess.run(command, cwd=cwd, check=True, env=os.environ.copy())


if __name__ == "__main__":
    try:
        main()
    except ValueError as exc:
        raise SystemExit(f"Configuration error: {exc}") from None
