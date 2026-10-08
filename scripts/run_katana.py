"""Run the SAPC_full synthesis pipeline on Katana."""
import argparse
import os
import subprocess
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
STAGES = ("cache_models", "generate", "prepare", "train", "infer")


def required_path(value, name):
    if not value:
        raise ValueError(f"Set {name} in the selected pipeline config")
    return Path(value).expanduser().resolve()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True, help="Pipeline YAML to run")
    ap.add_argument("--stage", choices=STAGES, help="Run one stage instead of run.stages")
    ap.add_argument("--check", action="store_true", help="Check paths and print commands only")
    args = ap.parse_args()
    settings = Path(args.config).expanduser().resolve()
    with settings.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    stages = [args.stage] if args.stage else cfg["run"]["stages"]
    if not stages or any(stage not in STAGES for stage in stages):
        ap.error(f"run.stages must contain stages from {STAGES}")
    if len(set(stages)) != len(stages) or stages != sorted(stages, key=STAGES.index):
        ap.error("run.stages must be unique and follow cache_models, generate, prepare, train, infer order")

    run_root = required_path(cfg["run"]["root"], "run.root")
    name = cfg["run"]["name"]
    if not name or Path(name).name != name or name in (".", ".."):
        ap.error("run.name must be one directory name")
    run_dir = run_root / name
    converted = run_dir / "converted_audio"
    prepared = run_dir / "prepared_data"
    training = run_dir / "training"
    outputs = run_dir / "outputs"

    dataset = required_path(cfg["dataset"]["path"], "dataset.path") if any(s in stages for s in ("generate", "prepare")) else None
    if dataset is not None and not (dataset / "dataset_dict.json").is_file() and not (dataset / "train/dataset_info.json").is_file():
        ap.error(f"SAPC DatasetDict/train split not found at {dataset}")
    if any(s in stages for s in ("generate", "prepare")):
        labels = cfg["dataset"].get("pathology_labels") or {}
        if not labels.get("mapping") and not labels.get("by_speaker") and not labels.get("by_category"):
            ap.error("Set dataset.pathology_labels mapping")
        condition_cfg = cfg["dataset"].get("condition_labels") or {}
        if condition_cfg.get("mode") == "joint_etiology_severity":
            severity_csv = required_path(condition_cfg.get("speaker_csv"), "dataset.condition_labels.speaker_csv")
            if not severity_csv.is_file():
                ap.error(f"Speaker severity CSV missing: {severity_csv}")
            severity_mapping = condition_cfg.get("severity_mapping") or {}
            expected_classes = len(labels.get("mapping") or {}) * len(severity_mapping)
            if int(condition_cfg.get("num_classes", -1)) != expected_classes:
                ap.error(
                    "dataset.condition_labels.num_classes must equal "
                    f"etiologies x severities ({expected_classes})"
                )

    model_dir = required_path(cfg["model"]["pretrained_dir"], "model.pretrained_dir") if any(s in stages for s in ("prepare", "train", "infer")) else None
    if model_dir is not None:
        needed = set()
        if "prepare" in stages:
            needed.update(("dvae.pth", "gpt.pth"))
        if "train" in stages:
            needed.update(("bpe.model", "gpt.pth"))
        if "infer" in stages:
            needed.update(("bpe.model", "dvae.pth", "bigvgan_generator.pth"))
        missing = [str(model_dir / item) for item in sorted(needed) if not (model_dir / item).is_file()]
        if missing:
            ap.error("Missing pretrained files: " + ", ".join(missing))
    if "train" in stages and "prepare" not in stages and not (prepared / "speaker_info.json").is_file():
        ap.error(f"Prepared data missing: {prepared / 'speaker_info.json'}")

    python = sys.executable
    condition_cfg = cfg["dataset"].get("condition_labels") or {}
    pathology_num_classes = int(
        condition_cfg.get("num_classes", cfg["dataset"]["pathology_labels"]["num_classes"])
    )
    commands = []
    if "cache_models" in stages:
        commands.append((ROOT, [python, "scripts/cache_seedvc_models.py"]))
    if "generate" in stages:
        gen = cfg["generate"]
        commands.append((ROOT, [python, "scripts/generate_sapc.py", "--settings", str(settings), "--out-dir", str(converted), "--num-shards", str(gen["num_shards"]), "--shard", str(gen["shard"])]))
    if "prepare" in stages:
        commands.append((ROOT, [python, "scripts/prepare_sapc.py", "--settings", str(settings), "--out-dir", str(prepared), "--model-dir", str(model_dir), "--config", str(ROOT / "configs/controllable_dysarthric_speech_synthesis.yaml"), "--converted-dir", str(converted)]))
    if "train" in stages:
        tr = cfg["train"]
        commands.append((ROOT, [python, "train.py", "--config", str(ROOT / "configs/controllable_dysarthric_speech_synthesis.yaml"), "--model-dir", str(model_dir), "--output-dir", str(training), "--data-dir", str(prepared), "--embedding-dir", str(prepared / "pathology_embedding"), "--num-pathology-classes", str(pathology_num_classes), "--epochs", str(tr["epochs"]), "--batch-size", str(tr["batch_size"]), "--num-workers", str(tr["num_workers"])]))
    if "infer" in stages:
        prompt = required_path(cfg["run"].get("prompt_wav"), "run.prompt_wav")
        if not prompt.is_file():
            ap.error(f"Prompt WAV missing: {prompt}")
        text = cfg["run"].get("text")
        if not text:
            ap.error("Set run.text for inference")
        checkpoint = training / "checkpoints_dys_spk_grl_exp1/gpt_best.pth"
        if "train" not in stages and not checkpoint.is_file():
            ap.error(f"Trained checkpoint missing: {checkpoint}")
        pathology_id = cfg["run"].get("condition_id", cfg["run"].get("pathology"))
        if pathology_id is None:
            ap.error("Set run.condition_id for inference")
        commands.append((ROOT, [python, "-m", "indextts.inference", "--cfg", str(ROOT / "configs/controllable_dysarthric_speech_synthesis.yaml"), "--model-dir", str(model_dir), "--gpt-ckpt", str(checkpoint), "--num-pathology-classes", str(pathology_num_classes), "--prompt", str(prompt), "--text", str(text), "--pathology", str(pathology_id), "--out", str(outputs / cfg["inference"]["output_name"])]))

    print(f"Run directory: {run_dir}", flush=True)
    for cwd, command in commands:
        print(f"[{cwd}] {' '.join(command)}", flush=True)
        if not args.check:
            run_dir.mkdir(parents=True, exist_ok=True)
            subprocess.run(command, cwd=cwd, check=True, env=os.environ.copy())


if __name__ == "__main__":
    try:
        main()
    except ValueError as exc:
        raise SystemExit(f"Configuration error: {exc}") from None
