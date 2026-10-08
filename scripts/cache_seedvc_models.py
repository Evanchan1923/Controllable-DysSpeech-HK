"""Preload Seed-VC's public model assets into the configured shared HF cache."""
import os

from huggingface_hub import hf_hub_download, snapshot_download


def main():
    seed_cache = os.environ["SEED_VC_CACHE_DIR"]
    hf_hub_download("Plachta/Seed-VC", "DiT_seed_v2_uvit_whisper_small_wavenet_bigvgan_pruned.pth", cache_dir=seed_cache)
    hf_hub_download("Plachta/Seed-VC", "config_dit_mel_seed_uvit_whisper_small_wavenet.yml", cache_dir=seed_cache)
    hf_hub_download("funasr/campplus", "campplus_cn_common.bin", cache_dir=seed_cache)
    snapshot_download("openai/whisper-small")
    snapshot_download("nvidia/bigvgan_v2_22khz_80band_256x")
    print("Seed-VC assets cached in", seed_cache, "and", os.environ["HF_HOME"])


if __name__ == "__main__":
    main()
