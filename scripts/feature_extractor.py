"""IndexTTS feature extraction shared by the TORGO and SAPC preparers."""
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

import numpy as np
import torch
import torchaudio
from omegaconf import OmegaConf

from indextts.utils.feature_extractors import MelSpectrogramFeatures
from indextts.vqvae.xtts_dvae import DiscreteVAE
from indextts.gpt.model import UnifiedVoice


class FeatureExtractor:
    def __init__(self, finetune_dir: str, config_path: str, device: str = "cuda"):
        self.device = device
        cfg = OmegaConf.load(config_path)
        self.mel_fn = MelSpectrogramFeatures().to(device)

        dvae_path = os.path.join(finetune_dir, cfg.dvae_checkpoint)
        self.dvae = DiscreteVAE(channels=cfg.vqvae.channels,
                                num_tokens=cfg.vqvae.num_tokens,
                                hidden_dim=cfg.vqvae.hidden_dim,
                                num_resnet_blocks=cfg.vqvae.num_resnet_blocks,
                                codebook_dim=cfg.vqvae.codebook_dim,
                                num_layers=cfg.vqvae.num_layers,
                                positional_dims=cfg.vqvae.positional_dims,
                                kernel_size=cfg.vqvae.kernel_size,
                                use_transposed_convs=cfg.vqvae.use_transposed_convs)
        dvae_sd = torch.load(dvae_path, map_location="cpu")
        dvae_sd = dvae_sd.get("model", dvae_sd)
        self.dvae.load_state_dict(dvae_sd, strict=False)
        self.dvae.eval().to(device)

        gpt_path = os.path.join(finetune_dir, "gpt.pth")
        gpt_sd = torch.load(gpt_path, map_location="cpu")
        gpt_sd = gpt_sd.get("model", gpt_sd)
        self.gpt = UnifiedVoice(**cfg.gpt)
        self.gpt.load_state_dict(gpt_sd, strict=False)
        self.gpt.eval().to(device)

    @torch.no_grad()
    def extract(self, wav_path: str):
        audio, sr = torchaudio.load(wav_path)
        return self.extract_audio(audio, sr)

    @torch.no_grad()
    def extract_audio(self, audio: torch.Tensor, sr: int):
        audio = audio[:1]  # mono
        if sr != 24000:
            audio = torchaudio.transforms.Resample(sr, 24000)(audio)
        duration = audio.shape[-1] / 24000.0
        if duration < 0.2 or duration > 20:
            return None
        if not torch.isfinite(audio).all():
            return None
        audio = audio.to(self.device)
        mel = self.mel_fn(audio)
        codes = self.dvae.get_codebook_indices(mel)
        cond_len = torch.tensor([mel.shape[-1]], device=self.device)
        cond = self.gpt.get_conditioning(mel, cond_len)
        if not (torch.isfinite(mel).all() and torch.isfinite(cond).all()):
            return None
        return (mel.cpu().numpy().astype(np.float32),
                codes.cpu().numpy().astype(np.int64),
                cond.cpu().numpy().astype(np.float32),
                duration)
