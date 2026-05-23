import torch
from torch import nn
from typing import Tuple, Optional, Any, List, Dict
from src.scheduler_model.film_mlp.noise_encoders import NoiseEncoder, StatsNoiseEncoder, LightConvNoiseEncoder, PyramidStatsNoiseEncoder, PatchEmbedNoiseEncoder


class PromptNoiseFiLMMlp(nn.Module):
    def __init__(self, out_dim: int, prompt_dim: int = 768, hidden_dim: int = 256,
                 noise_encoder: Optional[NoiseEncoder] = None,
                 noise_feat_dim: int = 3, noise_encoder_type: str = "stats",
                 in_channels: int = 3):
        super().__init__()
        # Создаём энкодер шума
        if noise_encoder is not None:
            self.noise_encoder = noise_encoder
        else:
            if noise_encoder_type == "stats":
                self.noise_encoder = StatsNoiseEncoder(noise_feat_dim, in_channels)
            elif noise_encoder_type == "lightconv":
                self.noise_encoder = LightConvNoiseEncoder(noise_feat_dim, in_channels)
            elif noise_encoder_type == "pyramid":
                self.noise_encoder = PyramidStatsNoiseEncoder(noise_feat_dim, in_channels)
            elif noise_encoder_type == "patch":
                self.noise_encoder = PatchEmbedNoiseEncoder(noise_feat_dim, in_channels)
            else:
                raise ValueError(f"Unknown noise_encoder_type: {noise_encoder_type}")
        actual_noise_dim = noise_feat_dim

        self.prompt_mlp = nn.Sequential(
            nn.Linear(prompt_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.noise_mlp = nn.Sequential(
            nn.Linear(actual_noise_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.film = nn.Linear(hidden_dim, 2 * hidden_dim)
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)

        self.out_norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, out_dim),
        )
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, noise: torch.Tensor, cond_emb: torch.Tensor) -> torch.Tensor:
        p = cond_emb.mean(dim=1)
        hp = self.prompt_mlp(p)

        n_feats = self.noise_encoder(noise).to(dtype=hp.dtype, device=hp.device)
        hn = self.noise_mlp(n_feats)

        gamma, beta = self.film(hn).chunk(2, dim=1)
        h = hp * (1.0 + gamma) + beta
        return self.head(h)

    