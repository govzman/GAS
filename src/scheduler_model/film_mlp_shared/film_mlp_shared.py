import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from typing import Tuple, Optional, Any, List, Dict


class SharedPromptNoiseFiLMBackbone(nn.Module):
    """
    Общий backbone для предсказания a/c коэффициентов.
    Возвращает одновременно (a_flat, c_flat) после прохождения общего FiLM-кодировщика и двух голов.
    """
    def __init__(self, a_out_dim: int, c_out_dim: int,
                 prompt_dim: int = 768, hidden_dim: int = 256,
                 noise_encoder = None,
                 noise_feat_dim: int = 256, latent_dim: int = 256,
                 head_hidden_dim: int = 128):
        super().__init__()
        self.noise_encoder = noise_encoder
        self.prompt_norm = nn.LayerNorm(prompt_dim)
        self.noise_norm = nn.LayerNorm(noise_feat_dim)

        self.prompt_mlp = nn.Sequential(
            nn.Linear(prompt_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.noise_mlp = nn.Sequential(
            nn.Linear(noise_feat_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.film = nn.Linear(hidden_dim, 2 * hidden_dim)
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)

        self.latent_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, latent_dim),
            nn.SiLU(),
            nn.Linear(latent_dim, latent_dim),
        )
        nn.init.zeros_(self.latent_proj[-1].weight)
        nn.init.zeros_(self.latent_proj[-1].bias)

        # Головы для A и C
        self.a_head = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, head_hidden_dim),
            nn.SiLU(),
            nn.Linear(head_hidden_dim, a_out_dim),
        )
        self.c_head = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, head_hidden_dim),
            nn.SiLU(),
            nn.Linear(head_hidden_dim, c_out_dim),
        )
        # Инициализируем последние слои голов в нуль (как и раньше)
        nn.init.zeros_(self.a_head[-1].weight)
        nn.init.zeros_(self.a_head[-1].bias)
        nn.init.zeros_(self.c_head[-1].weight)
        nn.init.zeros_(self.c_head[-1].bias)

    def forward(self, noise: torch.Tensor, cond_emb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        p = cond_emb.mean(dim=1)  # (B, 768)
        hp = self.prompt_mlp(p)

        n_feats = self.noise_encoder(noise).to(dtype=hp.dtype, device=hp.device)
        hn = self.noise_mlp(n_feats)

        gamma, beta = self.film(hn).chunk(2, dim=1)
        h = hp * (1.0 + gamma) + beta
        latent = self.latent_proj(h)  # (B, latent_dim)

        a_flat = self.a_head(latent)  # (B, a_out_dim)
        c_flat = self.c_head(latent)  # (B, c_out_dim)
        return a_flat, c_flat
