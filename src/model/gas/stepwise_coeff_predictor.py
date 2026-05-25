from typing import Optional, Tuple

import torch
from torch import nn

from src.scheduler_model.film_mlp.noise_encoders.noise_encoders import (
    LightConvNoiseEncoder,
    NoiseEncoder,
    PatchEmbedNoiseEncoder,
    PyramidStatsNoiseEncoder,
    StatsNoiseEncoder,
)


def _build_noise_encoder(
    noise_encoder: Optional[NoiseEncoder],
    noise_encoder_type: str,
    noise_feat_dim: int,
    in_channels: int,
) -> NoiseEncoder:
    if noise_encoder is not None:
        return noise_encoder
    if noise_encoder_type == "stats":
        return StatsNoiseEncoder(noise_feat_dim, in_channels)
    if noise_encoder_type == "lightconv":
        return LightConvNoiseEncoder(noise_feat_dim, in_channels)
    if noise_encoder_type == "pyramid":
        return PyramidStatsNoiseEncoder(noise_feat_dim, in_channels)
    if noise_encoder_type == "patch":
        return PatchEmbedNoiseEncoder(noise_feat_dim, in_channels)
    raise ValueError(f"Unknown noise_encoder_type: {noise_encoder_type}")


class StepwiseCoeffPredictor(nn.Module):
    """
    Lightweight stepwise head: predicts a/c coefficients for the current solver step
    from (x_t, t, cond_emb).
    """

    def __init__(
        self,
        order: int,
        latent_channels: int = 4,
        text_embed_dim: int = 768,
        hidden_dim: int = 256,
        time_embed_dim: int = 64,
        noise_encoder: Optional[NoiseEncoder] = None,
        noise_feat_dim: int = 256,
        noise_encoder_type: str = "lightconv",
    ):
        super().__init__()
        self.order = order
        self.noise_encoder = _build_noise_encoder(
            noise_encoder=noise_encoder,
            noise_encoder_type=noise_encoder_type,
            noise_feat_dim=noise_feat_dim,
            in_channels=latent_channels,
        )

        self.prompt_mlp = nn.Sequential(
            nn.Linear(text_embed_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.noise_mlp = nn.Sequential(
            nn.Linear(noise_feat_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.time_mlp = nn.Sequential(
            nn.Linear(1, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, hidden_dim),
        )
        self.film = nn.Linear(hidden_dim, 2 * hidden_dim)
        nn.init.normal_(self.film.weight, std=0.01)
        nn.init.zeros_(self.film.bias)

        self.a_head = nn.Linear(hidden_dim, order)
        self.c_head = nn.Linear(hidden_dim, order)
        nn.init.normal_(self.a_head.weight, std=0.01)
        nn.init.zeros_(self.a_head.bias)
        nn.init.normal_(self.c_head.weight, std=0.01)
        nn.init.zeros_(self.c_head.bias)

    def forward(
        self,
        x_t: torch.Tensor,
        t_current: torch.Tensor,
        cond_emb: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x_t: (B, C, H, W) current latent
            t_current: (B,) or (B, 1) continuous time at the current step
            cond_emb: (B, L, D) text embedding

        Returns:
            a_coefs: (B, order)
            c_coefs: (B, order)
        """
        if t_current.ndim == 1:
            t_in = t_current.unsqueeze(-1)
        else:
            t_in = t_current.reshape(x_t.shape[0], 1)

        p = cond_emb.mean(dim=1)
        hp = self.prompt_mlp(p)

        n_feats = self.noise_encoder(x_t).to(dtype=hp.dtype, device=hp.device)
        hn = self.noise_mlp(n_feats)
        ht = self.time_mlp(t_in.to(dtype=hp.dtype, device=hp.device))

        gamma, beta = self.film(hn + ht).chunk(2, dim=1)
        h = hp * (1.0 + gamma) + beta
        return self.a_head(h), self.c_head(h)
