"""FiLM MLP that predicts solver coefficients from noise + text prompt."""

import torch
from torch import nn


class PromptNoiseFiLMMlp(nn.Module):
    """
    Maps (latent noise, prompt embedding) → flat coefficient vector.

    Noise encoding is built into the module (small CNN + GAP), so there is no
    separate pluggable noise-encoder config. Prompt features are modulated by
    FiLM (γ, β) derived from the noise features, then passed through a head.
    The final linear layer is zero-initialized so the network starts near a
    constant (bias-only) prediction.
    """

    def __init__(
        self,
        out_dim: int,
        prompt_dim: int = 768,
        hidden_dim: int = 256,
        noise_feat_dim: int = 256,
        in_channels: int = 4,
        noise_encoder_width: int = 32,
    ):
        super().__init__()

        # Lightweight spatial encoder over the latent (or image) noise map.
        w = noise_encoder_width
        self.noise_encoder = nn.Sequential(
            nn.Conv2d(in_channels, w, kernel_size=3, padding=1, stride=2),
            nn.GroupNorm(8, w),
            nn.SiLU(),
            nn.Conv2d(w, w * 2, kernel_size=3, padding=1, stride=2),
            nn.GroupNorm(8, w * 2),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(w * 2, noise_feat_dim),
            nn.SiLU(),
        )
        nn.init.zeros_(self.noise_encoder[-2].weight)
        nn.init.zeros_(self.noise_encoder[-2].bias)

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

        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, out_dim),
        )
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, noise: torch.Tensor, cond_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            noise: (B, C, H, W) initial / current latent noise.
            cond_emb: (B, L, D) text token embeddings (mean-pooled over L).

        Returns:
            (B, out_dim) predicted coefficients.
        """
        # Mean-pool tokens → single prompt vector.
        p = cond_emb.mean(dim=1)
        hp = self.prompt_mlp(p)

        n_feats = self.noise_encoder(noise).to(dtype=hp.dtype, device=hp.device)
        hn = self.noise_mlp(n_feats)

        gamma, beta = self.film(hn).chunk(2, dim=1)
        h = hp * (1.0 + gamma) + beta
        return self.head(h)
