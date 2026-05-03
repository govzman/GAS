from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from src.scheduler_model.scheduler_transformer.modules import ImageEncoder, MLP


class StepWisePredictorLightweight(nn.Module):
    """Fast predictor: latent channel stats + text-pooled MLP."""

    def __init__(
        self,
        out_dim: int,
        latent_channels: int = 4,
        text_embed_dim: int = 768,
        num_mlp_layers: int = 3,
        hidden_dim: int = 256,
    ):
        super().__init__()
        stats_dim = 3 * latent_channels
        self.latent_proj = nn.Sequential(
            nn.Linear(stats_dim, hidden_dim // 2),
            nn.SiLU(),
        )
        self.text_proj = nn.Sequential(
            nn.LayerNorm(text_embed_dim),
            nn.Linear(text_embed_dim, hidden_dim // 2),
            nn.SiLU(),
        )

        mlp_layers = [
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
            )
        ]
        for _ in range(max(0, num_mlp_layers - 2)):
            mlp_layers.append(
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.SiLU(),
                )
            )
        final_layer = nn.Linear(hidden_dim, out_dim)
        nn.init.zeros_(final_layer.weight)
        nn.init.zeros_(final_layer.bias)
        mlp_layers.append(final_layer)
        self.mlp = nn.Sequential(*mlp_layers)

    @staticmethod
    def _compute_stats(x: torch.Tensor) -> torch.Tensor:
        b, c = x.shape[:2]
        x_flat = x.reshape(b, c, -1)
        mean = x_flat.mean(dim=2)
        std = x_flat.std(dim=2, unbiased=False)
        max_val = x_flat.max(dim=2)[0]
        return torch.cat([mean, std, max_val], dim=1)

    def forward(
        self,
        x_t: torch.Tensor,
        cond_emb: torch.Tensor,
        timestep: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        _ = timestep
        stats = self._compute_stats(x_t)
        latent_feat = self.latent_proj(stats)
        if cond_emb.ndim == 3:
            cond_emb = cond_emb.mean(dim=1)
        text_feat = self.text_proj(cond_emb)
        return self.mlp(torch.cat([latent_feat, text_feat], dim=-1))


class StepWisePredictorMedium(nn.Module):
    """Balanced predictor: conv encoder + cross-attn + MLP head."""

    def __init__(
        self,
        out_dim: int,
        latent_channels: int = 4,
        text_embed_dim: int = 768,
        encoder_width: int = 32,
        num_conv_blocks: int = 2,
        attention_heads: int = 4,
        num_mlp_layers: int = 3,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.input_proj = nn.Conv2d(latent_channels, encoder_width, kernel_size=1)
        conv_blocks = []
        channels = encoder_width
        for i in range(num_conv_blocks):
            conv_blocks.append(
                nn.Sequential(
                    nn.Conv2d(channels, channels, kernel_size=3, padding=1),
                    nn.GroupNorm(min(8, channels), channels),
                    nn.SiLU(),
                )
            )
            if i < num_conv_blocks - 1:
                conv_blocks.append(
                    nn.Sequential(
                        nn.Conv2d(channels, channels * 2, kernel_size=3, stride=2, padding=1),
                        nn.GroupNorm(min(8, channels * 2), channels * 2),
                        nn.SiLU(),
                    )
                )
                channels *= 2
        self.conv_encoder = nn.Sequential(*conv_blocks)
        self.query_proj = nn.Linear(channels, channels)
        self.kv_proj = nn.Linear(text_embed_dim, channels)
        self.cross_attn = nn.MultiheadAttention(channels, attention_heads, batch_first=True)

        mlp_layers = [
            nn.Sequential(
                nn.Linear(channels, hidden_dim),
                nn.SiLU(),
            )
        ]
        for _ in range(max(0, num_mlp_layers - 2)):
            mlp_layers.append(
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.SiLU(),
                )
            )
        final_layer = nn.Linear(hidden_dim, out_dim)
        nn.init.zeros_(final_layer.weight)
        nn.init.zeros_(final_layer.bias)
        mlp_layers.append(final_layer)
        self.mlp = nn.Sequential(*mlp_layers)

    def forward(
        self,
        x_t: torch.Tensor,
        cond_emb: torch.Tensor,
        timestep: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        _ = timestep
        x = self.input_proj(x_t)
        x = self.conv_encoder(x)
        b, c, h, w = x.shape
        x_flat = x.reshape(b, c, h * w).permute(0, 2, 1)
        x_query = self.query_proj(x_flat)
        x_kv = self.kv_proj(cond_emb)
        attn_out, _ = self.cross_attn(x_query, x_kv, x_kv)
        x_flat = F.layer_norm(x_flat + attn_out, [c])
        return self.mlp(x_flat.mean(dim=1))


class StepWisePredictorFull(nn.Module):
    """Heaviest predictor: adapted ImageEncoder + MLP head."""

    def __init__(
        self,
        out_dim: int,
        latent_channels: int = 4,
        text_embed_dim: int = 768,
        image_encoder_depth: int = 2,
        image_encoder_width: int = 32,
        cross_attention_heads: int = 4,
        attention_dim: int = 128,
        number_of_transformer_blocks: int = 3,
        num_mlp_layers: int = 3,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.image_encoder = ImageEncoder(
            image_encoder_depth=image_encoder_depth,
            image_encoder_width=image_encoder_width,
            text_embed_dim=text_embed_dim,
            cross_attention_heads=cross_attention_heads,
            attention_dim=attention_dim,
            number_of_transformer_blocks=number_of_transformer_blocks,
            input_channels=latent_channels,
        )
        final_xs_dim = sum(
            image_encoder_width * 2 ** min(4, i) for i in range(number_of_transformer_blocks)
        )
        self.mlp = MLP(
            num_mlp_layers=num_mlp_layers,
            in_channels=final_xs_dim,
            hidden_dim=hidden_dim,
            num_timesteps=out_dim,
        )
        if hasattr(self.mlp, "mlps") and len(self.mlp.mlps) > 0 and isinstance(self.mlp.mlps[-1], nn.Linear):
            nn.init.zeros_(self.mlp.mlps[-1].weight)
            if self.mlp.mlps[-1].bias is not None:
                nn.init.zeros_(self.mlp.mlps[-1].bias)

    def forward(
        self,
        x_t: torch.Tensor,
        cond_emb: torch.Tensor,
        timestep: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        _ = timestep
        features = self.image_encoder(x_t, cond_emb)
        return self.mlp(features)


def create_stepwise_predictor(
    out_dim: int,
    version: str = "medium",
    latent_channels: int = 4,
    text_embed_dim: int = 768,
    **kwargs,
) -> nn.Module:
    if version == "lightweight":
        return StepWisePredictorLightweight(
            out_dim=out_dim,
            latent_channels=latent_channels,
            text_embed_dim=text_embed_dim,
            num_mlp_layers=kwargs.get("num_mlp_layers", 3),
            hidden_dim=kwargs.get("hidden_dim", 256),
        )
    if version == "medium":
        return StepWisePredictorMedium(
            out_dim=out_dim,
            latent_channels=latent_channels,
            text_embed_dim=text_embed_dim,
            encoder_width=kwargs.get("encoder_width", 32),
            num_conv_blocks=kwargs.get("num_conv_blocks", 2),
            attention_heads=kwargs.get("attention_heads", 4),
            num_mlp_layers=kwargs.get("num_mlp_layers", 3),
            hidden_dim=kwargs.get("hidden_dim", 256),
        )
    if version == "full":
        return StepWisePredictorFull(
            out_dim=out_dim,
            latent_channels=latent_channels,
            text_embed_dim=text_embed_dim,
            image_encoder_depth=kwargs.get("image_encoder_depth", 2),
            image_encoder_width=kwargs.get("image_encoder_width", 32),
            cross_attention_heads=kwargs.get("cross_attention_heads", 4),
            attention_dim=kwargs.get("attention_dim", 128),
            number_of_transformer_blocks=kwargs.get("number_of_transformer_blocks", 3),
            num_mlp_layers=kwargs.get("num_mlp_layers", 3),
            hidden_dim=kwargs.get("hidden_dim", 256),
        )
    raise ValueError(f"Unknown stepwise predictor version: {version}")
