import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class NoiseEncoder(nn.Module):
    """Базовый класс для кодирования шума в вектор признаков."""
    def __init__(self, feat_dim: int = 256, in_channels: int = 3):
        super().__init__()
        self.feat_dim = feat_dim
        self.in_channels = in_channels

    def forward(self, noise: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class StatsNoiseEncoder(NoiseEncoder):
    """Оригинальные глобальные статистики (mean, std, rms)."""
    def __init__(self, feat_dim: int = 256, in_channels: int = 3):
        super().__init__(feat_dim, in_channels)
        # Проекция в feat_dim, т.к. исходно 3 числа
        self.proj = nn.Linear(3, feat_dim) if feat_dim != 3 else nn.Identity()

    def forward(self, noise: torch.Tensor) -> torch.Tensor:
        b = noise.shape[0]
        flat = noise.reshape(b, -1)
        mean = flat.mean(dim=1, keepdim=True)
        std = flat.std(dim=1, unbiased=False, keepdim=True)
        rms = (flat.pow(2).mean(dim=1, keepdim=True) + 1e-12).sqrt()
        stats = torch.cat([mean, std, rms], dim=1)  # (B, 3)
        return self.proj(stats)


class LightConvNoiseEncoder(NoiseEncoder):
    """
    Мелкая CNN с несколькими блоками conv + groupnorm + SiLU.
    Каждый блок уменьшает пространственное разрешение вдвое (stride=2).
    Завершается глобальным average pooling и линейным слоем.
    """
    def __init__(self, feat_dim: int = 256, in_channels: int = 3, base_width: int = 32):
        super().__init__(feat_dim, in_channels)
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, base_width, 3, padding=1, stride=2),
            nn.GroupNorm(8, base_width),
            nn.SiLU(),
            nn.Conv2d(base_width, base_width * 2, 3, padding=1, stride=2),
            nn.GroupNorm(8, base_width * 2),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(base_width * 2, feat_dim),
            nn.SiLU(),
        )
        # Инициализация близка к нулю
        nn.init.zeros_(self.body[-2].weight)
        nn.init.zeros_(self.body[-2].bias)

    def forward(self, noise: torch.Tensor) -> torch.Tensor:
        return self.body(noise)


class PyramidStatsNoiseEncoder(NoiseEncoder):
    """
    Делит изображение на регулярную сетку (например, 2×2, 4×4) и считает статистики
    внутри каждой ячейки. Все векторы агрегируются через небольшой MLP.
    """
    def __init__(self, feat_dim: int = 256, in_channels: int = 3,
                 grid_sizes=(2, 4, 8), pool_dim: int = 64):
        super().__init__(feat_dim, in_channels)
        self.grid_sizes = grid_sizes
        # Каждая ячейка даёт 3 статистики (mean, std, rms) для каждого канала
        # Для простоты: считаем mean, std по всему тензору ячейки (без канального разделения) -> 3 числа на ячейку
        total_stats = sum(g*g for g in grid_sizes) * 3
        self.mlp = nn.Sequential(
            nn.Linear(total_stats, pool_dim),
            nn.SiLU(),
            nn.Linear(pool_dim, feat_dim),
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, noise: torch.Tensor) -> torch.Tensor:
        b, c, h, w = noise.shape
        all_stats = []
        for g in self.grid_sizes:
            # Адаптивный средний пул до размера (g, g) — получаем "суперпиксели"
            pooled = F.adaptive_avg_pool2d(noise, (g, g))  # (B, C, g, g)
            # Для каждой ячейки считаем статистики по каналам и пространству? 
            # Упростим: флаттен в (B, C*g*g) и считаем mean, std, rms вдоль последней оси -> 3 числа на ячейку
            flat = pooled.reshape(b, c, g * g)
            mean = flat.mean(dim=2)  # (B, C)
            std = flat.std(dim=2, unbiased=False)
            rms = flat.pow(2).mean(dim=2).sqrt()
            # Усредняем по каналам? Или сохраняем поканально? Сохраним поканально: (B, C*3)
            stats = torch.cat([mean, std, rms], dim=1)  # (B, 3*C)
            all_stats.append(stats)
        combined = torch.cat(all_stats, dim=1)  # (B, total_stats)
        return self.mlp(combined)


class PatchEmbedNoiseEncoder(NoiseEncoder):
    """
    Разбивает изображение на патчи, линейно проецирует их,
    пропускает через один-два TransformerEncoder слоя, затем делает mean pooling токенов.
    """
    def __init__(self, feat_dim: int = 256, in_channels: int = 3,
                 patch_size: int = 8, embed_dim: int = 192, depth: int = 2):
        super().__init__(feat_dim, in_channels)
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        # Позиционные эмбеддинги (обучаемые)
        self.pos_embed = nn.Parameter(torch.randn(1, 256, embed_dim) * 0.02)  # макс. число патчей
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=4,
                                                   dim_feedforward=embed_dim*4,
                                                   activation='gelu', batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.out = nn.Linear(embed_dim, feat_dim)
        # Инициализация
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, noise: torch.Tensor) -> torch.Tensor:
        b, c, h, w = noise.shape
        # Проверка размера
        ph = h // self.patch_size
        pw = w // self.patch_size
        tokens = self.proj(noise)  # (B, embed_dim, ph, pw)
        tokens = tokens.flatten(2).transpose(1, 2)  # (B, N, embed_dim)
        N = tokens.shape[1]
        tokens = tokens + self.pos_embed[:, :N, :]
        tokens = self.transformer(tokens)  # (B, N, embed_dim)
        pooled = tokens.mean(dim=1)  # (B, embed_dim)
        return self.out(pooled)