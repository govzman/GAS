import threading
from typing import List, Optional, Sequence

import torch
from torch import nn


_HPS_MODEL = None
_HPS_LOCK = threading.Lock()


def _load_hps_model():
    global _HPS_MODEL
    with _HPS_LOCK:
        if _HPS_MODEL is not None:
            return _HPS_MODEL
        try:
            import hpsv2  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "hpsv2 is required for reward training. Install it before enabling loss_config.hpsv2.enabled=true."
            ) from exc
        _HPS_MODEL = hpsv2
        return _HPS_MODEL


class HPSv2RewardLoss(nn.Module):
    """Negative HPSv2 reward as scalar loss."""

    def __init__(self, reward_scale: float = 1.0, input_range: str = "auto"):
        super().__init__()
        self.reward_scale = float(reward_scale)
        self.input_range = input_range

    def _to_hps_range(self, images: torch.Tensor) -> torch.Tensor:
        x = images
        if self.input_range == "minus_one_to_one":
            x = (x + 1.0) * 0.5
        elif self.input_range == "zero_to_one":
            x = x
        else:
            # auto: assume [-1, 1] if negatives are present.
            if torch.any(x < 0):
                x = (x + 1.0) * 0.5
        return x.clamp(0.0, 1.0)

    def reward(
        self,
        images: torch.Tensor,
        prompts: Sequence[str],
        grad: bool = True,
    ) -> torch.Tensor:
        if not isinstance(prompts, (list, tuple)):
            raise TypeError(f"prompts must be list/tuple[str], got {type(prompts)}")
        if len(prompts) != images.shape[0]:
            raise ValueError(f"prompts length {len(prompts)} != batch {images.shape[0]}")

        hps = _load_hps_model()
        x = self._to_hps_range(images)
        x = x.to(device=images.device, dtype=images.dtype)
        scores = hps.score(x, list(prompts))
        if not isinstance(scores, torch.Tensor):
            scores = torch.as_tensor(scores, device=images.device, dtype=images.dtype)
        scores = scores.reshape(-1).to(device=images.device, dtype=images.dtype)
        if not grad:
            scores = scores.detach()
        return scores

    def forward(self, images: torch.Tensor, prompts: List[str]) -> torch.Tensor:
        reward = self.reward(images=images, prompts=prompts, grad=True)
        return -reward.mean() * self.reward_scale
