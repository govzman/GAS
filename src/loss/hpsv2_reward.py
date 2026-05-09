import threading
from typing import List, Sequence

import torch
from torch import nn
import torch.nn.functional as F


_HPS_MODEL = None
_HPS_LOCK = threading.Lock()


def _load_hps():
    global _HPS_MODEL

    with _HPS_LOCK:
        if _HPS_MODEL is not None:
            return _HPS_MODEL

        from hpsv2.src.open_clip import (
            create_model_and_transforms,
            get_tokenizer,
        )

        device = "cuda" if torch.cuda.is_available() else "cpu"

        model, _, _ = create_model_and_transforms(
            'ViT-H-14',
            'laion2B-s32B-b79K',
            precision='amp',
            device=device,
            jit=False,
            output_dict=True,
            with_score_predictor=False,
            with_region_predictor=False,
        )

        import huggingface_hub
        from hpsv2.utils import hps_version_map

        ckpt_path = huggingface_hub.hf_hub_download(
            "xswu/HPSv2",
            hps_version_map["v2.0"]
        )

        checkpoint = torch.load(ckpt_path, map_location=device)

        model.load_state_dict(checkpoint["state_dict"])

        tokenizer = get_tokenizer("ViT-H-14")

        model.eval()

        _HPS_MODEL = {
            "model": model,
            "tokenizer": tokenizer,
        }

        return _HPS_MODEL


class HPSv2RewardLoss(nn.Module):

    def __init__(
        self,
        reward_scale: float = 1.0,
        input_range: str = "auto",
    ):
        super().__init__()

        self.reward_scale = reward_scale
        self.input_range = input_range

        self.register_buffer(
            "mean",
            torch.tensor(
                [0.48145466, 0.4578275, 0.40821073]
            ).view(1, 3, 1, 1)
        )

        self.register_buffer(
            "std",
            torch.tensor(
                [0.26862954, 0.26130258, 0.27577711]
            ).view(1, 3, 1, 1)
        )

    def _to_01(self, x: torch.Tensor):
        if self.input_range == "minus_one_to_one":
            x = (x + 1.0) * 0.5

        elif self.input_range == "auto":
            if x.min() < 0:
                x = (x + 1.0) * 0.5

        return x.clamp(0, 1)

    def reward(
        self,
        images: torch.Tensor,
        prompts: Sequence[str],
    ):
        hps = _load_hps()

        model = hps["model"]
        tokenizer = hps["tokenizer"]

        x = self._to_01(images)

        # differentiable resize
        x = F.interpolate(
            x,
            size=(224, 224),
            mode="bicubic",
            align_corners=False,
        )

        # CLIP normalize
        x = (x - self.mean) / self.std

        text = tokenizer(list(prompts)).to(images.device)

        outputs = model(x, text)

        image_features = outputs["image_features"]
        text_features = outputs["text_features"]

        image_features = F.normalize(image_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)

        scores = torch.sum(
            image_features * text_features,
            dim=-1,
        )

        return scores

    def forward(
        self,
        images: torch.Tensor,
        prompts: List[str],
    ):
        reward = self.reward(images, prompts)

        return -reward.mean() * self.reward_scale