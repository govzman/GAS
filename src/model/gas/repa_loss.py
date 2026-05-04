"""REPA (Representation Alignment): frozen DINOv2 multi-layer feature alignment."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

_DEFAULT_LAYER_INDICES: Dict[str, List[int]] = {
    "dinov2_vitb14": [3, 7, 11],
    "dinov2_vitl14": [8, 16, 23],
    "dinov2_vitg14": [13, 26, 39],
}

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


class DINOv2FeatureExtractor(nn.Module):
    """Frozen DINOv2 with forward hooks on selected transformer blocks."""

    def __init__(
        self,
        model_name: str,
        layer_indices: Optional[List[int]] = None,
        freeze: bool = True,
        pretrained: bool = True,
    ):
        super().__init__()
        if layer_indices is None:
            if model_name not in _DEFAULT_LAYER_INDICES:
                raise ValueError(
                    f"layer_indices is required for model_name={model_name!r}. "
                    f"Known defaults: {list(_DEFAULT_LAYER_INDICES)}"
                )
            layer_indices = list(_DEFAULT_LAYER_INDICES[model_name])
        self.model_name = model_name
        self.layer_indices = [int(i) for i in layer_indices]

        self.dino = torch.hub.load(
            "facebookresearch/dinov2",
            model_name,
            pretrained=pretrained,
            trust_repo=True,
        )
        n_blocks = len(self.dino.blocks)
        for idx in self.layer_indices:
            if idx < 0 or idx >= n_blocks:
                raise ValueError(
                    f"layer index {idx} out of range for {model_name} (num_blocks={n_blocks})"
                )

        if freeze:
            self.dino.eval()
            for p in self.dino.parameters():
                p.requires_grad = False

        self._hook_outputs: Dict[int, torch.Tensor] = {}
        self._hook_handles: List[Any] = []
        for layer_idx in self.layer_indices:
            block = self.dino.blocks[layer_idx]

            def _make_hook(li: int):
                def _hook(_module, _inp, out):
                    self._hook_outputs[li] = out

                return _hook

            h = block.register_forward_hook(_make_hook(layer_idx))
            self._hook_handles.append(h)

        mean = torch.tensor(_IMAGENET_MEAN).view(1, 3, 1, 1)
        std = torch.tensor(_IMAGENET_STD).view(1, 3, 1, 1)
        self.register_buffer("_imagenet_mean", mean, persistent=False)
        self.register_buffer("_imagenet_std", std, persistent=False)

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Map (B, 3, H, W) in [-1, 1] to DINOv2 tensor (B, 3, 224, 224) ImageNet-normalized."""
        x01 = x.clamp(-1.0, 1.0).mul(0.5).add(0.5)
        x224 = F.interpolate(x01, size=(224, 224), mode="bilinear", align_corners=False)
        mean = self._imagenet_mean.to(device=x224.device, dtype=x224.dtype)
        std = self._imagenet_std.to(device=x224.device, dtype=x224.dtype)
        return (x224 - mean) / std

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        self.dino.eval()
        self._hook_outputs.clear()
        x_in = self.preprocess(x)
        # Runs blocks; hooks populate _hook_outputs
        _ = self.dino.forward_features(x_in)
        out: Dict[str, torch.Tensor] = {}
        for idx in self.layer_indices:
            if idx not in self._hook_outputs:
                raise RuntimeError(f"DINOv2 hook missing output for block {idx}")
            out[f"block_{idx}"] = self._hook_outputs[idx]
        return out


def _pair_loss(
    s: torch.Tensor,
    t: torch.Tensor,
    loss_type: str,
    do_l2norm: bool,
) -> torch.Tensor:
    """s, t: same shape, last dim D. Return per-sample loss (B,)."""
    if do_l2norm:
        s = F.normalize(s, dim=-1, eps=1e-6)
        t = F.normalize(t, dim=-1, eps=1e-6)
    if loss_type == "cosine":
        if s.dim() == 2:
            cos = (s * t).sum(dim=-1).clamp(-1.0, 1.0)
            return 1.0 - cos
        # (B, N, D)
        cos = (s * t).sum(dim=-1).clamp(-1.0, 1.0)
        return 1.0 - cos.mean(dim=1)
    if loss_type == "mse":
        return (s - t).pow(2).mean(dim=tuple(range(1, s.dim())))
    if loss_type == "huber":
        return F.smooth_l1_loss(s, t, reduction="none").mean(dim=tuple(range(1, s.dim())))
    raise ValueError(f"Unknown loss_type={loss_type!r}")


def _branch_loss(
    s_feat: torch.Tensor,
    t_feat: torch.Tensor,
    use_cls: bool,
    use_patch: bool,
    loss_type: str,
    normalize_for_loss: bool,
    patch_start: int,
) -> torch.Tensor:
    """Per-sample (B,) combined CLS + patch loss for one layer."""
    b = s_feat.shape[0]
    device, dtype = s_feat.device, s_feat.dtype
    parts: List[torch.Tensor] = []
    if use_cls:
        parts.append(
            _pair_loss(
                s_feat[:, 0, :],
                t_feat[:, 0, :],
                loss_type,
                normalize_for_loss,
            )
        )
    if use_patch:
        parts.append(
            _pair_loss(
                s_feat[:, patch_start:, :],
                t_feat[:, patch_start:, :],
                loss_type,
                normalize_for_loss,
            )
        )
    if not parts:
        return torch.zeros(b, device=device, dtype=dtype)
    return torch.stack(parts, dim=0).mean(dim=0)


class FinalREPALoss(nn.Module):
    """Align student / teacher RGB images in frozen DINOv2 feature space."""

    def __init__(
        self,
        model_name: str,
        layer_indices: Optional[List[int]] = None,
        loss_type: str = "cosine",
        layer_weights: Optional[List[float]] = None,
        use_cls_token: bool = True,
        use_patch_tokens: bool = True,
        normalize_features: bool = True,
        pretrained: bool = True,
    ):
        super().__init__()
        if loss_type not in ("cosine", "mse", "huber"):
            raise ValueError(f"loss_type must be cosine|mse|huber, got {loss_type!r}")
        if not use_cls_token and not use_patch_tokens:
            raise ValueError("At least one of use_cls_token, use_patch_tokens must be True")
        self.loss_type = loss_type
        self.use_cls_token = use_cls_token
        self.use_patch_tokens = use_patch_tokens
        self.normalize_features = normalize_features

        self.extractor = DINOv2FeatureExtractor(
            model_name=model_name,
            layer_indices=layer_indices,
            freeze=True,
            pretrained=pretrained,
        )
        idxs = list(self.extractor.layer_indices)
        if layer_weights is None:
            if len(idxs) == 3:
                lw = [0.25, 0.3, 0.45]
            else:
                lw = [1.0 / len(idxs)] * len(idxs)
        else:
            lw = list(layer_weights)
        if len(lw) != len(idxs):
            raise ValueError(f"layer_weights len {len(lw)} != num layers {len(idxs)}")
        self.register_buffer("_layer_weights", torch.tensor(lw, dtype=torch.float32))

    def forward(
        self,
        student_images: torch.Tensor,
        teacher_images: torch.Tensor,
        return_detailed: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        if student_images.shape != teacher_images.shape:
            raise ValueError(
                f"shape mismatch student {student_images.shape} vs teacher {teacher_images.shape}"
            )
        normalize_for_loss = self.normalize_features or self.loss_type == "cosine"

        with torch.set_grad_enabled(student_images.requires_grad):
            fs = self.extractor(student_images)
        with torch.no_grad():
            ft = self.extractor(teacher_images)

        n_reg = int(getattr(self.extractor.dino, "num_register_tokens", 0) or 0)
        patch_start = 1 + n_reg

        w = self._layer_weights.to(device=student_images.device, dtype=torch.float32)
        total = torch.zeros(student_images.shape[0], device=student_images.device, dtype=student_images.dtype)
        info: Optional[Dict[str, torch.Tensor]] = {} if return_detailed else None

        for k, idx in enumerate(self.extractor.layer_indices):
            layer_name = f"block_{idx}"
            s = fs[layer_name]
            t = ft[layer_name]
            weight = float(w[k].item())
            layer_loss = _branch_loss(
                s,
                t,
                self.use_cls_token,
                self.use_patch_tokens,
                self.loss_type,
                normalize_for_loss,
                patch_start,
            )
            total = total + weight * layer_loss.to(total.dtype)
            if info is not None:
                info[f"repa/{layer_name}"] = layer_loss.detach().mean()

        mean_loss = total.mean()
        if info is not None:
            info["repa/total"] = mean_loss.detach()
        return mean_loss, info
