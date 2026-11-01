"""Evaluate a checkpoint with correct and permuted predictor prompts."""
import os
import pickle

import hydra
import torch
from omegaconf import OmegaConf

from src.model.gas.models import get_gs_wrapper, load_base_model


@hydra.main(version_base=None, config_path="src/configs", config_name="gs_synt")
def main(cfg):
    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    checkpoint = os.environ.get("GAS_CHECKPOINT")
    if not checkpoint:
        raise ValueError("Set GAS_CHECKPOINT to the conditional checkpoint path")
    with open(cfg.datasets.teacher_pkl, "rb") as f:
        data = pickle.load(f)  # trusted dataset only
    n = len(data["noise"])
    val_size = cfg.datasets.validation_size
    indices = list(range(n - val_size, n))
    if val_size < 2 or cfg.datasets.dataset_shift + cfg.datasets.train_size > n - val_size:
        raise ValueError("Validation must contain at least two samples and not overlap train")
    device = torch.device(cfg.trainer.device)
    cfg.model.t_eps = cfg.student_solver.t_eps
    cfg.model.guidance_scale = cfg.student_solver.guidance_scale
    model = load_base_model(cfg.model, device)
    wrapper = get_gs_wrapper(model, cfg)
    if wrapper.ac_coeff_model is None:
        raise ValueError("Checkpoint must use conditional a/c parametrization")
    state = torch.load(checkpoint, map_location="cpu")
    wrapper.load_state_dict(state["model"], strict=True)
    original = wrapper._predict_ac_flat
    print(f"Checkpoint: {checkpoint}; saved step: {state.get('step', 'unknown')}")
    print(f"Dataset: {cfg.datasets.teacher_pkl}; validation samples: {val_size}")
    print("Permutation: cyclic shift by one across the complete validation set")
    for weights in ("raw", "ema"):
        if weights == "ema":
            wrapper.load_checkpoint(checkpoint)
        correct, permuted = [], []
        with torch.no_grad():
            for position, idx in enumerate(indices):
                other = indices[(position + 1) % len(indices)]
                noise = data["noise"][idx:idx+1].to(device)
                target = data["latents"][idx:idx+1].to(device)
                prompt = data["condition"][idx]
                other_prompt = data["condition"][other]
                model.set_condition([other_prompt])
                other_emb = model.model_fn.condition
                # Sampling sets the original SD conditioning; only the predictor
                # receives the other prompt's embedding.
                wrapper._predict_ac_flat = original
                latent, _ = wrapper.student_sampler_fn(noise, condition=[prompt])
                correct.append((latent - target).abs().mean().item())
                wrapper._predict_ac_flat = lambda x, emb: original(x, other_emb)
                latent, _ = wrapper.student_sampler_fn(noise, condition=[prompt])
                permuted.append((latent - target).abs().mean().item())
        wrapper._predict_ac_flat = original
        a, b = torch.tensor(correct), torch.tensor(permuted)
        diff = b - a
        print(f"{weights}: correct L1={a.mean():.8g}, permuted L1={b.mean():.8g}, "
              f"paired delta={diff.mean():+.8g}, median delta={diff.median():+.8g}, "
              f"worse={int((diff > 0).sum())}/{val_size}")


if __name__ == "__main__":
    main()
