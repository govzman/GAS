"""Evaluate coefficient-supervised checkpoint against a train-mean baseline."""
import os
import pickle

import hydra
import torch
from omegaconf import OmegaConf

from src.model.gas.models import get_gs_wrapper, load_base_model


def correlation(x, y):
    x, y = x.double(), y.double()
    x, y = x - x.mean(), y - y.mean()
    denom = x.norm() * y.norm()
    return float((x @ y / denom).item()) if denom > 1e-15 else float("nan")


@hydra.main(version_base=None, config_path="src/configs", config_name="gs_synt")
def main(cfg):
    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    checkpoint = os.environ.get("GAS_CHECKPOINT")
    if not checkpoint:
        raise ValueError("Set GAS_CHECKPOINT to the checkpoint path")
    with open(cfg.datasets.teacher_pkl, "rb") as f:
        data = pickle.load(f)  # Only load trusted datasets.
    count = len(data["noise"])
    start = int(cfg.datasets.dataset_shift)
    train = list(range(start, start + int(cfg.datasets.train_size)))
    val = list(range(count - int(cfg.datasets.validation_size), count))
    if not train or not val or train[-1] >= val[0]:
        raise ValueError("Train and validation indices are empty or overlap")
    feature = data.get("synthetic_prompt_feature")
    if not isinstance(feature, torch.Tensor) or feature.numel() != count:
        raise ValueError("Expected synthetic_prompt_feature for each sample")
    steps, order = int(cfg.student_solver.steps), int(cfg.student_solver.order)
    keys = [f"{side}{i}_diff" for side in ("a", "c") for i in range(1, order + 1)]
    target = {}
    for key in keys:
        value = data.get("manual_solver_params." + key)
        if not isinstance(value, torch.Tensor) or tuple(value.shape) != (count, steps):
            raise ValueError(f"Expected {key} shape {(count, steps)}")
        target[key] = value.float()
    baseline = {key: value[train].mean(dim=0) for key, value in target.items()}

    device = torch.device(cfg.trainer.device)
    cfg.model.t_eps = cfg.student_solver.t_eps
    cfg.model.guidance_scale = cfg.student_solver.guidance_scale
    model = load_base_model(cfg.model, device)
    wrapper = get_gs_wrapper(model, cfg)
    if wrapper.ac_coeff_model is None:
        raise ValueError("Expected conditional a/c predictor")
    state = torch.load(checkpoint, map_location="cpu")
    wrapper.load_state_dict(state["model"], strict=True)
    print(f"Checkpoint: {checkpoint}; step={state.get('step', 'unknown')}")
    print(f"Dataset: {cfg.datasets.teacher_pkl}; train={len(train)}, validation={len(val)}")
    print("Baseline: coefficient mean fitted only on train; active elements only")

    for weight_type in ("raw", "ema"):
        if weight_type == "ema":
            wrapper.load_checkpoint(checkpoint)
        for split, indices in (("train", train), ("validation", val)):
            predictions = {k: [] for k in keys}
            latent_l1 = []
            with torch.no_grad():
                for idx in indices:
                    noise = data["noise"][idx:idx+1].to(device)
                    latent, _ = wrapper.student_sampler_fn(noise, condition=[data["condition"][idx]])
                    latent_l1.append((latent - data["latents"][idx:idx+1].to(device)).abs().mean().item())
                    for key in keys:
                        predictions[key].append(getattr(wrapper.solver, key)[0].detach().float().cpu())
            pred_active, gt_active, mean_active = [], [], []
            per_coeff = []
            for key in keys:
                order_index = int(key[1:-5]) - 1
                pred = torch.stack(predictions[key])[:, order_index:]
                gt = target[key][indices, order_index:]
                base = baseline[key][order_index:].expand_as(gt)
                pred_active.append(pred)
                gt_active.append(gt)
                mean_active.append(base)
                for step in range(order_index, steps):
                    p = torch.stack(predictions[key])[:, step]
                    t = target[key][indices, step]
                    per_coeff.append((key, step, correlation(feature[indices], p),
                                      correlation(feature[indices], t),
                                      float((p-t).square().mean())))
            pred = torch.cat(pred_active, dim=1)
            gt = torch.cat(gt_active, dim=1)
            base = torch.cat(mean_active, dim=1)
            print(f"\n{weight_type} / {split}:")
            print(f"  predictor active coefficient MSE={(pred-gt).square().mean().item():.9g}")
            print(f"  train-mean baseline active coefficient MSE={(base-gt).square().mean().item():.9g}")
            print(f"  predictor latent L1={sum(latent_l1)/len(latent_l1):.9g}")
            print("  coefficient step: corr(feature,pred), corr(feature,target), MSE")
            for key, step, pred_corr, target_corr, mse in per_coeff:
                print(f"    {key}[{step}]: {pred_corr:+.5f}, {target_corr:+.5f}, {mse:.9g}")


if __name__ == "__main__":
    main()
