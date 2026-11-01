"""Replay a synthetic teacher and measure the effect of global coefficients."""

import pickle

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from src.model.gas.models import get_gs_wrapper, load_base_model


def report(label, values):
    values = torch.tensor(values, dtype=torch.float64)
    print(f"{label}: n={len(values)} mean={values.mean():.8g} "
          f"median={values.median():.8g} max={values.max():.8g}")


@hydra.main(version_base=None, config_path="src/configs", config_name="generate_synt_prompt_dependent")
def main(config: DictConfig):
    config = OmegaConf.create(OmegaConf.to_container(config, resolve=True))
    path = config.outdir + "/dataset.pkl"
    with open(path, "rb") as f:
        data = pickle.load(f)  # Load only trusted datasets.

    n = len(data["noise"])
    train_size = 80
    validation_size = 20
    if n < train_size + validation_size:
        raise ValueError(f"Need at least {train_size + validation_size} samples, found {n}")
    steps = int(config.synthetic_gs_dataset.nfe)
    order = int(config.synthetic_gs_dataset.order)
    keys = [f"{side}{i}_diff" for side in ("a", "c") for i in range(1, order + 1)]
    params = {}
    for key in keys:
        value = data.get("manual_solver_params." + key)
        if not isinstance(value, torch.Tensor) or tuple(value.shape) != (n, steps):
            raise ValueError(f"Expected manual_solver_params.{key} shape {(n, steps)}")
        params[key] = value
    if not isinstance(data["latents"], torch.Tensor):
        raise ValueError("Dataset must contain latent tensors")

    # Mean is fitted on train only; validation remains held out.
    global_params = {key: value[:train_size].mean(dim=0, keepdim=True)
                     for key, value in params.items()}
    config.student_solver.steps = steps
    config.student_solver.order = order
    config.model.t_eps = config.student_solver.t_eps
    config.model.guidance_scale = config.student_solver.guidance_scale
    device = torch.device(config.device)
    model = load_base_model(config.model, device)
    wrapper = get_gs_wrapper(model, config)

    print(f"Dataset: {path}; samples={n}; teacher steps={steps}; order={order}")
    print("Global coefficients: mean over samples 0..79 only")
    print("Assumes the current generation config, SD checkpoint and code match those used to create the dataset.")
    print("Using batch size 1, matching the generation config.")
    with torch.no_grad():
        for split, indices in (("train", range(train_size)),
                               ("validation", range(n - validation_size, n))):
            replay_errors, global_errors, effects, max_errors = [], [], [], []
            for idx in indices:
                noise = data["noise"][idx:idx + 1].to(device)
                target = data["latents"][idx:idx + 1].to(device)
                condition = [data["condition"][idx]]
                oracle = {k: v[idx:idx + 1].to(device) for k, v in params.items()}
                common = {k: v.to(device) for k, v in global_params.items()}
                replay, _ = wrapper.student_sampler_fn(noise, condition=condition, manual_solver_params=oracle)
                global_latent, _ = wrapper.student_sampler_fn(noise, condition=condition, manual_solver_params=common)
                replay_errors.append((replay - target).abs().mean().item())
                max_errors.append((replay - target).abs().max().item())
                global_errors.append((global_latent - target).abs().mean().item())
                effects.append((global_latent - replay).abs().mean().item())
            print(f"\n{split}:")
            report("  oracle replay vs stored latent, L1", replay_errors)
            report("  oracle replay vs stored latent, max absolute error", max_errors)
            report("  train-mean coefficients vs stored latent, L1", global_errors)
            report("  train-mean coefficients vs oracle replay, L1", effects)


if __name__ == "__main__":
    main()
