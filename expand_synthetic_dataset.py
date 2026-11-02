"""Extend the existing 80/20 synthetic dataset to 500/20, preserving validation."""
import os
import pickle

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf

from generate import StackedRandomGenerator, SyntheticCoefficientMapper
from src.model.gas.models import get_gs_wrapper, load_base_model


@hydra.main(version_base=None, config_path="src/configs", config_name="generate_synt_prompt_dependent")
def main(config):
    config = OmegaConf.create(OmegaConf.to_container(config, resolve=True))
    old_path = "data/teachers/sd-v1/synthetic/prompt_dependent/dataset.pkl"
    output_path = "data/teachers/sd-v1/synthetic/prompt_dependent_500/dataset.pkl"
    if os.path.exists(output_path):
        raise FileExistsError(f"Refusing to overwrite {output_path}")
    with open(old_path, "rb") as f:
        old = pickle.load(f)  # Only load trusted datasets.
    if len(old["noise"]) != 100 or len(old["condition"]) != 100:
        raise ValueError("Expected the original 100-sample dataset with 80 train and 20 validation")
    required = [f"manual_solver_params.{side}{i}_diff" for side in ("a", "c") for i in range(1, 4)]
    required += ["noise", "images", "latents", "condition", "synthetic_prompt_feature"]
    if set(old) != set(required):
        raise ValueError(f"Unexpected original dataset keys: {set(old) ^ set(required)}")
    if config.checkpoint_path is not None:
        raise ValueError("Expected synthetic teacher without a checkpoint")

    cfg = config.synthetic_gs_dataset
    if cfg.coefficient_mode != "prompt_projection" or int(cfg.nfe) != 4 or int(cfg.order) != 3:
        raise ValueError("Expected prompt-projection teacher with 4 steps and order 3")
    config.student_solver.steps = int(cfg.nfe)
    config.student_solver.order = int(cfg.order)
    config.model.t_eps = config.student_solver.t_eps
    config.model.guidance_scale = config.student_solver.guidance_scale
    device = torch.device(config.device)
    model = load_base_model(config.model, device)
    wrapper = get_gs_wrapper(model, config)
    mapper = SyntheticCoefficientMapper(wrapper, cfg, device)

    # Validate the exact teacher mapping on all original prompts before any output.
    with torch.no_grad():
        for i in range(100):
            model.set_condition([old["condition"][i]])
            params, feature = mapper(model.model_fn.condition)
            torch.testing.assert_close(feature.cpu(), old["synthetic_prompt_feature"][i:i+1], rtol=1e-4, atol=1e-6)
            for key, value in params.items():
                torch.testing.assert_close(value.cpu(), old["manual_solver_params." + key][i:i+1], rtol=1e-4, atol=1e-6)
    print("Verified teacher mapping on all 100 original prompts.", flush=True)

    with open(config.model.prompts_path, encoding="utf-8") as f:
        prompts = [line.strip() for line in f if line.strip()]
    used = set(old["condition"])
    rng = np.random.default_rng(2026)
    new_prompts = []
    for idx in rng.permutation(len(prompts)):
        prompt = prompts[idx]
        if prompt not in used:
            new_prompts.append(prompt)
            used.add(prompt)
            if len(new_prompts) == 420:
                break
    if len(new_prompts) != 420:
        raise ValueError("Not enough new unique prompts")

    new = {key: [] for key in required}
    shape = [1, model.image_channels, model.image_size, model.image_size]
    with torch.no_grad():
        for i, prompt in enumerate(new_prompts):
            noise = StackedRandomGenerator(device, [20000 + i]).randn(shape, device=device)
            model.set_condition([prompt])
            params, feature = mapper(model.model_fn.condition)
            latent, image = wrapper.student_sampler_fn(noise, condition=[prompt], decode=True, manual_solver_params=params)
            for key, value in (("noise", noise), ("images", image), ("latents", latent), ("synthetic_prompt_feature", feature)):
                new[key].append(value.cpu())
            new["condition"].append(prompt)
            for key, value in params.items():
                new["manual_solver_params." + key].append(value.cpu())
            if (i + 1) % 25 == 0:
                print(f"Generated {i + 1}/420", flush=True)

    result = {}
    for key in required:
        if key == "condition":
            result[key] = old[key][:80] + new[key] + old[key][-20:]
        else:
            result[key] = torch.cat((old[key][:80], torch.cat(new[key], dim=0), old[key][-20:]), dim=0)
        assert len(result[key]) == 520
        if key != "condition":
            assert torch.equal(result[key][-20:], old[key][-20:]), key
        else:
            assert result[key][-20:] == old[key][-20:]
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "xb") as f:
        pickle.dump(result, f)
    print(f"Saved {output_path}: 500 train, original 20 validation", flush=True)


if __name__ == "__main__":
    main()
