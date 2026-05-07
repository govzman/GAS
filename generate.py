import os
import re
from functools import partial
from typing import Any, Dict, List, Optional

import hydra
import numpy as np
import PIL.Image
import torch
import tqdm
from omegaconf import DictConfig, OmegaConf

from src.model.gas.models import get_gs_wrapper, load_base_model
from src.model.gas.sampling_algs import SAMPLING_ALGS
from torch_utils import distributed as dist


def custom_to_np(x: torch.Tensor) -> np.array:
    # saves the batch in adm style as in https://github.com/openai/guided-diffusion/blob/main/scripts/image_sample.py
    sample = x.detach().cpu()
    sample = ((sample + 1) * 127.5).clamp(0, 255).to(torch.uint8)
    sample = sample.permute(0, 2, 3, 1)
    sample = sample.numpy()
    return sample


# ----------------------------------------------------------------------------
# Wrapper for torch.Generator that allows specifying a different random seed
# for each sample in a minibatch.


class StackedRandomGenerator:
    def __init__(self, device, seeds):
        super().__init__()
        self.generators = [
            torch.Generator(device).manual_seed(int(seed) % (1 << 32)) for seed in seeds
        ]

    def randn(self, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack(
            [torch.randn(size[1:], generator=gen, **kwargs) for gen in self.generators]
        )

    def randn_like(self, input):
        return self.randn(
            input.shape, dtype=input.dtype, layout=input.layout, device=input.device
        )


# ----------------------------------------------------------------------------
# Parse a comma separated list of numbers or ranges and return a list of ints.
# Example: '1,2,5-10' returns [1, 2, 5, 6, 7, 8, 9, 10]


def parse_int_list(s):
    if isinstance(s, list):
        return s
    ranges = []
    range_re = re.compile(r"^(\d+)-(\d+)$")
    for p in s.split(","):
        m = range_re.match(p)
        if m:
            ranges.extend(range(int(m.group(1)), int(m.group(2)) + 1))
        else:
            ranges.append(int(p))
    return ranges


def parse_prompts_list(prompts: Optional[Any]) -> Optional[List[str]]:
    if prompts is None:
        return None
    if isinstance(prompts, list):
        return [str(p).strip() for p in prompts if str(p).strip()]
    return [p.strip() for p in str(prompts).split("|") if p.strip()]


def sample_manual_gs_params(
    gs_wrapper,
    batch_size: int,
    device: torch.device,
    cfg: DictConfig,
) -> Dict[str, torch.Tensor]:
    coef_std = float(getattr(cfg, "coef_std", 0.01))
    t_couple_std = float(getattr(cfg, "t_couple_std", coef_std))
    t_std = float(getattr(cfg, "t_std", coef_std))
    include_t_couple = bool(getattr(cfg, "include_t_couple", True))
    include_timesteps = bool(getattr(cfg, "include_timesteps", False))

    steps = int(gs_wrapper.steps)
    order = int(gs_wrapper.order)
    params = {}

    for i in range(1, order + 1):
        params[f"a{i}_diff"] = torch.randn(batch_size, steps, device=device) * coef_std
        params[f"c{i}_diff"] = torch.randn(batch_size, steps, device=device) * coef_std

    if include_t_couple:
        params["t_couple"] = torch.randn(batch_size, steps, device=device) * t_couple_std

    if include_timesteps:
        base_t = torch.linspace(1.0, gs_wrapper.t_eps, steps + 1, device=device).flip(0)
        base_logits = gs_wrapper.get_inv_t_steps(base_t).reshape(1, -1).repeat(batch_size, 1)
        logits = base_logits + torch.randn_like(base_logits) * t_std
        params["timesteps"] = gs_wrapper.get_mu_t_steps(logits).flip(1)

    return params


# ----------------------------------------------------------------------------


@hydra.main(version_base=None, config_path="src/configs", config_name="generate")
def main(config: DictConfig):
    config = OmegaConf.create(OmegaConf.to_container(config, resolve=True))

    outdir = config.outdir
    seeds = parse_int_list(config.seeds)
    max_batch_size = int(config.max_batch_size)
    # num_steps = config.num_steps
    checkpoint_path = config.checkpoint_path
    create_dataset = bool(config.create_dataset)
    device = torch.device(config.device)
    dist.init()

    num_batches = (
        (len(seeds) - 1) // (max_batch_size * dist.get_world_size()) + 1
    ) * dist.get_world_size()
    all_batches = torch.as_tensor(seeds).tensor_split(num_batches)
    rank_batches = all_batches[dist.get_rank() :: dist.get_world_size()]

    # Rank 0 goes first.
    if dist.get_rank() != 0:
        torch.distributed.barrier()

    if create_dataset:
        synt_dir = os.path.join(outdir, "dataset")
        os.makedirs(synt_dir, exist_ok=True)

    outdir = os.path.join(outdir, "images")
    os.makedirs(outdir, exist_ok=True)

    # Prepare configs and generation settings.
    model_config = config.model
    synthetic_cfg = config.get("synthetic_gs_dataset", None)
    use_synthetic_gs = bool(getattr(synthetic_cfg, "enabled", False))
    gs_solver = checkpoint_path is not None or use_synthetic_gs
    solver_config = (
        config.student_solver if gs_solver else config.teacher_solver
    )

    if use_synthetic_gs:
        synthetic_count = int(getattr(synthetic_cfg, "count", 100))
        fixed_noise_seed = int(getattr(synthetic_cfg, "fixed_noise_seed", 0))
        seeds = list(range(synthetic_count))
        num_batches = (
            (len(seeds) - 1) // (max_batch_size * dist.get_world_size()) + 1
        ) * dist.get_world_size()
        all_batches = torch.as_tensor(seeds).tensor_split(num_batches)
        rank_batches = all_batches[dist.get_rank() :: dist.get_world_size()]

    num_steps = solver_config.steps # !
    # assert (num_steps is None) != (
    #     solver_config.steps is None
    # ), "Students steps should be specified in one and only one of both generate script and solver config"

    # Load base model.
    model_config.t_eps = solver_config.t_eps
    model_config.guidance_scale = solver_config.guidance_scale

    model = load_base_model(model_config, device)

    # Generating using GS checkpoint.
    if gs_solver:
        config.loss_config.loss_type = "GS"
        if use_synthetic_gs:
            nfe = int(getattr(synthetic_cfg, "nfe", 4))
            # For NFE=K, use coefficients up to K-1 by default (a1..a(K-1), c1..c(K-1)).
            # This can be overridden via synthetic_gs_dataset.order.
            synth_order = int(getattr(synthetic_cfg, "order", max(1, nfe - 1)))
            solver_config.steps = nfe
            solver_config.order = min(synth_order, nfe)
        else:
            solver_config.steps = num_steps
            # Keep configured order (e.g. teacher order=3) unless it exceeds steps.
            solver_config.order = min(int(solver_config.order), int(solver_config.steps))

        gs_wrapper = get_gs_wrapper(model, config)
        if checkpoint_path is not None:
            gs_wrapper.load_checkpoint(checkpoint_path=checkpoint_path)
        else:
            dist.print0(
                "Running GS generation without checkpoint: using wrapper initialization "
                "and manual synthetic coefficients."
            )
        sampler_fn = partial(gs_wrapper.student_sampler_fn, decode=True)

    # Generating images with UniPC/iPNDM solvers.
    else:
        if num_steps is not None:
            solver_config.steps = num_steps

        sampler_fn = SAMPLING_ALGS[model_config.type]
        sampler_fn = partial(sampler_fn, model=model, solver_config=solver_config)

    # Other ranks follow.
    if dist.get_rank() == 0:
        torch.distributed.barrier()

    dist.print0(
        f"Generation setup: gs_solver={gs_solver}, synthetic_gs={use_synthetic_gs}, "
        f"steps(NFE)={solver_config.steps}, order={solver_config.order}, outdir={outdir}"
    )

    shape = [None, model.image_channels, model.image_size, model.image_size]
    fixed_noise = None
    prompts_override = None
    if use_synthetic_gs:
        fixed_noise_gen = torch.Generator(device=device).manual_seed(
            fixed_noise_seed % (1 << 32)
        )
        fixed_noise = torch.randn(
            [1, model.image_channels, model.image_size, model.image_size],
            generator=fixed_noise_gen,
            device=device,
        )

        prompts_override = parse_prompts_list(getattr(synthetic_cfg, "prompts", None))
        if prompts_override is None:
            prompts_path = getattr(model_config, "prompts_path", None)
            if prompts_path is None:
                raise ValueError("synthetic_gs_dataset needs prompts or model.prompts_path.")
            with open(prompts_path, "r", encoding="utf-8") as f:
                prompts_override = [line.strip() for line in f if line.strip()]
        if len(prompts_override) < len(seeds):
            raise ValueError(
                f"Not enough prompts ({len(prompts_override)}) for {len(seeds)} samples."
            )
        prompt_seed = getattr(synthetic_cfg, "prompt_seed", None)
        if prompt_seed is not None:
            prompt_rng = np.random.default_rng(int(prompt_seed))
            perm = prompt_rng.permutation(len(prompts_override))
            prompts_override = [prompts_override[i] for i in perm]
        prompts_override = prompts_override[: len(seeds)]

    # Loop over batches.
    dist.print0(f'Generating {len(seeds)} images to "{outdir}"...')
    for batch_seeds in tqdm.tqdm(
        rank_batches, unit="batch", disable=(dist.get_rank() != 0)
    ):
        torch.distributed.barrier()

        batch_size = len(batch_seeds)
        if batch_size == 0:
            continue

        shape[0] = batch_size

        # Pick latents and labels.
        if use_synthetic_gs:
            noise = fixed_noise.repeat(batch_size, 1, 1, 1)
        else:
            rnd = StackedRandomGenerator(device, batch_seeds)
            noise = rnd.randn(shape, device=device)

        condition = None
        if model_config.conditional:
            if use_synthetic_gs:
                condition = [prompts_override[int(i)] for i in batch_seeds.tolist()]
            else:
                condition = model.iterate_condition(batch_seeds.tolist())

        # Generate images.
        with torch.no_grad():
            manual_solver_params = None
            if use_synthetic_gs:
                manual_solver_params = sample_manual_gs_params(
                    gs_wrapper=gs_wrapper,
                    batch_size=batch_size,
                    device=device,
                    cfg=synthetic_cfg,
                )
            latents, images = sampler_fn(
                noise=noise,
                condition=condition,
                manual_solver_params=manual_solver_params,
            )

        if create_dataset:
            latents = [None] * batch_size if latents is None else latents
            condition = [None] * batch_size if condition is None else condition

            dataset = {
                "noise": noise.detach().cpu(),
                "latents": (
                    latents.detach().cpu()
                    if isinstance(latents, torch.Tensor)
                    else latents
                ),
                "images": images.detach().cpu(),
                "condition": (
                    condition.detach().cpu()
                    if isinstance(condition, torch.Tensor)
                    else condition
                ),
                "manual_solver_params": {
                    k: (v.detach().cpu() if isinstance(v, torch.Tensor) else v)
                    for k, v in (manual_solver_params or {}).items()
                },
            }
            torch.save(dataset, os.path.join(synt_dir, f"{batch_seeds[0]}.pt"))

        # Save images.
        if model_config.type == "EDM":
            # Saves the batch in EDM style as in https://github.com/NVlabs/edm/blob/main/generate.py
            images_np = (
                (images * 127.5 + 128)
                .clip(0, 255)
                .to(torch.uint8)
                .permute(0, 2, 3, 1)
                .cpu()
                .numpy()
            )
        else:
            # Saves the batch in LDM style as in https://github.com/CompVis/latent-diffusion/blob/main/scripts/sample_diffusion.py
            images_np = custom_to_np(images)

        for seed, image_np in zip(batch_seeds, images_np):
            image_path = os.path.join(outdir, f"{seed:06d}.png")
            PIL.Image.fromarray(image_np, "RGB").save(image_path)

    # Done.
    torch.distributed.barrier()
    dist.print0("Done.")


# ----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

# ----------------------------------------------------------------------------
