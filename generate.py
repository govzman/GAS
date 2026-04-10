import os
import re
from functools import partial

import hydra
import numpy as np
import PIL.Image
import torch
import tqdm
from omegaconf import DictConfig, OmegaConf

from model.gas.models import get_gs_wrapper, load_base_model
from model.gas.sampling_algs import SAMPLING_ALGS
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


# ----------------------------------------------------------------------------


@hydra.main(version_base=None, config_path="conf", config_name="generate")
def main(cfg: DictConfig):
    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))

    outdir = cfg.outdir
    seeds = parse_int_list(cfg.seeds)
    max_batch_size = int(cfg.max_batch_size)
    num_steps = cfg.num_steps
    checkpoint_path = cfg.checkpoint_path
    create_dataset = bool(cfg.create_dataset)
    device = torch.device(cfg.device)
    dist.init()

    num_batches = (
        (len(seeds) - 1) // (max_batch_size * dist.get_world_size()) + 1
    ) * dist.get_world_size()
    all_batches = torch.as_tensor(seeds).tensor_split(num_batches)
    rank_batches = all_batches[dist.get_rank() :: dist.get_world_size()]

    # Rank 0 goes first.
    if dist.get_rank() != 0:
        torch.distributed.barrier()

    config = cfg.config

    if create_dataset:
        synt_dir = os.path.join(outdir, "dataset")
        os.makedirs(synt_dir, exist_ok=True)

    outdir = os.path.join(outdir, "images")
    os.makedirs(outdir, exist_ok=True)

    # Prepare configs and generation settings.
    gs_solver = checkpoint_path is not None
    model_config = config.model
    solver_config = (
        config.student_solver_config if gs_solver else config.teacher_solver_config
    )

    assert (num_steps is None) != (
        solver_config.steps is None
    ), "Students steps should be specified in one and only one of both generate script and solver config"

    # Load base model.
    model_config.t_eps = solver_config.t_eps
    model_config.guidance_scale = solver_config.guidance_scale

    model = load_base_model(model_config, device)

    # Generating using GS checkpoint.
    if gs_solver:
        solver_config.loss_config.loss_type = "GS"
        solver_config.steps = num_steps
        solver_config.order = num_steps

        gs_wrapper = get_gs_wrapper(model, solver_config)
        gs_wrapper.load_checkpoint(checkpoint_path=checkpoint_path)
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

    shape = [None, model.image_channels, model.image_size, model.image_size]

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
        rnd = StackedRandomGenerator(device, batch_seeds)
        noise = rnd.randn(shape, device=device)

        condition = None
        if model_config.conditional:
            condition = model.iterate_condition(batch_seeds.tolist())

        # Generate images.
        with torch.no_grad():
            latents, images = sampler_fn(noise=noise, condition=condition)

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
