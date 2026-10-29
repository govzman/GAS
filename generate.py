import os
import re
from functools import partial
from typing import Any, Dict, List, Optional, Tuple

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


def _make_generator(device: torch.device, seed: int) -> torch.Generator:
    return torch.Generator(device=device).manual_seed(int(seed) % (1 << 32))


class SyntheticCoefficientMapper:
    """Deterministic teacher a/c targets for controlled context experiments.

    ``constant`` implements the original GAS assumption: all samples share one
    coefficient table. ``prompt_projection`` uses one fixed scalar feature of
    the frozen SD text embedding, so the target depends on prompt and on nothing
    else. Only active solver entries are perturbed; inactive high-order entries
    remain zero because they cannot affect a trajectory.
    """

    def __init__(self, gs_wrapper, cfg: DictConfig, device: torch.device):
        self.steps = int(gs_wrapper.steps)
        self.order = int(gs_wrapper.order)
        self.mode = str(getattr(cfg, "coefficient_mode", "constant"))
        if self.mode not in {"constant", "prompt_projection"}:
            raise ValueError(f"Unknown synthetic coefficient_mode={self.mode!r}")

        seed = int(getattr(cfg, "coefficient_seed", 0))
        generator = _make_generator(device, seed)
        self.base_std = float(getattr(cfg, "base_std", 0.01))
        self.residual_std = float(getattr(cfg, "residual_std", 0.01))
        self.projection_gain = float(getattr(cfg, "projection_gain", 12.0))

        active = torch.zeros(self.order, self.steps, device=device)
        for order_idx in range(self.order):
            active[order_idx, order_idx:] = 1.0
        self.active = active

        self.base_a = torch.randn(self.order, self.steps, generator=generator, device=device)
        self.base_c = torch.randn(self.order, self.steps, generator=generator, device=device)
        self.direction_a = torch.randn(self.order, self.steps, generator=generator, device=device)
        self.direction_c = torch.randn(self.order, self.steps, generator=generator, device=device)
        self.projection = torch.randn(768, generator=generator, device=device)

        self.base_a = self.base_a * self.active * self.base_std
        self.base_c = self.base_c * self.active * self.base_std
        self.direction_a = self._unit_rms(self.direction_a * self.active)
        self.direction_c = self._unit_rms(self.direction_c * self.active)
        self.projection = self.projection / self.projection.norm().clamp_min(1e-12)

    @staticmethod
    def _unit_rms(x: torch.Tensor) -> torch.Tensor:
        norm = x.pow(2).mean().sqrt().clamp_min(1e-12)
        return x / norm

    def text_feature(self, cond_emb: torch.Tensor) -> torch.Tensor:
        """Bounded prompt-only scalar from mean-pooled CLIP token embeddings."""
        pooled = cond_emb.float().mean(dim=1)
        pooled = pooled / pooled.norm(dim=1, keepdim=True).clamp_min(1e-12)
        cosine = pooled @ self.projection
        return torch.tanh(self.projection_gain * cosine)

    def __call__(self, cond_emb: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        batch_size = cond_emb.shape[0]
        feature = self.text_feature(cond_emb)
        if self.mode == "constant":
            feature = torch.zeros_like(feature)

        a_all = self.base_a.unsqueeze(0) + feature[:, None, None] * self.residual_std * self.direction_a
        c_all = self.base_c.unsqueeze(0) + feature[:, None, None] * self.residual_std * self.direction_c
        params: Dict[str, torch.Tensor] = {}
        for i in range(self.order):
            params[f"a{i + 1}_diff"] = a_all[:, i, :]
            params[f"c{i + 1}_diff"] = c_all[:, i, :]
        return params, feature


def synthetic_noise_seeds(cfg: DictConfig) -> List[int]:
    count = int(getattr(cfg, "count", 100))
    mode = str(getattr(cfg, "noise_mode", "unique"))
    seed = int(getattr(cfg, "noise_seed", 0))
    if mode == "unique":
        return list(range(seed, seed + count))
    if mode == "fixed":
        return [seed] * count
    raise ValueError(f"Unknown synthetic noise_mode={mode!r}")


def synthetic_prompts(cfg: DictConfig, prompts_path: str, count: int) -> List[str]:
    prompts = parse_prompts_list(getattr(cfg, "prompts", None))
    if prompts is None:
        with open(prompts_path, "r", encoding="utf-8") as f:
            prompts = [line.strip() for line in f if line.strip()]
    if len(prompts) < count:
        raise ValueError(f"Need {count} distinct prompts, got {len(prompts)}")
    rng = np.random.default_rng(int(getattr(cfg, "prompt_seed", 0)))
    indices = rng.permutation(len(prompts))[:count]
    return [prompts[i] for i in indices]


# ----------------------------------------------------------------------------


@hydra.main(version_base=None, config_path="src/configs", config_name="generate")
def main(config: DictConfig):
    config = OmegaConf.create(OmegaConf.to_container(config, resolve=True))

    outdir = config.outdir
    noise_seeds = parse_int_list(config.seeds)
    max_batch_size = int(config.max_batch_size)
    # num_steps = config.num_steps
    checkpoint_path = config.checkpoint_path
    create_dataset = bool(config.create_dataset)
    device = torch.device(config.device)
    dist.init()

    sample_ids = list(range(len(noise_seeds)))

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
        noise_seeds = synthetic_noise_seeds(synthetic_cfg)
        sample_ids = list(range(len(noise_seeds)))

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

    shape = [None, model.image_channels, model.image_size, model.image_size]
    prompts_per_sample = None
    coefficient_mapper = None
    if use_synthetic_gs:
        prompts_path = getattr(model_config, "prompts_path", None)
        if prompts_path is None:
            raise ValueError("synthetic_gs_dataset needs model.prompts_path or explicit prompts.")
        prompts_per_sample = synthetic_prompts(
            synthetic_cfg, prompts_path, len(sample_ids)
        )
        coefficient_mapper = SyntheticCoefficientMapper(gs_wrapper, synthetic_cfg, device)

    num_batches = (
        (len(sample_ids) - 1) // (max_batch_size * dist.get_world_size()) + 1
    ) * dist.get_world_size()
    all_batches = torch.as_tensor(sample_ids).tensor_split(num_batches)
    rank_batches = all_batches[dist.get_rank() :: dist.get_world_size()]

    # Loop over batches.
    dist.print0(f'Generating {len(sample_ids)} images to "{outdir}"...')
    for shard_idx, batch_ids in tqdm.tqdm(
        enumerate(rank_batches), unit="batch", disable=(dist.get_rank() != 0)
    ):
        torch.distributed.barrier()

        batch_size = len(batch_ids)
        if batch_size == 0:
            continue
        batch_ids_list = batch_ids.tolist()
        batch_noise_seeds = [noise_seeds[i] for i in batch_ids_list]

        shape[0] = batch_size

        rnd = StackedRandomGenerator(device, batch_noise_seeds)
        noise = rnd.randn(shape, device=device)

        condition = None
        if model_config.conditional:
            if use_synthetic_gs:
                condition = [prompts_per_sample[i] for i in batch_ids_list]
            else:
                condition = model.iterate_condition(batch_noise_seeds)

        # Generate images.
        with torch.no_grad():
            manual_solver_params = None
            synthetic_feature = None
            if use_synthetic_gs:
                model.set_condition(condition)
                manual_solver_params, synthetic_feature = coefficient_mapper(
                    model.model_fn.condition
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
            if synthetic_feature is not None:
                dataset["synthetic_prompt_feature"] = synthetic_feature.detach().cpu()
            torch.save(dataset, os.path.join(synt_dir, f"{batch_ids_list[0]:06d}.pt"))

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

        for sample_id, image_np in zip(batch_ids_list, images_np):
            image_path = os.path.join(outdir, f"{sample_id:06d}.png")
            PIL.Image.fromarray(image_np, "RGB").save(image_path)

    # Done.
    torch.distributed.barrier()
    dist.print0("Done.")


# ----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

# ----------------------------------------------------------------------------
