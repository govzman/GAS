import hydra
from hydra.utils import instantiate

import torch
from omegaconf import DictConfig, OmegaConf
from torch_ema import ExponentialMovingAverage
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs

from src.model.gas.models import get_gs_wrapper, load_base_model
from src.model.gas.synt_data import SyntDataLoaders
from src.model.gas.utils.random import set_global_seed
from training import train


def _resolve_device(device_cfg: str) -> torch.device:
    if device_cfg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_cfg)


def split_parameters(gs_wrapper):
    a_params = []
    other_params = []

    for name, p in gs_wrapper.named_parameters():
        if not p.requires_grad:
            print('not require grad!', name)
            continue

        # a conditional model
        # if name.startswith("a_diff_model") or name.startswith("_a_bias"):
        if name.startswith("a") or name.startswith("_a"):
            a_params.append(p)
            print('a_params:', name)
        else:
            other_params.append(p)
            print('other_params:', name)

    print("A params:", len(a_params))
    print("Other params:", len(other_params))
    return a_params, other_params


@hydra.main(version_base=None, config_path="src/configs", config_name="config")
def main(config: DictConfig) -> None:
    # Freeze/resolve interpolations early (and make cfg printable).
    config = OmegaConf.create(OmegaConf.to_container(config, resolve=True))

    # Accelerator handles device / DDP / mixed precision / grad accumulation.
    # Keep device resolution only as a fallback for non-accelerate paths.
    device = _resolve_device(config.trainer.device)

    accumulation_steps = getattr(config.trainer, "accumulation_steps", None)
    if accumulation_steps is None:
        accumulation_steps = getattr(config.trainer, "iters_to_accumulate", 1)
    amp = getattr(config.trainer, "amp", "no")

    accelerator = Accelerator(
        gradient_accumulation_steps=accumulation_steps,
        mixed_precision=amp,
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)],
    )
    device = accelerator.device

    # Setup seed (kept separate from model/dataset config on purpose).
    set_global_seed(config.trainer.seed)

    # Auto-name run if not provided.
    solver_config = config.student_solver
    dataset_config = config.datasets
    loss_type = config.loss_config.loss_type
    if getattr(solver_config, "student_name", None) is None:
        solver_config.student_name = "_".join(
            f"{k}={v}" for k, v in solver_config.items() if k != "loss_config"
        )
    if config.writer.run_name is None:
        config.writer.run_name = f"{solver_config.student_name}_{dataset_config.teacher_pkl}_{loss_type}"

    # Setup dataset
    data = SyntDataLoaders(dataset_config)

    # Setup model
    model_config = config.model
    model_config.t_eps = solver_config.t_eps
    model_config.guidance_scale = solver_config.guidance_scale

    base_model = load_base_model(model_config, device)
    gs_wrapper = get_gs_wrapper(base_model, config)

    # Setup training
    a_params, other_params = split_parameters(gs_wrapper)
    optim = instantiate(config.optimizer, params=other_params) if other_params else None
    optim_a_params = instantiate(config.optimizer_a_params, params=a_params) if a_params else None
    
    ema = ExponentialMovingAverage(gs_wrapper.parameters(), decay=config.trainer.ema_decay)
    n_iters = config.trainer.n_iters
    config.trainer.epoch_num = n_iters // len(data.train_loader) + int(
        n_iters % len(data.train_loader) != 0
    )

    train(
        config=config,
        gs_wrapper=gs_wrapper,
        ema=ema,
        data=data,
        optim=optim,
        optim_a_params=optim_a_params,
        device=device,
        accelerator=accelerator,
    )


# ----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

# ----------------------------------------------------------------------------
