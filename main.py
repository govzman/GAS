import hydra
from hydra.utils import instantiate

import torch
from omegaconf import DictConfig, OmegaConf
from torch_ema import ExponentialMovingAverage

from src.models.gas.models import get_gs_wrapper, load_base_model
from src.models.gas.synt_data import SyntDataLoaders
from src.models.gas.utils.random import set_global_seed
from training import train


def _resolve_device(device_cfg: str) -> torch.device:
    if device_cfg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_cfg)


@hydra.main(version_base=None, config_path="src/configs", config_name="config")
def main(config: DictConfig) -> None:
    # Freeze/resolve interpolations early (and make cfg printable).
    config = OmegaConf.create(OmegaConf.to_container(config, resolve=True))

    device = _resolve_device(config.trainer.device)

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
    optim = instantiate(config.optimizer, params=gs_wrapper.parameters())
    
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
        device=device,
    )


# ----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

# ----------------------------------------------------------------------------
