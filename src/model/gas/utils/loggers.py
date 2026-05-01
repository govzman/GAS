import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.utils import make_grid

import comet_ml
from src.model.gas.gs_wrapper import GSWrapper


def log_plt_fig(exp: comet_ml.Experiment, fig, key: str, global_step: int) -> None:
    fig.tight_layout()

    exp.log_figure(
        figure=fig,
        figure_name=key,
        step=global_step,
    )
    # comet_ml.log_metrics({key: comet_ml.Image(fig)}, step=global_step)
    plt.close("all")


@torch.no_grad()
def log_t_steps_plot(
    exp: comet_ml.Experiment, t_steps: torch.Tensor, global_step: int = None, key: str = None
) -> None:
    t_steps = t_steps.detach().cpu()

    # Normalize shapes:
    # - (S,) -> (S, 1)
    # - (B, S) -> (S, B)
    # - (S, B) -> (S, B)
    if t_steps.ndim == 1:
        t_steps = t_steps[:, None]
    elif t_steps.ndim == 2:
        # Heuristic: if first dim looks like batch (usually small) and second like steps (usually >= 2)
        if t_steps.shape[0] < t_steps.shape[1]:
            t_steps = t_steps.T
    else:
        # Unexpected, just flatten batch dims into one and keep steps dimension
        t_steps = t_steps.reshape(t_steps.shape[0], -1)

    t_steps = t_steps.numpy()
    steps, batch_size = t_steps.shape

    fig, ax = plt.subplots(1, 1, figsize=(4, 4))

    # Рисуем каждую траекторию из батча
    for i in range(batch_size):
        color = plt.cm.tab10(i % 10)  # циклические цвета
        ax.plot(t_steps[:, i], alpha=0.5, linestyle='--', color=color)

    # Рисуем усреднённую линию
    mean_vals = t_steps.mean(axis=1)
    ax.plot(mean_vals, color='black', linewidth=2, label='Mean')

    ax.set_xlabel("Step")
    ax.set_ylabel("Time")
    ax.grid(True)
    ax.legend()

    if global_step is not None:
        log_plt_fig(exp=exp, fig=fig, key=key, global_step=global_step)


@torch.no_grad()
def vis_grid(a: torch.Tensor, ax=None) -> None:
    a = a.detach().cpu()

    nrow = int(np.around(np.sqrt(a.shape[0])))
    a = make_grid(a, nrow=nrow).permute(1, 2, 0).numpy()
    a = a / 2 + 0.5
    a = np.clip(a, 0, 1)
    if ax is None:
        plt.imshow(a)
    else:
        ax.imshow(a)


@torch.no_grad()
def log_end_img(
    exp: comet_ml.Experiment, x_s: torch.Tensor, x_t: torch.Tensor, global_step: int = None, key: str = None
) -> None:
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    vis_grid(x_s, ax=ax[0])
    ax[0].axis("off")
    ax[0].set_title("Student")

    vis_grid(x_t, ax=ax[1])
    ax[1].axis("off")
    ax[1].set_title("Teacher")

    if global_step is None:
        return

    log_plt_fig(exp=exp, fig=fig, key=key, global_step=global_step)


@torch.no_grad()
def log_weights(exp: comet_ml.Experiment, model: GSWrapper, global_step: int, suff: str = "") -> None:
    d = {}
    key = f"weights_stats{suff}"

    for t, p in model.named_parameters():
        if p.requires_grad:
            data = p.data.detach().clone().cpu().numpy()
            if data.ndim == 0:
                d[f"{key}_{t}/scalar"] = float(data)
                continue
            if np.prod(data.shape) > 12:
                d[f"{key}/{t}_norm"] = np.linalg.norm(data)
                continue
            for i, v in enumerate(data):
                d[f"{key}_{t}/{i:02d}"] = v

    exp.log_metrics(d, step=global_step)


@torch.no_grad()
def log_grads(exp: comet_ml.Experiment, model: GSWrapper, global_step: int) -> None:
    d = {}
    key = "grads_stats"
    for t, p in model.named_parameters():
        if p.requires_grad and p.grad is not None:
            data = p.grad.detach().clone().cpu().numpy()
            if data.ndim == 0:
                d[f"{key}_{t}/scalar"] = data.item()
                continue
            if np.prod(data.shape) > 12:
                d[f"{key}/{t}_norm"] = np.linalg.norm(data)
                continue
            for i, v in enumerate(data):
                d[f"{key}_{t}/{i:02d}"] = v

    exp.log_metrics(d, step=global_step)


@torch.no_grad()
def log_t_steps(exp: comet_ml.Experiment, t_steps: torch.Tensor, global_step: int, key: str = "t_stats") -> None:
    """
    Logs:
    - Per-step stats across batch: mean/std/min/max
    - A few individual trajectories (first K batch items)

    Supports input shapes:
    - (S,)
    - (B, S)
    - (S, B)
    """
    t_steps = t_steps.detach().float().cpu()

    # Normalize shapes to (S, B)
    if t_steps.ndim == 1:
        t_steps = t_steps[:, None]
    elif t_steps.ndim == 2:
        if t_steps.shape[0] < t_steps.shape[1]:
            # likely (B, S)
            t_steps = t_steps.T
    else:
        # Keep first dim as steps, collapse the rest into batch
        t_steps = t_steps.reshape(t_steps.shape[0], -1)

    S, B = t_steps.shape
    d = {}

    # Stats across batch
    mean = t_steps.mean(dim=1)
    std = t_steps.std(dim=1, unbiased=False)
    tmin = t_steps.min(dim=1).values
    tmax = t_steps.max(dim=1).values

    for i in range(S):
        d[f"{key}/t_{i:02d}_mean"] = float(mean[i])
        d[f"{key}/t_{i:02d}_std"] = float(std[i])
        d[f"{key}/t_{i:02d}_min"] = float(tmin[i])
        d[f"{key}/t_{i:02d}_max"] = float(tmax[i])

    # A few individual trajectories
    k = min(3, B)
    for b in range(k):
        for i in range(S):
            d[f"{key}/traj_{b:02d}/t_{i:02d}"] = float(t_steps[i, b])

    exp.log_metrics(d, step=global_step)