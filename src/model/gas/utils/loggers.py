import matplotlib.pyplot as plt
import numpy as np
import torch
from typing import Dict, List, Mapping, Optional, Union
from torchvision.utils import make_grid

import comet_ml
from src.model.gas.gs_wrapper import GSWrapper

CoeffArray = Union[np.ndarray, torch.Tensor]


def _to_numpy_1d(data: CoeffArray) -> np.ndarray:
    if isinstance(data, torch.Tensor):
        data = data.detach().float().cpu().numpy()
    else:
        data = np.asarray(data, dtype=np.float64)
    return np.atleast_1d(data).reshape(-1)


def reduce_coeff_tensor(
    tensor: torch.Tensor,
    reduction: str = "mean",
) -> np.ndarray:
    """Reduce batch-dependent solver coeffs to a 1D per-step vector for logging."""
    t = tensor.detach().float()
    if t.ndim == 0:
        return t.cpu().numpy().reshape(1)
    if t.ndim == 1:
        return t.cpu().numpy()
    if reduction == "batch0":
        return t[0].cpu().numpy()
    return t.mean(dim=0).cpu().numpy()


def coeffs_to_metric_dict(
    coeffs: Mapping[str, CoeffArray],
    prefix: str,
    suff: str = "",
) -> Dict[str, float]:
    """Build Comet scalars: ``{prefix}{name}/{step:02d}`` (e.g. weights_stats_a1_diff/00)."""
    d: Dict[str, float] = {}
    key_base = f"{prefix}{suff}"
    for name, values in coeffs.items():
        arr = _to_numpy_1d(values)
        if arr.size > 12:
            d[f"{key_base}/{name}_norm"] = float(np.linalg.norm(arr))
            continue
        for i, v in enumerate(arr):
            d[f"{key_base}_{name}/{i:02d}"] = float(v)
    return d


def print_final_solver_coeffs(
    coeffs: Mapping[str, CoeffArray],
    header: str = "",
    reduction_label: str = "batch-mean",
    max_samples: int = 0,
) -> None:
    lines = []
    if header:
        lines.append(header)
    lines.append(f"=== final solver coefficients ({reduction_label}) ===")
    for name in sorted(coeffs.keys()):
        tensor_or_arr = coeffs[name]
        if isinstance(tensor_or_arr, torch.Tensor) and tensor_or_arr.ndim == 2 and max_samples > 0:
            t = tensor_or_arr.detach().cpu().numpy()
            samples_to_show = min(max_samples, t.shape[0])
            for b in range(samples_to_show):
                vals = ", ".join(f"{v:.6g}" for v in t[b])
                lines.append(f"  {name}[{b}]: {vals}")
        else:
            arr = _to_numpy_1d(tensor_or_arr)
            if arr.size <= 8:
                vals = ", ".join(f"{v:.6g}" for v in arr)
            else:
                vals = (
                    f"[{', '.join(f'{v:.6g}' for v in arr[:4])}, ..., "
                    f"{', '.join(f'{v:.6g}' for v in arr[-2:])}] (n={arr.size})"
                )
            lines.append(f"  {name}: {vals}")
    print("\n".join(lines))


@torch.no_grad()
def log_final_solver_coeffs(
    exp: comet_ml.Experiment,
    coeffs: Mapping[str, CoeffArray],
    global_step: int,
    suff: str = "",
    prefix: str = "weights_stats",
) -> None:
    exp.log_metrics(coeffs_to_metric_dict(coeffs, prefix=prefix, suff=suff), step=global_step)


def log_final_solver_grads(
    exp: comet_ml.Experiment,
    grads: Mapping[str, CoeffArray],
    global_step: int,
    suff: str = "",
    prefix: str = "grad_stats",
) -> None:
    if not grads:
        return
    exp.log_metrics(coeffs_to_metric_dict(grads, prefix=prefix, suff=suff), step=global_step)


@torch.no_grad()
def log_stepwise_vis_metrics(
    exp: comet_ml.Experiment,
    trace: Mapping[str, List[float]],
    global_step: int,
    key_prefix: str = "stepwise_vis",
    suff: str = "",
) -> None:
    """Log per-solver-step scalar traces collected on the vis batch."""
    d: Dict[str, float] = {}
    prefix = f"{key_prefix}{suff}"
    for name, values in trace.items():
        if not values:
            continue
        arr = np.asarray(values, dtype=np.float64)
        for step_i, v in enumerate(arr):
            d[f"{prefix}/{name}/step_{step_i:02d}"] = float(v)
        d[f"{prefix}/{name}/mean"] = float(arr.mean())
        d[f"{prefix}/{name}/std"] = float(arr.std()) if arr.size > 1 else 0.0
        d[f"{prefix}/{name}/delta_last"] = float(arr[-1] - arr[0]) if arr.size > 1 else 0.0
    if d:
        exp.log_metrics(d, step=global_step)


@torch.no_grad()
def log_stepwise_vis_plot(
    exp: comet_ml.Experiment,
    trace: Mapping[str, List[float]],
    global_step: int,
    key: str = "stepwise_vis/trajectory",
) -> None:
    if not trace:
        return
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    for name, values in sorted(trace.items()):
        if not values:
            continue
        ax.plot(values, marker="o", markersize=3, label=name, alpha=0.85)
    ax.set_xlabel("Solver step")
    ax.set_ylabel("Coefficient (batch mean)")
    ax.set_title("Stepwise coefficient trajectory (vis batch)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    log_plt_fig(exp=exp, fig=fig, key=key, global_step=global_step)


def log_plt_fig(exp: comet_ml.Experiment, fig, key: str, global_step: int) -> None:
    fig.tight_layout()

    exp.log_figure(
        figure=fig,
        figure_name=key,
        step=global_step,
    )
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


def log_grads(
    exp: comet_ml.Experiment,
    model: GSWrapper,
    global_step: int,
    suff: str = "",
    solver_grads: Optional[Mapping[str, CoeffArray]] = None,
) -> None:
    d = {}
    key = f"grad_stats{suff}"
    skip_param_names = set(solver_grads.keys()) if solver_grads and model.should_log_final_solver_coeffs() else set()
    for t, p in model.named_parameters():
        if t in skip_param_names:
            continue
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

    if solver_grads:
        d.update(coeffs_to_metric_dict(solver_grads, prefix=key, suff=""))

    if d:
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
