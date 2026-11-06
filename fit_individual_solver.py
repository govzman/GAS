"""Fit per-example four-step a/c and a matched shared control from a global checkpoint.

This is target-fitting diagnostics, including when the selected examples belong
to validation. Best losses here are not estimates of generalization or lower bounds.
"""
import csv
import datetime
import json
import math
import os
from pathlib import Path
import pickle
import random

import hydra
import torch
from omegaconf import OmegaConf

from src.model.gas.models import get_gs_wrapper, load_base_model
from src.model.gas.utils.random import set_global_seed


def select_indices(count, dataset_cfg, diagnostic):
    train_start = int(dataset_cfg.dataset_shift)
    train_stop = train_start + int(dataset_cfg.train_size)
    val_start = count - int(dataset_cfg.validation_size)
    if not (0 <= train_start < train_stop <= val_start < count):
        raise ValueError("Expected valid non-overlapping train/validation splits")
    if diagnostic.split == "validation":
        pool = range(val_start, count)
    elif diagnostic.split == "train":
        pool = range(train_start, train_stop)
    else:
        raise ValueError("diagnostic.split must be train or validation")
    n = int(diagnostic.count)
    if not 1 <= n <= len(pool):
        raise ValueError(f"Choose diagnostic.count between 1 and {len(pool)}")
    return sorted(random.Random(int(diagnostic.selection_seed)).sample(list(pool), n))


def active_parameters(base):
    """Allocate only active entries; order i first appears at zero-based step i-1."""
    return {
        name: torch.nn.Parameter(value[int(name[1:-5]) - 1:].clone())
        for name, value in base.items()
    }


def coefficient_tables(base, params):
    return {
        name: torch.cat((value[:int(name[1:-5]) - 1], params[name]))
        for name, value in base.items()
    }


def snapshot(params):
    return {name: value.detach().clone() for name, value in params.items()}


def sample_loss(wrapper, entry, tables):
    # Restore the cached original SD conditioning for this sample. The context
    # manager overrides a/c only: checkpoint t/t_couple stay fixed throughout.
    wrapper.model.model_fn.condition = entry["embedding"]
    wrapper.model.model_fn.unconditional_condition = entry["unconditional"]
    prediction, _ = wrapper.student_sampler_fn(
        noise=entry["noise"], manual_solver_params=tables,
    )
    return (prediction - entry["target"]).abs().mean()


def fit(wrapper, entries, base, settings, label, history):
    params = active_parameters(base)
    optimizer = torch.optim.Adam(params.values(), lr=float(settings.lr), weight_decay=0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=int(settings.iterations), eta_min=float(settings.min_lr),
    )
    best_loss = float("inf")
    best_step = 0
    best_params = snapshot(params)
    best_per_sample = None
    initial = None
    final = None
    status = "complete"
    for step in range(int(settings.iterations) + 1):
        optimizer.zero_grad(set_to_none=True)
        losses = []
        finite = True
        # Each example contributes equally. Shared fitting updates once after
        # all examples; individual fitting uses exactly one example per update.
        for entry in entries:
            with torch.set_grad_enabled(step < int(settings.iterations)):
                loss = sample_loss(wrapper, entry, coefficient_tables(base, params))
            if not torch.isfinite(loss).item():
                finite = False
                break
            losses.append(loss.detach().item())
            if step < int(settings.iterations):
                (loss / len(entries)).backward()
        if not finite:
            status = f"nonfinite_loss_at_step_{step}"
            break
        mean_loss = sum(losses) / len(losses)
        final = losses
        if initial is None:
            initial = losses
        if mean_loss < best_loss:
            best_loss, best_step = mean_loss, step
            best_params = snapshot(params)
            best_per_sample = losses
        if step % int(settings.log_every) == 0 or step == int(settings.iterations):
            print(f"{label} step={step} mean_L1={mean_loss:.9g} best={best_loss:.9g}", flush=True)
        history.writerow([label, step, mean_loss, best_loss, optimizer.param_groups[0]["lr"]])
        if step == int(settings.iterations):
            break
        grad_norm = torch.nn.utils.clip_grad_norm_(list(params.values()), float(settings.grad_clip))
        if not torch.isfinite(grad_norm).item():
            status = f"nonfinite_gradient_at_step_{step}"
            break
        optimizer.step()
        scheduler.step()
    if initial is None:
        raise RuntimeError(f"{label}: checkpoint produced nonfinite initial loss")
    tables = {k: v.detach().cpu() for k, v in coefficient_tables(base, best_params).items()}
    return dict(initial=initial, best=best_per_sample, final=final,
                best_step=best_step, status=status), tables


@hydra.main(version_base=None, config_path="src/configs", config_name="diagnose_synt_5to4_individual")
def main(config):
    cfg = OmegaConf.create(OmegaConf.to_container(config, resolve=True))
    settings = cfg.diagnostic
    checkpoint = os.environ.get("GAS_CHECKPOINT")
    if not checkpoint:
        raise ValueError("Set GAS_CHECKPOINT to the global checkpoint path")
    if int(settings.iterations) < 1 or int(settings.log_every) < 1:
        raise ValueError("iterations and log_every must be positive")
    if not 0 < float(settings.min_lr) <= float(settings.lr) or float(settings.grad_clip) <= 0:
        raise ValueError("Require 0 < min_lr <= lr and positive grad_clip")
    solver_cfg = cfg.student_solver
    if (solver_cfg.a_parametrization != "diff" or solver_cfg.c_parametrization != "diff"
            or solver_cfg.t_parametrization not in ("diff", "mu_logit")
            or solver_cfg.t_couple_parametrization != "diff"):
        raise ValueError("Use a global a/c checkpoint with static t and t_couple")
    set_global_seed(int(cfg.trainer.seed))
    with open(cfg.datasets.teacher_pkl, "rb") as f:
        data = pickle.load(f)  # trusted local datasets only
    indices = select_indices(len(data["noise"]), cfg.datasets, settings)
    device = torch.device(cfg.trainer.device)
    cfg.model.t_eps = solver_cfg.t_eps
    cfg.model.guidance_scale = solver_cfg.guidance_scale
    model = load_base_model(cfg.model, device)
    wrapper = get_gs_wrapper(model, cfg)
    state = torch.load(checkpoint, map_location="cpu")
    wrapper.load_state_dict(state["model"], strict=True)
    if settings.weights == "ema":
        wrapper.load_checkpoint(checkpoint)
    elif settings.weights != "raw":
        raise ValueError("diagnostic.weights must be raw or ema")
    wrapper.requires_grad_(False)
    wrapper.eval()
    base = {
        f"{side}{i}_diff": getattr(wrapper, f"{side}{i}_diff").detach().clone()
        for side in ("a", "c") for i in range(1, int(wrapper.order) + 1)
    }
    frozen_times = wrapper.get_t_steps().detach().clone()
    frozen_couple = wrapper.t_couple.detach().clone()
    entries = []
    with torch.no_grad():
        for idx in indices:
            model.set_condition([data["condition"][idx]])
            entries.append(dict(index=idx, noise=data["noise"][idx:idx+1].to(device),
                                target=data["latents"][idx:idx+1].to(device),
                                embedding=model.model_fn.condition.detach().clone(),
                                unconditional=model.model_fn.unconditional_condition.detach().clone()))

    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output = Path(settings.output_dir) / stamp
    output.mkdir(parents=True, exist_ok=False)
    OmegaConf.save(cfg, output / "config.yaml")
    print(f"Checkpoint: {checkpoint}; step={state.get('step')}; weights={settings.weights}", flush=True)
    print(f"Dataset: {cfg.datasets.teacher_pkl}; fitting indices={indices}; output={output}", flush=True)
    print("Target fitting only, NOT held-out evaluation or a theoretical lower bound.", flush=True)
    individual = []
    saved = {"base": {k: v.cpu() for k, v in base.items()}, "individual": {}}
    with (output / "history.csv").open("w", newline="") as stream:
        history = csv.writer(stream)
        history.writerow(["fit", "step", "mean_L1", "best_mean_L1", "lr"])
        if settings.shared_control:
            shared, saved["shared"] = fit(wrapper, entries, base, settings, "shared", history)
            stream.flush()
        else:
            shared = None
        for entry in entries:
            result, tables = fit(wrapper, [entry], base, settings, f"individual_{entry['index']}", history)
            individual.append(result)
            saved["individual"][entry["index"]] = tables
            stream.flush()
    assert torch.equal(wrapper.get_t_steps(), frozen_times)
    assert torch.equal(wrapper.t_couple, frozen_couple)
    rows = []
    for position, entry in enumerate(entries):
        result = individual[position]
        if shared is not None and not math.isclose(result["initial"][0], shared["initial"][position], rel_tol=1e-5, abs_tol=1e-6):
            raise RuntimeError("Shared and individual fits did not start from the same loss")
        row = dict(index=entry["index"], initial_L1=result["initial"][0],
                   shared_best_L1=shared["best"][position] if shared else None,
                   individual_best_L1=result["best"][0], individual_final_L1=result["final"][0],
                   individual_best_step=result["best_step"], status=result["status"])
        rows.append(row)
        print(json.dumps(row), flush=True)
    summary = dict(checkpoint=checkpoint, weights=settings.weights, split=settings.split,
                   checkpoint_step=state.get("step"), dataset=cfg.datasets.teacher_pkl,
                   shared=shared, samples=rows,
                   mean_initial_L1=sum(r["initial_L1"] for r in rows) / len(rows),
                   mean_individual_best_L1=sum(r["individual_best_L1"] for r in rows) / len(rows))
    if shared:
        summary["mean_shared_best_L1"] = sum(shared["best"]) / len(entries)
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with (output / "samples.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    torch.save(saved, output / "best_coefficients.pt")
    print(f"Mean initial L1: {summary['mean_initial_L1']:.9g}", flush=True)
    if shared:
        print(f"Mean shared-best L1: {summary['mean_shared_best_L1']:.9g}", flush=True)
    print(f"Mean individual-best L1: {summary['mean_individual_best_L1']:.9g}", flush=True)


if __name__ == "__main__":
    main()
