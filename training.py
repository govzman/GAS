import datetime
import os
import time
from typing import Optional
import math

import comet_ml
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch_ema import ExponentialMovingAverage
from tqdm import tqdm

from evaluate import NOT_LOG_KEYS, evaluate_wrapper
from src.model.gas.gs_wrapper import GSWrapper
from src.model.gas.synt_data import SyntDataset, move_batch_to_device
from src.model.gas.utils.loggers import (
    log_end_img,
    log_final_solver_coeffs,
    log_grads,
    log_t_steps,
    log_weights,
    print_final_solver_coeffs,
)


def restore_training_state(checkpoint, model, optim, ema, lr_scheduler, target_step, device):
    required = {"model", "ema", "step"} | ({"optim"} if optim is not None else set())
    missing = required - set(checkpoint)
    if missing:
        raise ValueError(f"Resume checkpoint is missing: {sorted(missing)}")
    step = int(checkpoint["step"])
    if not 0 <= step < int(target_step):
        raise ValueError("trainer.n_iters is the TOTAL target step and must exceed checkpoint step")
    if (lr_scheduler is not None) != ("lr_scheduler" in checkpoint):
        raise ValueError("Resume scheduler configuration must match the saved checkpoint")
    model.load_state_dict(checkpoint["model"], strict=True)
    if optim is not None:
        optim.load_state_dict(checkpoint["optim"])
    if lr_scheduler is not None:
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
    ema.load_state_dict(checkpoint["ema"])
    ema.to(device=device)
    return step


def train(
    config,
    gs_wrapper: GSWrapper,
    ema: ExponentialMovingAverage,
    data: SyntDataset,
    optim: torch.optim.Adam,
    device: torch.device,
    accelerator: Optional[object] = None,
):
    ct = datetime.datetime.now()
    date_str = ct.strftime("%m_%d_%H_%M_%S")

    is_main = True
    if accelerator is not None:
        is_main = bool(getattr(accelerator, "is_main_process", True))
        device = getattr(accelerator, "device", device)

        if optim:
            optim = accelerator.prepare(optim)

        gs_wrapper, data.train_loader, data.test_loader = accelerator.prepare(
            gs_wrapper, data.train_loader, data.test_loader
        )

    lr_scheduler = None
    if optim and hasattr(config, "lr_scheduler") and config.lr_scheduler is not None:
        lr_scheduler = instantiate(config.lr_scheduler, optimizer=optim)

    global_step = 0
    resume_path = getattr(config.trainer, "resume_checkpoint", None)
    if resume_path:
        # Old checkpoints do not contain pending gradients, AMP scaler state
        # or distributed sampler positions. Support the current single-GPU setup.
        accumulation = int(getattr(config.trainer, "accumulation_steps",
                                   getattr(config.trainer, "iters_to_accumulate", 1)))
        if accumulation != 1 or str(getattr(config.trainer, "amp", "no")) != "no":
            raise ValueError("Resume currently requires accumulation_steps=1 and amp=no")
        if accelerator is not None and accelerator.num_processes != 1:
            raise ValueError("Resume currently supports a single process/GPU")
        checkpoint = torch.load(resume_path, map_location="cpu")
        model_to_restore = accelerator.unwrap_model(gs_wrapper) if accelerator is not None else gs_wrapper
        global_step = restore_training_state(
            checkpoint, model_to_restore, optim, ema, lr_scheduler,
            config.trainer.n_iters, device,
        )
        if is_main:
            print(f"RESUME: {resume_path}, step {global_step} -> {config.trainer.n_iters}", flush=True)
            if optim is not None:
                print(f"Restored optimizer learning rates: {[g['lr'] for g in optim.param_groups]}", flush=True)
            print("Data order restarts with a fresh shuffle; old checkpoints do not store the sampler position.", flush=True)
        del checkpoint

    if is_main:
        dir = os.path.join("./checkpoints", date_str)
        os.makedirs(dir, exist_ok=False)
        config.trainer.checkpoints_dir = dir

        print(f"\nSTART TRAINING: {date_str}")
        print("=" * 40 + " Config Info " + "=" * 40)
        print(config)
        print("=" * 90 + "\n")

    exp = None
    if is_main:
        comet_ml.login()

        if config.writer.mode == "offline":
            exp_class = comet_ml.OfflineExperiment
        else:
            exp_class = comet_ml.Experiment

        exp = exp_class(
            project_name=config.writer.project_name,
            workspace=getattr(config.writer, "workspace", None),
            experiment_key=None,
            log_code=True,
        )
        exp.set_name(config.writer.run_name)
        exp.log_parameters(parameters=OmegaConf.to_container(config, resolve=True))

    pbar = tqdm(total=config.trainer.n_iters, initial=global_step, dynamic_ncols=True) if is_main else None

    if resume_path and exp is not None:
        # Check restored validation losses before the first optimizer update.
        evaluate_wrapper(exp, gs_wrapper, data, device, "", global_step)
        with ema.average_parameters():
            evaluate_wrapper(exp, gs_wrapper, data, device, "_ema", global_step)

    remaining_epochs = math.ceil((config.trainer.n_iters - global_step) / len(data.train_loader))
    for _ in range(remaining_epochs):
        for batch in data.train_loader:
            if global_step >= config.trainer.n_iters:
                break
            global_step += 1

            t_start = time.time()
            batch = move_batch_to_device(batch, device)

            log_solver_coeffs = (
                is_main
                and gs_wrapper.should_log_final_solver_coeffs()
                and global_step % config.writer.log_weights_freq == 0
            )
            gs_wrapper.set_retain_solver_coeff_grads(log_solver_coeffs)

            if accelerator is None:
                res_d = gs_wrapper.forward(batch=batch, return_timesteps=True)
                loss = res_d["loss_total"].mean() / config.trainer.iters_to_accumulate
                loss.backward()
                log_d = {"optim/time": time.time() - t_start}

                if global_step % config.trainer.iters_to_accumulate == 0:
                    if exp is not None and global_step % config.writer.log_weights_freq == 0:
                        solver_grads = (
                            gs_wrapper.get_final_solver_coeff_grads_for_logging()
                            if log_solver_coeffs
                            else None
                        )
                        log_grads(
                            exp=exp,
                            model=gs_wrapper,
                            global_step=global_step,
                            solver_grads=solver_grads,
                        )

                    grad_norm = torch.nn.utils.clip_grad_norm_(gs_wrapper.parameters(), 1.0)

                    if optim:
                        optim.step()
                        if lr_scheduler is not None:
                            lr_scheduler.step()
                        optim.zero_grad()
                    ema.update(gs_wrapper.parameters())

                    if exp is not None and global_step % config.writer.log_weights_freq == 0:
                        log_t_steps(exp, res_d["timesteps"], global_step=global_step)
                        log_weights(exp, model=gs_wrapper, global_step=global_step)
                        if log_solver_coeffs:
                            final_coeffs = gs_wrapper.get_final_solver_coeffs_for_logging()
                            log_final_solver_coeffs(
                                exp, final_coeffs, global_step=global_step
                            )

                    log_d["optim/grad_norm"] = grad_norm
                    if optim:
                        log_d["optim/lr"] = optim.param_groups[0]["lr"]
            else:
                with accelerator.accumulate(gs_wrapper):
                    with accelerator.autocast():
                        res_d = gs_wrapper.forward(batch=batch, return_timesteps=True)
                        loss = res_d["loss_total"].mean()

                    accelerator.backward(loss)
                    log_d = {"optim/time": time.time() - t_start}

                    if accelerator.sync_gradients:
                        if exp is not None and global_step % config.writer.log_weights_freq == 0:
                            solver_grads = (
                                gs_wrapper.get_final_solver_coeff_grads_for_logging()
                                if log_solver_coeffs
                                else None
                            )
                            log_grads(
                                exp=exp,
                                model=gs_wrapper,
                                global_step=global_step,
                                solver_grads=solver_grads,
                            )

                        grad_norm = accelerator.clip_grad_norm_(gs_wrapper.parameters(), 1.0)

                        if optim:
                            optim.step()
                            if lr_scheduler is not None:
                                lr_scheduler.step()
                            optim.zero_grad()
                        ema.update(gs_wrapper.parameters())

                        if exp is not None and global_step % config.writer.log_weights_freq == 0:
                            log_t_steps(exp, res_d["timesteps"], global_step=global_step)
                            log_weights(exp, model=gs_wrapper, global_step=global_step)
                            if log_solver_coeffs:
                                final_coeffs = gs_wrapper.get_final_solver_coeffs_for_logging()
                                log_final_solver_coeffs(
                                    exp, final_coeffs, global_step=global_step
                                )

                        log_d["optim/grad_norm"] = float(grad_norm)
                        if optim:
                            log_d["optim/lr"] = optim.param_groups[0]["lr"]

            for k, v in res_d.items():
                if k not in NOT_LOG_KEYS:
                    log_d[f"train/{k}"] = v.mean().item()

            if log_solver_coeffs:
                final_coeffs = gs_wrapper.get_final_solver_coeff_tensors()
                print_final_solver_coeffs(
                    final_coeffs,
                    header=f"[train step {global_step}]",
                    max_samples=2,
                )

            # Synthetic teachers may carry GT manual_solver_params.*; log MSE periodically.
            print_every = int(getattr(config.trainer, "print_gt_solver_every", 0))
            if is_main and print_every > 0 and global_step % print_every == 0:
                gt_mse_keys = sorted(
                    k for k in res_d.keys() if k.startswith("gt_mse_") and k != "gt_mse_mean"
                )
                if gt_mse_keys:
                    mean_line = ""
                    if "gt_mse_mean" in res_d:
                        mean_line = (
                            f"gt_mse_mean(batch avg)={res_d['gt_mse_mean'].mean().item():.6g} | "
                        )
                    detail = " | ".join(
                        f"{k[len('gt_mse_'):]}={res_d[k].mean().item():.6g}"
                        for k in gt_mse_keys
                    )
                    print(
                        f"[train step {global_step}] GT coeff MSE vs predicted — {mean_line}{detail}"
                    )

            if exp is not None:
                exp.log_metrics(log_d, step=global_step)

            if exp is not None and (global_step % config.writer.eval_freq == 0 or global_step == 1):
                if "x0_s" not in res_d:
                    with torch.no_grad():
                        res_d["x0_s"] = gs_wrapper.model.decode(res_d["latents_s"])
                log_end_img(
                    exp,
                    res_d["x0_s"],
                    res_d["x0_t"],
                    global_step=global_step,
                    key="train/backward_end_inter",
                )

                evaluate_wrapper(
                    exp=exp,
                    gs_wrapper=gs_wrapper,
                    data=data,
                    device=device,
                    suff="",
                    global_step=global_step,
                )

                with ema.average_parameters():
                    evaluate_wrapper(
                        exp=exp,
                        gs_wrapper=gs_wrapper,
                        data=data,
                        device=device,
                        suff="_ema",
                        global_step=global_step,
                    )
                    log_weights(exp=exp, model=gs_wrapper, global_step=global_step, suff="_ema")
                    if gs_wrapper.should_log_final_solver_coeffs():
                        final_coeffs_ema = gs_wrapper.get_final_solver_coeffs_for_logging()
                        log_final_solver_coeffs(
                            exp,
                            final_coeffs_ema,
                            global_step=global_step,
                            suff="_ema",
                        )

            gs_wrapper.set_retain_solver_coeff_grads(False)

            if is_main and (global_step % config.writer.checkpoint_freq == 0 or global_step == 1
                            or global_step == config.trainer.n_iters):
                model_to_save = gs_wrapper
                if accelerator is not None:
                    model_to_save = accelerator.unwrap_model(gs_wrapper)

                ckpt = {
                    "ema": ema.state_dict(),
                    "model": model_to_save.state_dict(),
                    "step": global_step,
                }
                if optim:
                    ckpt["optim"] = optim.state_dict()
                if lr_scheduler is not None:
                    ckpt["lr_scheduler"] = lr_scheduler.state_dict()
                ckpt_path = os.path.join(dir, f"{global_step}.pt")
                if accelerator is not None:
                    accelerator.save(ckpt, ckpt_path)
                else:
                    torch.save(ckpt, ckpt_path)

            if pbar is not None:
                pbar.update(1)
