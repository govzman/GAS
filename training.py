import datetime
import os
import time

import torch
from torch_ema import ExponentialMovingAverage
from tqdm import tqdm

import comet_ml
from hydra.utils import instantiate
from omegaconf import OmegaConf
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
from typing import Optional


def train(
    config,
    gs_wrapper: GSWrapper,
    ema: ExponentialMovingAverage,
    data: SyntDataset,
    optim: torch.optim.Adam,
    optim_a_params: torch.optim.Adam,
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
        if optim_a_params:
            optim_a_params = accelerator.prepare(optim_a_params)
        
        gs_wrapper, data.train_loader, data.test_loader = accelerator.prepare(
            gs_wrapper, data.train_loader, data.test_loader
        )
        
    freeze_steps = getattr(config.student_solver, 'freeze_mlp_steps', 0)
    if freeze_steps > 0 and is_main:
        print(f"🧊 MLP will be DISABLED for first {freeze_steps} steps (masking, not freezing)")

    lr_scheduler = None
    lr_scheduler_a_params = None
    if optim and hasattr(config, "lr_scheduler") and config.lr_scheduler is not None:
        lr_scheduler = instantiate(config.lr_scheduler, optimizer=optim)
    if optim_a_params and hasattr(config, "lr_scheduler_a_params") and config.lr_scheduler_a_params is not None:
        lr_scheduler_a_params = instantiate(config.lr_scheduler_a_params, optimizer=optim_a_params)

    if is_main:
        dir = os.path.join("./checkpoints", date_str)
        os.makedirs(dir, exist_ok=False)
        config.trainer.checkpoints_dir = dir

        print(f"\n🚀 START TRAINING: {date_str}")
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

    global_step = 0
    pbar = tqdm(range(config.trainer.n_iters), dynamic_ncols=True) if is_main else None

    for _ in range(config.trainer.epoch_num):
        for batch in data.train_loader:
            if global_step == config.trainer.n_iters:
                break
            global_step += 1
            # ⭐ КЛЮЧЕВОЕ ИЗМЕНЕНИЕ: просто переключаем флаг
            if freeze_steps > 0 and global_step == freeze_steps:
                gs_wrapper.mlp_disabled = False
                if is_main:
                    print(f"🔓 MLP ENABLED at step {global_step}")
            
            # ⭐ ДИАГНОСТИКА после включения MLP
            if freeze_steps > 0 and global_step == freeze_steps + 1 and is_main:
                print("\n🔍 DIAGNOSTIC: Checking MLP after enabling")

                # Проверка a_diff_model
                if hasattr(gs_wrapper, 'a_diff_model') and gs_wrapper.a_diff_model is not None:
                    print("=== a_diff_model ===")
                    for name, param in gs_wrapper.a_diff_model.named_parameters():
                        if param.grad is not None:
                            print(f"  {name}: grad_norm={param.grad.norm().item():.6f}, "
                                  f"param_norm={param.norm().item():.6f}, "
                                  f"requires_grad={param.requires_grad}")
                        else:
                            print(f"  {name}: grad=None, requires_grad={param.requires_grad}")

                # Проверка c_diff_model
                if hasattr(gs_wrapper, 'c_diff_model') and gs_wrapper.c_diff_model is not None:
                    print("=== c_diff_model ===")
                    for name, param in gs_wrapper.c_diff_model.named_parameters():
                        if param.grad is not None:
                            print(f"  {name}: grad_norm={param.grad.norm().item():.6f}, "
                                  f"param_norm={param.norm().item():.6f}, "
                                  f"requires_grad={param.requires_grad}")
                        else:
                            print(f"  {name}: grad=None, requires_grad={param.requires_grad}")

                # Проверка базовых diff параметров
                print("=== base diff params ===")
                for i in range(1, gs_wrapper.order + 1):
                    for prefix in ['a', 'c']:
                        pname = f'{prefix}{i}_diff'
                        if hasattr(gs_wrapper, pname):
                            param = getattr(gs_wrapper, pname)
                            if param.grad is not None:
                                print(f"  {pname}: grad_norm={param.grad.norm().item():.6f}, "
                                      f"param_norm={param.norm().item():.6f}")
                print()

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
                    if optim_a_params:
                        optim_a_params.step()
                        if lr_scheduler_a_params is not None:
                            lr_scheduler_a_params.step()
                        optim_a_params.zero_grad()
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
                    if optim_a_params:
                        log_d["optim_a_params/lr"] = optim_a_params.param_groups[0]["lr"]
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
                        if optim_a_params:
                            optim_a_params.step()
                            if lr_scheduler_a_params is not None:
                                lr_scheduler_a_params.step()
                            optim_a_params.zero_grad()
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
                        if optim_a_params:
                            log_d["optim_a_params/lr"] = optim_a_params.param_groups[0]["lr"]

            for k, v in res_d.items():
                if k not in NOT_LOG_KEYS:
                    log_d[f"train/{k}"] = v.mean().item()

            if log_solver_coeffs:
                final_coeffs = gs_wrapper.get_final_solver_coeffs_for_logging()
                print_final_solver_coeffs(
                    final_coeffs,
                    header=f"[train step {global_step}]",
                )

            print_every = int(getattr(config.trainer, "print_gt_solver_every", 0))
            if is_main and print_every > 0 and global_step % print_every == 0:
                gt_mse_keys = sorted(
                    k for k in res_d.keys() if k.startswith("gt_mse_") and k != "gt_mse_mean"
                )
                if gt_mse_keys:
                    mean_line = ""
                    if "gt_mse_mean" in res_d:
                        mean_line = f"gt_mse_mean(batch avg)={res_d['gt_mse_mean'].mean().item():.6g} | "
                    detail = " | ".join(
                        f"{k[len('gt_mse_'):]}={res_d[k].mean().item():.6g}"
                        for k in gt_mse_keys
                    )
                    print(f"[train step {global_step}] GT coeff MSE vs predicted — {mean_line}{detail}")

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

            if is_main and (global_step % config.writer.checkpoint_freq == 0 or global_step == 1):
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
                if optim_a_params:
                    ckpt["optim_a_params"] = optim_a_params.state_dict()
                if lr_scheduler_a_params is not None:
                    ckpt["lr_scheduler_a_params"] = lr_scheduler_a_params.state_dict()
                ckpt_path = os.path.join(dir, f"{global_step}.pt")
                if accelerator is not None:
                    accelerator.save(ckpt, ckpt_path)
                else:
                    torch.save(ckpt, ckpt_path)

            if pbar is not None:
                pbar.update(1)

    # comet_ml.finish()
