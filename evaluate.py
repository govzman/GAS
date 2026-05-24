from collections import defaultdict

import torch

import comet_ml
from src.model.gas.gs_wrapper import GSWrapper
from src.model.gas.synt_data import SyntDataLoaders, move_batch_to_device
from src.model.gas.utils.loggers import (
    log_end_img,
    log_final_solver_coeffs,
    log_stepwise_vis_metrics,
    log_stepwise_vis_plot,
    log_t_steps_plot,
    print_final_solver_coeffs,
)

NOT_LOG_KEYS = ["timesteps", "x0_s", "x0_t", "latents_s"]


@torch.no_grad()
def evaluate_wrapper(
    exp: comet_ml.Experiment,
    gs_wrapper: GSWrapper, 
    data: SyntDataLoaders, 
    device: torch.device, 
    suff: str, 
    global_step: int
) -> None:
    """Evaluating GS on test dataset and visualization batch for logging."""
    batch = move_batch_to_device(data.vis_batch, device)

    d_res = {}

    if gs_wrapper.use_stepwise_coeff:
        gs_wrapper.enable_stepwise_vis_trace(True)

    out_d = gs_wrapper.forward(batch=batch, return_timesteps=True, is_train=False)

    stepwise_trace = gs_wrapper.consume_stepwise_vis_trace()
    if stepwise_trace:
        log_stepwise_vis_metrics(
            exp=exp,
            trace=stepwise_trace,
            global_step=global_step,
            suff=suff,
        )
        log_stepwise_vis_plot(
            exp=exp,
            trace=stepwise_trace,
            global_step=global_step,
            key=f"stepwise_vis{suff}/trajectory",
        )

    if gs_wrapper.should_log_final_solver_coeffs():
        final_coeffs = gs_wrapper.get_final_solver_coeffs_for_logging()
        print_final_solver_coeffs(
            final_coeffs,
            header=f"[eval vis step {global_step}{suff}]",
        )
        log_final_solver_coeffs(
            exp,
            final_coeffs,
            global_step=global_step,
            suff=suff,
        )
    log_t_steps_plot(
        exp=exp,
        t_steps=out_d["timesteps"],
        global_step=global_step,
        key=f"eval_image{suff}/t_steps",
    )
    for k, v in out_d.items():
        if k not in NOT_LOG_KEYS:
            d_res[f"vis_stat/{k}{suff}"] = v.mean().item()

    if "x0_s" not in out_d:
        out_d["x0_s"] = gs_wrapper.model.decode(out_d["latents_s"])
    log_end_img(
        exp,
        out_d["x0_s"],
        out_d["x0_t"],
        global_step=global_step,
        key=f"vis_stat{suff}/backward_end_inter",
    )

    log_d = defaultdict(float)
    num_elements = 0
    for batch in data.test_loader:
        batch = move_batch_to_device(batch, device)
        out_d = gs_wrapper.forward(batch=batch, return_timesteps=False, is_train=False)
        bs = batch[0].shape[0]
        num_elements += bs
        for k, v in out_d.items():
            if k not in NOT_LOG_KEYS:
                log_d[k] += v.mean().item() * bs

    for k, v in log_d.items():
        if k not in NOT_LOG_KEYS:
            d_res[f"val_stat/{k}{suff}"] = v / num_elements

    exp.log_metrics(d_res, step=global_step)
    if "x0_s" not in out_d:
        out_d["x0_s"] = gs_wrapper.model.decode(out_d["latents_s"])
    log_end_img(
        exp,
        out_d["x0_s"],
        out_d["x0_t"],
        global_step=global_step,
        key=f"val_stat{suff}/backward_end_inter",
    )
