import os
import pickle
from collections import defaultdict
from typing import Any, Dict

import click
import torch
from tqdm import tqdm

MANUAL_PREFIX = "manual_solver_params"


def _batch_size_from_checkpoint(checkpoint: Dict[str, Any]) -> int:
    noise = checkpoint.get("noise")
    if isinstance(noise, torch.Tensor):
        return int(noise.shape[0])
    cond = checkpoint.get("condition")
    if isinstance(cond, list):
        return len(cond)
    latents = checkpoint.get("latents")
    if isinstance(latents, torch.Tensor):
        return int(latents.shape[0])
    raise ValueError(
        "Cannot infer batch size: expected tensor 'noise' or list 'condition' in checkpoint."
    )


def _append_manual_solver_params(
    data: defaultdict,
    params: Any,
    batch_size: int,
) -> None:
    if params is None:
        return
    if not isinstance(params, dict):
        raise NotImplementedError(
            f"{MANUAL_PREFIX} must be dict[str, Tensor], got {type(params)}"
        )
    for sub_k, sub_v in sorted(params.items()):
        flat_key = f"{MANUAL_PREFIX}.{sub_k}"
        if isinstance(sub_v, torch.Tensor):
            if int(sub_v.shape[0]) != batch_size:
                raise ValueError(
                    f"{flat_key}: batch dim {sub_v.shape[0]} != inferred batch_size {batch_size}"
                )
            data[flat_key].append(sub_v)
        else:
            raise NotImplementedError(
                f"Unsupported type {type(sub_v)} for {flat_key} (expected Tensor)"
            )


def _tensor_summary(x: torch.Tensor) -> str:
    xf = x.detach().float().cpu()
    return (
        f"shape={tuple(x.shape)} dtype={x.dtype} "
        f"mean={xf.mean().item():.6g} std={xf.std(unbiased=False).item():.6g} "
        f"min={xf.min().item():.6g} max={xf.max().item():.6g}"
    )


def print_ground_truth_solver_params(data: Dict[str, Any]) -> None:
    """Print collated optimal / teacher GS tensors saved under manual_solver_params.*"""
    gt_keys = sorted(k for k in data if k.startswith(f"{MANUAL_PREFIX}."))
    print("\n=== Ground-truth GS params (manual_solver_params.*, collated) ===")
    if not gt_keys:
        print("(no manual_solver_params tensors found)")
        return
    for k in gt_keys:
        v = data[k]
        if isinstance(v, torch.Tensor):
            print(f"{k}: {_tensor_summary(v)}")
        else:
            print(f"{k}: (non-tensor, len={len(v)})")


def compare_with_learned(
    gt_data: Dict[str, Any],
    learned_path: str,
    learned_prefix: str,
    gt_prefix: str,
) -> None:
    """
    Compare tensors in learned pkl to ground-truth keys.

    Expected layout:
      - gt pkl: keys like ``manual_solver_params.a1_diff``
      - learned pkl: same suffix after replacing gt_prefix -> learned_prefix,
        e.g. ``predicted_solver_params.a1_diff``
    """
    with open(learned_path, "rb") as f:
        learned = pickle.load(f)

    print("\n=== Learned vs ground-truth (per-key MSE / MAE) ===")
    n_compared = 0
    for k_gt in sorted(gt_data.keys()):
        if not k_gt.startswith(f"{gt_prefix}."):
            continue
        suffix = k_gt[len(gt_prefix) + 1 :]
        k_learned = f"{learned_prefix}.{suffix}"
        if k_learned not in learned:
            print(f"skip {k_gt}: no learned key {k_learned}")
            continue
        a = gt_data[k_gt]
        b = learned[k_learned]
        if not isinstance(a, torch.Tensor) or not isinstance(b, torch.Tensor):
            print(f"skip {k_gt}: non-tensor types")
            continue
        if a.shape != b.shape:
            print(f"skip {k_gt}: shape mismatch gt{a.shape} vs learned{b.shape}")
            continue
        af = a.detach().float()
        bf = b.detach().float()
        mse = torch.mean((af - bf) ** 2).item()
        mae = torch.mean(torch.abs(af - bf)).item()
        print(f"{suffix}: MSE={mse:.6g} MAE={mae:.6g} {_tensor_summary(af)} vs learned {_tensor_summary(bf)}")
        n_compared += 1
    if n_compared == 0:
        print("(no overlapping tensor keys to compare)")


@click.command()
@click.option(
    "--synt_dir",
    help="Path to the teacher dir",
    metavar="PATH",
    type=str,
    required=True,
)
@click.option(
    "--out_pkl", help="Path to pkl dataset", metavar="PATH", type=str, required=True
)
@click.option(
    "--num_samples",
    help="Number of samples to add to the final dataset",
    type=int,
    default=50000,
    show_default=True,
)
@click.option(
    "--print_gt_params/--no_print_gt_params",
    default=True,
    show_default=True,
    help="Print summary stats for all collated manual_solver_params.* tensors.",
)
@click.option(
    "--compare_learned_pkl",
    type=str,
    default=None,
    help="Optional second pickle with predicted coeffs (same batch order / shapes).",
)
@click.option(
    "--learned_key_prefix",
    type=str,
    default="predicted_solver_params",
    show_default=True,
    help="Prefix for keys in compare_learned_pkl (e.g. predicted_solver_params.a1_diff).",
)
@click.option(
    "--gt_key_prefix",
    type=str,
    default=MANUAL_PREFIX,
    show_default=True,
    help="Prefix for ground-truth keys in collated data (default: manual_solver_params).",
)
def main(
    synt_dir,
    out_pkl,
    num_samples,
    print_gt_params,
    compare_learned_pkl,
    learned_key_prefix,
    gt_key_prefix,
):
    """Collate teacher ``.pt`` shards into one pickle.

    Nested ``manual_solver_params`` dicts are flattened to keys
    ``manual_solver_params.<param_name>`` so batches concatenate like other tensors.

    For comparing a trained predictor to these targets, dump predictions with keys
    ``<learned_key_prefix>.<param_name>`` (same ``param_name`` as in teacher shards).

    Example:

    \b
    python collate.py --synt_dir=dir_synt --out_pkl=out_name.pkl --num_samples=100 \\
      --compare_learned_pkl preds.pkl --learned_key_prefix predicted_solver_params
    """
    assert os.path.splitext(out_pkl)[1] == ".pkl"

    paths = sorted(
        f for f in os.listdir(synt_dir) if f.endswith(".pt") or f.endswith(".pth")
    )
    data = defaultdict(list)

    pbar = tqdm(total=num_samples)
    total_samples = 0

    for p in paths:
        ckpt_path = os.path.join(synt_dir, p)
        checkpoint = torch.load(ckpt_path, weights_only=False)
        batch_size = _batch_size_from_checkpoint(checkpoint)

        for k, v in checkpoint.items():
            if k == MANUAL_PREFIX:
                _append_manual_solver_params(data, v, batch_size)
                continue
            if isinstance(v, torch.Tensor):
                if int(v.shape[0]) != batch_size:
                    raise ValueError(
                        f"{k}: leading dim {v.shape[0]} != batch_size {batch_size}"
                    )
                data[k].append(v)
            elif isinstance(v, list):
                if len(v) != batch_size:
                    raise ValueError(
                        f"{k}: list len {len(v)} != batch_size {batch_size}"
                    )
                data[k].extend(v)
            else:
                raise NotImplementedError(f"Unknown {type(v)} type for key {k!r}")

        total_samples += batch_size
        pbar.update(batch_size)

        if total_samples >= num_samples:
            break

    pbar.close()

    for k, v in list(data.items()):
        if not v:
            continue
        if isinstance(v[0], torch.Tensor):
            data[k] = torch.cat(v, dim=0)
        else:
            data[k] = v
        data[k] = data[k][:num_samples]

        expected_len = num_samples
        if isinstance(data[k], torch.Tensor):
            actual = data[k].shape[0]
        else:
            actual = len(data[k])
        assert actual == expected_len, (
            f"Key {k!r}: length {actual}, expected {expected_len}"
        )

    if print_gt_params:
        print_ground_truth_solver_params(data)

    if compare_learned_pkl:
        compare_with_learned(data, compare_learned_pkl, learned_key_prefix, gt_key_prefix)

    with open(out_pkl, "wb") as f:
        pickle.dump(dict(data), f)

    print(f"\nWrote {out_pkl} with keys: {sorted(data.keys())}")


# ----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

# ----------------------------------------------------------------------------
