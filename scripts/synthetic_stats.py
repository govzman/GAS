"""Print statistics for a collated synthetic teacher dataset."""

import argparse
import pickle

import torch


def summary(name, values):
    values = values.detach().float().reshape(-1)
    print(
        f"{name}: n={values.numel()} mean={values.mean():.6g} "
        f"std={values.std(unbiased=False):.6g} "
        f"min={values.min():.6g} max={values.max():.6g}"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", help="Path to the collated dataset.pkl")
    args = parser.parse_args()

    # Only load trusted datasets: pickle can execute code during loading.
    with open(args.dataset, "rb") as f:
        data = pickle.load(f)

    feature = data.get("synthetic_prompt_feature")
    if isinstance(feature, torch.Tensor):
        summary("synthetic_prompt_feature", feature)
    else:
        print("synthetic_prompt_feature: missing")

    prefix = "manual_solver_params."
    active_stds = []
    for name in sorted(k for k in data if k.startswith(prefix)):
        values = data[name]
        if not isinstance(values, torch.Tensor) or values.ndim != 2:
            print(f"{name}: skipped (expected tensor [samples, steps])")
            continue
        short_name = name[len(prefix):]
        if not (short_name.startswith(("a", "c")) and short_name.endswith("_diff")):
            continue
        try:
            order = int(short_name[1:-5])
        except ValueError:
            continue
        print(f"\n{short_name} shape={tuple(values.shape)}")
        for step in range(values.shape[1]):
            is_active = step >= order - 1
            per_sample = values[:, step].float()
            summary(f"  step {step} ({'active' if is_active else 'inactive'})", per_sample)
            if is_active:
                active_stds.append(per_sample.std(unbiased=False))

    if active_stds:
        summary("\nactive coefficient std across samples", torch.stack(active_stds))
    else:
        print("No compatible a/c coefficient tensors found.")


if __name__ == "__main__":
    main()
