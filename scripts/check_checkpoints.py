"""Inspect recent trusted checkpoints on CPU, without creating any model."""
import argparse
from collections.abc import Mapping
import gc
from pathlib import Path


DEFAULT_KEYS = [f"{side}{i}_diff" for side in ("a", "c") for i in range(1, 4)]


def recent_files(root, limit, per_run):
    files = sorted(
        (p for p in root.rglob("*") if p.is_file() and p.suffix in (".pt", ".pth")),
        key=lambda p: (p.stat().st_mtime_ns, str(p)), reverse=True,
    )
    if per_run:
        parents, selected = set(), []
        for path in files:
            if path.parent not in parents:
                selected.append(path)
                parents.add(path.parent)
        files = selected
    return files[:limit]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?", default="checkpoints", type=Path)
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--per-run", action="store_true",
                        help="Inspect the most recently modified file in each of the latest run folders")
    parser.add_argument("--required", nargs="+", default=DEFAULT_KEYS,
                        help="Required model state_dict keys; defaults to global a/c parameters")
    args = parser.parse_args()
    if not args.root.is_dir() or args.limit < 1:
        parser.error("root must be an existing directory and --limit must be positive")
    paths = recent_files(args.root, args.limit, args.per_run)
    if not paths:
        parser.error(f"No .pt/.pth files found under {args.root}")

    import torch

    print("Ordered by file modification time, newest first.")
    print("Checking field presence only, not complete architecture or dataset compatibility.")
    matches = []
    for path in paths:
        checkpoint = state = None
        print(f"\n{path}", flush=True)
        try:
            # The project uses ordinary torch.save checkpoints. CPU mapping
            # avoids CUDA initialization; only inspect your own trusted files.
            checkpoint = torch.load(str(path), map_location="cpu", weights_only=False)
            if not isinstance(checkpoint, Mapping):
                raise ValueError("Expected checkpoint dictionary")
            state = checkpoint.get("model", checkpoint.get("state_dict", checkpoint))
            if not isinstance(state, Mapping):
                raise ValueError("Expected model state_dict dictionary")
            missing = [key for key in args.required if key not in state]
            conditional = any(key.startswith("ac_coeff_model.") for key in state)
            print(f"  saved step: {checkpoint.get('step', 'unknown')}; EMA: {'ema' in checkpoint}")
            print(f"  a/c predictor network: {conditional}")
            print(f"  required fields: {'PRESENT' if not missing else 'MISSING'}")
            for key in args.required:
                value = state.get(key)
                print(f"    {key}: {tuple(value.shape) if hasattr(value, 'shape') else 'missing/non-tensor'}")
            if missing:
                print("  missing: " + ", ".join(missing))
            global_keys = [key for key in state if key.endswith("_global")]
            if global_keys:
                print("  conditional global-base fields: " + ", ".join(sorted(global_keys)))
            if not missing:
                matches.append(str(path))
        except Exception as error:
            print(f"  ERROR: {type(error).__name__}: {error}")
        finally:
            checkpoint = state = None
            gc.collect()
    print("\nFiles containing all required fields:")
    print("\n".join(matches) if matches else "None among inspected files. Try --per-run or a larger --limit.")


if __name__ == "__main__":
    main()
