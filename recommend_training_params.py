"""
recommend_training_params.py
-----------------------------
Standalone advisor: detects your GPU and reads your real dataset manifests,
then PRINTS a recommended set of training hyperparameters for
configs/experiment/train_seg.yaml (the SegFormer pipeline).

This script NEVER writes or modifies any file. You review the recommendation
and paste the values into the YAML yourself.

WHY THIS EXISTS: segformer_training.py used to auto-scale batch_size /
gradient_accumulation_steps to the detected GPU at runtime. That was removed
deliberately -- a fixed, explicit YAML you can read and quote is worth more
than a value that silently depends on which GPU happened to run it. This
script gives you the same calculation, but as a one-time, reviewed-by-you
recommendation instead of hidden runtime behaviour.

--data_dir has NO default and is REQUIRED: point it at your real
data/seg_training_<resize_mode> directory (train.jsonl/val.jsonl/test.jsonl,
mode-named since seg_map_calculations.py wrote it) -- there is no sensible
default across different machines/datasets/modes, so it's always explicit.

QUICK COMMANDS (run from repo root with conda loradapter env active):
  python recommend_training_params.py --data_dir data/seg_training_letterbox --epochs 10
  python recommend_training_params.py --data_dir data/seg_training_letterbox --device cuda:1

NOTE ON CONFIDENCE: the batch_size/gradient_checkpointing recommendation for a
~12GB GPU is MEASURED (real training runs, this project, 2026-07-02). For any
other GPU size, the recommendation is a REASONED linear extrapolation of that
measurement, NOT independently verified on such hardware -- always sanity
check with a short real dry run (see segformer_training.py's docstring) before
committing to a long training run.
"""

import argparse
from pathlib import Path

import torch


def _count_images(jsonl_path: Path) -> int | None:
    """Count non-empty lines in a JSONL manifest. Returns None if missing."""
    if not jsonl_path.exists():
        return None
    with open(jsonl_path, "r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def recommend_batch_and_accum(
    total_gb: float,
    baseline_gb: float = 12.0,
    baseline_batch: int = 4,
    baseline_accum: int = 4,
) -> tuple[int, int, int, int]:
    """
    Scale batch_size UP and gradient_accumulation_steps DOWN by the same
    ratio, so the EFFECTIVE batch (batch_size * accum) stays constant and
    learning_rate stays valid -- same formula validated (and bug-fixed: capped
    at the baseline effective batch once accum hits 1) in this project's
    now-removed runtime auto-scaler. Here it's just arithmetic for a printed
    recommendation, not something applied automatically.

    Returns: (recommended_batch, recommended_accum, uncapped_desired_batch,
              baseline_effective_batch)
    """
    baseline_effective = baseline_batch * baseline_accum
    ratio = total_gb / baseline_gb
    desired_batch = round(baseline_batch * ratio)
    # Cap at baseline_effective: beyond that point accum would need to be < 1
    # to hold the effective batch, which isn't possible -- see the printed
    # note below for what this means and how to use extra VRAM deliberately.
    new_batch = max(baseline_batch, min(desired_batch, baseline_effective))
    new_accum = max(1, round(baseline_effective / new_batch))
    return new_batch, new_accum, desired_batch, baseline_effective


def main():
    parser = argparse.ArgumentParser(
        description="Recommend training hyperparameters for your GPU + dataset (prints only).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--data_dir", type=str, default=None, required=True,
        help="Folder with train.jsonl/val.jsonl/test.jsonl to count real images from. "
             "REQUIRED, no default -- there is no sensible default across "
             "different machines/datasets. Example: --data_dir data/seg_training_letterbox.",
    )
    parser.add_argument("--epochs", type=int, default=5, help="Epochs to compute step totals for. Default: 5.")
    parser.add_argument("--device", type=str, default=None, help="cuda / cuda:N / cpu. Default: auto-detect.")
    args = parser.parse_args()

    # ---- 1. Detect GPU ----------------------------------------------------- #
    device = args.device if args.device is not None else ("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 70)
    print("GPU")
    print("=" * 70)
    total_gb = None
    if device == "cpu" or not torch.cuda.is_available():
        print("  No CUDA GPU detected -- training needs a GPU. Showing the 12GB-baseline")
        print("  values below as a placeholder; re-run this script once a GPU is visible.")
    else:
        idx = int(device.split(":")[1]) if ":" in device else 0
        props = torch.cuda.get_device_properties(idx)
        total_gb = props.total_memory / 1024**3
        print(f"  {props.name}  ({total_gb:.1f} GB)  [{device}]")

    # ---- 2. Read real dataset counts --------------------------------------- #
    data_dir = Path(args.data_dir)
    train_n = _count_images(data_dir / "train.jsonl")
    val_n = _count_images(data_dir / "val.jsonl")
    test_n = _count_images(data_dir / "test.jsonl")

    print()
    print("=" * 70)
    print(f"DATASET  ({data_dir})")
    print("=" * 70)
    if train_n is None:
        print(f"  No train.jsonl found in {data_dir} -- put your SegFormer")
        print(f"  (raw_image_path, seg_path, prompt) manifests there, then re-run.")
    else:
        print(f"  train: {train_n} images")
        print(f"  val  : {val_n if val_n is not None else '?'} images")
        print(f"  test : {test_n if test_n is not None else '?'} images")

    # ---- 3. Recommend batch_size / gradient_accumulation_steps ------------- #
    BASELINE_GB, BASELINE_BATCH, BASELINE_ACCUM = 12.0, 4, 4

    print()
    print("=" * 70)
    print("RECOMMENDED HYPERPARAMETERS")
    print("=" * 70)

    if total_gb is not None:
        batch_size, accum, desired_batch, baseline_effective = recommend_batch_and_accum(
            total_gb, BASELINE_GB, BASELINE_BATCH, BASELINE_ACCUM
        )
        confidence = (
            "MEASURED (real training runs, this project)"
            if abs(total_gb - BASELINE_GB) <= 1.0
            else "REASONED extrapolation -- NOT independently measured on this GPU size"
        )
        print(f"  data.batch_size              : {batch_size}   [{confidence}]")
        print(f"  gradient_accumulation_steps  : {accum}")
        print(f"  effective batch (batch*accum): {batch_size * accum}")
        print(f"  gradient_checkpointing       : true")
        if desired_batch > baseline_effective:
            print(f"\n  NOTE: this GPU could technically support batch_size~{desired_batch}, but that")
            print(f"  would need gradient_accumulation_steps < 1 (not possible) -- capped at")
            print(f"  batch_size={batch_size} to hold the SAME effective batch as the validated 12GB")
            print(f"  baseline. Using the rest of this GPU's VRAM would mean a genuinely LARGER")
            print(f"  effective batch -- a training-dynamics decision that also needs a reconsidered")
            print(f"  learning_rate. This tool won't choose that for you; it's a deliberate call.")
        if total_gb >= 24.0:
            print(f"\n  OPTIONAL: this GPU has {total_gb / BASELINE_GB:.1f}x the 12GB baseline VRAM --")
            print(f"  gradient_checkpointing=false would save ~20% compute time if you want to try it,")
            print(f"  but this has NOT been measured on a GPU this size. Sanity-check with a short")
            print(f"  real dry run before committing to a full training run.")
    else:
        batch_size, accum = BASELINE_BATCH, BASELINE_ACCUM
        print(f"  data.batch_size              : {batch_size}   [placeholder, no GPU detected]")
        print(f"  gradient_accumulation_steps  : {accum}")
        print(f"  gradient_checkpointing       : true")

    # ---- 4. Learning rate (never auto-scaled) ------------------------------ #
    print()
    print(f"  learning_rate                : 1.0e-4   [fixed]")
    print(f"    Not scaled with batch size. This project's effective batch (16) matches what's")
    print(f"    already been validated by real training runs at this LR. A LARGER effective batch")
    print(f"    is a genuine training-dynamics decision needing its own LR reconsideration -- your")
    print(f"    call to make, not something this tool decides (no verified scaling result exists).")

    # ---- 5. Epoch / step math from the REAL dataset size ------------------- #
    if train_n is not None:
        effective_batch = batch_size * accum
        steps_per_epoch = -(-train_n // effective_batch)  # ceil division
        total_steps = steps_per_epoch * args.epochs
        val_steps = max(1, round(steps_per_epoch / 7))    # ~7 val checks/epoch
        ckpt_steps = max(1, round(steps_per_epoch / 3.5))  # ~3-4 checkpoint saves/epoch

        print()
        print(f"  epochs                       : {args.epochs}")
        print(f"  steps_per_epoch              : {steps_per_epoch}  (ceil({train_n} / {effective_batch}))")
        print(f"  total optimizer steps        : {total_steps}")
        print(f"  val_steps                    : {val_steps}   (~7 val/loss checks per epoch;")
        print(f"                                  round to a clean number like 500 if close enough)")
        print(f"  ckpt_steps                   : {ckpt_steps}  (~3-4 checkpoint saves per epoch;")
        print(f"                                  round to a clean number like 1000 if close enough)")

    print()
    print(f"  n_grid_images                : 10   (5 fixed + 5 fresh -- already-validated default)")
    print(f"  size                         : [512, 320]  (width, height -- non-square, no pad band,")
    print(f"                                  caps long side at SD1.5's native 512)")
    print(f"  resize_mode                  : aspect")
    print(f"    aspect: direct resize to a non-square (width, height) target chosen close to")
    print(f"    the source aspect ratio -- NO pad, NO crop.")
    print(f"    letterbox (SquarePad) keeps 100% of the scene but adds a pad band.")
    print(f"    CenterCrop (original stock LoRAdapter recipe) has no pad band, crops scene edges.")
    print(f"    UNLIKE the grounded_sam branch, this mode is baked into the SAVED MAP -- you")
    print(f"    must run seg_map_calculations.py with the SAME --resize_mode (and, for aspect,")
    print(f"    the SAME --width/--height) before training.")

    # ---- 6. Ready-to-paste snippet ------------------------------------------ #
    print()
    print("=" * 70)
    print(f"PASTE INTO configs/experiment/train_seg.yaml")
    print("=" * 70)
    print(f"gradient_checkpointing: true")
    print(f"gradient_accumulation_steps: {accum}")
    if train_n is not None:
        print(f"epochs: {args.epochs}")
        print(f"val_steps: {val_steps}")
        print(f"ckpt_steps: {ckpt_steps}")
    print(f"size: [512, 320]")
    print(f"resize_mode: aspect")
    print(f"data:")
    print(f"  batch_size: {batch_size}")
    print()


if __name__ == "__main__":
    main()
