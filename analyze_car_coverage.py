"""
STANDALONE DIAGNOSTIC — not part of any of the 4 pipeline stages.

PURPOSE
--------
Checks training-data COVERAGE for a given class: a warped/generic object
blob at generation time, for a class whose seg map region is structurally
fine, points at a coverage gap rather than a segmentation- or
checkpoint-quality problem — the model may simply not have seen enough
training examples where that class fills a large fraction of the frame.

This script walks your seg training manifest (the same train.json Stage D
reads) and, for every sample, computes what fraction of the seg map's pixels
are the target class. It buckets that fraction into a histogram so you can
see, at a glance, whether "large/close object" frames (e.g. class-pixel
fraction > 15%) are rare or absent in your real training set.

CAR CLASS ID — pass --car_class_id, don't trust a default: this project's
locked CARLA taxonomy (configs/grounded_sam_classes.json) has "Car" at id
14. There is deliberately no default to fall back on — using the wrong id
silently measures a DIFFERENT class with no error, so it's always explicit.

If large-car-fraction frames turn out to be rare: the fix is adding/upsampling
such examples in training, NOT more epochs on the current data mix (more
passes over data that lacks the pattern won't teach the model the pattern).
If large-car-fraction frames turn out to be common: this hypothesis is wrong
and we look elsewhere (e.g. classifier-free-guidance behavior, LoRA rank/
capacity for high-frequency vehicle detail, etc.) — report the histogram back
before concluding anything either way.

USAGE (run on whichever machine holds the real grounded_sam manifest+PNGs —
this repo's local 913-image folder is a TEST set, not the real training data):
    python analyze_car_coverage.py --json_file data/grounded_sam/train.json --car_class_id 14
"""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def _resolve(p: str, image_root: str | None) -> Path:
    path = Path(p)
    if image_root and not path.is_absolute():
        path = Path(image_root) / path
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json_file", required=True, help="grounded_sam manifest, e.g. data/grounded_sam/train.json")
    ap.add_argument("--image_root", default=None, help="prefix for relative seg_path entries, if any")
    ap.add_argument(
        "--car_class_id", type=int, required=True,
        help="REQUIRED, no default -- this project's locked CARLA taxonomy "
             "(configs/grounded_sam_classes.json) has 'Car' at id 14. Passing "
             "the wrong id silently measures a different class, so there is no "
             "safe default to fall back on.",
    )
    args = ap.parse_args()
    CAR_CLASS_ID = args.car_class_id

    manifest_path = Path(args.json_file)
    entries = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    print(f"[analyze_car_coverage] {len(entries)} entries in {manifest_path}")

    fractions: list[float] = []
    missing = 0
    for item in entries:
        seg_path = _resolve(item["seg_path"], args.image_root)
        if not seg_path.exists():
            missing += 1
            continue
        # seg PNGs are 8-bit single-channel class-ID maps (raw pixel value == class id).
        ids = np.array(Image.open(seg_path).convert("L"))
        car_fraction = float((ids == CAR_CLASS_ID).mean())
        fractions.append(car_fraction)

    if missing:
        print(f"[analyze_car_coverage] WARNING: {missing} seg_path files not found, skipped")
    if not fractions:
        print("[analyze_car_coverage] No usable entries — nothing to report.")
        return

    arr = np.array(fractions)
    bins = [0.0, 0.01, 0.05, 0.10, 0.15, 0.20, 0.30, 1.01]
    labels = ["0-1%", "1-5%", "5-10%", "10-15%", "15-20%", "20-30%", "30%+"]
    counts, _ = np.histogram(arr, bins=bins)

    print("\ncar-class pixel-fraction histogram (per training image):")
    for label, count in zip(labels, counts):
        pct = 100 * count / len(arr)
        bar = "#" * int(pct / 2)
        print(f"  {label:>7}: {count:6d} ({pct:5.1f}%)  {bar}")

    dominant = int((arr > 0.15).sum())
    print(f"\nTotal images: {len(arr)}")
    print(f"Images with car covering >15% of frame (CARLA-truck-like): {dominant} "
          f"({100*dominant/len(arr):.2f}%)")
    print(f"Mean car-pixel fraction: {arr.mean():.4f}   Max: {arr.max():.4f}")


if __name__ == "__main__":
    main()
