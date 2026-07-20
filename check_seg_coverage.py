"""
STANDALONE DIAGNOSTIC — not part of any of the 4 pipeline stages.

WHY THIS EXISTS
----------------
analyze_car_coverage.py answered one fixed question ("how much car-class
coverage exists across the whole training set?") for one fixed class (car).
This script generalises that into a per-IMAGE, per-CLASS check: given any
single seg map (e.g. a CARLA test frame's segmentation output), it reports,
for every class present in that image, how that image's coverage compares to
the ENTIRE training set's distribution for that same class. That tells you,
before you even generate anything, whether a given scene's composition is
something the model has actually seen examples of, or is likely
out-of-distribution (the same failure mode diagnosed for the CARLA truck:
17.8% was the training set's OWN maximum car coverage, and the CARLA frame
almost certainly exceeds it).

WHAT YOU PASS IN
-----------------
--seg_map   : the SINGLE query segmentation map to check (e.g. the seg map
              you already have for a CARLA frame). Either:
                (a) a raw class-ID PNG (single-channel, values 0..18) --
                    exactly what seg_map_calculations.py / seg_path entries
                    in the training manifest already are, or
                (b) an RGB colourised seg map (e.g. grounded_sam_inference.py's
                    visual output, or a palette-coloured PNG) -- pixels are
                    matched to the nearest SEG_CITYSCAPES_PALETTE colour to
                    recover class IDs. A JPEG photo of a screen (heavy
                    compression + moire) will give NOISY nearest-colour
                    matches -- prefer an actual PNG from the pipeline output
                    when you can, and treat (b) results as approximate.
--json_file : the TRAINING manifest to compare against (e.g.
              data/grounded_sam/train.json) -- same file Stage D reads.
--cache_file: OPTIONAL. Precomputing per-class coverage for tens of
              thousands of training images is the slow part; pass a path
              (e.g. data/grounded_sam/train_coverage_cache.npz) to save it
              once and reuse instantly on every later query against the same
              manifest. Deleted/ignored automatically if the manifest's
              entry count no longer matches (e.g. you rebuilt the manifest).

USAGE
------
    python check_seg_coverage.py \\
        --seg_map path/to/carla_frame_seg.png \\
        --json_file data/grounded_sam/train.json \\
        --cache_file data/grounded_sam/train_coverage_cache.npz
"""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

# SEG_CITYSCAPES_PALETTE is the project's single source of truth for
# class-id -> colour (src/encoders/seg_encoder.py). Importing it here (rather
# than re-typing the table) means this diagnostic can never silently drift
# out of sync with the real encoder. The import is cheap: seg_encoder.py only
# imports torch at module level; the heavy `transformers` import lives inside
# a method and is never triggered just by reading this constant.
from src.encoders.seg_encoder import SEG_CITYSCAPES_PALETTE

# Class NAMES are diagnostic-only labels, not used by training -- they must
# stay in the exact order of SEG_CITYSCAPES_PALETTE (index == class id).
CLASS_NAMES = [
    "road", "sidewalk", "building", "wall", "fence", "pole",
    "traffic light", "traffic sign", "vegetation", "terrain", "sky",
    "person", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle",
]
NUM_CLASSES = len(SEG_CITYSCAPES_PALETTE)
assert len(CLASS_NAMES) == NUM_CLASSES, "CLASS_NAMES / SEG_CITYSCAPES_PALETTE length mismatch"


def _class_fractions_from_ids(ids: np.ndarray) -> np.ndarray:
    """ids: HxW array of class indices 0..NUM_CLASSES-1 -> per-class pixel fraction."""
    counts = np.bincount(ids.ravel(), minlength=NUM_CLASSES)[:NUM_CLASSES]
    return counts / counts.sum()


def load_query_seg_map(path: Path) -> np.ndarray:
    """Load a query seg map (raw ID PNG or RGB colourised PNG/JPEG) -> per-class fractions."""
    img = Image.open(path)
    if img.mode in ("L", "I", "I;16"):
        ids = np.array(img.convert("L"))
        if ids.max() >= NUM_CLASSES:
            raise ValueError(
                f"{path}: grayscale image has values up to {ids.max()}, but only "
                f"{NUM_CLASSES} classes exist (0..{NUM_CLASSES - 1}). This looks like "
                f"a normal photo, not a class-ID map -- pass a raw seg_path PNG or an "
                f"RGB colourised seg map instead."
            )
        return _class_fractions_from_ids(ids)

    # RGB path: nearest-palette-colour matching.
    print(f"[check_seg_coverage] '{path.name}' is RGB -> matching pixels to nearest "
          f"SEG_CITYSCAPES_PALETTE colour (approximate; noisier if this came from a "
          f"JPEG or a photographed screen).")
    rgb = np.array(img.convert("RGB")).reshape(-1, 3).astype(np.int32)
    palette = np.array(SEG_CITYSCAPES_PALETTE, dtype=np.int32)  # [19,3]
    # squared distance from every pixel to every palette colour -> [N,19], argmin -> [N]
    dists = ((rgb[:, None, :] - palette[None, :, :]) ** 2).sum(axis=2)
    ids = dists.argmin(axis=1)
    return _class_fractions_from_ids(ids)


def build_or_load_training_distribution(json_file: Path, cache_file: Path | None) -> tuple[np.ndarray, int]:
    """Returns (fractions [N,NUM_CLASSES], N). Uses cache_file if present and matching."""
    entries = [json.loads(line) for line in json_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    n = len(entries)

    if cache_file is not None and cache_file.exists():
        cached = np.load(cache_file)
        if int(cached["n"]) == n:
            print(f"[check_seg_coverage] Loaded cached distribution ({n} images) from {cache_file}")
            return cached["fractions"], n
        print(f"[check_seg_coverage] Cache at {cache_file} has {int(cached['n'])} images, "
              f"manifest now has {n} -- recomputing.")

    print(f"[check_seg_coverage] Computing per-class coverage for {n} training images "
          f"(one-time cost; pass --cache_file to reuse this next time)...")
    fractions = np.zeros((n, NUM_CLASSES), dtype=np.float32)
    missing = 0
    for i, item in enumerate(entries):
        seg_path = Path(item["seg_path"])
        if not seg_path.exists():
            missing += 1
            continue
        ids = np.array(Image.open(seg_path).convert("L"))
        fractions[i] = _class_fractions_from_ids(ids)
        if (i + 1) % 5000 == 0:
            print(f"  ...{i + 1}/{n}")
    if missing:
        print(f"[check_seg_coverage] WARNING: {missing} seg_path files not found, left as zero-rows")

    if cache_file is not None:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache_file, fractions=fractions, n=n)
        print(f"[check_seg_coverage] Cached to {cache_file}")

    return fractions, n


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seg_map", required=True, help="query seg map to check (raw ID PNG or RGB colourised)")
    ap.add_argument("--json_file", required=True, help="training manifest, e.g. data/grounded_sam/train.json")
    ap.add_argument("--cache_file", default=None, help="optional .npz cache path to reuse across queries")
    args = ap.parse_args()

    query_fractions = load_query_seg_map(Path(args.seg_map))
    train_fractions, n = build_or_load_training_distribution(
        Path(args.json_file), Path(args.cache_file) if args.cache_file else None
    )

    print(f"\nComparing '{args.seg_map}' against {n} training images:\n")
    header = f"{'class':<14}{'query %':>10}{'train mean %':>14}{'train max %':>13}{'images >= query':>18}"
    print(header)
    print("-" * len(header))

    # Only report classes actually present in the query image, largest first --
    # a class the query doesn't have can't be out-of-distribution for this scene.
    present = [c for c in range(NUM_CLASSES) if query_fractions[c] > 0]
    present.sort(key=lambda c: query_fractions[c], reverse=True)

    for c in present:
        q = query_fractions[c] * 100
        train_mean = train_fractions[:, c].mean() * 100
        train_max = train_fractions[:, c].max() * 100
        # % of training images whose coverage of this class meets or beats the query --
        # low = rare/out-of-distribution composition, same logic as the car-coverage finding.
        pct_meeting_or_beating = 100 * (train_fractions[:, c] >= query_fractions[c]).mean()
        flag = "  <-- RARE IN TRAINING" if pct_meeting_or_beating < 1.0 else ""
        print(f"{CLASS_NAMES[c]:<14}{q:>9.2f}%{train_mean:>13.2f}%{train_max:>12.2f}%"
              f"{pct_meeting_or_beating:>17.2f}%{flag}")

    print("\n'images >= query' = % of the training set with AT LEAST this much of that "
          "class in frame. Under ~1% means this scene's composition for that class is "
          "rare or absent in training -- the model likely wasn't taught to render it well "
          "(this is exactly what happened with the CARLA truck: max car coverage in "
          "59,766 training images was 17.8%, so anything above that is guaranteed rare).")


if __name__ == "__main__":
    main()
