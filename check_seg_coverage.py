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
--seg_map     : the SINGLE query segmentation map to check (e.g. the seg map
                you already have for a CARLA frame). Either:
                  (a) a raw class-ID PNG (single-channel, values 0..28 for the
                      locked CARLA taxonomy) -- exactly what seg_path entries
                      in the training manifest already are, or
                  (b) an RGB colourised seg map (e.g. grounded_sam_inference.py's
                      visual output) -- pixels are matched to the nearest
                      colour in --classes_file's palette to recover class IDs.
                      A JPEG photo of a screen (heavy compression + moire)
                      will give NOISY nearest-colour matches -- prefer an
                      actual PNG from the pipeline output when you can, and
                      treat (b) results as approximate.
--json_file   : the TRAINING manifest to compare against (e.g.
                data/grounded_sam/train.json).
--classes_file: the CARLA class-definition JSON (id->name/colour). Default
                configs/grounded_sam_classes.json — this project's LOCKED
                29-class taxonomy. MUST match whatever classes_file the seg
                maps were actually saved/coloured with, or class ids and
                names will disagree.
--cache_file  : OPTIONAL. Precomputing per-class coverage for tens of
                thousands of training images is the slow part; pass a path
                (e.g. data/grounded_sam/train_coverage_cache.npz) to save it
                once and reuse instantly on every later query against the
                same manifest. Deleted/ignored automatically if the
                manifest's entry count no longer matches (e.g. you rebuilt
                the manifest).

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

# The real, LOCKED CARLA class taxonomy (id->name/colour) lives in a JSON
# file, not a hardcoded table -- this project's class set is open/configured,
# not a fixed pretrained taxonomy. Reusing load_grounded_sam_palette (the
# SAME function training and inference use) means this diagnostic can never
# silently drift out of sync with the real palette.
from src.encoders.grounded_sam_encoder import load_grounded_sam_palette

DEFAULT_CLASSES_FILE = "configs/grounded_sam_classes.json"


def _class_fractions_from_ids(ids: np.ndarray, num_classes: int) -> np.ndarray:
    """ids: HxW array of class indices 0..num_classes-1 -> per-class pixel fraction."""
    counts = np.bincount(ids.ravel(), minlength=num_classes)[:num_classes]
    return counts / counts.sum()


def load_query_seg_map(path: Path, palette: np.ndarray, num_classes: int) -> np.ndarray:
    """Load a query seg map (raw ID PNG or RGB colourised PNG/JPEG) -> per-class fractions."""
    img = Image.open(path)
    if img.mode in ("L", "I", "I;16"):
        ids = np.asarray(img).astype(np.int64)   # raw read -- handles 8-bit "L" and 16-bit "I;16" alike
        if ids.max() >= num_classes:
            raise ValueError(
                f"{path}: grayscale image has values up to {ids.max()}, but only "
                f"{num_classes} classes exist (0..{num_classes - 1}). This looks like "
                f"a normal photo, not a class-ID map -- pass a raw seg_path PNG or an "
                f"RGB colourised seg map instead."
            )
        return _class_fractions_from_ids(ids, num_classes)

    # RGB path: nearest-palette-colour matching.
    print(f"[check_seg_coverage] '{path.name}' is RGB -> matching pixels to nearest "
          f"classes_file colour (approximate; noisier if this came from a "
          f"JPEG or a photographed screen).")
    rgb = np.array(img.convert("RGB")).reshape(-1, 3).astype(np.int32)
    # squared distance from every pixel to every palette colour -> [N,num_classes], argmin -> [N]
    dists = ((rgb[:, None, :] - palette[None, :, :]) ** 2).sum(axis=2)
    ids = dists.argmin(axis=1)
    return _class_fractions_from_ids(ids, num_classes)


def build_or_load_training_distribution(
    json_file: Path, cache_file: Path | None, num_classes: int
) -> tuple[np.ndarray, int]:
    """Returns (fractions [N,num_classes], N). Uses cache_file if present and matching."""
    entries = [json.loads(line) for line in json_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    n = len(entries)

    if cache_file is not None and cache_file.exists():
        cached = np.load(cache_file)
        if int(cached["n"]) == n and int(cached["fractions"].shape[1]) == num_classes:
            print(f"[check_seg_coverage] Loaded cached distribution ({n} images) from {cache_file}")
            return cached["fractions"], n
        print(f"[check_seg_coverage] Cache at {cache_file} doesn't match (image count or "
              f"class count changed) -- recomputing.")

    print(f"[check_seg_coverage] Computing per-class coverage for {n} training images "
          f"(one-time cost; pass --cache_file to reuse this next time)...")
    fractions = np.zeros((n, num_classes), dtype=np.float32)
    missing = 0
    for i, item in enumerate(entries):
        seg_path = Path(item["seg_path"])
        if not seg_path.exists():
            missing += 1
            continue
        ids = np.asarray(Image.open(seg_path)).astype(np.int64)   # raw read, see load_query_seg_map
        fractions[i] = _class_fractions_from_ids(ids, num_classes)
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
    ap.add_argument("--classes_file", default=DEFAULT_CLASSES_FILE,
                     help=f"CARLA class-definition JSON (id->name/colour). Default: {DEFAULT_CLASSES_FILE}. "
                          f"MUST match whatever classes_file the seg maps were saved/coloured with.")
    ap.add_argument("--cache_file", default=None, help="optional .npz cache path to reuse across queries")
    args = ap.parse_args()

    class_names, palette_list = load_grounded_sam_palette(Path(args.classes_file))
    num_classes = len(class_names)
    palette = np.array(palette_list, dtype=np.int32)   # [num_classes, 3]
    print(f"[check_seg_coverage] loaded {num_classes} classes from {args.classes_file}")

    query_fractions = load_query_seg_map(Path(args.seg_map), palette, num_classes)
    train_fractions, n = build_or_load_training_distribution(
        Path(args.json_file), Path(args.cache_file) if args.cache_file else None, num_classes
    )

    print(f"\nComparing '{args.seg_map}' against {n} training images:\n")
    header = f"{'class':<16}{'query %':>10}{'train mean %':>14}{'train max %':>13}{'images >= query':>18}"
    print(header)
    print("-" * len(header))

    # Only report classes actually present in the query image, largest first --
    # a class the query doesn't have can't be out-of-distribution for this scene.
    present = [c for c in range(num_classes) if query_fractions[c] > 0]
    present.sort(key=lambda c: query_fractions[c], reverse=True)

    for c in present:
        q = query_fractions[c] * 100
        train_mean = train_fractions[:, c].mean() * 100
        train_max = train_fractions[:, c].max() * 100
        # % of training images whose coverage of this class meets or beats the query --
        # low = rare/out-of-distribution composition, same logic as the car-coverage finding.
        pct_meeting_or_beating = 100 * (train_fractions[:, c] >= query_fractions[c]).mean()
        flag = "  <-- RARE IN TRAINING" if pct_meeting_or_beating < 1.0 else ""
        print(f"{class_names[c]:<16}{q:>9.2f}%{train_mean:>13.2f}%{train_max:>12.2f}%"
              f"{pct_meeting_or_beating:>17.2f}%{flag}")

    print("\n'images >= query' = % of the training set with AT LEAST this much of that "
          "class in frame. Under ~1% means this scene's composition for that class is "
          "rare or absent in training -- the model likely wasn't taught to render it well "
          "(this is exactly what happened with the CARLA truck: max car coverage in "
          "59,766 training images was 17.8%, so anything above that is guaranteed rare).")


if __name__ == "__main__":
    main()
