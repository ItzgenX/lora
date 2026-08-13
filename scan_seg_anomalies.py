"""
STANDALONE DIAGNOSTIC — not part of any of the 4 pipeline stages.

WHY THIS EXISTS
----------------
At 70K images, nobody can hand-pick which ones to check. This script scans
the WHOLE manifest and flags images whose seg map looks like a SegFormer
failure (a single class swallowing most of the frame, or almost no distinct
classes present) — the same pattern seen in a real night-highway sample where
a lime-green "vegetation" blob covered most of a scene with no vegetation in
it. It also tags each image day/night by raw-image brightness, so you get an
empirical answer to "is this really a night-specific problem" instead of a
guess.

FLAG RULES (both cheap, both explainable — no per-class hand-tuned
thresholds):
  1. DOMINANT-CLASS: one class covers more than --dominance_threshold
     (default 60%) of the frame.
  2. LOW-DIVERSITY: fewer than --min_classes (default 3) distinct classes
     present at all in the seg map.
  3. NO-ROAD: the "road" class (id 0) covers less than --road_min_fraction
     (default 1%) of the frame — implausible for a forward-facing dashcam.
An image flagged by ANY rule is reported. Flagging by more than one rule is
a stronger signal, and is reported as such.

WHAT THIS DOES NOT DO
-----------------------
It does not delete or modify anything. It writes a CSV report of flagged
entries and, if --out_filtered_jsonl is given, a NEW manifest that excludes
them — your original train.jsonl and seg maps are untouched either way.

CACHING
-------
--cache_file (.npz: fractions[N,NUM_CLASSES] + n) avoids recomputing
per-class coverage on a re-run against the same manifest. Brightness is
cached separately (--brightness_cache) since it reads the RAW image, not the
seg map. Fully standalone -- no dependency on any other script in this repo.

USAGE
------
    python scan_seg_anomalies.py \\
        --json_file data/seg_training_aspect/train.jsonl \\
        --cache_file data/seg_training_aspect/train_coverage_cache.npz \\
        --brightness_cache data/seg_training_aspect/train_brightness_cache.npz \\
        --out_report data/seg_training_aspect/anomaly_report.csv \\
        --out_filtered_jsonl data/seg_training_aspect/train_filtered.jsonl

Add --dry_run_n N to scan only the first N entries for a quick check before
committing to a full 70K-image run.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image

# SEG_CITYSCAPES_PALETTE is the project's single source of truth for
# class-id -> colour (src/encoders/seg_encoder.py). Importing it here (rather
# than re-typing the table) means this diagnostic can never silently drift
# out of sync with the real encoder.
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

ROAD_CLASS_ID = CLASS_NAMES.index("road")

# Downsample size for brightness sampling — mean brightness is stable at this
# resolution and decoding+resizing every raw image at full size for 70K
# images is the dominant cost of this script otherwise.
_BRIGHTNESS_SAMPLE_SIZE = (32, 32)


def _read_entries(json_file: Path) -> list[dict]:
    with open(json_file, "r", encoding="utf-8-sig") as f:
        return [json.loads(line) for line in f if line.strip()]


def _resolve(p: str, image_root: str | None) -> Path:
    path = Path(p)
    if image_root and not path.is_absolute():
        path = Path(image_root) / path
    return path


def _class_fractions_from_ids(ids: np.ndarray) -> np.ndarray:
    """ids: HxW array of class indices 0..NUM_CLASSES-1 -> per-class pixel fraction."""
    counts = np.bincount(ids.ravel(), minlength=NUM_CLASSES)[:NUM_CLASSES]
    return counts / counts.sum()


def build_or_load_training_distribution(json_file: Path, cache_file: Path | None) -> tuple[np.ndarray, int]:
    """Returns (fractions [N,NUM_CLASSES], N). Uses cache_file if present and matching."""
    entries = _read_entries(json_file)
    n = len(entries)

    if cache_file is not None and cache_file.exists():
        cached = np.load(cache_file)
        if int(cached["n"]) == n:
            print(f"[scan_seg_anomalies] Loaded cached distribution ({n} images) from {cache_file}")
            return cached["fractions"], n
        print(f"[scan_seg_anomalies] Cache at {cache_file} has {int(cached['n'])} images, "
              f"manifest now has {n} -- recomputing.")

    print(f"[scan_seg_anomalies] Computing per-class coverage for {n} training images "
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
        print(f"[scan_seg_anomalies] WARNING: {missing} seg_path files not found, left as zero-rows")

    if cache_file is not None:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache_file, fractions=fractions, n=n)
        print(f"[scan_seg_anomalies] Cached to {cache_file}")

    return fractions, n


def build_or_load_brightness(
    entries: list[dict], image_key: str, image_root: str | None, cache_file: Path | None
) -> np.ndarray:
    """Returns mean brightness in [0,255] per entry, using a downsampled raw-image read."""
    n = len(entries)
    if cache_file is not None and cache_file.exists():
        cached = np.load(cache_file)
        if int(cached["n"]) == n:
            print(f"[scan_seg_anomalies] Loaded cached brightness ({n} images) from {cache_file}")
            return cached["brightness"]
        print(f"[scan_seg_anomalies] Brightness cache at {cache_file} has {int(cached['n'])} "
              f"images, manifest now has {n} -- recomputing.")

    print(f"[scan_seg_anomalies] Computing raw-image brightness for {n} images "
          f"(one-time cost; pass --brightness_cache to reuse this next time)...")
    brightness = np.full(n, -1.0, dtype=np.float32)   # -1 = unreadable, excluded from day/night stats
    missing = 0
    for i, item in enumerate(entries):
        img_path = _resolve(item[image_key], image_root)
        if not img_path.exists():
            missing += 1
            continue
        img = Image.open(img_path).convert("L").resize(_BRIGHTNESS_SAMPLE_SIZE)
        brightness[i] = float(np.array(img).mean())
        if (i + 1) % 5000 == 0:
            print(f"  ...{i + 1}/{n}")
    if missing:
        print(f"[scan_seg_anomalies] WARNING: {missing} {image_key} files not found, left as -1")

    if cache_file is not None:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache_file, brightness=brightness, n=n)
        print(f"[scan_seg_anomalies] Cached to {cache_file}")

    return brightness


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json_file", required=True, help="training manifest, e.g. data/seg_training_aspect/train.jsonl")
    ap.add_argument("--image_key", default="raw_image_path", help="JSONL key for the raw RGB image. Default: raw_image_path")
    ap.add_argument("--image_root", default=None, help="prefix for relative paths, if any")
    ap.add_argument("--cache_file", default=None, help="optional .npz cache for per-class fractions")
    ap.add_argument("--brightness_cache", default=None, help="optional .npz cache for raw-image brightness")
    ap.add_argument("--dominance_threshold", type=float, default=0.60, help="flag if one class covers more than this fraction. Default: 0.60")
    ap.add_argument("--min_classes", type=int, default=3, help="flag if fewer than this many distinct classes are present. Default: 3")
    ap.add_argument("--road_min_fraction", type=float, default=0.01, help="flag if road-class fraction is below this. Default: 0.01")
    ap.add_argument("--night_threshold", type=float, default=60.0, help="mean brightness (0-255) below which an image is tagged 'night'. Default: 60.0")
    ap.add_argument("--out_report", default=None, help="CSV path to write flagged entries + reasons")
    ap.add_argument("--out_filtered_jsonl", default=None, help="if given, write a new manifest excluding flagged entries")
    ap.add_argument("--dry_run_n", type=int, default=None, help="only process the first N entries")
    args = ap.parse_args()

    json_file = Path(args.json_file)
    entries = _read_entries(json_file)
    if args.dry_run_n:
        entries = entries[: args.dry_run_n]
        print(f"[DRY RUN] First {len(entries)} entries only.")
    n = len(entries)
    print(f"[scan_seg_anomalies] {n} entries in {json_file}")

    # build_or_load_training_distribution reads json_file itself, so give it
    # a manifest containing exactly `entries` when --dry_run_n trims the set.
    dist_source = json_file
    if args.dry_run_n:
        dist_source = json_file.with_name(json_file.stem + "_dryrun_tmp.jsonl")
        with open(dist_source, "w", encoding="utf-8") as f:
            for e in entries:
                f.write(json.dumps(e) + "\n")

    fractions, n_check = build_or_load_training_distribution(
        dist_source, Path(args.cache_file) if args.cache_file else None
    )
    assert n_check == n, f"fraction count {n_check} != entry count {n}"

    brightness = build_or_load_brightness(
        entries, args.image_key, args.image_root,
        Path(args.brightness_cache) if args.brightness_cache else None,
    )

    if args.dry_run_n:
        dist_source.unlink(missing_ok=True)

    dominant_class = fractions.argmax(axis=1)                     # [N]
    dominant_frac = fractions.max(axis=1)                          # [N]
    n_classes_present = (fractions > 0).sum(axis=1)                # [N]
    road_frac = fractions[:, ROAD_CLASS_ID]                        # [N]

    is_night = brightness >= 0
    is_night &= brightness < args.night_threshold

    flags: list[list[str]] = [[] for _ in range(n)]
    for i in range(n):
        if dominant_frac[i] > args.dominance_threshold:
            flags[i].append(f"dominant_class={CLASS_NAMES[dominant_class[i]]}({dominant_frac[i]*100:.1f}%)")
        if n_classes_present[i] < args.min_classes:
            flags[i].append(f"low_diversity({n_classes_present[i]}_classes)")
        if road_frac[i] < args.road_min_fraction:
            flags[i].append(f"no_road({road_frac[i]*100:.2f}%)")

    flagged_idx = [i for i in range(n) if flags[i]]

    print(f"\n{'='*70}")
    print(f"RESULTS: {len(flagged_idx)}/{n} images flagged ({100*len(flagged_idx)/n:.2f}%)")
    print(f"{'='*70}")

    valid_bright = brightness >= 0
    n_night = int(is_night.sum())
    n_day = int((valid_bright & ~is_night).sum())
    n_night_flagged = sum(1 for i in flagged_idx if is_night[i])
    n_day_flagged = sum(1 for i in flagged_idx if valid_bright[i] and not is_night[i])

    print(f"\nNight images (brightness < {args.night_threshold}): {n_night}/{int(valid_bright.sum())}")
    if n_night:
        print(f"  flagged: {n_night_flagged}/{n_night} ({100*n_night_flagged/n_night:.1f}%)")
    if n_day:
        print(f"Day images: {n_day}")
        print(f"  flagged: {n_day_flagged}/{n_day} ({100*n_day_flagged/n_day:.1f}%)")
    if n_night and n_day:
        night_rate = n_night_flagged / n_night
        day_rate = n_day_flagged / n_day if n_day else 0.0
        if day_rate > 0:
            print(f"\nNight images are flagged {night_rate/day_rate:.1f}x more often than "
                  f"day images ({100*night_rate:.1f}% vs {100*day_rate:.1f}%).")
        else:
            print(f"\nNight images are flagged at {100*night_rate:.1f}%; day images have "
                  f"zero flags.")

    if args.out_report and flagged_idx:
        report_path = Path(args.out_report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["index", "raw_image_path", "seg_path", "brightness", "is_night", "reasons"])
            for i in flagged_idx:
                w.writerow([
                    i, entries[i].get(args.image_key, ""), entries[i].get("seg_path", ""),
                    f"{brightness[i]:.1f}", is_night[i], "; ".join(flags[i]),
                ])
        print(f"\nFlagged-entry report written to {report_path}")

    if args.out_filtered_jsonl:
        out_path = Path(args.out_filtered_jsonl)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        flagged_set = set(flagged_idx)
        with open(out_path, "w", encoding="utf-8") as f:
            for i, e in enumerate(entries):
                if i not in flagged_set:
                    f.write(json.dumps(e) + "\n")
        print(f"Filtered manifest ({n - len(flagged_idx)}/{n} entries kept) written to {out_path}")
        print("Original manifest and seg maps were NOT modified.")


if __name__ == "__main__":
    main()
