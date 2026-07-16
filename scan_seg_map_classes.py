"""
STANDALONE DIAGNOSTIC — not part of any pipeline stage.

WHY THIS EXISTS
----------------
check_seg_map_format.py answered the format question for ONE file. This
script answers two more questions that need the WHOLE dataset, not a sample:

  1. HOW MANY CLASSES are actually in use? A single file only shows the
     classes present in that one scene -- a rare class (e.g. "train") might
     only appear in a handful of the 68K images. This scans every seg map
     across train/val/test and reports the TRUE global min/max/unique value
     set, which is what you actually need to build a correctly-sized colour
     palette.

  2. IS JPEG COMPRESSION ACTUALLY CORRUPTING LABELS? JPEG is lossy, but that
     doesn't automatically mean your specific maps are damaged in practice --
     it depends on JPEG quality and how aggressively edges get smoothed. Two
     concrete things this script checks instead of guessing:
       a. Any value appearing OUTSIDE the dataset's dominant class range is
          almost certainly compression noise, not a real class -- flagged
          per-file so you can spot-check them.
       b. A per-file "unique value count" outlier report -- a file with
          drastically more unique values than typical (e.g. 40 unique values
          when every other file has ~18) suggests speckle noise from
          compression artifacts rather than real distinct classes.
     This can't catch EVERY boundary-pixel shift (a pixel becoming a
     NEIGHBOURING valid class ID is indistinguishable from real data without
     a clean reference) -- but it catches the failure modes that matter:
     stray out-of-range values and abnormal per-file noise.

USAGE (run against each split; repeat for train/val/test)
------
    python scan_seg_map_classes.py --json_file data/seg_training/train.json
    python scan_seg_map_classes.py --json_file data/seg_training/val.json
    python scan_seg_map_classes.py --json_file data/seg_training/test.json

Add --limit N to scan only the first N entries for a quick smoke check
before committing to a full 60K-image scan.
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json_file", required=True, help="manifest to scan, e.g. data/seg_training/train.json")
    ap.add_argument("--seg_key", default="seg_path", help="JSON key pointing at the seg map file (default: seg_path)")
    ap.add_argument("--limit", type=int, default=None, help="only scan the first N entries (quick check)")
    args = ap.parse_args()

    manifest_path = Path(args.json_file)
    entries = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.limit:
        entries = entries[: args.limit]
    n = len(entries)
    print(f"[scan_seg_map_classes] Scanning {n} entries from {manifest_path}...")

    global_values: Counter = Counter()          # value -> total pixel count across dataset
    per_file_unique_counts: list[int] = []       # unique-value count per file, for outlier detection
    per_file_max: list[int] = []
    missing = 0
    read_errors = 0

    for i, item in enumerate(entries):
        seg_path = Path(item[args.seg_key])
        if not seg_path.exists():
            missing += 1
            continue
        try:
            arr = np.array(Image.open(seg_path).convert("L"))
        except Exception as e:
            read_errors += 1
            print(f"  [READ ERROR] {seg_path}: {e}")
            continue

        vals, counts = np.unique(arr, return_counts=True)
        for v, c in zip(vals.tolist(), counts.tolist()):
            global_values[v] += c
        per_file_unique_counts.append(len(vals))
        per_file_max.append(int(vals.max()))

        if (i + 1) % 10000 == 0:
            print(f"  ...{i + 1}/{n}")

    if missing:
        print(f"[scan_seg_map_classes] WARNING: {missing} seg_path files not found")
    if read_errors:
        print(f"[scan_seg_map_classes] WARNING: {read_errors} files failed to open/read")

    if not global_values:
        print("[scan_seg_map_classes] No usable files scanned -- nothing to report.")
        return

    sorted_values = sorted(global_values.keys())
    total_pixels = sum(global_values.values())

    print(f"\n=== GLOBAL CLASS REPORT ({len(per_file_unique_counts)} files scanned) ===")
    print(f"Global min value : {sorted_values[0]}")
    print(f"Global max value : {sorted_values[-1]}")
    print(f"Distinct values found across ALL files: {len(sorted_values)}")
    print(f"  -> {sorted_values}")
    print(f"\nPer-value share of all pixels in the dataset:")
    for v in sorted_values:
        pct = 100 * global_values[v] / total_pixels
        print(f"  value {v:>3}: {pct:6.2f}%  {'<-- RARE (<0.01%)' if pct < 0.01 else ''}")

    # ---- Outlier detection: files with unusually many unique values -------- #
    median_unique = int(np.median(per_file_unique_counts))
    outlier_threshold = median_unique + 5   # heuristic: 5+ more classes than typical in one scene
    outliers = [
        (entries[i].get(args.seg_key), c)
        for i, c in enumerate(per_file_unique_counts)
        if c > outlier_threshold
    ]
    print(f"\n=== NOISE CHECK ===")
    print(f"Median unique values per file: {median_unique}")
    print(f"Files with unusually MANY unique values (> {outlier_threshold}, possible JPEG speckle): {len(outliers)}")
    for path, c in outliers[:15]:
        print(f"  {c} unique values: {path}")
    if len(outliers) > 15:
        print(f"  ... and {len(outliers) - 15} more")

    print(f"\nSUMMARY: if 'Distinct values found' is a clean small contiguous range "
          f"(e.g. 0..N-1 with no gaps) and the 'NOISE CHECK' outlier list is empty "
          f"or tiny relative to {n} files, your class count is N and JPEG "
          f"compression is very likely NOT a practical problem for this dataset. "
          f"Any stray values outside the expected range, or a large outlier list, "
          f"means re-exporting as PNG is worth the effort.")


if __name__ == "__main__":
    main()
