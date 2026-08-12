"""
STANDALONE DIAGNOSTIC — not part of any pipeline stage.

WHY THIS EXISTS
----------------
Before wiring up a Grounded-SAM training pipeline, we need to know what
format your ALREADY-GENERATED seg maps are actually saved in: 8-bit
grayscale class-ID PNGs (what the SegFormer pipeline uses), 16-bit, RGB
colourised, or something else entirely. Guessing this wrong silently
corrupts every class label, so this script just tells you the facts about
one real file instead.

USAGE
------
    python check_seg_map_format.py --seg_map path/to/one_real_seg_map.png
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seg_map", required=True, help="path to ONE real seg map file to inspect")
    args = ap.parse_args()

    path = Path(args.seg_map)
    img = Image.open(path)
    arr = np.array(img)

    print(f"File            : {path}")
    print(f"PIL mode        : {img.mode}")
    print(f"Size (W x H)    : {img.size}")
    print(f"Array shape     : {arr.shape}")
    print(f"Array dtype     : {arr.dtype}")
    print(f"Min pixel value : {arr.min()}")
    print(f"Max pixel value : {arr.max()}")

    if img.mode in ("L", "P", "I", "I;16"):
        unique = np.unique(arr)
        print(f"Unique values   : {len(unique)}  -> {unique[:30]}{' ...' if len(unique) > 30 else ''}")
        print()
        if len(unique) <= 256 and arr.max() < 256:
            print("VERDICT: looks like a class-ID map (small integer range) — "
                  "this is the RECOMMENDED format, no conversion needed.")
        else:
            print("VERDICT: single-channel but with a value range too large/odd for "
                  "plain class IDs — inspect the 'Unique values' list above by hand.")
    elif img.mode == "RGB":
        flat = arr.reshape(-1, 3)
        unique_colors = np.unique(flat, axis=0)
        print(f"Unique RGB colours: {len(unique_colors)}")
        print(unique_colors[:30])
        print()
        if len(unique_colors) <= 60:
            print("VERDICT: RGB image with a SMALL number of distinct colours — "
                  "consistent with a colourised class map using a fixed palette. "
                  "You'll need to give me the exact colour -> class mapping.")
        else:
            print("VERDICT: RGB image with a LARGE number of distinct colours — "
                  "this looks like a real photo, or JPEG compression has "
                  "introduced colour noise (re-export as PNG if so), not a clean "
                  "class map.")
    else:
        print(f"VERDICT: unexpected PIL mode '{img.mode}' — describe this to me directly.")

    if path.suffix.lower() in (".jpg", ".jpeg"):
        print("\nWARNING: this file is a JPEG. JPEG is LOSSY — even 'solid colour' "
              "regions get subtle compression noise, which corrupts exact class-ID "
              "or exact-colour recovery at boundaries. Seg maps should be saved as "
              "PNG (lossless), never JPEG.")


if __name__ == "__main__":
    main()
