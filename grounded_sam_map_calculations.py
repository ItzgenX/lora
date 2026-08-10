"""
grounded_sam_map_calculations.py
---------------------------------
STAGE C for this branch -- but NOT a model-running calc script.

WHY THIS FILE IS DIFFERENT FROM seg_map_calculations.py (the segformer
branch's calc script):
  On the segformer branch, the calc script RUNS SegFormer on every training
  photo to PRODUCE a segmentation map. On THIS branch, Grounded-SAM maps
  (class_map.png) are made EXTERNALLY -- you already have them (confirmed:
  GROUNDED_SAM.md / GENERATION_QUALITY_GROUNDED_SAM.md document the real
  format as 16-bit PNG, PIL mode I;16, 1280x800, raw CARLA class ids 0..28).
  This script's only job is PATH ASSEMBLY + VERIFICATION: pair each image in
  your source manifests with its already-existing mask, and write the
  (raw_image_path, seg_path, prompt) manifests grounded_sam_training.py /
  grounded_sam_inference.py actually read. It never opens an image for
  processing, never loads a model, never needs a GPU.

WHERE THE MASK COMES FROM, for each image path taken from a source JSONL:
  1. If --mask_key is given AND the source entry has that key, its value is
     used directly (already-known path from your source data).
  2. Otherwise (the default path): the mask is a SIBLING FILE in the SAME
     folder as the image, named --mask_name (default "class_map.png") --
     the confirmed real convention on this branch (e.g.
     .../000417/raw_image.jpg and .../000417/class_map.png together).

Geometry (resize_mode / squaring) is DELIBERATELY not this script's concern:
unlike segformer, this branch squares BOTH the RGB and the mask LIVE at load
time (see src/data/transforms.py square_id_map, used by src/data/local_seg.py
and grounded_sam_inference.py) -- the calc step never touches pixels, so
there is nothing here to keep in sync with a resize_mode choice.

USAGE — from your source manifests (recommended):
  python grounded_sam_map_calculations.py --data_dir data/
      # reads data/{train,val,test}.jsonl, image path key --image_path
      # (default "target" -- matches this repo's real data/train.jsonl;
      # override if your real dataset's source JSONL uses a different key,
      # e.g. --image_path raw_image_path), locates each image's sibling
      # class_map.png, writes data/grounded_sam/{train,val,test}.jsonl.

  # --- Dry run: first 15 entries per split, sanity-check before the full set ---
  python grounded_sam_map_calculations.py --data_dir data/ --dry_run_n 15

  # --- Your source JSONL already has a mask-path field ---
  python grounded_sam_map_calculations.py --data_dir data/ --mask_key mask_path

  # --- Different sibling mask filename ---
  python grounded_sam_map_calculations.py --data_dir data/ --mask_name seg_mask.png

OUTPUT (data/grounded_sam/{train,val,test}.jsonl by default):
  {"raw_image_path": "...jpg", "seg_path": ".../class_map.png", "prompt": "..."}
"""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def _find_split_jsonl(data_dir: Path, split: str) -> "Path | None":
    """
    Locate the JSONL file for a given split in data_dir.
    Exact match first (data_dir/{split}.jsonl), then any *.jsonl whose stem
    contains the split name; among those pick the shortest stem. Mirrors the
    identical helper on the segformer branch (seg_map_calculations.py) so
    behaviour is predictable across branches.
    """
    exact = data_dir / f"{split}.jsonl"
    if exact.exists():
        return exact
    candidates = sorted(
        [p for p in data_dir.glob("*.jsonl") if split.lower() in p.stem.lower()],
        key=lambda p: len(p.stem),
    )
    return candidates[0] if candidates else None


def _resolve_mask_path(
    image_path: Path, entry: dict, mask_key: str | None, mask_name: str,
) -> Path:
    """
    Decide where this image's mask lives.

    1. --mask_key given AND present on this entry -> use it directly
       (resolved relative to the image's own folder if not absolute, since
       that is the only anchor this script has for a bare relative path).
    2. Otherwise -> sibling file in the SAME folder as the image, named
       mask_name (the confirmed real convention on this branch).
    """
    if mask_key and entry.get(mask_key):
        p = Path(entry[mask_key])
        if p.is_absolute():
            return p
        return image_path.parent / p
    return image_path.parent / mask_name


def _verify_mask(mask_path: Path, num_classes: int) -> tuple[bool, str]:
    """
    Open the mask and check it holds valid class ids, WITHOUT any geometric
    processing (no resize here -- that happens live at load time on this
    branch). Raw pixel read (np.asarray, no .convert("L")): the real masks
    are 16-bit PNGs (PIL mode I;16); .convert("L") on I;16 is
    Pillow-version-dependent, reading raw values is not (matches
    src/data/local_seg.py's _load_seg_colormap exactly).

    Returns (ok, reason). reason is "" when ok is True.
    """
    try:
        ids = np.asarray(Image.open(mask_path))
    except Exception as e:
        return False, f"cannot open mask: {e}"
    if ids.ndim != 2:
        return False, f"mask is not single-channel (shape {ids.shape})"
    max_id = int(ids.max())
    if max_id >= num_classes:
        return False, f"mask contains class id {max_id} >= num_classes {num_classes}"
    if max_id > 255:
        # local_seg.py's _load_seg_colormap round-trips through an 8-bit "L"
        # PIL image -- a class id above 255 would silently corrupt there.
        return False, f"max pixel value {max_id} exceeds the 8-bit round-trip limit (255)"
    return True, ""


def build_grounded_sam_manifests(
    data_dir: Path,
    output_dir: Path,
    image_path_key: str,
    prompt_key: str,
    mask_key: str | None,
    mask_name: str,
    num_classes: int,
    subset_n: int | None,
) -> None:
    """
    Core routine: for each split (train/val/test), read the source JSONL,
    resolve + verify each entry's mask, and write the standard-key manifest
    grounded_sam_training.py / grounded_sam_inference.py read.

    Fails loudly (not silently skips) on a missing/invalid mask by default --
    printing every problem entry -- because a manifest that silently drops
    rows changes your dataset size without telling you, and a manifest that
    silently pairs the wrong mask poisons training with no crash at all (the
    exact bug class this project's other verifiers were built to catch).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    for split in ("train", "val", "test"):
        src_path = _find_split_jsonl(data_dir, split)
        if src_path is None:
            print(f"\n[WARN] No JSONL for split '{split}' found in {data_dir} — skipping.")
            continue

        with open(src_path, "r", encoding="utf-8") as f:
            entries = [json.loads(line) for line in f if line.strip()]
        if subset_n:
            entries = entries[:subset_n]

        print(f"\n{'='*56}")
        print(f"  {split}: {len(entries)} entries from {src_path}"
              + (" (dry-run subset)" if subset_n else ""))
        print(f"{'='*56}")

        out_entries = []
        seen_masks = {}   # normalised mask path (str) -> first image that used it
        n_missing_img = n_missing_mask = n_bad_mask = n_dup = 0

        for entry in entries:
            if image_path_key not in entry:
                print(f"  [FAIL] entry missing '{image_path_key}': {entry}")
                n_missing_img += 1
                continue
            img_path = Path(entry[image_path_key])
            if not img_path.is_absolute():
                img_path = (data_dir / img_path).resolve()
            if not img_path.exists():
                print(f"  [WARN] image not found, skipping: {img_path}")
                n_missing_img += 1
                continue

            mask_path = _resolve_mask_path(img_path, entry, mask_key, mask_name)
            if not mask_path.exists():
                print(f"  [FAIL] mask not found for {img_path.name}: expected {mask_path}")
                n_missing_mask += 1
                continue

            ok, reason = _verify_mask(mask_path, num_classes)
            if not ok:
                print(f"  [FAIL] {mask_path}: {reason}")
                n_bad_mask += 1
                continue

            # Global-uniqueness check (the stem-collision bug class found
            # earlier on the segformer branch: a fixed mask filename per
            # folder is exactly the shape of bug where two images could
            # accidentally resolve to the SAME mask file). Report but do not
            # silently drop -- a genuine same-mask-for-two-images case (e.g.
            # two crops of one frame) is plausible on this branch and the
            # user should see it and decide, not have it hidden.
            key = str(mask_path.resolve()).lower()
            if key in seen_masks:
                n_dup += 1
                print(f"  [WARN] {mask_path} is ALSO used by {seen_masks[key]} "
                      f"(now also by {img_path}) -- confirm this is intentional.")
            else:
                seen_masks[key] = img_path

            out_entries.append({
                "raw_image_path": img_path.as_posix(),
                "seg_path":       mask_path.resolve().as_posix(),
                "prompt":         entry.get(prompt_key, ""),
            })

        out_path = output_dir / f"{split}.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            for e in out_entries:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")

        print(f"\n  Written {len(out_entries)}/{len(entries)} entries -> {out_path}")
        if n_missing_img or n_missing_mask or n_bad_mask or n_dup:
            print(f"  Problems: {n_missing_img} missing image, {n_missing_mask} missing mask, "
                  f"{n_bad_mask} invalid mask, {n_dup} duplicate mask reference(s)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--data_dir", type=str, default="data",
        help="Folder with train.jsonl/val.jsonl/test.jsonl. Default: data/",
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Where to write the standard-key manifests. Default: <data_dir>/grounded_sam",
    )
    parser.add_argument(
        "--image_path", type=str, default="target",
        help="Key in the SOURCE JSONL holding the image path. Default 'target' "
             "(matches this repo's real data/train.jsonl -- confirmed by reading "
             "it, not assumed). Override for your real dataset's source JSONL if "
             "it uses a different key, e.g. --image_path raw_image_path.",
    )
    parser.add_argument(
        "--prompt_key", type=str, default="prompt",
        help="Key in the source JSONL holding the text caption. Default 'prompt'.",
    )
    parser.add_argument(
        "--mask_key", type=str, default=None,
        help="Optional key in the source JSONL that ALREADY holds the mask path "
             "for that entry. When unset (default) or absent on a given entry, "
             "the mask is located via --mask_name instead (sibling-file mode).",
    )
    parser.add_argument(
        "--mask_name", type=str, default="class_map.png",
        help="Sibling filename to look for in the image's own folder when "
             "--mask_key doesn't resolve an entry. Default 'class_map.png' -- "
             "the confirmed real filename (GROUNDED_SAM.md).",
    )
    parser.add_argument(
        "--classes_file", type=str, default="configs/grounded_sam_classes.json",
        help="Class-definition JSON, used only to determine num_classes for the "
             "mask pixel-value verification check. Default: the locked 29-class "
             "CARLA taxonomy.",
    )
    parser.add_argument(
        "--dry_run_n", type=int, default=None,
        help="Process only the first N entries per split (sanity run before the full dataset).",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve()
    if not data_dir.exists():
        parser.error(f"--data_dir not found: {data_dir}")
    output_dir = Path(args.output_dir).resolve() if args.output_dir else data_dir / "grounded_sam"

    classes_file = Path(args.classes_file)
    if not classes_file.is_absolute():
        classes_file = Path(__file__).parent / classes_file
    from src.encoders.grounded_sam_encoder import load_grounded_sam_palette
    class_names, palette = load_grounded_sam_palette(classes_file)
    num_classes = len(palette)

    print(f"Source dir   : {data_dir}")
    print(f"Output dir   : {output_dir}")
    print(f"Image key    : {args.image_path!r}")
    print(f"Mask key     : {args.mask_key!r} (fallback: sibling {args.mask_name!r})")
    print(f"Classes      : {num_classes} (from {classes_file})")
    if args.dry_run_n:
        print(f"[DRY RUN] capped at first {args.dry_run_n} entries per split")

    build_grounded_sam_manifests(
        data_dir=data_dir,
        output_dir=output_dir,
        image_path_key=args.image_path,
        prompt_key=args.prompt_key,
        mask_key=args.mask_key,
        mask_name=args.mask_name,
        num_classes=num_classes,
        subset_n=args.dry_run_n,
    )

    print(f"\nNext step — train:")
    print(f"  python grounded_sam_training.py experiment=train_grounded_sam "
          f"data.json_file={output_dir}/train.jsonl data.val_json_file={output_dir}/val.jsonl")


if __name__ == "__main__":
    main()
