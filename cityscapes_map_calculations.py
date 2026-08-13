"""
cityscapes_map_calculations.py
--------------------------------
Build (raw_image_path, seg_path, prompt) train/val/test.jsonl manifests from
a real downloaded Cityscapes dataset -- the same standard schema every other
entry point in this project reads (seg_map_calculations.py, segformer_training.py,
segformer_inference.py).

WHAT THIS SCRIPT NEEDS ON DISK
  --cityscapes_root pointing at the extracted Cityscapes package root, i.e.
  the folder that directly contains leftImg8bit/ and gtFine/:
      <cityscapes_root>/leftImg8bit/{train,val,test}/<city>/<city>_..._leftImg8bit.png
      <cityscapes_root>/gtFine/{train,val,test}/<city>/<city>_..._gtFine_labelIds.png
  From: leftImg8bit_trainvaltest.zip + gtFine_trainvaltest.zip (cityscapes-dataset.com,
  free account required). Only the "train" and "val" splits carry real labels --
  Cityscapes' own "test" split's ground truth is withheld for the benchmark, so
  it is NEVER read here even though leftImg8bit/test/ has images.

TWO THINGS THIS SCRIPT DOES THAT A NAIVE COPY WOULDN'T
  1. LABEL REMAP (labelIds -> trainIds): gtFine's *_labelIds.png files use
     Cityscapes' full 34-class raw id scheme (things like "ego vehicle",
     "rectification border" that must be ignored, not treated as real
     classes). This project's SegmentationEncoder / SEG_CITYSCAPES_PALETTE
     use the standard 19-class "trainId" scheme instead (the same one
     nvidia/segformer-b5-finetuned-cityscapes-1024-1024 itself was trained
     on) -- ids 0..18, everything else (including "ignore" regions) mapped
     to 255. Feeding raw labelIds through unconverted would silently teach
     the model 34 classes that don't match this project's 19-class palette.
     The remapped PNGs are written to a NEW sibling folder
     (<cityscapes_root>_trainids/) -- the original gtFine/ tree is never
     written into.
  2. RE-SPLIT into train/val/test: Cityscapes' own val split (500 images) has
     real labels but this project needs its own train/val/test triplet, and
     Cityscapes' "test" has none at all usable here. So Cityscapes' labeled
     train+val (2,975 + 500 = 3,475 images) are pooled and re-split by
     --val_frac/--test_frac (default 0.1/0.1), not passed through as-is.

CAPTIONS: BLIP (Salesforce/blip-image-captioning-large), same model this
  project's other captioning step (data/dataset_preparation/generate_prompts.py)
  uses, batched for speed.

USAGE:
  python cityscapes_map_calculations.py \\
      --cityscapes_root /path/to/cityscapes \\
      --output_dir data/cityscapes_prepared

  # Quick sanity check before the full ~3,475-image run:
  python cityscapes_map_calculations.py --cityscapes_root ... --dry_run_n 20

OUTPUT (--output_dir, default data/cityscapes_prepared/):
  train.jsonl / val.jsonl / test.jsonl, each line:
    {"raw_image_path": "<cityscapes_root>/leftImg8bit/...png",
     "seg_path":       "<cityscapes_root>_trainids/...png",
     "prompt":         "<BLIP caption>"}
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

# Official Cityscapes labelId (0..33, -1) -> trainId (0..18, 255=ignore) map.
# Source: cityscapesscripts/helpers/labels.py -- the standard public table,
# same order as this project's SEG_CITYSCAPES_PALETTE (index == trainId).
_LABELID_TO_TRAINID = {
    0: 255, 1: 255, 2: 255, 3: 255, 4: 255, 5: 255, 6: 255,   # unlabeled..ground
    7: 0,    # road
    8: 1,    # sidewalk
    9: 255,  # parking
    10: 255, # rail track
    11: 2,   # building
    12: 3,   # wall
    13: 4,   # fence
    14: 255, # guard rail
    15: 255, # bridge
    16: 255, # tunnel
    17: 5,   # pole
    18: 255, # polegroup
    19: 6,   # traffic light
    20: 7,   # traffic sign
    21: 8,   # vegetation
    22: 9,   # terrain
    23: 10,  # sky
    24: 11,  # person
    25: 12,  # rider
    26: 13,  # car
    27: 14,  # truck
    28: 15,  # bus
    29: 255, # caravan
    30: 255, # trailer
    31: 16,  # train
    32: 17,  # motorcycle
    33: 18,  # bicycle
    -1: 255, # license plate
}


def _build_lut() -> np.ndarray:
    """256-entry lookup table (index = labelId, value = trainId), 255=ignore
    default for any id not in the official table (safety net, not expected
    to ever be hit on real Cityscapes files)."""
    lut = np.full(256, 255, dtype=np.uint8)
    for raw_id, train_id in _LABELID_TO_TRAINID.items():
        if 0 <= raw_id <= 255:
            lut[raw_id] = train_id
    return lut


_LUT = _build_lut()


def center_crop_box(w: int, h: int, target_ratio: float) -> tuple:
    """
    (left, top, right, bottom) box that center-crops a (w, h) image to
    target_ratio (width/height), keeping the FULL height and trimming width
    (Cityscapes' 2:1 is wider than this project's 1.6:1 target, never the
    reverse, so height is always the limiting dimension here).
    """
    crop_w = round(h * target_ratio)
    crop_w = min(crop_w, w)   # no-op if the source is already narrower than target
    x0 = (w - crop_w) // 2
    return (x0, 0, x0 + crop_w, h)


def remap_labelids_to_trainids(label_ids_path: Path, out_path: Path, crop_box: tuple = None) -> None:
    """Read a *_gtFine_labelIds.png, optionally center-crop, remap via _LUT, write a trainId PNG."""
    ids_img = Image.open(label_ids_path)
    if crop_box is not None:
        ids_img = ids_img.crop(crop_box)
    ids = np.array(ids_img, dtype=np.uint8)
    train_ids = _LUT[ids]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(train_ids, mode="L").save(out_path)


def crop_and_save_rgb(image_path: Path, out_path: Path, crop_box: tuple) -> None:
    """Center-crop a raw Cityscapes RGB image to crop_box and save a copy.
    Never writes into the original leftImg8bit/ tree."""
    img = Image.open(image_path).convert("RGB").crop(crop_box)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)


def find_pairs(cityscapes_root: Path) -> list[dict]:
    """
    Walk leftImg8bit/{train,val} and match each image to its gtFine labelIds
    file by filename stem. Cityscapes' own "test" split is skipped entirely
    -- its ground truth is withheld for the benchmark, no gtFine/test labels
    exist to match against.
    """
    pairs = []
    for split in ("train", "val"):
        img_root = cityscapes_root / "leftImg8bit" / split
        gt_root = cityscapes_root / "gtFine" / split
        if not img_root.exists():
            print(f"[WARN] {img_root} not found -- skipping Cityscapes split '{split}'.")
            continue
        for city_dir in sorted(p for p in img_root.iterdir() if p.is_dir()):
            for img_path in sorted(city_dir.glob("*_leftImg8bit.png")):
                stem = img_path.name[: -len("_leftImg8bit.png")]
                label_path = gt_root / city_dir.name / f"{stem}_gtFine_labelIds.png"
                if not label_path.exists():
                    print(f"[WARN] no matching gtFine labelIds for {img_path} -- skipped.")
                    continue
                pairs.append({
                    "image_path": img_path,
                    "label_ids_path": label_path,
                    "cityscapes_split": split,
                    "stem": stem,
                })
    return pairs


def caption_images(image_paths: list[Path], device: str, batch_size: int,
                    local_files_only: bool) -> list[str]:
    """BLIP captions for a list of real images, batched. Same model this
    project's data/dataset_preparation/generate_prompts.py already uses."""
    from transformers import BlipForConditionalGeneration, BlipProcessor

    model_id = "Salesforce/blip-image-captioning-large"
    processor = BlipProcessor.from_pretrained(model_id, local_files_only=local_files_only)
    model = BlipForConditionalGeneration.from_pretrained(
        model_id, local_files_only=local_files_only,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    ).to(device)
    model.eval()

    captions: list[str] = []
    for start in tqdm(range(0, len(image_paths), batch_size), desc="Captioning"):
        batch_paths = image_paths[start: start + batch_size]
        images = [Image.open(p).convert("RGB") for p in batch_paths]
        inputs = processor(images=images, return_tensors="pt").to(device)
        with torch.no_grad():
            output_ids = model.generate(**inputs, max_new_tokens=50, num_beams=4)
        captions.extend(c.strip() for c in processor.batch_decode(output_ids, skip_special_tokens=True))
    return captions


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cityscapes_root", required=True, type=str,
                     help="Folder containing leftImg8bit/ and gtFine/ (extracted Cityscapes zips).")
    ap.add_argument("--output_dir", type=str, default="data/cityscapes_prepared",
                     help="Where to write train.jsonl/val.jsonl/test.jsonl. Default: data/cityscapes_prepared")
    ap.add_argument("--seg_output_dir", type=str, default=None,
                     help="Where to write remapped trainId PNGs. Default: "
                          "<cityscapes_root>_trainids (sibling of --cityscapes_root, never written INTO it).")
    ap.add_argument("--rgb_output_dir", type=str, default=None,
                     help="Where to write center-cropped RGB copies. Default: "
                          "<cityscapes_root>_cropped (sibling of --cityscapes_root, never written INTO it).")
    ap.add_argument("--target_ratio", type=float, default=512 / 320,
                     help="width/height ratio to center-crop Cityscapes images (2:1 native) to before "
                          "they reach the shared resize step -- MUST match this project's training "
                          "size ratio (default 512/320=1.6, this repo's locked resolution) or the "
                          "whole point of cropping (avoiding stretch distortion downstream) is lost.")
    ap.add_argument("--val_frac", type=float, default=0.1, help="Fraction of the pooled labeled set for val. Default 0.1.")
    ap.add_argument("--test_frac", type=float, default=0.1, help="Fraction of the pooled labeled set for test. Default 0.1.")
    ap.add_argument("--seed", type=int, default=42, help="Shuffle seed for the train/val/test split. Default 42.")
    ap.add_argument("--batch_size", type=int, default=8, help="BLIP captioning batch size. Default 8.")
    ap.add_argument("--device", type=str, default=None, help="cuda / cuda:N / cpu. Default: auto-detect, refuse if no GPU.")
    ap.add_argument("--local_files_only", type=lambda s: s.lower() != "false", default=True, metavar="True|False",
                     help="True (default) = load BLIP strictly from local cache (offline). False = allow download.")
    ap.add_argument("--no_skip", action="store_true", help="Recompute trainId PNGs even if they already exist.")
    ap.add_argument("--dry_run_n", type=int, default=None, help="Process only the first N pairs (quick sanity check).")
    args = ap.parse_args()

    cityscapes_root = Path(args.cityscapes_root).resolve()
    if not cityscapes_root.exists():
        raise FileNotFoundError(f"--cityscapes_root not found: {cityscapes_root}")

    seg_output_dir = (
        Path(args.seg_output_dir).resolve() if args.seg_output_dir
        else cityscapes_root.parent / (cityscapes_root.name + "_trainids")
    )
    rgb_output_dir = (
        Path(args.rgb_output_dir).resolve() if args.rgb_output_dir
        else cityscapes_root.parent / (cityscapes_root.name + "_cropped")
    )
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[cityscapes_map_calculations] cityscapes_root = {cityscapes_root}")
    print(f"[cityscapes_map_calculations] seg_output_dir  = {seg_output_dir}")
    print(f"[cityscapes_map_calculations] rgb_output_dir  = {rgb_output_dir}")
    print(f"[cityscapes_map_calculations] output_dir      = {output_dir}")
    print(f"[cityscapes_map_calculations] target_ratio    = {args.target_ratio:.4f} "
          f"(Cityscapes' native 2048x1024 is 2.0 -- center-cropped to this ratio BEFORE "
          f"the shared resize step, so nothing gets stretched downstream)")

    pairs = find_pairs(cityscapes_root)
    print(f"[cityscapes_map_calculations] found {len(pairs)} labeled (image, gtFine) pairs "
          f"across Cityscapes' train+val splits (test split skipped -- no labels).")
    if args.dry_run_n:
        pairs = pairs[: args.dry_run_n]
        print(f"[DRY RUN] first {len(pairs)} pairs only.")
    if not pairs:
        print("[ERROR] No pairs found -- check --cityscapes_root points at the folder "
              "containing leftImg8bit/ and gtFine/.")
        return

    # ---- 1. Center-crop RGB + label to target_ratio, then remap labels ------
    # Cityscapes is natively 2:1 (2048x1024); this project's locked training
    # ratio is 1.6:1 (512x320). Feeding 2:1 images through the shared
    # resize_mode="aspect" step (a direct, non-cropping resize) would stretch
    # them -- confirmed by measurement: w_scale/h_scale skew of 1.25x, visibly
    # egg-shaping round objects. Cropping to target_ratio HERE, once, before
    # either file is saved, means nothing downstream ever needs to stretch --
    # the shared pipeline stays exactly as simple as it is for this project's
    # own native 1.6:1 photos.
    print("\n[Step 1/3] Center-cropping to target_ratio, remapping labels ...")
    seg_paths: list[Path] = []
    rgb_paths: list[Path] = []
    processed = skipped = 0
    for item in tqdm(pairs, desc="Cropping + remapping"):
        w, h = Image.open(item["image_path"]).size
        crop_box = center_crop_box(w, h, args.target_ratio)

        rel = item["label_ids_path"].relative_to(cityscapes_root / "gtFine")
        seg_out = seg_output_dir / rel.parent / (item["stem"] + "_trainIds.png")
        rgb_rel = item["image_path"].relative_to(cityscapes_root / "leftImg8bit")
        rgb_out = rgb_output_dir / rgb_rel.parent / (item["stem"] + "_leftImg8bit.png")

        if seg_out.exists() and rgb_out.exists() and not args.no_skip:
            skipped += 1
        else:
            remap_labelids_to_trainids(item["label_ids_path"], seg_out, crop_box=crop_box)
            crop_and_save_rgb(item["image_path"], rgb_out, crop_box)
            processed += 1
        seg_paths.append(seg_out)
        rgb_paths.append(rgb_out)
    print(f"  processed {processed}, skipped {skipped} (already present)")

    # ---- 2. Captions (BLIP) --------------------------------------------------
    # Caption the CROPPED image, not the original -- that's what the model
    # actually trains on, and the crop only trims empty margin (see the
    # measured-and-visually-confirmed comparison this was based on), so the
    # caption is still accurate to what's kept.
    print("\n[Step 2/3] Captioning with BLIP ...")
    device = args.device if args.device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cpu":
        print("[WARN] No CUDA GPU selected/detected -- BLIP captioning on CPU will be slow "
              "for a full Cityscapes run. Pass --device cuda:N if a GPU should be visible.")
    prompts = caption_images(rgb_paths, device, args.batch_size, args.local_files_only)

    # ---- 3. Pool + re-split into train/val/test, write JSONLs ---------------
    print("\n[Step 3/3] Splitting and writing train/val/test.jsonl ...")
    entries = [
        {
            "raw_image_path": str(rgb_paths[i]),
            "seg_path": str(seg_paths[i]),
            "prompt": prompts[i],
        }
        for i in range(len(pairs))
    ]
    rng = random.Random(args.seed)
    rng.shuffle(entries)

    n = len(entries)
    n_val = round(n * args.val_frac)
    n_test = round(n * args.test_frac)
    n_train = n - n_val - n_test
    if n_train <= 0:
        raise ValueError(f"--val_frac + --test_frac ({args.val_frac + args.test_frac}) leaves no train data "
                          f"for {n} entries.")
    splits = {
        "train": entries[:n_train],
        "val": entries[n_train: n_train + n_val],
        "test": entries[n_train + n_val:],
    }
    for split_name, split_entries in splits.items():
        out_path = output_dir / f"{split_name}.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            for e in split_entries:
                f.write(json.dumps(e) + "\n")
        print(f"  {split_name}.jsonl: {len(split_entries)} entries -> {out_path}")

    print(f"\nDone. {n} total entries "
          f"({n_train} train / {n_val} val / {n_test} test).")
    print(f"Next: python segformer_training.py experiment=train_seg "
          f"data.json_file={output_dir}/train.jsonl data.val_json_file={output_dir}/val.jsonl "
          f"resize_mode=aspect")


if __name__ == "__main__":
    main()
