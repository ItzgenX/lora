"""
seg_make_draft_masks.py — Generate DRAFT label masks with the pretrained SegFormer.

============================================================================
READ THIS — what "draft" means and why it exists
============================================================================
There is NO way to produce *correct* masks automatically — if there were, you
wouldn't need to finetune. This script produces ROUGH DRAFTS you then CORRECT in
a labeling tool (CVAT + SAM). The point is to start from a mostly-right canvas
instead of a blank one:
  * on EASY frames (daytime, no hood confusion) the draft is largely correct,
    so correction is fast;
  * on HARD frames (night, hood) the draft is poor — exactly the frames you must
    fix by hand. The draft still gives you a starting outline.
The hood region is auto-set to 255 (ignore) here, so that problem is handled for
free on every frame.

It uses the SAME pretrained SegFormer already in this repo, so it needs NO extra
dependencies (no Grounded-SAM, no SAM install).

LIMITATION (be aware): SegFormer only knows the 19 Cityscapes classes. This
script can only DRAFT classes that map from those 19 (see --remap). Any class of
yours that has no Cityscapes equivalent must be annotated by hand — the draft
leaves those pixels as 255 (ignore).

============================================================================
WHAT IT WRITES
============================================================================
For each <images_dir>/<name>.jpg it writes <out_dir>/<name>.png — a
single-channel mask, SAME size as the original image, pixel value = YOUR class
id (0..N-1), 255 = ignore. That pairs directly with the original image for
seg_finetune.py, and can be uploaded to CVAT as a pre-annotation to correct.

============================================================================
CLASS REMAP  (Cityscapes id  ->  YOUR id)
============================================================================
Pass --remap pointing to a JSON that maps each Cityscapes id (0..18) you care
about to YOUR target id. Anything not listed becomes 255 (ignore).

  remap.json  (example: reduce 19 Cityscapes classes -> a 10-class CARLA set):
  {
    "0": 0,    "1": 1,    "2": 2,               # road, sidewalk, building
    "5": 3,                                      # pole            -> 3
    "6": 4,    "7": 5,                           # traffic light/sign -> 4,5
    "8": 6,    "10": 7,   "11": 8,               # vegetation, sky, person
    "13": 9,   "14": 9,   "15": 9,   "17": 9,    # car/truck/bus/motorcycle -> "vehicle" 9
    "18": 9                                      # bicycle -> vehicle 9
  }

Cityscapes ids for reference:
  0 road 1 sidewalk 2 building 3 wall 4 fence 5 pole 6 traffic light
  7 traffic sign 8 vegetation 9 terrain 10 sky 11 person 12 rider 13 car
  14 truck 15 bus 16 train 17 motorcycle 18 bicycle

============================================================================
RUN
============================================================================
  conda activate loradapter
  python seg_make_draft_masks.py \
      --images_dir data/seg_finetune/images/train \
      --out_dir    data/seg_finetune/masks/train \
      --remap      data/seg_finetune/remap.json \
      --hood_frac  0.18            # force bottom 18% (the hood) to ignore
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import SegformerForSemanticSegmentation

from src.utils import print_gpu_diagnostics, resolve_device
from src.data.transforms import SquarePad

IGNORE_ID = 255
NUM_CITYSCAPES = 19
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _square_pad_amounts(w: int, h: int):
    """Pad (left, top, right, bottom) to square — same math as SquarePad."""
    if w == h:
        return (0, 0, 0, 0)
    if w > h:
        t = (w - h) // 2
        return (0, t, 0, w - h - t)
    l = (h - w) // 2
    return (l, 0, h - w - l, 0)


def build_remap_lut(remap_path: str) -> np.ndarray:
    """Build a length-19 lookup: cityscapes_id -> your_id (default 255)."""
    with open(remap_path, "r", encoding="utf-8") as f:
        remap = {int(k): int(v) for k, v in json.load(f).items()}
    lut = np.full(NUM_CITYSCAPES, IGNORE_ID, dtype=np.uint8)
    for cs_id, my_id in remap.items():
        if not (0 <= cs_id < NUM_CITYSCAPES):
            raise ValueError(f"remap key {cs_id} is not a valid Cityscapes id 0..18")
        lut[cs_id] = my_id
    print(f"[remap] {remap}  (unmapped Cityscapes ids -> {IGNORE_ID} ignore)")
    return lut


@torch.no_grad()
def predict_cityscapes_ids(model, img: Image.Image, size: int, device: str) -> np.ndarray:
    """Run SegFormer on a letterboxed image, return 19-class ids at ORIGINAL size.

    Letterbox for the prediction (best accuracy + matches deployment), then
    UN-letterbox: crop the pad band out of the 512 prediction and NEAREST-resize
    the real-content region back to the original (w, h) so the mask aligns with
    the original photo pixel-for-pixel.
    """
    w, h = img.size
    l, t, r, b = _square_pad_amounts(w, h)
    S = max(w, h)                                   # square side after padding

    # --- letterbox -> 512 -> ImageNet-normalized tensor ---
    sq = SquarePad(fill_mode="mean")(img).resize((size, size), Image.BILINEAR)
    arr = np.asarray(sq, dtype=np.float32) / 255.0
    x = torch.from_numpy(arr).permute(2, 0, 1)
    x = ((x - IMAGENET_MEAN) / IMAGENET_STD).unsqueeze(0).to(device)

    logits = model(pixel_values=x).logits          # [1,19,128,128]
    logits = F.interpolate(logits, size=(size, size), mode="bilinear",
                           align_corners=False)
    ids512 = logits.argmax(dim=1)[0].to("cpu").numpy().astype(np.uint8)  # [512,512]

    # --- un-letterbox: crop the padded band in 512 space ---
    scale = size / S
    x0, y0 = round(l * scale), round(t * scale)
    x1, y1 = size - round(r * scale), size - round(b * scale)
    content = ids512[y0:y1, x0:x1]                  # real image region, no pad

    # NEAREST back to the original photo size (never blend class ids)
    content_pil = Image.fromarray(content).resize((w, h), Image.NEAREST)
    return np.asarray(content_pil)                  # [h, w] Cityscapes ids


def main():
    ap = argparse.ArgumentParser(description="Generate draft masks with pretrained SegFormer")
    ap.add_argument("--images_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--remap", required=True, help="JSON: cityscapes_id -> your_id")
    ap.add_argument("--model_path",
                    default="checkpoints/local_models/segformer-b5-cityscapes")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--hood_frac", type=float, default=0.0,
                    help="fraction of image HEIGHT at the bottom to force to 255 "
                         "(the fixed hood region). e.g. 0.18")
    ap.add_argument("--device", default=None)
    ap.add_argument("--local_files_only", default="True")
    args = ap.parse_args()

    print_gpu_diagnostics()
    device = resolve_device(args.device)
    local_only = str(args.local_files_only).lower() in ("true", "1", "yes")
    lut = build_remap_lut(args.remap)

    model = SegformerForSemanticSegmentation.from_pretrained(
        args.model_path, local_files_only=local_only).to(device).eval()

    images_dir, out_dir = Path(args.images_dir), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    imgs = sorted(p for p in images_dir.iterdir()
                  if p.is_file() and p.suffix.lower() in IMG_EXTS)
    if not imgs:
        raise RuntimeError(f"no images found in {images_dir}")

    for i, path in enumerate(imgs, 1):
        try:
            img = Image.open(path).convert("RGB")
        except Exception as e:
            print(f"[WARN] skip {path.name}: {e}")
            continue

        cs_ids = predict_cityscapes_ids(model, img, args.size, device)  # 0..18
        my_ids = lut[cs_ids]                        # remap -> your ids (255 = ignore)

        # Fixed-camera hood: bottom band is always hood -> ignore on every frame.
        if args.hood_frac > 0:
            h = my_ids.shape[0]
            my_ids[int(round(h * (1 - args.hood_frac))):, :] = IGNORE_ID

        Image.fromarray(my_ids, mode="L").save(out_dir / f"{path.stem}.png")
        if i % 25 == 0 or i == len(imgs):
            print(f"  [{i}/{len(imgs)}] {path.name}")

    print(f"\n[done] {len(imgs)} draft masks -> {out_dir}")
    print("[next] CORRECT these in CVAT (esp. night + object boundaries) before "
          "finetuning. Drafts are a starting canvas, not final labels.")


if __name__ == "__main__":
    main()
