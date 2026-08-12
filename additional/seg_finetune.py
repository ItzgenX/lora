"""
seg_finetune.py — Finetune SegFormer-Cityscapes (b5) on YOUR real images.

============================================================================
WHAT THIS IS (and what it is NOT)
============================================================================
This script trains the *segmentation model itself* so it produces correct seg
maps on YOUR domain (car-hood in frame, night scenes) instead of the messy
output the stock Cityscapes model gives you.

  THIS FILE          : image  +  correct label mask  ->  better SegFormer weights
  segformer_training.py    : (different job) trains the LoRAdapter diffusion model,
                       USING a frozen SegFormer to make conditioning maps.

Do not confuse the two. This one has nothing to do with diffusion — it is a
plain supervised semantic-segmentation finetune (HuggingFace Trainer).

============================================================================
WHAT DATA IT NEEDS  (read this before running)
============================================================================
Two parallel folders, images paired to masks BY FILENAME STEM:

    <data_root>/
      images/train/0001.png  0002.jpg ...
      images/val/  9001.png ...
      masks/train/ 0001.png  0002.png ...   # SAME stem as its image
      masks/val/   9001.png ...

RULES for the mask files (this is the format the model learns from):
  * single-channel PNG (grayscale / palette-index), NOT an RGB colour image.
  * pixel value = the integer CLASS ID for that pixel (0,1,2,... N-1).
  * value 255 = "ignore" (e.g. the car hood, or any unlabeled pixel). The loss
    SKIPS these pixels, so they are never learned from.
  * mask must be the SAME height x width as its image.

============================================================================
CLASS TAXONOMY  (this is the "reduce to CARLA classes" part)
============================================================================
You control your class set with a JSON file passed via --classes (see the
load_classes docstring for the format) — no code editing. The pretrained head
has 19 classes; when your JSON has a different number, the script drops the old
19-class head and attaches a fresh head with YOUR classes
(ignore_mismatched_sizes=True) while keeping all the pretrained backbone
features. Your MASK files must use exactly these ids (0..N-1, plus 255 ignore).

The original 19 Cityscapes ids (for reference when building your subset):
  0 road 1 sidewalk 2 building 3 wall 4 fence 5 pole 6 traffic light
  7 traffic sign 8 vegetation 9 terrain 10 sky 11 person 12 rider 13 car
  14 truck 15 bus 16 train 17 motorcycle 18 bicycle

============================================================================
SIZE HANDLING  (matches the rest of this repo, on purpose)
============================================================================
Images are letterboxed to a 512x512 square (SquarePad -> uniform resize) — the
SAME geometry src/data/transforms.py uses everywhere else — so the finetuned
model is IN-DISTRIBUTION when the pipeline later feeds it letterboxed images.
The MASK gets the SAME geometry but with two segmentation-specific overrides:
  * padded with 255 (ignore), never a colour fill (pad has no real class).
  * resized with NEAREST, never bilinear (blending class ids invents fake
    classes, e.g. id 3 + id 7 -> "5").

============================================================================
DOWNSTREAM NOTE (do not skip)
============================================================================
After finetuning to a NEW class count, the pipeline's SegmentationEncoder
(src/encoders/seg_encoder.py) and its 19-class SEG_CITYSCAPES_PALETTE will no
longer match this model. To actually USE these weights for conditioning you
must update that encoder's num_labels + palette to your new class set. That is
a separate integration step, intentionally not done here.

============================================================================
RUN
============================================================================
  # activate env first (see project notes): conda activate loradapter
  python seg_finetune.py --data_root data/seg_finetune --epochs 30

  python seg_finetune.py --data_root data/seg_finetune \
      --output_dir checkpoints/local_models/segformer-b5-carla \
      --epochs 40 --batch_size 2 --lr 6e-5
"""

import argparse
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from transformers import (
    SegformerForSemanticSegmentation,
    Trainer,
    TrainingArguments,
)

# Reuse this project's helpers so behaviour/conventions match the rest of the
# repo (esp. the GPU silent-CPU-fallback protection — a hard-won fix here).
from src.utils import print_gpu_diagnostics, resolve_device, auto_batch_size
from src.data.transforms import SquarePad   # letterbox pad used everywhere in this repo


# ============================================================================ #
#  1. YOUR CLASS TAXONOMY  — YOU control this, no code editing needed           #
# ============================================================================ #
# You define your classes in a JSON file and pass it with --classes. That file
# is the SINGLE SOURCE OF TRUTH for your taxonomy — full control, versionable,
# no touching this script. Format is {"id": "label"} with ids 0..N-1:
#
#   classes.json  (example CARLA subset):
#   {
#     "0": "road", "1": "sidewalk", "2": "building", "3": "pole",
#     "4": "traffic_light", "5": "traffic_sign", "6": "vegetation",
#     "7": "sky", "8": "person", "9": "vehicle"
#   }
#
# Your MASK PNGs must use exactly these ids (0..N-1), plus 255 for ignore.
# If you DON'T pass --classes, this default (full 19 Cityscapes classes) is used
# so the script still runs on stock Cityscapes-id masks.
DEFAULT_ID2LABEL = {
    0: "road", 1: "sidewalk", 2: "building", 3: "wall", 4: "fence",
    5: "pole", 6: "traffic light", 7: "traffic sign", 8: "vegetation",
    9: "terrain", 10: "sky", 11: "person", 12: "rider", 13: "car",
    14: "truck", 15: "bus", 16: "train", 17: "motorcycle", 18: "bicycle",
}

IGNORE_ID = 255   # pixels with this id contribute ZERO loss (hood, unlabeled)


def load_classes(classes_path: str | None) -> dict:
    """Load the id->label taxonomy from a JSON file, or fall back to the default.

    Validates that ids are a contiguous 0..N-1 block — SegFormer's head has N
    output channels indexed 0..N-1, so a gap (e.g. skipping id 3) would silently
    mis-map every class above the gap. We refuse loudly instead.
    """
    if not classes_path:
        print("[classes] no --classes given; using default 19 Cityscapes classes")
        return dict(DEFAULT_ID2LABEL)

    import json
    with open(classes_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    id2label = {int(k): str(v) for k, v in raw.items()}   # JSON keys are strings

    expected = set(range(len(id2label)))
    if set(id2label) != expected:
        raise ValueError(
            f"class ids in {classes_path} must be exactly 0..{len(id2label) - 1} "
            f"with no gaps; got {sorted(id2label)}. (255=ignore is implicit, "
            f"do NOT list it here.)")
    print(f"[classes] loaded {len(id2label)} classes from {classes_path}: "
          f"{[id2label[i] for i in range(len(id2label))]}")
    return id2label

# ImageNet normalization — what SegFormer was pretrained with. Must match, or
# the pretrained backbone sees inputs in the wrong range and learns poorly.
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


# ============================================================================ #
#  2. DATASET — pairs each image with its label mask, letterboxes both         #
# ============================================================================ #
def _square_pad_amounts(w: int, h: int):
    """Pad amounts (left, top, right, bottom) to make a w x h image square.

    IDENTICAL math to SquarePad (src/data/transforms.py) so the mask ends up
    padded with exactly the same geometry as the image and stays pixel-aligned.
    """
    if w == h:
        return (0, 0, 0, 0)
    if w > h:                      # landscape: pad top + bottom
        t = (w - h) // 2
        return (0, t, 0, w - h - t)
    l = (h - w) // 2               # portrait: pad left + right
    return (l, 0, h - w - l, 0)


def _list_pairs(images_dir: Path, masks_dir: Path):
    """Match every image to a mask with the same filename stem."""
    # Map stem -> mask path (masks are the label PNGs; stems must line up).
    masks = {p.stem: p for p in masks_dir.iterdir() if p.is_file()}
    pairs = []
    for img in sorted(images_dir.iterdir()):
        if not img.is_file():
            continue
        m = masks.get(img.stem)
        if m is None:
            print(f"[WARN] no mask for image {img.name} — skipping")
            continue
        pairs.append((img, m))
    if not pairs:
        raise RuntimeError(f"No image/mask pairs found in {images_dir} + {masks_dir}")
    return pairs


class SegFinetuneDataset(Dataset):
    """Returns {'pixel_values': [3,size,size] float, 'labels': [size,size] long}."""

    def __init__(self, images_dir: Path, masks_dir: Path, size: int):
        self.pairs = _list_pairs(images_dir, masks_dir)
        self.size = size
        # Image letterbox = repo's SquarePad (flat local-mean fill) -> matches
        # exactly how the pipeline will preprocess images at deployment time.
        self.square_pad = SquarePad(fill_mode="mean")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, mask_path = self.pairs[idx]

        # ---- IMAGE: letterbox -> 512 (bilinear) -> tensor -> ImageNet norm ----
        img = Image.open(img_path).convert("RGB")
        img_sq = self.square_pad(img)                              # pad shorter side -> square
        img_sq = img_sq.resize((self.size, self.size), Image.BILINEAR)
        arr = np.asarray(img_sq, dtype=np.float32) / 255.0         # [H,W,3] in [0,1]
        pixel_values = torch.from_numpy(arr).permute(2, 0, 1)      # -> [3,H,W]
        pixel_values = (pixel_values - IMAGENET_MEAN) / IMAGENET_STD

        # ---- MASK: same geometry, but 255-pad + NEAREST (never blend ids) ----
        # Open WITHOUT convert() — np.asarray gives the raw class ids directly
        # (convert("L") on a palette PNG would remap ids to grey levels = wrong).
        raw = Image.open(mask_path)
        mask = np.asarray(raw)
        if mask.ndim != 2:      # someone handed us an RGB mask by mistake
            raise ValueError(
                f"Mask {mask_path.name} has shape {mask.shape} (PIL mode "
                f"{raw.mode!r}); label masks must be single-channel (pixel value "
                f"= class id). Convert it first.")

        # NORMALIZE DTYPE — do not rely on the source mode.
        # Real cases seen in this project:
        #   Grounded-SAM masks  -> PIL mode "I;16" (uint16)
        #   Cityscapes-style    -> PIL mode "L"    (uint8)
        #   palette PNGs        -> PIL mode "P"    (uint8 palette INDICES = the ids)
        # np.asarray yields the raw ids for all of them, but the dtype differs
        # (uint8/uint16/int32). Casting to int32 makes every downstream step
        # (pad, fromarray, resize, torch) behave identically regardless of the
        # source mode, instead of working only by dtype coincidence.
        mask = mask.astype(np.int32)
        h, w = mask.shape
        assert (w, h) == img.size, (
            f"size mismatch: image {img.size} vs mask {(w, h)} for {img_path.name}")
        l, t, r, b = _square_pad_amounts(w, h)
        mask_sq = np.pad(mask, ((t, b), (l, r)),
                         mode="constant", constant_values=IGNORE_ID)
        mask_pil = Image.fromarray(mask_sq)
        mask_pil = mask_pil.resize((self.size, self.size), Image.NEAREST)
        # np.array (not asarray) -> owns its buffer; torch.from_numpy on a
        # read-only PIL view warns about undefined write behaviour otherwise.
        labels = torch.from_numpy(np.array(mask_pil)).long()       # [H,W] class ids

        return {"pixel_values": pixel_values, "labels": labels}


# ============================================================================ #
#  2b. MASK VALIDATION — run ONCE up front, fail before wasting GPU hours       #
# ============================================================================ #
def validate_masks(ds: "SegFinetuneDataset", num_labels: int, split: str,
                   max_report: int = 5) -> None:
    """Scan every mask ONCE and refuse to train if any class id is out of range.

    WHY THIS EXISTS (this is the failure it prevents):
      The loss is CrossEntropy over `num_labels` channels. If a mask contains an
      id >= num_labels (e.g. masks still in the 19 Cityscapes ids while you
      declared 10 CARLA classes), PyTorch does NOT give you a clear error — you
      get a "device-side assert triggered" CUDA crash, often many minutes into
      training, with a stack trace that points nowhere useful. Worse, a subtly
      wrong id can train "successfully" and silently learn the wrong class.
      Catching it here turns an opaque late crash into a precise early message.

    Also prints the real PIL mode/dtype so the source format is a logged FACT,
    not an assumption (Grounded-SAM masks arrive as "I;16"/uint16, not "L").
    """
    print(f"[validate:{split}] scanning {len(ds.pairs)} masks ...")
    modes, bad_files = {}, []
    for _, mask_path in ds.pairs:
        raw = Image.open(mask_path)
        arr = np.asarray(raw)
        modes[f"{raw.mode}/{arr.dtype}"] = modes.get(f"{raw.mode}/{arr.dtype}", 0) + 1
        if arr.ndim != 2:
            bad_files.append((mask_path.name, f"not single-channel (shape {arr.shape})"))
            continue
        uniq = np.unique(arr.astype(np.int64))
        # Valid = a real class id 0..N-1, OR the ignore id. Anything else is a bug.
        bad = uniq[(uniq < 0) | ((uniq >= num_labels) & (uniq != IGNORE_ID))]
        if bad.size:
            bad_files.append((mask_path.name, f"out-of-range ids {bad[:10].tolist()}"))

    for k, v in modes.items():
        print(f"[validate:{split}]   format {k}: {v} files")

    if bad_files:
        lines = "\n".join(f"    {n}: {why}" for n, why in bad_files[:max_report])
        more = f"\n    ... and {len(bad_files) - max_report} more" if len(bad_files) > max_report else ""
        raise ValueError(
            f"\n{len(bad_files)} mask(s) in '{split}' contain class ids outside "
            f"0..{num_labels - 1} (plus {IGNORE_ID}=ignore):\n{lines}{more}\n"
            f"FIX: either your --classes JSON doesn't match the ids in your masks, "
            f"or the masks need remapping to your taxonomy.")
    print(f"[validate:{split}] OK — all ids within 0..{num_labels - 1} (+{IGNORE_ID}=ignore)")


# ============================================================================ #
#  3. METRICS — mean IoU (the honest segmentation metric; accuracy lies)        #
# ============================================================================ #
def _preprocess_logits(logits, labels):
    """Turn raw logits into predicted id maps BEFORE they're accumulated.

    SegFormer emits logits at H/4 x W/4, so upsample to the label size, then
    argmax to a single id per pixel. Doing this here (on-device, per batch)
    keeps memory low — we accumulate small id maps, not full class-logit cubes.
    """
    up = F.interpolate(logits, size=labels.shape[-2:], mode="bilinear",
                       align_corners=False)
    return up.argmax(dim=1)      # [B,H,W] predicted ids


def make_compute_metrics(num_labels: int):
    def compute(eval_pred):
        preds, labels = eval_pred                 # preds already argmaxed above
        preds = np.asarray(preds).reshape(-1)
        labels = np.asarray(labels).reshape(-1)

        valid = labels != IGNORE_ID               # never score ignore pixels
        preds, labels = preds[valid], labels[valid]

        # Confusion matrix via a single bincount, then IoU per class.
        conf = np.bincount(num_labels * labels + preds,
                           minlength=num_labels ** 2).reshape(num_labels, num_labels)
        inter = np.diag(conf).astype(np.float64)
        union = conf.sum(1) + conf.sum(0) - inter
        iou = inter / np.maximum(union, 1)
        present = conf.sum(1) > 0                  # only average classes that appear
        return {
            "mean_iou": float(iou[present].mean()) if present.any() else 0.0,
            "pixel_accuracy": float(inter.sum() / np.maximum(conf.sum(), 1)),
        }
    return compute


# ============================================================================ #
#  4. MAIN                                                                      #
# ============================================================================ #
def main():
    ap = argparse.ArgumentParser(description="Finetune SegFormer-b5 on custom data")
    ap.add_argument("--data_root", required=True,
                    help="folder holding images/{train,val} and masks/{train,val}")
    ap.add_argument("--classes", default=None,
                    help="path to a JSON {id: label} taxonomy file; unset = default 19")
    ap.add_argument("--model_path",
                    default="checkpoints/local_models/segformer-b5-cityscapes",
                    help="local pretrained SegFormer folder (or a HF hub id)")
    ap.add_argument("--output_dir",
                    default="checkpoints/local_models/segformer-b5-finetuned")
    ap.add_argument("--size", type=int, default=512, help="square input size")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=None,
                    help="unset -> auto from GPU VRAM (base 2)")
    ap.add_argument("--lr", type=float, default=6e-5, help="typical SegFormer finetune lr")
    ap.add_argument("--device", default=None,
                    help="None=require a GPU (loud error if none), or 'cpu'/'cuda:N'")
    ap.add_argument("--local_files_only", default="True",
                    help="True=load model from local folder only, False=allow HF hub")
    args = ap.parse_args()

    # GPU safety: print diagnostics every run, and REFUSE to silently use CPU
    # (the exact bug this repo hit before — a slow run that looked normal).
    print_gpu_diagnostics()
    device = resolve_device(args.device)          # raises if no GPU and device!=cpu
    local_only = str(args.local_files_only).lower() in ("true", "1", "yes")
    batch_size = args.batch_size or auto_batch_size(default=2, device=device)

    # YOUR taxonomy — full control via the --classes JSON file.
    id2label = load_classes(args.classes)
    label2id = {v: k for k, v in id2label.items()}
    num_labels = len(id2label)

    data_root = Path(args.data_root)
    train_ds = SegFinetuneDataset(data_root / "images/train",
                                  data_root / "masks/train", args.size)
    val_ds = SegFinetuneDataset(data_root / "images/val",
                                data_root / "masks/val", args.size)
    print(f"[data] train={len(train_ds)}  val={len(val_ds)}  "
          f"classes={num_labels}  size={args.size}  batch={batch_size}")

    # Fail NOW on bad ids/format, not 20 minutes into training with a CUDA assert.
    validate_masks(train_ds, num_labels, "train")
    validate_masks(val_ds, num_labels, "val")

    # Load pretrained weights but SWAP the 19-class head for a num_labels one.
    # ignore_mismatched_sizes=True = "keep the backbone, re-init only the head".
    model = SegformerForSemanticSegmentation.from_pretrained(
        args.model_path,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
        local_files_only=local_only,
    )
    # Make the internal CrossEntropy loss ignore our 255 pixels (hood/unlabeled).
    model.config.semantic_loss_ignore_index = IGNORE_ID

    targs = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=args.lr,
        lr_scheduler_type="polynomial",     # SegFormer's usual poly-decay schedule
        warmup_ratio=0.05,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=3,
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="mean_iou",   # keep the checkpoint with best mIoU
        greater_is_better=True,
        remove_unused_columns=False,        # our dataset returns custom keys
        report_to="tensorboard",
        dataloader_num_workers=2,
    )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=make_compute_metrics(num_labels),
        preprocess_logits_for_metrics=_preprocess_logits,
    )

    trainer.train()

    # Save best model + its config so it can be reloaded / plugged in later.
    trainer.save_model(args.output_dir)
    print(f"\n[done] finetuned SegFormer saved to: {args.output_dir}")
    print("[reminder] to USE these weights for conditioning, update "
          "src/encoders/seg_encoder.py num_labels + palette to the new class set.")


if __name__ == "__main__":
    main()
