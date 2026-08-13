# Quickstart — segformer branch

Copy-paste commands only. Run from the repo root with the `loradapter` conda
env active (`conda activate loradapter`).

Your fix compared to what you had drafted: `--width`/`--height` are **not
optional** in step 1 — without them the calc script silently falls back to a
plain `--size 512` **square**, not the locked 512×320 target. Also,
`letterbox` no longer exists as an option anywhere in this branch — only
`aspect` is supported now, so `resize_mode=letterbox` in your draft would
error out.

## 1. Compute segmentation maps

```
python seg_map_calculations.py --data_dir data/ --resize_mode aspect --width 512 --height 320
```

- Builds `data/seg_training_aspect/{train,val,test}.jsonl` from
  `data/{train,val,test}.jsonl`.
- `--width`/`--height` are **required** together to get the non-square
  512×320 target — this is the one thing you can't rely on defaults for.
- If your source JSONL uses a key other than `raw_image_path` for the image
  path (check with `head -1 data/train.jsonl`), add `--image_path <your_key>`.

## 2. Train

```
python segformer_training.py experiment=train_seg
```

- No extra args needed — `size: [512, 320]` and `resize_mode: aspect` are
  already the defaults in `configs/experiment/train_seg.yaml`, and they
  match step 1 exactly.
- Checkpoints land in `outputs/train/seg_aspect/runs/<date>/<time>/`.
  `best_model/` inside that folder is what you'll point inference at.

## 3. Run inference

```
python segformer_inference.py ckpt_path=outputs/train/seg_aspect/runs/YYYY-MM-DD/HH-MM-SS/best_model inference.json_file=data/seg_training_aspect/test.jsonl
```

- Replace the `YYYY-MM-DD/HH-MM-SS` path with your real run's folder from step 2.
- `ckpt_path` is the only value you must supply — everything else
  (`resize_mode=aspect`, `size=[512,320]`, base model, VAE) already defaults
  to match training.
- Output grids land in `outputs/inference/seg_aspect/results/<timestamp>/`.

## Optional — using Cityscapes as extra training data

```
python cityscapes_map_calculations.py --cityscapes_root /path/to/extracted/cityscapes --output_dir data/cityscapes_prepared
```

- `--cityscapes_root` must point at the folder that directly contains
  `leftImg8bit/` and `gtFine/` (extract the two zips there first).
- Handles the label remap (Cityscapes' 34-class raw ids → this project's
  19-class scheme) and the 2:1 → 1.6:1 center-crop automatically — nothing
  else to pass.
- Then train on it the same way as step 2, pointing `data.json_file`/
  `data.val_json_file` at `data/cityscapes_prepared/{train,val}.jsonl`
  instead of (or merged with) your own manifests.

## Common mistakes this guide already avoids

| Wrong | Right | Why |
|---|---|---|
| `--resize_mode aspect` alone | `--resize_mode aspect --width 512 --height 320` | without both, falls back to a square 512×512 |
| `resize_mode=letterbox` | `resize_mode=aspect` | letterbox/CenterCrop were removed from this branch entirely |
| `outputs/train/seg_letterbox/...` | `outputs/train/seg_aspect/...` | output folders are named after `resize_mode`, which is now always `aspect` |
| `data/seg_training_letterbox/...` | `data/seg_training_aspect/...` | same mode-naming, applies to the calc script's output too |
