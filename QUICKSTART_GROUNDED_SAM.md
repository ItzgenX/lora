# Quickstart — grounded_sam branch

Copy-paste commands only. Run from the repo root with the `loradapter` conda
env active (`conda activate loradapter`).

Unlike the segformer branch, there's no `--width`/`--height` step here — this
branch resizes the RGB image and its mask live at load time (not baked into
a saved file), so the calc script below never touches resolution at all,
only file paths. Also: `letterbox`/`CenterCrop` don't exist anywhere on this
branch either — only `aspect` is supported.

## 1. Assemble the training manifest

```
python grounded_sam_map_calculations.py --data_dir data/
```

- Reads `data/{train,val,test}.jsonl` (your source manifests), pairs each
  image with its sibling `class_map.png` mask, writes
  `data/grounded_sam/{train,val,test}.jsonl`.
- Default `--image_path target` already matches this repo's real source
  manifests — only override with `--image_path raw_image_path` (or whatever
  key you actually use) if your source JSONL uses a different key.
- This script never opens an image or loads a model — it's pure path
  assembly + verification, safe to re-run any time.

## 2. Train

```
python grounded_sam_training.py experiment=train_grounded_sam
```

- No extra args needed — `size: [512, 320]` and `resize_mode: aspect` are
  already the defaults in `configs/experiment/train_grounded_sam.yaml`.
- Checkpoints land in `outputs/train/grounded_sam_aspect/runs/<date>/<time>/`.
  `best_model/` inside that folder is what you'll point inference at.
- 4-GPU cluster: `accelerate launch --num_processes=4 grounded_sam_training.py experiment=train_grounded_sam`

## 3. Run inference

```
python grounded_sam_inference.py ckpt_path=outputs/train/grounded_sam_aspect/runs/YYYY-MM-DD/HH-MM-SS/best_model inference.json_file=data/grounded_sam/test.jsonl
```

- Replace the `YYYY-MM-DD/HH-MM-SS` path with your real run's folder from step 2.
- `ckpt_path` is the only value you must supply — `resize_mode=aspect`,
  `size=[512,320]`, `classes_file`, base model, and VAE all default to match
  training.
- Output grids land in `outputs/inference/grounded_sam_aspect/results/<timestamp>/`.
- Inference cross-checks the resize_mode your checkpoint actually trained
  with (reading `best_model/info.txt`) against what you pass here, and warns
  loudly if they don't match — you don't have to track this by hand.

## Notes specific to this branch

- **Masks are never computed here** — no calc-time model, no GPU used in
  step 1. Your `class_map.png` files are the ground truth this entire
  pipeline reads; if they're wrong, nothing downstream can fix that.
- **A live Tier-2 encoder exists** (`src/encoders/grounded_sam_encoder.py`,
  real GroundingDINO+SAM, `live=true`) for generating fresh masks on new
  images or scoring the mIoU metric — but training and the standard
  inference flow above never use it; they always read your pre-saved masks.
- **1280×800 in, 512×320 out, zero distortion** — confirmed by real
  execution: your actual mask format (16-bit, CARLA 29-class ids) resizes to
  exactly 512×320 with no stretch, since 1280/800 = 512/320 = 1.6 exactly.
