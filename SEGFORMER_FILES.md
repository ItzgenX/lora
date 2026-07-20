# SegFormer — File Map

**If you're setting up a fresh clone of the original `CompVis/LoRAdapter`
repo to run this SegFormer pipeline: copy REPLACE + ADD below onto it, at
the exact same relative paths. That's the whole checklist.**

For the deep-dive explanations behind any of these files, see
[SEGMENTATION.md](SEGMENTATION.md) (the pipeline itself) and
[SBATCH_ZERO_TO_HERO.md](SBATCH_ZERO_TO_HERO.md) (running it on the JUSUF
cluster) — this file is just the copy checklist.

---

## REPLACE these (already exist in the original repo — overwrite at the same path)
```
src/data/local_seg.py
src/data/transforms.py
src/utils.py
```

## ADD these (brand new — copy them in at these exact paths)
```
segformer_training.py
segformer_inference.py
seg_map_calculations.py
src/encoders/seg_encoder.py
configs/experiment/train_seg.yaml
configs/data/local_seg.yaml
configs/inference_seg.yaml
configs/lora/encoder/segformer.yaml
configs/train_seg.yaml
slurm/train_seg_jusuf.sbatch
slurm/calc_seg_segformer_jusuf.sbatch
```

**Optional — docs, diagnostics, and the labeling-assist scripts, not needed
to *run* training/inference:**
```
SEGMENTATION.md
SEG_TRAINING_GUIDE.md
SBATCH_ZERO_TO_HERO.md
LORA_ARCHITECTURE.md
GENERATION_QUALITY_SEGFORMER.md
SEGFORMER_FILES.md
recommend_training_params.py
scan_seg_map_classes.py
check_seg_map_format.py
analyze_car_coverage.py
check_seg_coverage.py
squarepad_vs_stretch.py
seg_finetune.py             (finetunes SegFormer itself on your data — separate job)
seg_make_draft_masks.py     (generates draft masks for hand-correction)
```

## Don't touch anything else
Already in the original repo, 100% unchanged — no action needed:
```
train.py
sample.py
src/model.py
src/lora.py
src/mapper_network.py
src/data/local.py
configs/train.yaml
configs/model/sd15.yaml
configs/lora/struct.yaml
configs/data/local.yaml
```

---

## One thing that works differently here vs the grounded_sam branch

The `resize_mode` (letterbox/CenterCrop) toggle exists on both branches, but
**SegFormer needs an extra calc step grounded_sam doesn't**: SegFormer
*computes* the segmentation map from a squared RGB image, so
`seg_map_calculations.py` must run first, with `--resize_mode` matching
whatever you'll train with — the squaring gets baked into the saved map
permanently. On grounded_sam, maps are pre-existing (CARLA's own output) and
get squared live at load time, so no separate calc-time mode exists there.
See [SEGMENTATION.md §5.9](SEGMENTATION.md) for the full explanation.

## Renamed 2026-07-20 (heads up if you're diffing against an older copy)

| Old name | New name |
|---|---|
| `seg_training.py` | `segformer_training.py` |
| `seg_inference.py` | `segformer_inference.py` |
