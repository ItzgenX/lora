# Grounded-SAM — File Map

**If you're setting up a fresh clone of the original `CompVis/LoRAdapter`
repo to run Grounded-SAM: copy REPLACE + ADD below onto it, at the exact
same relative paths. That's the whole checklist.**

For the deep-dive explanations behind any of these files, see
[GROUNDED_SAM.md](GROUNDED_SAM.md) (the pipeline itself, Part A onward) and
[SBATCH_ZERO_TO_HERO.md](SBATCH_ZERO_TO_HERO.md) (running it on the JUSUF
cluster) — this file is just the copy checklist.

---

## REPLACE these (already exist in the original repo — overwrite at the same path)
```
src/data/local_seg.py
src/data/transforms.py
src/utils.py
configs/train_seg.yaml
recommend_training_params.py
src/model.py            # CHANGED 2026-08: vae_path hook (ModelBase.__init__, mirrors
                         # the existing tiny_vae pattern) + sample_custom() gained
                         # optional height/width params (was hardcoded-square with NO
                         # override possible at all -- a real bug caught only by reading
                         # grounded_sam_training.py's checkpoint-monitoring calls in
                         # full). Both changes are additive/backward-compatible.
configs/model/sd15.yaml # CHANGED 2026-08: gained vae_path: null (see src/model.py above)
```

## ADD these (brand new — copy them in at these exact paths)
```
grounded_sam_training.py
grounded_sam_inference.py
grounded_sam_map_calculations.py   # NEW 2026-08: manifest-assembly from your ALREADY
                                    # EXISTING class_map.png masks + source JSONLs.
                                    # Runs NO model, needs NO GPU -- this branch's masks
                                    # are made externally, this only pairs+verifies them.
src/encoders/grounded_sam_encoder.py
src/data/seg_palette.py
configs/experiment/train_grounded_sam.yaml
configs/data/local_grounded_sam.yaml
configs/lora/encoder/grounded_sam.yaml
configs/inference_grounded_sam.yaml
configs/grounded_sam_classes.json
download_sd15.py         # CHANGED 2026-08: skip-existing check (safe to re-run any
                          # time), + downloads epiCRealism + sd-vae-ft-mse
slurm/train_grounded_sam_jusuf.sbatch
```

**No SegFormer/Cityscapes code remains on this branch** —
`src/encoders/seg_encoder.py` (the live SegFormer encoder + a Cityscapes
fallback palette that could never actually work for this branch's CARLA
data) was deleted entirely. The genuinely shared, taxonomy-agnostic palette
math it also contained now lives in `src/data/seg_palette.py` above.

**Optional — docs and diagnostics, not needed to *run* training/inference:**
```
GROUNDED_SAM.md
SBATCH_ZERO_TO_HERO.md
LORA_ARCHITECTURE.md
GENERATION_QUALITY_GROUNDED_SAM.md
GROUNDED_SAM_FILES.md
scan_seg_map_classes.py
check_seg_map_format.py
analyze_car_coverage.py
check_seg_coverage.py
```

## Don't touch anything else
Already in the original repo, 100% unchanged — no action needed:
```
train.py
sample.py
src/lora.py
src/mapper_network.py
src/data/local.py
configs/train.yaml
configs/lora/struct.yaml
configs/data/local.yaml
```
(`src/model.py` and `configs/model/sd15.yaml` moved to REPLACE above 2026-08
— they are NOT stock-unchanged anymore. `src/lora.py`/`src/mapper_network.py`
ARE still confirmed byte-identical to upstream — diffed directly against a
fresh `CompVis/LoRAdapter` clone 2026-08, not assumed.)

---

## Renamed (heads up if you're diffing against an older copy)

| Old name | New name |
|---|---|
| `seg_training.py` | `grounded_sam_training.py` |
| `seg_inference.py` | `grounded_sam_inference.py` |
