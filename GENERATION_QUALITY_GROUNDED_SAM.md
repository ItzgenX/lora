# Generation-Quality Diagnosis — Grounded-SAM pipeline

Findings scoped to the **grounded_sam branch only**, based on real
generated-image samples from a CARLA-domain run (`data/info/test1_carla_image/*`)
and the actual, confirmed real training-mask format. This file does
**not** carry over any finding from an unused file — every claim below is
checked against what this pipeline actually trains on today. (See
`GENERATION_QUALITY_SEGFORMER.md` on the `segformer` branch for that
pipeline's own, separate findings — the two pipelines have different data
and different fixes; do not cross-apply them.)

---

## 1. What works (confirmed from real output)

`test1_carla_image/predicted with prompt.jpeg` shows a CARLA-map-conditioned
generation that follows the input layout closely — road, buildings, vehicle
placement all track the seg map. This confirms the shared injection mechanism
(`NewStructLoRAConv` FiLM residual, `skip_encode` inference path — see
`LORA_ARCHITECTURE.md`) works correctly on this pipeline too. Remaining
issues below trace to data format and coverage, not the injection code.

## 2. Bug (fixed) — mask format was not being read correctly

**The real training masks** (confirmed via a
`check_seg_map_format.py` scan of `class_map.png`): PNG (lossless), PIL mode
**I;16** (16-bit), **1280x800 — non-square**, raw CARLA class ids.

Before commit `5db3624`, the loader (`SegJsonDataset._load_seg_colormap` and
`seg_inference._load_seg_map`) had two problems with this exact format:
1. `.convert("L")` on an I;16 image is Pillow-version-dependent — risked
   silently wrong ids on a different Pillow version than this one.
2. The non-square map was NEAREST-**stretched** to 512x512, while the
   paired RGB is **letterboxed** (`SquarePad`) — measured misalignment up to
   **96px / 18.8% of the frame** at the top and bottom (0px only at
   mid-frame), meaning conditioning and target disagreed about where content
   sits, worst right at the top/bottom of every training image.

**Fixed** in `5db3624`: raw pixel read (format-independent) + letterbox
non-square maps with `pad_id` (default 0 = CARLA `Unlabeled`) matching
`SquarePad`'s exact geometry, instead of stretching. Verified by execution:
real `SegJsonDataset` + real CARLA palette + real 1280x800 RGB + a synthetic
mask in the exact I;16 format — pad rows equal `palette[pad_id]`, a boundary
at mask row 400/800 lands at square row 256 on both the map and RGB paths,
train/inference loaders produce byte-identical output. Full detail:
`GROUNDED_SAM.md` §5.0a / §5.1d.

**Consequence for any checkpoint trained BEFORE `5db3624`:** its structure
conditioning near the top/bottom of frames was trained on misaligned pairs.
Retrain after this fix before judging quality near frame edges.

## 3. Artifact — flat pastel band at the top/bottom of generated images

Same root cause as the segformer pipeline (`SquarePad`'s flat local-mean
letterbox fill — see `GENERATION_QUALITY_SEGFORMER.md` §2 for the measured
colour values on a real image) — this is shared engine code
(`src/data/transforms.py`), not pipeline-specific. Now that §2's fix makes
the map's pad region consistently `Unlabeled` (rather than stretched real
content), the model has a *learnable, consistent* pad-band mapping instead of
a contradictory one — an improvement, though the band itself still exists
until a native-aspect-ratio fix (512x320, see the segformer doc §2) is
applied on this branch too.

## 4. Object/vehicle rendering quality — not separately assessed here

The segformer branch's car-coverage scan (98.5% of images <1% car pixels)
was run against the SegFormer pipeline's own dataset. **This pipeline's real
training set has not been scanned for class-coverage the same way** — if you
want that number for your CARLA dataset, `analyze_car_coverage.py` on this
branch already supports `--car_class_id 14` (CARLA's Car id; confirmed in
`configs/grounded_sam_classes.json`). Car
rendering quality is explicitly OUT OF SCOPE for active work regardless of
what a scan would show.

## 5. What's already available, no retraining needed

Same shared inference knobs as the segformer pipeline
(`inference.lora_scale_start/_end/_decay_start_frac`,
`inference.conditioning_kernel_size` — see `GROUNDED_SAM.md` §5.1d for the
run-recipe/output-layout docs and §5.1a for the full mechanism). Both
default to no-op.

## 6. Verification status

- §2 (mask format bug + fix): verified by execution (real classes, real
  dataset RGB, synthetic mask in the exact confirmed real format).
- §3 (pad band, shared cause): the RGB-side measurement was done on the
  segformer branch's real data (same `SquarePad` code, so the mechanism
  transfers); not independently re-measured on a real CARLA frame here.
- §4: explicitly not scanned for this pipeline — stated as unknown, not
  assumed to match the segformer pipeline's numbers.
- No claim here about output quality at real training scale — this pipeline
  has only been trained here in small smoke tests (real code path, synthetic/
  minimal data); the real verdict requires a real run on your training
  machine.
