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

## 3. Artifact — flat pastel band at the top/bottom of generated images — FIXED 2026-08

Same root cause as the segformer pipeline (`SquarePad`'s flat local-mean
letterbox fill — see `GENERATION_QUALITY_SEGFORMER.md` §2 for the measured
colour values on a real image) — this is shared engine code
(`src/data/transforms.py`), not pipeline-specific. The earlier fix (§2's
`pad_id` change) made the pad region *consistent* rather than contradictory,
but did not remove it -- the model was still trained on a real, if
consistent, artificial band.

**APPLIED FIX (2026-08), same as the segformer branch:** new
`resize_mode: "aspect"` -- direct resize to `size: [512, 320]` (1280/800 =
1.6 = 512/320 exactly; divisible by 64; caps the long side at exactly
SD1.5's native 512, deliberately more conservative than a larger same-ratio
target like 832x512). No pad at all -> nothing to learn. Ported into
`square_id_map()` (`src/data/transforms.py`, this branch's own live-at-load-
time squaring function -- architecturally different from segformer's
calc-time squaring, see `_square_rgb_steps`/`square_id_map`), `local_seg.py`,
`grounded_sam_training.py`, `grounded_sam_inference.py`, and both configs.
Locked in as the new default; `letterbox`/`CenterCrop` remain available for
comparison via `resize_mode=letterbox`/`CenterCrop`.

Also fixed while implementing this: `SD15.sample_custom()` (used by
`grounded_sam_training.py`'s checkpoint-monitoring images) had NO way at all
to produce non-square output -- `height`/`width` were computed
unconditionally from `unet.config.sample_size`, ignoring any override. Every
checkpoint-monitoring grid during non-square training would have silently
come out square and misaligned without this fix.

UNVERIFIED BY EXECUTION with real masks/real weights as of this writing --
the shape/axis-order logic was verified with synthetic data (16/16 checks
pass, including a real subprocess run of the new
`grounded_sam_map_calculations.py` against a synthetic dataset matching the
confirmed real class_map.png format), but the actual retrain on real CARLA/
real-photo data + real generated output has not been observed yet.

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

## 6b. New — live Grounded-SAM encoder (Tier 2) [2026-08]

Previously `GroundedSamEncoder` was a training-only stub (Tier 1) --
`forward()`/`label_ids()` raised `NotImplementedError` by design, since
masks were assumed to always be pre-made externally. Built for real now,
opt-in via `lora.struct.encoder.live: true`:

- Uses transformers' NATIVE Grounding DINO + SAM support
  (`AutoModelForZeroShotObjectDetection` + `SamModel`/`SamProcessor`) --
  NOT the standalone GroundingDINO repo + `segment-anything` package an
  earlier version of this module's docs assumed. No compiled CUDA
  extension, no second package ecosystem -- `transformers` is already a
  pinned dependency, and every other encoder in this project already uses
  the identical `from_pretrained()` pattern.
- Your 29-class taxonomy becomes the GroundingDINO text prompt
  automatically (`build_grounding_dino_prompt`), CamelCase names
  normalized to natural phrases ("TrafficLight" -> "traffic light"),
  4 classes excluded by default (Unlabeled/Static/Dynamic/Other -- no
  clear visual referent to search for).
- Compositing: "stuff" classes (Roads, Sky, Building, ...) painted first,
  "thing" classes (Car, Pedestrian, Pole, ...) painted on top by
  descending confidence -- standard panoptic-segmentation practice, so a
  detected car isn't erased by an overlapping road mask.
- Defaults to the LARGEST non-giant variants (`grounding-dino-base`,
  `sam-vit-huge`) -- chosen deliberately over smaller/faster variants
  after a real test showed the tiny GroundingDINO variant missing an
  obvious detection that a larger model would very likely catch.

**VERIFIED BY REAL EXECUTION (2026-08, not synthetic-logic-only like most of
this doc's other entries)** -- actual GPU, actual downloaded weights
(`grounding-dino-tiny` + `sam-vit-base`, small variants for a fast test),
run against a synthetic test image:
- The full detect -> segment -> composite pipeline runs end-to-end and
  produces correctly-shaped, correctly-ranged output.
- **A real bug was caught this way, not by reasoning**: the CFG-dropout
  all-zero input (a flat mid-grey image after the `[-1,1] -> [0,1]`
  conversion) caused GroundingDINO to hallucinate a confident "sky"
  detection on an image with zero real content. Fixed with an explicit
  flat-image short-circuit (`arr.std() < 1.0` -> return all-Unlabeled
  without running detection at all) -- this is a real, load-bearing check,
  not defensive dead code; it was observed failing before the fix existed.
- A second real, non-blocking finding: GroundingDINO sometimes merges
  adjacent prompt phrases into one returned label on a multi-class prompt
  (e.g. "traffic sign road line" as one detection). Handled safely --
  unmatched labels are DROPPED, not guessed at -- but this means recall on
  a many-class prompt is imperfect. Worth watching once run on real data;
  not something a code fix should paper over with a guess.

**NOT YET VERIFIED**: real-world detection/segmentation quality on actual
driving photos -- particularly "stuff" classes, since GroundingDINO is
fundamentally an object detector, not a dense per-pixel classifier the way
SegFormer is. This is the real open risk flagged when Tier 2 was first
discussed; only real photos will answer it.

## 7. New — grounded_sam_map_calculations.py [2026-08]

Until now, this branch had **no calc script at all** -- masks are made
externally (confirmed: this branch's Tier-1 design assumes `class_map.png`
already exists per image), so there was nothing to assemble the
`(raw_image_path, seg_path, prompt)` manifests `grounded_sam_training.py`/
`grounded_sam_inference.py` actually read. `grounded_sam_map_calculations.py`
fills that gap -- it runs NO model and needs NO GPU, it only pairs each
image (read from your source `data/{train,val,test}.jsonl`) with its
already-existing mask and writes + verifies the standard-key manifest.

Mask location strategy: sibling file in the image's own folder, named
`class_map.png` by default (`--mask_name` to override), OR read directly
from a field already on the source entry (`--mask_key`). Confirmed against
this repo's real `data/train.jsonl` on disk: the image-path key is `target`,
not `raw_image_path` -- `--image_path` defaults to `target` for that reason;
override it if your real dataset's source JSONL uses a different key.

Verified by execution (synthetic data: two fake images + sibling
`class_map.png` masks in the confirmed real format, run as a real
subprocess, real output manifest inspected) -- not yet run against your
actual CARLA/real-photo dataset.

## 8. Artifact — garbled/nonsensical structure-marking shapes [2026-08]

Same class of finding as the segformer branch's §6 (see
`GENERATION_QUALITY_SEGFORMER.md`): where the seg map's conditioning is flat/
uninformative (e.g. a CARLA "Roads" or "RoadLine" region carries no lane-
marking detail), SD1.5's own learned prior can intrude and render it
incoherently -- a base-model capability ceiling, not a pipeline bug. `negative_prompt`
is now threaded through `grounded_sam_inference.py` /
`configs/inference_grounded_sam.yaml` (`inference.negative_prompt`) for the
same zero-retrain first test as the segformer branch.

## 9. Base checkpoint + VAE — changed 2026-08

Stock `runwayml/stable-diffusion-v1-5` swapped for **epiCRealism**
(`emilianJR/epiCRealism`), matching the segformer branch's decision --
same UNet architecture, `add_lora_to_unet` needs zero changes. Paired with
`stabilityai/sd-vae-ft-mse` via the new `model.vae_path` config
(`src/model.py` `ModelBase.__init__`). See `GENERATION_QUALITY_SEGFORMER.md`
§7 for the full reasoning (chosen over Realistic Vision V6.0 because it
ships in ready-to-use diffusers format) and the honest architecture-ceiling
note (this does not make the result comparable to a 32B transformer model
like Flux.2 -- still the same UNet family, just better weights).

UNVERIFIED BY EXECUTION: has not been run against a real trained checkpoint
on this branch yet.
