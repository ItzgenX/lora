# Generation-Quality Diagnosis — SegFormer pipeline [2026-07-20]

Findings from the user's own real generated-image samples
(`data/info/test/*`, `data/info/test2_real_world_iamge/*`) and a real
`analyze_car_coverage.py` scan of the actual 59,766-image training set
(`data/info/analysis res.jpeg`). Scoped to the **segformer branch only** —
its own calc script, its own 512x512 pre-squared maps, no CARLA/Grounded-SAM
content here (see `GENERATION_QUALITY_GROUNDED_SAM.md` on the `grounded_sam`
branch for that pipeline's own findings).

---

## 1. What works (confirmed from the user's real outputs)

Generated layouts follow the seg map closely — road position/curvature,
grass verges, building masses, pole placement all match the conditioning
across every sample shown. This confirms the core mechanism (LoRA injection,
mapper, `skip_encode` inference path) works correctly. The problems below are
all traceable to specific properties of the training data, not the injection
code.

## 2. Artifact — flat pastel band at the top/bottom of every output

**Cause (verified by executing the real training transform on a real dataset
image, `custome_dataset/000000/raw_image.jpg`):** `SquarePad`
(`src/data/transforms.py`) letterboxes every 1280x800 frame with a **flat
local-mean fill colour**. Measured on that real image: pad band mean RGB
`(6, 4, 3)` top / `(81, 53, 25)` bottom at 96 rows (512-px scale) top and
bottom. The model was trained on tens of thousands of targets containing that
exact kind of flat band, and reproduces it at generation time — this is
learned behaviour, not a rendering glitch.

**Fix options** (not yet applied — pick one before your next real run):
- Train/infer at the data's NATIVE aspect ratio, 512x320 (1280/800 = 1.6 =
  512/320 exactly; both are multiples of 64, SD1.5-legal) instead of
  letterboxing to a square. No pad band exists in the data → nothing to learn
  to reproduce, and ~37% of every batch stops being wasted flat fill.
- Cheaper/cosmetic alternative: crop the known pad rows off every generated
  output at inference time.

## 3. Artifact — mangled/melted vehicles

**Cause (verified from the user's own scan, not a guess):**
`analyze_car_coverage.py` run on the real 59,766-image training set found
**98.5% of images have under 1% car pixels; mean car-pixel fraction is
0.08%; zero images have a car covering >20% of the frame.** The LoRA has
essentially never seen a large, clearly-visible car, so it has no learned
appearance for that region even though the seg map declares "car here."

**User decision (2026-07-19): this is explicitly OUT OF SCOPE for now** — no
further work on car coverage is being done. Documented here only so the
cause isn't re-investigated from scratch later. If revisited, the fix is
data-side only (oversample car-rich frames / add more of them); no code or
hyperparameter change compensates for absent training signal.

## 4. Artifact — hood ghost at the bottom of generated images

**Cause:** every dashcam frame includes the ego vehicle's hood; SegFormer
(Cityscapes-trained, hoods are cropped out of that dataset) misclassifies it
as road/sidewalk/car. The model learned the resulting dark reflective smear
as part of those classes and reproduces it.

**Fix (not yet applied):** crop the bottom ~10-12% (hood region) from frames
at dataset-prep time, before `seg_map_calculations.py` runs on them.

## 5. What's already improved, no data change needed

Two no-retrain inference knobs exist today (`model.py`'s `sample_easy`,
wired through `segformer_inference.py`'s `inference.*` config keys — see
`SEG_TRAINING_GUIDE.md` §11 for the full run-recipe/output-layout docs):
- `lora_scale_start=1.0, lora_scale_end=0.4` (`lora_scale_decay_start_frac`
  controls where the fade begins) lets the late denoising steps lean on
  SD1.5's own prior for object appearance instead of a flat, information-poor
  conditioning region — directly targets artifacts like §3/§4 without
  retraining.
- `conditioning_kernel_size=3` (or 5) softens hard seg-map silhouette edges.

Both default to no-op (0 / equal start-end) so existing commands are
unaffected until explicitly set.

## 6. Verification status

- §1, §2, §3, §4 causes: each verified either by executing the real
  `SquarePad` transform on real data, or by reading the user's own
  `analyze_car_coverage.py` output on the real dataset — not guessed.
- Fix option in §2 (512x320 training) and the hood crop in §4: arithmetic/
  reasoning verified, NOT executed as an actual training run in this
  environment.
- No claim here about output quality after any fix is applied — that
  requires a real training run on the training machine.
