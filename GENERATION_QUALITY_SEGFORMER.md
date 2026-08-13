# Generation-Quality Diagnosis — SegFormer pipeline [2026-07-20]

**This is now the single doc for this branch** (2026-08 consolidation —
`SEGMENTATION.md`, `LORA_ARCHITECTURE.md`, `SBATCH_ZERO_TO_HERO.md`,
`SEG_TRAINING_GUIDE.md`, `SEGFORMER_FILES.md` were removed; their load-bearing
content that wasn't already duplicated elsewhere is folded in below).

## 0. Where things live now

- **LoRA/FiLM architecture, taught from zero** (what used to be
  `LORA_ARCHITECTURE.md`): this is now a published interactive field guide
  with diagrams, not a markdown file — ask for the link if you don't have it,
  or see `src/lora.py` (`NewStructLoRAConv.forward`, ~line 177) directly, which
  is the ground truth either way.
- **Training parameters for your actual dataset/GPU**: run
  `python recommend_training_params.py --data_dir <your data>` — it counts
  your real manifest and detects your real GPU rather than using a stale
  worked example. Dataset-math tables that used to live in a doc are
  intentionally NOT reproduced here for that reason (a hand-copied number
  goes stale; the script never does).
- **Cluster setup**: the old cluster doc covered JUSUF (V100 16GB nodes) —
  not your actual 4x95GB machine, and no longer relevant. Ask fresh for
  instructions specific to your real hardware if/when needed, rather than
  trusting anything V100-specific left over.
- **Fresh-clone file checklist** (what used to be `SEGFORMER_FILES.md`): see
  §9 at the bottom of this file.

## 0a. Core architecture facts worth keeping without opening code

- Vanilla LoRA: `y = W·x + (B·A)·x` — a FIXED correction, same regardless of
  input.
- This repo's conditional LoRA (`NewStructLoRAConv`, `src/lora.py:143-197`)
  adds ONE thing on top: the low-rank branch gets FiLM-modulated by the
  conditioning map `c` on every forward call —
  `a_cond = A(x) * (gamma(c)+1.0) + beta(c)` — before going through `B`. That
  one line is the entire novelty; everything else is standard LoRA.
- It physically REPLACES the original `nn.Conv2d` inside the live UNet
  (`setattr`, `src/model.py:313-317`) — not a wrapper, not a hook.
- Confirmed 2026-08 by diffing directly against a fresh `CompVis/LoRAdapter`
  clone: `src/lora.py`, `src/mapper_network.py`, `configs/lora/struct.yaml`
  are byte-identical to upstream. This architecture is the real paper
  mechanism, not a modified/drifted copy.

Findings from the user's own real generated-image samples
(`data/info/test/*`, `data/info/test2_real_world_iamge/*`) and a real
`analyze_car_coverage.py` scan of the actual 59,766-image training set
(`data/info/analysis res.jpeg`). Scoped to the **segformer branch only** —
its own calc script, no CARLA/Grounded-SAM content here (see
`GENERATION_QUALITY_GROUNDED_SAM.md` on the `grounded_sam` branch for that
pipeline's own findings). The samples these findings are drawn from were
generated under the OLD 512x512 letterbox default — that's what exposed the
pad-band bug in §2 below. Current default is 512x320 `aspect` (no pad, no
crop); see §2.

---

## 1. What works (confirmed from the user's real outputs)

Generated layouts follow the seg map closely — road position/curvature,
grass verges, building masses, pole placement all match the conditioning
across every sample shown. This confirms the core mechanism (LoRA injection,
mapper, `skip_encode` inference path) works correctly. The problems below are
all traceable to specific properties of the training data, not the injection
code.

## 2. Artifact — flat pastel band at the top/bottom of every output — FIXED 2026-08

**Cause (verified by executing the real training transform on a real dataset
image, `custome_dataset/000000/raw_image.jpg`):** `SquarePad`
(`src/data/transforms.py`) letterboxes every 1280x800 frame with a **flat
local-mean fill colour**. Measured on that real image: pad band mean RGB
`(6, 4, 3)` top / `(81, 53, 25)` bottom at 96 rows (512-px scale) top and
bottom. The model was trained on tens of thousands of targets containing that
exact kind of flat band, and reproduces it at generation time — this is
learned behaviour, not a rendering glitch. Independently RE-confirmed 2026-08
from `AM.jpeg` (a real checkpoint grid): the band is visible in the ORIGINAL,
SEG MAP, and PREDICTED panels alike.

**APPLIED FIX (2026-08):** train/infer at the data's native aspect ratio
instead of letterboxing to a square — exactly the first option below, now
implemented, not just proposed:
- New `resize_mode: "aspect"` (`src/data/transforms.py`, `seg_map_calculations.py`
  `--width`/`--height` flags, `SegmentationEncoder`, `src/data/local_seg.py`,
  both training/inference scripts) — direct resize to `size: [512, 320]`
  (1280/800 = 1.6 = 512/320 exactly; both divisible by 64, SD1.5-legal; caps
  the long side at exactly SD1.5's native 512, deliberately more conservative
  than a larger same-ratio target like 832x512). No pad band exists in the
  data anymore → nothing to learn to reproduce.
- Locked in as the new default in `configs/experiment/train_seg.yaml` /
  `configs/inference_seg.yaml`. `letterbox`/`CenterCrop` remain available for
  comparison via `resize_mode=letterbox`/`CenterCrop`.
- Also fixed while implementing this: `SD15.sample_custom()` (used by
  `segformer_training.py`'s checkpoint-monitoring images) had NO way at all to
  produce non-square output — `height`/`width` were computed unconditionally
  from `unet.config.sample_size`, ignoring any override. Every
  checkpoint-monitoring grid during non-square training would have silently
  come out square and misaligned without this fix. Caught by reading
  `segformer_training.py` in full, not by testing the inference script alone.
- UNVERIFIED BY EXECUTION with real data/real weights as of this writing —
  the shape/axis-order logic (PIL vs torchvision vs `F.interpolate` argument
  order, a real bug class) was verified with synthetic data (18/18 checks
  pass, see verification pass 2026-08), but the actual retrain on real driving
  data + real generated output has not been observed yet. Next step: run
  `seg_map_calculations.py --resize_mode aspect --width 512 --height 320`,
  retrain, and visually confirm the pad band is gone AND no new artifact
  appeared at the aspect-mode seams.
- Cheaper/cosmetic alternative (not applied, kept as a fallback idea): crop
  the known pad rows off every generated output at inference time.

## 3. Artifact — mangled/melted vehicles

**Cause (verified from the user's own scan, not a guess):**
`analyze_car_coverage.py` (later removed from the repo; this finding stands
on its own) run on the real 59,766-image training set found
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
wired through `segformer_inference.py`'s `inference.*` config keys — the
comments directly above those keys in `configs/inference_seg.yaml` are the
authoritative run-recipe/output-layout reference now):
- `lora_scale_start=1.0, lora_scale_end=0.4` (`lora_scale_decay_start_frac`
  controls where the fade begins) lets the late denoising steps lean on
  SD1.5's own prior for object appearance instead of a flat, information-poor
  conditioning region — directly targets artifacts like §3/§4 without
  retraining.
- `conditioning_kernel_size=3` (or 5) softens hard seg-map silhouette edges.

Both default to no-op (0 / equal start-end) so existing commands are
unaffected until explicitly set.

## 6. Artifact — garbled/nonsensical road-marking shapes [2026-08]

**Cause (read from real `result.jpeg`/`result_1.jpeg` output, not guessed):**
the segmentation map's road region is one flat, uniform class with zero
marking information — nothing in the conditioning tells the model to draw
lane arrows. The garbled shapes are SD1.5's OWN learned prior ("photos of
roads usually have lane markings") intruding, combined with SD1.5's well-known
weakness at rendering small precise symbolic detail coherently (same failure
class as text/hands). Not a segmentation-pipeline bug — a base-model
capability ceiling.

**Fix (added 2026-08, not yet exercised on real output):** `negative_prompt`
is now threaded through `segformer_inference.py` / `configs/inference_seg.yaml`
(`inference.negative_prompt`) — forwarded via `sample_easy`'s existing
`**kwargs` passthrough to the underlying diffusers pipeline; needed zero
`model.py` changes. Try
`negative_prompt: "road markings, lane markings, arrows, text, symbols, watermark"`
first, before any retrain.

## 7. Base checkpoint + VAE — changed 2026-08

Stock `runwayml/stable-diffusion-v1-5` swapped for **epiCRealism**
(`emilianJR/epiCRealism`), a photorealistic SD1.5-ARCHITECTURE finetune —
same UNet, same state-dict keys, `add_lora_to_unet` needs zero changes.
Chosen over Realistic Vision V6.0 because it ships as a ready-to-use diffusers
pipeline (V6.0 is safetensors-only, needs `from_single_file()` support this
repo doesn't have). Paired with `stabilityai/sd-vae-ft-mse` via the new
`model.vae_path` config (`src/model.py` `ModelBase.__init__`, mirrors the
existing `tiny_vae` pattern) — this is ALSO required if you ever switch to a
"noVAE"-style checkpoint that ships with no VAE weights at all.

Honest ceiling, stated once: this combination is the strongest realistic
target achievable by EXTENDING this codebase's architecture (still a
~1B-parameter UNet + one CLIP text encoder). It is not comparable to a 32B
transformer model like Flux.2 — that would require a different
architecture family, not a config change.

UNVERIFIED BY EXECUTION: the checkpoint swap has not yet been run against a
real trained LoRA to confirm transfer quality (LoRA weights were trained
against stock SD1.5's activations; most finetunes are continued-training from
stock SD1.5 so transfer is *expected* to work, but expected is not the same
as observed). Cheapest first test: load an EXISTING trained checkpoint against
the new `base_model_path=checkpoints/local_models/epicrealism` with zero
retraining, and look at the output before committing cluster time to a full
retrain.

## 9. Fresh-clone file checklist (was `SEGFORMER_FILES.md`)

Setting up this pipeline on a clean `CompVis/LoRAdapter` clone:

**REPLACE** (already exist upstream — overwrite at the same path):
`src/data/local_seg.py`, `src/data/transforms.py`, `src/utils.py`,
`src/model.py` (gained `vae_path` + `sample_custom` height/width — see §7/§2),
`src/encoders/seg_encoder.py` (gained non-square size support).

**ADD** (brand new): `segformer_training.py`, `segformer_inference.py`,
`seg_map_calculations.py`, `configs/experiment/train_seg.yaml`,
`configs/data/local_seg.yaml`, `configs/inference_seg.yaml`,
`configs/lora/encoder/segformer.yaml`, `configs/train_seg.yaml`,
`configs/model/sd15.yaml` (gained `vae_path: null`), `download_sd15.py`
(gained skip-existing check + epiCRealism/sd-vae-ft-mse), plus the cluster
`.sbatch` scripts if using a Slurm cluster.

**Confirmed unchanged from upstream** (verified by diff, 2026-08, not
assumed): `src/lora.py`, `src/mapper_network.py`, `configs/lora/struct.yaml`.
Also untouched: `train.py`, `sample.py`, `src/data/local.py`,
`configs/train.yaml`, `configs/data/local.yaml`.

**Resolution note**: `seg_map_calculations.py` needs an extra calc step
`grounded_sam` doesn't — SegFormer *computes* the map from a squared RGB
image, so it must run first, with `--resize_mode`/`--width`/`--height`
matching whatever you'll train with (the geometry is baked into the saved map
permanently, unlike `grounded_sam` which squares live at load time).

**`--image_path` gotcha, confirmed by execution 2026-08-11**: the script's
own `--data_dir` default (`--image_path raw_image_path`) does NOT match this
project's real local `data/{train,val,test}.jsonl` — those entries use the
key `target`. Running the docstring's own "TYPICAL WORKFLOW" command exactly
as written (`seg_map_calculations.py --data_dir data/ --resize_mode ...`)
raises `KeyError: Entry has neither 'raw_image_path' nor 'raw_image_path'.`
against this real file. Always add `--image_path target` for this dataset —
already the default `seg_map_calculations.py --dataset_dir ...` scan-mode
example shows, but the plain `--data_dir` quick command doesn't.

## 6. Verification status

- §1, §2, §3, §4 causes: each verified either by executing the real
  `SquarePad` transform on real data, or by reading the user's own
  `analyze_car_coverage.py` output on the real dataset — not guessed.
- Fix option in §2 (512x320 training) and the hood crop in §4: arithmetic/
  reasoning verified, NOT executed as an actual training run in this
  environment.
- No claim here about output quality after any fix is applied — that
  requires a real training run on the training machine.
