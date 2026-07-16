# Grounded-SAM for LoRAdapter Segmentation — 0 to Hero

A from-scratch guide to what Grounded-SAM is, how it differs from SegFormer, its
real strengths and weaknesses **for this project specifically**, and exactly how
it is wired into this repo. Written for a diffusion-model beginner — every term
is explained the first time it appears.

---

## 0 · The one-paragraph summary

Grounded-SAM is not one model — it is **two models chained together**: an
open-vocabulary *detector* (GroundingDINO) that finds objects you name in plain
text, and a *mask generator* (SAM, "Segment Anything Model") that traces the
exact pixel outline of each found object. Together they turn a sentence like
`"car. person. traffic light."` plus an image into pixel masks for those things.
In this project we do **not** use it to condition the diffusion model directly —
we use it (offline, elsewhere) to produce **segmentation maps**, which are then
fed to the LoRAdapter model as a "paint the scene in this layout" instruction.

---

## 1 · Concepts from zero

### 1.1 What a "segmentation map" is
A **segmentation map** is an image where every pixel is labelled with *what kind
of thing it is* — pixel (12, 40) is "road", pixel (300, 15) is "sky", and so on.
Stored as an 8-bit grayscale PNG, each pixel's value IS its class number
(0 = road, 1 = sidewalk, …). It has the same width/height as the photo it
describes. It throws away colour and texture, keeping only *the shape and
position of each category*.

### 1.2 Why a diffusion model wants one (structure conditioning)
LoRAdapter generates images from noise. Left alone it invents any layout. A
segmentation map is a **structure condition**: "put a road here, a building
there, sky along the top." The model then fills in realistic texture while
obeying that layout. This is how you turn a CARLA (simulator) frame into a
photorealistic one *that keeps the same scene geometry* — the map pins the
structure, the model supplies realism. (The map is injected via the LoRA
mapper — a small Conv2d that reads the colour map as the conditioning signal;
see §5 and the code in seg_training.py.)

### 1.3 The two families of segmentation model
This distinction is the single most important thing to understand, because it is
where Grounded-SAM and SegFormer fundamentally differ.

- **Semantic segmentation (SegFormer)** — one neural network looks at the whole
  image once and assigns *every pixel* to one of a **fixed** list of classes
  (SegFormer-Cityscapes: 19 classes). Every pixel gets a label, always. There is
  no "I didn't find anything here" — the softmax always picks a winner.

- **Detection + promptable segmentation (Grounded-SAM)** — first GroundingDINO
  *detects* objects matching a **text prompt** you supply, drawing a box around
  each. Then SAM traces a precise mask inside each box. The class list is
  **open** — whatever words you put in the prompt. Pixels belonging to nothing
  you asked for (or nothing detected) get **no label**.

### 1.4 "Thing" classes vs "stuff" classes
- **Things** = countable objects with clear boundaries: car, person, pole,
  traffic light. Detectors excel here — there's a discrete object to box.
- **Stuff** = amorphous regions with no object boundary: road, sky, vegetation,
  building façade. Detectors are structurally *weak* here — there's no single
  "object" to draw a box around; where does "the road" begin and end?

SegFormer handles stuff and things equally (it just classifies pixels).
Grounded-SAM is naturally strong on things and naturally weak on stuff. Keep
this in mind — it drives most of the trade-offs below.

---

## 2 · How the two compare, concretely

| Property | SegFormer-Cityscapes | Grounded-SAM |
|---|---|---|
| Approach | One-pass per-pixel classifier | Detector (GroundingDINO) → mask refiner (SAM) |
| Class list | Fixed 19 Cityscapes classes | **Open** — whatever you prompt |
| Every pixel labelled? | Yes, always | No — undetected pixels are unlabelled |
| "Stuff" (road/sky/veg) | Handled natively | Weak (no object to detect) |
| "Thing" boundaries (car/person) | Good (b5 backbone) | Often **sharper** (SAM masks are crisp) |
| Determinism | Deterministic (one forward pass) | Threshold-dependent (box/text confidence, NMS) |
| Speed | One model, fast | Two heavy models, slower |
| Setup cost | One HF checkpoint | GroundingDINO + SAM packages + 2 checkpoints |

---

## 3 · Positives (why you might prefer Grounded-SAM)

1. **Sharper object masks.** SAM produces very tight, high-quality outlines on
   *things* — a car's silhouette, a pedestrian's contour. If the point of your
   conditioning is crisp object structure, this can beat SegFormer's boundaries.

2. **Open vocabulary — your own classes.** You are not stuck with Cityscapes' 19
   categories. Need "traffic cone", "delivery van", "crosswalk stripe"? Just add
   the words. This is the reason you chose it: your class set is *yours*, defined
   by the prompts you generated the maps with, not by a pretrained taxonomy.

3. **Class set can match your target domain.** Because you pick the vocabulary,
   you can align it exactly with the objects that matter for CARLA→real, rather
   than accepting whatever Cityscapes happened to include.

4. **Decoupled from the training run.** You already generated the maps offline.
   Training just reads them — so the (heavy, slow) Grounded-SAM cost was paid
   once, not on every training step. (This is the same "pre-saved maps" design
   the SegFormer pipeline uses: the encoder never runs during training.)

---

## 4 · Negatives (the honest downsides — do not skip this)

1. **"Stuff" coverage is the weak point.** Road, sky, vegetation, building — the
   backbone of a driving scene — are exactly what a detector-first pipeline
   struggles to segment cleanly. If your Grounded-SAM maps have ragged or missing
   stuff regions, the conditioning quality suffers there. **Check your maps for
   this** before trusting them (see §6).

2. **Unlabelled / background pixels.** Any pixel Grounded-SAM didn't assign needs
   *some* value. If that "background" is a real class in your list, fine — but if
   large chunks of frame are an undefined catch-all, the model is being told
   "structure unknown here", which weakens control in those regions. You noted
   you don't yet know how background is handled in your maps — **resolve this**:
   decide whether there's an explicit `unlabelled`/`background` class ID.

3. **Non-determinism.** Grounded-SAM's output depends on confidence thresholds
   and non-max-suppression. The *same* image can yield a slightly different mask
   set run-to-run or version-to-version. Since your maps are already generated
   and frozen on disk, this doesn't affect *training* — but it matters the day
   you build live inference (Tier 2): the live maps must match how the training
   maps were made, or train/inference distributions diverge.

4. **Heavier live path.** For inference on a *new* image (a fresh CARLA frame),
   two large models must run (GroundingDINO + SAM ViT), needing their packages,
   checkpoints, and your class prompts. That is materially slower and more setup
   than SegFormer's single forward pass. (This is **Tier 2**, not yet built —
   see §5.)

5. **It does not fix the CARLA-truck artifact.** This project already proved (by
   scanning all 59,766 training maps) that the warped-vehicle problem is a
   **training-data coverage gap** — only 3 images had a car covering >15% of the
   frame; the dataset max was 17.8%. The generator was never taught to render a
   large close-up vehicle. That is independent of *which* segmentation model made
   the maps — swapping SegFormer for Grounded-SAM cannot fix it. (Diagnostics:
   [analyze_car_coverage.py](analyze_car_coverage.py),
   [check_seg_coverage.py](check_seg_coverage.py).)

---

## 5 · How Grounded-SAM is wired into THIS repo

The design **reuses the entire segmentation pipeline** and swaps only the parts
that must differ (the class palette and the encoder slot). Nothing is duplicated.

### 5.1 What was built — "Tier 1": train on pre-saved maps
This is everything needed to train the LoRAdapter model on the
`(raw_image, seg_map, prompt)` pairs you already generated with Grounded-SAM.

- **[src/encoders/grounded_sam_encoder.py](src/encoders/grounded_sam_encoder.py)**
  - `load_grounded_sam_palette(classes_file)` — reads your class list and builds
    a colour palette.
  - `generate_distinct_palette(n)` — auto-assigns well-separated colours so you
    don't hand-pick them (colours only need to be distinct + consistent).
  - `GroundedSamEncoder` — a **training-only slot filler**. Training never calls
    the encoder (it feeds pre-saved maps via `skip_encode=True`), so this holds
    only config and loads **no** heavy models. `live_available = False`.

- **[configs/grounded_sam_classes.json](configs/grounded_sam_classes.json)** —
  the class-definition file **you fill in**: `{"0": "road", "1": "car", …}`, in
  the exact ID order your saved maps use.

- **[configs/experiment/train_grounded_sam.yaml](configs/experiment/train_grounded_sam.yaml)**,
  **[configs/data/local_grounded_sam.yaml](configs/data/local_grounded_sam.yaml)**,
  **[configs/lora/encoder/grounded_sam.yaml](configs/lora/encoder/grounded_sam.yaml)**
  — the config that points training at your class set + manifests.

- Shared loader **[src/data/local_seg.py](src/data/local_seg.py)** gained a
  `classes_file` (build palette from your classes) and manifest-key flags
  `image_key` / `seg_key` / `prompt_key` (so your JSONL can use any key names).

### 5.2 What was NOT built — "Tier 2": live map generation
A live `GroundedSamEncoder` that actually runs GroundingDINO + SAM to make a map
for a brand-new image. Needed for **inference on new frames** and for the **mIoU
metric** (which segments the generated image to score layout adherence — auto-
skipped for now). This requires the packages, checkpoints, and your class-name
prompts, none of which are set up yet. Deliberately deferred.

---

## 6 · Before you train — a checklist grounded in what we found

1. **Confirm your class count and IDs.** Run
   `python scan_seg_map_classes.py --json_file data/grounded_sam/train.jsonl`.
   It reports the true global class set across ALL maps (a rare class may only
   appear in a few images). Use that count to fill the class file.

2. **Confirm the map format.** Run
   `python check_seg_map_format.py --seg_map <one_real_map>`. You already did
   this once — it showed mode `L`, uint8, clean IDs 0–17. Good. But your files
   were `.jpeg` (lossy). If the scan in step 1 shows stray out-of-range values or
   many per-file noise outliers, **re-export as PNG** (lossless) before training.

3. **Decide the background/unlabelled policy.** Find out whether your maps leave
   undetected pixels as a sentinel value; if so, make it an explicit class in the
   class file so the palette covers it.

4. **Fill [configs/grounded_sam_classes.json](configs/grounded_sam_classes.json)**
   with your real classes (delete the `__README__` key), then:
   `python seg_training.py experiment=train_grounded_sam`

5. **Do a short smoke run first.** Training here is code- and config-verified but
   has **not been executed** on real data on the target machine. Run a few steps,
   watch val/loss and the first checkpoint's monitoring images, before committing
   to a full run.

---

## 7 · File inventory — what each file is and why it's here

The Grounded-SAM pipeline **reuses the existing segmentation engine** rather than
duplicating it. That engine keeps its original `seg_*` names (segmentation is
segmentation, whatever model made the maps). So some `seg_*` files are the shared
engine we depend on, and others are SegFormer-only.

### 7.1 Grounded-SAM–specific (new in this work)
| File | Why it's here |
|---|---|
| [src/encoders/grounded_sam_encoder.py](src/encoders/grounded_sam_encoder.py) | Palette loader, distinct-colour generator, and the training-only `GroundedSamEncoder` slot module. |
| [configs/grounded_sam_classes.json](configs/grounded_sam_classes.json) | Your class list (id → name). **You fill this in.** |
| [configs/experiment/train_grounded_sam.yaml](configs/experiment/train_grounded_sam.yaml) | The training experiment: wires encoder + data + your manifests. |
| [configs/data/local_grounded_sam.yaml](configs/data/local_grounded_sam.yaml) | Data config: class file + manifest key names. |
| [configs/lora/encoder/grounded_sam.yaml](configs/lora/encoder/grounded_sam.yaml) | Encoder slot config (no HF model to load). |
| [GROUNDED_SAM.md](GROUNDED_SAM.md) | This guide. |

### 7.2 Shared segmentation engine — REUSED by Grounded-SAM (must keep)
| File | Why Grounded-SAM needs it |
|---|---|
| [seg_training.py](seg_training.py) | **The training script itself** — run with `experiment=train_grounded_sam`. |
| [src/data/local_seg.py](src/data/local_seg.py) | The dataset loader — reads your maps, colourises with your palette. |
| [src/encoders/seg_encoder.py](src/encoders/seg_encoder.py) | Provides the palette math (`seg_palette_tensor`, `seg_colorize_ids`, `seg_ids_from_colormap`) that local_seg.py + seg_training.py import. (Its `SegmentationEncoder` class is SegFormer-only, but the file is load-bearing.) |
| [configs/train_seg.yaml](configs/train_seg.yaml) | Base config `seg_training.py` loads (`config_name`). Experiment layers on top. |
| [src/data/transforms.py](src/data/transforms.py) | `SquarePad` letterbox preprocessing, shared by all image loading. |
| [src/utils.py](src/utils.py) | LoRA build, checkpoint save, GPU diagnostics, metrics — the training backbone. |

### 7.3 Diagnostics built for this investigation (keep — model-agnostic)
| File | Purpose |
|---|---|
| [scan_seg_map_classes.py](scan_seg_map_classes.py) | Global class count + JPEG-noise check across a manifest. |
| [check_seg_map_format.py](check_seg_map_format.py) | Inspect one map file's format (mode, dtype, value range). |
| [analyze_car_coverage.py](analyze_car_coverage.py) | Car-class coverage histogram (found the CARLA-truck data gap). |
| [check_seg_coverage.py](check_seg_coverage.py) | Per-image, per-class coverage vs the training distribution. |

### 7.4 SegFormer-only (candidates for removal on this branch)
These are used ONLY by the SegFormer approach and are **not** imported by any
Grounded-SAM code:
| File | What it did (SegFormer) |
|---|---|
| configs/lora/encoder/segformer.yaml | SegFormer encoder config. |
| configs/experiment/train_seg.yaml | SegFormer training experiment. |
| configs/inference_seg.yaml | Live SegFormer inference config. |
| configs/data/local_seg.yaml | SegFormer data config (Grounded-SAM uses local_grounded_sam.yaml). |
| seg_map_calculations.py | Computed SegFormer maps offline (Grounded-SAM maps are made externally). |
| seg_inference.py | Live SegFormer inference (Tier 2 not built yet). |
| seg_finetune.py / seg_make_draft_masks.py | Earlier SegFormer-on-CARLA finetuning exploration. |
| SEGMENTATION.md / SEG_TRAINING_GUIDE.md | SegFormer pipeline docs. |

Note: `seg_encoder.py` is NOT in this list — it stays, because the Grounded-SAM
path imports its palette functions. Only its `SegmentationEncoder` *class* is
SegFormer-specific (it can optionally be stripped, but the file must remain).

---

## 8 · Bottom line

Grounded-SAM is the right tool when you need **your own object vocabulary** and
**crisp object masks**, and you accept weaker "stuff" coverage, a non-
deterministic and heavier live path, and more setup. For dense structure
conditioning on driving scenes it is a legitimate alternative to SegFormer —
worth testing on its own merits. Just keep two facts in view: it does **not**
address the separate CARLA-truck data-coverage problem, and the live-inference
half (Tier 2) is real additional work still ahead.
