# Segmentation Pipeline — Zero to Hero Guide

**Pipeline**: CTRLorALTer segmentation-conditioning arm (ECCV 2024, arXiv:2405.07913)  
**What it does**: Fine-tunes LoRA blocks inside a Stable Diffusion 1.5 UNet so the model generates images that respect a user-supplied semantic segmentation map. The seg signal (19-class Cityscapes colour palette) is injected via the same mapper-network architecture used by the depth pipeline, keeping the base model frozen.

---

## 0 · Seg Concepts From Zero

For exactly where and how LoRA sits inside the model — which UNet layers get
wrapped, how the FiLM conditioning math works, how the segmentation map's
signal actually reaches the frozen UNet — see the dedicated deep-dive doc:
**[LORA_ARCHITECTURE.md](LORA_ARCHITECTURE.md)**. (Historical note: this
section used to point at a `DEPTH.md §0`; that file was removed when the
depth pipeline was deleted from this branch. `LORA_ARCHITECTURE.md` is its
replacement for the LoRA/FiLM/architecture material — it applies identically
to segmentation and Grounded-SAM, since both share the same underlying
`src/lora.py`/`src/model.py` code.) This section covers only what is
*different* about segmentation.

### 0.1 What semantic segmentation is

A segmentation model answers, for **every pixel**, "what kind of thing is this?" — not "how far away" (that's depth, a number per pixel) but "which category" (a label per pixel). Our model is **SegFormer-b5** trained on **Cityscapes**, a driving-scene dataset with **19 classes**: road, sidewalk, building, wall, fence, pole, traffic light, traffic sign, vegetation, terrain, sky, person, rider, car, truck, bus, train, motorcycle, bicycle (IDs 0–18, in exactly this order — verified against the model's own `id2label`).

### 0.2 Why the conditioning is a COLOUR map, not a grayscale ID ramp

Class IDs are categorical — class 13 (car) is not "one more than" class 12 (rider). If we fed the raw IDs as a grayscale image (id/18), the conv network would treat car≈rider as *numerically similar*, inventing an ordering that doesn't exist. Instead each class gets a fixed, well-separated RGB colour (`SEG_CITYSCAPES_PALETTE` in `src/encoders/seg_encoder.py` — the single source of truth): road is always purple, person always red, sky always steel-blue. Same approach as ControlNet's seg conditioning.

### 0.3 Why we SAVE raw IDs but TRAIN on colours

On disk the maps are 8-bit grayscale PNGs holding raw IDs 0–18 (they look almost black in a viewer — that's correct!). The dataset colourises them at load time (`seg_colorize_ids`). Why not save colour PNGs directly? IDs are canonical and tiny, can be re-coloured without re-running the 82M-param model, and are hand-editable. The palette is applied by ONE shared function everywhere, so a class can never get two different colours.

### 0.4 Why NEAREST interpolation is a hard rule

Resizing an ID map with bilinear interpolation averages neighbouring pixels: between a road pixel (0) and a car pixel (13) it would invent value 6 — "traffic light", a class that isn't there. NEAREST copies the closest pixel instead, so only real labels survive. This is enforced in `src/data/local_seg.py` (`Image.NEAREST`) and is a silent-corruption bug if it ever regresses.

### 0.5 The parity trick: one brain, two mouths

`SegmentationEncoder` has two entry points sharing the same internals (`_predict_ids()`): `label_ids()` (returns raw IDs — used ONLY by the offline calc script) and `forward()` (returns the colour map — what live inference calls). Because both run byte-for-byte the same prediction code and the same palette, the map saved for training and the map computed live at inference are guaranteed identical — verified on real images: palette-colourised IDs vs `forward()` output differ by exactly 0.0.

### 0.6 b5, not b0

SegFormer comes in sizes b0 (3.7M params) to b5 (82M). b5 is locked in because conditioning quality IS the point of this pipeline: b0 loses thin structures (poles, pedestrians, traffic lights) that matter most in driving scenes. The cost is only in the offline calc step and at inference — during training the encoder never runs at all (`skip_encode=True`), which is why depth and seg training measured *identical* VRAM.

### 0.7 Segmentation is NOT in the paper

CTRLorALTer's paper conditions on depth, HED edges, and human pose — never segmentation. This whole pipeline is our extension, built to mirror the depth pipeline exactly (same LoRA rank, mapper, injection points, configs) so the two can be compared fairly. That's also why "does it match the paper?" is answerable for depth but meaningless for seg — seg's correctness standard is the verification suite in 9 of the audit (encoder-slot contract, parity, class-order check, coherence on real images).

### 0.8 The domain-mismatch caveat (know this before judging outputs)

SegFormer-b5-Cityscapes knows *driving scenes*. On the local dev dataset (lifestyle/indoor photos) it produces coherent but oddly-labeled regions — walls become "building", a laptop desk becomes "road"-ish blobs. That is NOT a bug; the model is doing its job on out-of-domain input. On the real driving dataset the labels are semantically right (see AM.jpeg's map: road/buildings/sky exactly where they should be).

### 0.9 Self-test — seg edition

1. Why do the saved seg PNGs look black in an image viewer? *(0.3: they hold class IDs 0–18, not colours — 18/255 is nearly black.)*
2. Why colour palette instead of feeding IDs directly? *(0.2: IDs are categorical; a ramp fakes an ordering; colours give each class a distinct identity.)*
3. What breaks if someone changes NEAREST to bilinear? *(0.4: fabricated in-between classes at every boundary — silent corruption.)*
4. How is train/inference parity guaranteed for seg? *(0.5: label_ids() and forward() share _predict_ids() + one shared palette function; verified 0.0 diff.)*
5. Why doesn't the heavy b5 encoder slow training? *(0.6: skip_encode=True — training loads pre-saved PNGs; the encoder runs only offline and at inference.)*
6. Why can't MiDaS be reused as the seg encoder? *(4.1: regression vs classification — it has no class information at all.)*
7. Why is "does seg match the paper?" the wrong question? *(0.7: seg isn't in the paper; it's our extension, verified by execution instead.)*
8. Why did indoor photos give weird classes like "fence" on furniture? *(0.8: Cityscapes domain mismatch — expected, not a bug.)*

---

## 1 · What & Why

The segmentation pipeline is the structural twin of the depth pipeline. Where depth injects monocular depth maps, seg injects 19-class Cityscapes colour maps. Both use identical LoRA architecture (`NewStructLoRAConv`, rank=128, `only_res_conv`), the same mapper network, and the same training loop — only the encoder and offline preprocessing differ.

Why segmentation as a conditioning signal?
- Depth is scale-ambiguous (near and far objects look similar if they happen to have similar relative depths). Segmentation identifies semantic categories (road, sky, pedestrian, vehicle), giving the model sharper spatial structure for complex outdoor scenes.
- The two pipelines can be evaluated under identical conditions for a fair comparison (same dataset, same architecture, same hyperparameters).

---

## 2 · Files in Order

| Stage | File | When |
|-------|------|------|
| C — offline preprocessing | `seg_map_calculations.py` | **Once**, before any training |
| D — training | `seg_training.py` | Iterative; resumes from checkpoint |
| D — inference | `seg_inference.py` | After training; per-image generation |

Supporting files:

| File | Role |
|------|------|
| `src/encoders/seg_encoder.py` | `SegmentationEncoder`, `SEG_CITYSCAPES_PALETTE` (SSOT), `seg_colorize_ids()` |
| `src/data/local_seg.py` | `SegJsonDataset`, `SegJsonDataModule` |
| `src/data/transforms.py` | `build_seg_square_preprocess()` — single shared factory for seg preprocessing |
| `configs/train_seg.yaml` | Base training config |
| `configs/experiment/train_seg.yaml` | Unified experiment overrides (use this one) |
| `configs/inference_seg.yaml` | Inference config |
| `configs/data/local_seg.yaml` | Seg data module config |
| `configs/lora/encoder/segformer.yaml` | SegFormer-b5 encoder config |

---

## 3 · Data Flow Diagram

```
RAW IMAGE (arbitrary aspect ratio)
        |
        v
  [Stage C -- seg_map_calculations.py]  (run ONCE offline)
        |
        +-- build_seg_square_preprocess(size=512)   # letterbox squaring built in (fixed)
        |     (same factory used by seg_inference.py -- parity guaranteed)
        +-- SegmentationEncoder.label_ids(tensor)
        |     (SegFormer-b5 prediction --> raw class IDs [0..18])
        +-- Save as 8-bit grayscale PNG (mode "L", values 0..18)
        |       --> <dataset>_seg_map/000417/000417_seg_map.png  (sibling folder)
        +-- Write data/seg_training/{train,val,test}.jsonl
               (keys: raw_image_path, seg_path, prompt)

        |
        v
  [Stage D -- seg_training.py]  (per training step)
        |
        +-- SegJsonDataset.__getitem__:
        |     read seg PNG (L mode) --> NEAREST resize --> seg_colorize_ids() with palette
        |     --> 3-channel colour map Tensor [0,1]
        |     (NEAREST interpolation is REQUIRED -- bilinear would corrupt class IDs)
        |
        +-- batch["seg"]  --->  StructMapper  --->  LoRA delta-weights
        |   (skip_encode=True: SegFormer NOT called during training)
        |
        +-- model.forward_easy(imgs, prompts, [seg_maps], skip_encode=True)
        |
        +-- val/loss logged every val_steps; checkpoint+grid every ckpt_steps

        |
        v
  [Stage D -- seg_inference.py]
        |
        +-- build_seg_square_preprocess(size=512)   # letterbox squaring built in (fixed)
        |   (SAME factory function as Stage C -- parity by construction)
        +-- LIVE encoder call: SegmentationEncoder.forward(image) --> colour map
        |   (skip_encode=False: SegFormer runs in model.sample() path)
        |
        +-- Generates 4-panel grid: ORIGINAL | SEG MAP | PREDICTED | RAW SEG GEN
```

---

## 4 · SegFormer-b5 Encoder

**Model**: `nvidia/segformer-b5-finetuned-cityscapes-1024-1024`  
**Class**: `src.encoders.seg_encoder.SegmentationEncoder`

Key behaviours:

- **Locked model (b5 only)**: SegFormer-b5 was chosen over b0–b4 for best boundary precision on driving-scene classes (pedestrians, vehicles, traffic lights). The model is frozen (`requires_grad=False`). b0 gives worse boundary accuracy and must not be used — see the bug note in 7.
- **Manual ImageNet normalisation**: The encoder applies ImageNet mean/std normalisation manually in `_predict_ids()` rather than using `SegformerImageProcessor`. This is intentional: the Hugging Face image processor resizes internally to its own resolution, breaking compatibility with our fixed preprocessing chain.
- **19 Cityscapes classes**: IDs 0–18. Out-of-range values are impossible by construction — the ID is an argmax over exactly 19 class scores, so every pixel always gets one of the 19 real classes (there is no "unknown" class; see 0.8 for what happens on out-of-domain images).
- **Output of `label_ids()`**: raw class IDs `[B, H, W]` int64 — used ONLY by the offline calc script.
- **Output of `forward()`**: colour map `[B, 3, H, W]` float `[0, 1]` — used at live inference.
- **Parity guarantee**: both `label_ids()` and `forward()` share the same `_predict_ids()` internals. Only the final step differs (IDs vs colour lookup). Running `label_ids()` then `seg_colorize_ids()` on the saved PNG produces bit-identical output to calling `forward()` live.

### 4.1 Q&A — Can MiDaS be reused for segmentation instead of SegFormer?

**Q: We already have a MiDaS encoder for depth. Can we swap it into the segmentation pipeline instead of building/using SegFormer? Would it work?**

**A: No — not "works worse," it cannot work at all. This is an architecture mismatch, not a quality tradeoff.**

MiDaS is a **regression** model: its DPT decoder predicts one continuous depth value per pixel ("how far away is this point"), which `src/annotators/midas.py` min-max normalises to `[0,1]` and replicates across 3 channels. There is no class information anywhere in its weights or output — it was never trained to distinguish "this pixel is a pedestrian" from "this pixel is a building," only "near" from "far."

Segmentation needs a **classifier**: a per-pixel probability distribution over the 19 Cityscapes classes, with the highest-probability class picked as the label. SegFormer's decode head does exactly this; MiDaS's does not and cannot, regardless of which checkpoint is loaded.

Concretely, if MiDaS were forced into the seg encoder slot:
- Output would still be `[B, 1, H, W]` continuous depth in `[0,1]`, replicated to 3 channels — not class IDs.
- Colourising that with `SEG_CITYSCAPES_PALETTE` would produce "near = shade A, far = shade B" gradients, not road/car/person/sky regions.
- The LoRA mapper would learn a depth-shaped conditioning signal mislabelled as segmentation — zero semantic content, not noisy semantic content.

This is why `SegmentationEncoder` (`src/encoders/seg_encoder.py`) had to be built as a brand-new class rather than just pointing the existing `midas` encoder slot at a different model file — see 4 above and the encoder-slot-contract comment at the top of that file for how it satisfies the same input/output shape contract while doing fundamentally different (classification, not regression) work internally.

---

### 4.2 Q&A — Do we use `SegformerFeatureExtractor`? Did we write our own?

**Q: The HuggingFace example uses two objects — `SegformerFeatureExtractor` and `SegformerForSemanticSegmentation`. Do we use both?**

**A: We use `SegformerForSemanticSegmentation` (the model). We do NOT use `SegformerFeatureExtractor`. We wrote our own preprocessing that replaces it — three lines instead of one function call, with two deliberate differences.**

#### What the HuggingFace example does

```python
# Step 1 — feature extractor = preprocessing wrapper
feature_extractor = SegformerFeatureExtractor.from_pretrained("nvidia/segformer-b5-...")
# Internally: resize to 1024x1024, convert [0,255] -> [0,1], apply ImageNet mean/std

# Step 2 — the actual model
model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b5-...")
# Takes pixel_values -> logits [B, 19, H/4, W/4]
```

#### What our encoder does (from `src/encoders/seg_encoder.py:213-298`)

```python
# __init__: we load the MODEL only — no SegformerFeatureExtractor
from transformers import SegformerForSemanticSegmentation   # imported
# SegformerFeatureExtractor / SegformerImageProcessor       # NOT imported

self.seg_model = SegformerForSemanticSegmentation.from_pretrained(
    model, local_files_only=local_files_only
)

# _predict_ids(): we do the feature extractor's job manually
x = (imgs + 1.0) / 2.0                               # our [-1,1] -> [0,1]
x = F.interpolate(x, size=(512, 512), mode="bilinear")  # resize to OUR size, not 1024
x = (x - self._seg_mean) / self._seg_std             # ImageNet normalize (buffers, on GPU)
logits = self.seg_model(pixel_values=x).logits        # same as the HF example from here
logits = F.interpolate(logits, size=(512, 512))       # upsample logits
ids    = logits.argmax(dim=1)                         # class IDs [B, 512, 512]
```

So yes — `_predict_ids()` IS our own implementation of `SegformerFeatureExtractor`. It does the same three operations (range convert, resize, normalize) with two intentional differences:

| | `SegformerFeatureExtractor` | Our `_predict_ids()` |
|---|---|---|
| Input range | `[0, 255]` uint8 or `[0, 1]` float | `[-1, 1]` (our pipeline's training format) |
| Resize target | `1024 × 1024` (checkpoint's training size) | `512 × 512` (our pipeline's size) |
| Normalization | ImageNet mean/std | Same values, as GPU buffers |
| Output | `{"pixel_values": tensor}` dict | tensor directly |
| Numerical match | — | **2.4e-7** absolute diff (verified, documented in file header) |

#### Why resize to 512 instead of 1024

SegFormer's encoder uses overlapping patch embeddings with a /4 stride — it can accept any spatial size that is a multiple of 4. 512 is valid. The accuracy drop vs 1024 is small for driving scenes; the benefit is that Stage C (`seg_map_calculations.py`) and live inference use the **same code path at the same resolution**, so the training maps and inference maps are pixel-identical rather than differing by an extra resize step.

---

### 4.3 Proof — MiDaS encoder also skips its feature extractor (verified against original repo)

**Verified by fetching `https://github.com/CompVis/LoRAdapter` directly (2026-07-01).**

Three facts confirmed from the original CompVis/LoRAdapter repository and paper:

1. **`DPTImageProcessor` is commented out in the original repo's `midas.py`** — this is not something we added. The skip-processor pattern was already present in the code as shipped by the original authors.
2. **`src/encoders/seg_encoder.py` does not exist in the original repo (HTTP 404)** — segmentation conditioning is entirely our custom addition. The original repo has no SegFormer encoder, no Cityscapes pipeline, nothing seg-related.
3. **The paper (CTRLorALTer, arXiv 2405.07913) only implements depth and style conditioning** — segmentation is not mentioned in the abstract, contributions, experiments, or demos. The project page confirms only depth-based structure conditioning and style conditioning are demonstrated.

#### Code evidence from the original repo (`src/annotators/midas.py`)

```python
# Lines 6-8: DPTImageProcessor IS imported at the top...
from transformers import (
    DPTImageProcessor,         # imported
    DPTForDepthEstimation,
)

# Line 28 in __init__: processor instantiation is COMMENTED OUT
#   self.feature_extractor = DPTImageProcessor.from_pretrained(...)

# Lines 44-48 in forward: processor call is COMMENTED OUT
#   depth_dict = self.feature_extractor(imgs, do_rescale=False, return_tensors="pt")
#   for k, v in depth_dict.items():
#       if isinstance(v, torch.Tensor):
#           depth_dict[k] = v.to(device=imgs.device)

# Instead — manual preprocessing, direct model call:
imgs = (imgs + 1.0) / 2.0
imgs = better_resize(imgs, self.model_size)   # 384
depth_map = self.depth_estimator(pixel_values=imgs).predicted_depth
```

This is the **original CompVis code as published**. The authors chose to skip `DPTImageProcessor` and do manual preprocessing. Our `SegmentationEncoder` mirrors this exact decision for the same reason.

#### Side by side — both encoders skip their processor

| | `DepthEstimator` (original repo, midas.py) | `SegmentationEncoder` (our addition, seg_encoder.py) |
|---|---|---|
| Source | Original CompVis/LoRAdapter | Custom — does not exist in original repo |
| Range convert | `(imgs + 1.0) / 2.0` | `(imgs + 1.0) / 2.0` (identical) |
| Resize | `better_resize(imgs, 384)` | `F.interpolate(imgs, 512)` |
| Normalize | none (DPT handles it internally) | ImageNet mean/std (SegFormer requires it) |
| Feature extractor | `DPTImageProcessor` — imported, **commented out** | `SegformerImageProcessor` — not imported at all |
| Model call | `.predicted_depth` | `.logits` |
| Exists in paper | Yes — depth conditioning is the paper's core | No — segmentation is our custom extension |

#### Why we followed this pattern

When we built `SegmentationEncoder`, the MiDaS encoder in the repo was already doing manual preprocessing with the processor commented out. Matching that pattern ensures:
- Both encoders accept the same `[-1, 1]` input contract
- Both are audited the same way when the preprocessing changes
- No special cases: "encoder 1 uses the HF processor, encoder 2 doesn't"

---

## 5 · Key Design Concepts

### 5.1 Fixed Colour Palette — Why Not Per-Image Normalisation

**The critical difference from depth**: depth uses per-image min-max normalisation because the DPT output is unbounded and varies per scene. Segmentation must NOT do this.

Seg maps are discrete class assignments. Applying per-image normalisation would:
- Make the colour of class 0 (road) depend on which other classes appear in the image.
- The mapper network would see different colours for the same class in different images.
- Training would fail to learn a stable class-colour-to-latent mapping.

**Solution**: a fixed 19-entry colour palette (`SEG_CITYSCAPES_PALETTE` in `src/encoders/seg_encoder.py`) maps each class ID to a fixed RGB colour. This palette is the **single source of truth (SSOT)** — used identically in `SegmentationEncoder.forward()` (live inference) and `SegJsonDataset._load_seg_colormap()` (training data loading).

### 5.2 `SEG_CITYSCAPES_PALETTE` — The SSOT

Defined in `src/encoders/seg_encoder.py` as a `list[tuple]` with 19 entries. Helper functions:

```python
seg_palette_tensor(palette)     # --> [19, 3] float tensor in [0, 1]
seg_colorize_ids(ids, palette)  # [B,H,W] long --> [B,3,H,W] float [0,1]
```

Both the offline calc script (via `SegJsonDataset`) and the live encoder (`SegmentationEncoder.forward()`) call `seg_colorize_ids()` with this same palette. The palette is NOT stored in a JSON or YAML file — code is the SSOT.

### 5.3 Image Discovery and Path Control — how the pipeline finds your images

Same mechanics as DEPTH.md 5.2 — read that for the complete explanation. This section states the seg-specific values (sibling folder `_seg_map`, raw-ID PNGs).

#### The JSONL file types and their key names

```
data/train.jsonl                          ← source split
  {"target": "data/raw/000417/raw_image.jpg", "prompt": "..."}
   ▲ key name set by --image_path (default "source", commonly "target")

data/seg_training/train.jsonl             ← seg training manifest (output of Stage C)
  {"raw_image_path": ".../custome_dataset/000417/raw_image.jpg",
   "seg_path":       ".../custome_dataset_seg_map/000417/000417_seg_map.png",
   "prompt": "..."}
```

The seg-map is always `.png` (lossless, preserves integer class-ID values). The seg map now lives in a **sibling folder** named `_seg_map`, mirroring the dataset structure (see below).

#### `--image_path` — which key holds the image path

```bash
python seg_map_calculations.py --data_dir data/ --image_path target
```

Pass whatever key your JSONL uses. Default is `"source"`; your dataset uses `"target"`.

#### Where the seg maps are saved — a SIBLING folder, derived automatically (2026-07-07)

Exactly like depth (DEPTH.md 5.2). `--data_dir` looks at the image paths in your JSONLs, finds the folder common to all of them (the dataset root, e.g. `.../custome_dataset`), and saves each seg map into a **sibling folder** next to it — `.../custome_dataset_seg_map/` — mirroring the structure, named after the image's **folder**:

```
.../custome_dataset/000417/raw_image.jpg  →  .../custome_dataset_seg_map/000417/000417_seg_map.png
```

This is IDENTICAL to `--dataset_dir` scan mode (5.3b) — the two were unified on 2026-07-07 and proven to produce byte-identical manifest entries. The old design saved to `data/raw_seg/` named after the image filename, which (every image being `raw_image.jpg`) overwrote all maps into one file. Two guards now prevent that: folder-based naming, and a **collision guard** that aborts before any GPU work if two images would share a PNG. `--image_root` still exists but is only for resolving **relative** JSONL paths; your paths are absolute, so it is not needed.

#### The full seg discovery → output flow (VERIFIED by execution)

```
data/train.jsonl                               ← --data_dir
  {"target": ".../custome_dataset/000417/raw_image.jpg", "prompt": "..."}
       │
       │ --image_path target
       ▼
  _get_image_path(entry, "target")   → ".../custome_dataset/000417/raw_image.jpg"  (absolute)
       ▼
  dataset root = folder common to ALL images  →  .../custome_dataset
  sibling      = dataset root + "_seg_map"     →  .../custome_dataset_seg_map
       ▼
  SegmentationEncoder.label_ids(image)   → class-ID map [H, W] integer
       ▼
  saved AS RAW IDs — 8-bit grayscale PNG, values 0..18 (colour is applied
       │  later, at training-data LOAD time, by seg_colorize_ids — 0.3)
       │  → .../custome_dataset_seg_map/000417/000417_seg_map.png
       ▼
  data/seg_training/train.jsonl  (output — VERIFIED mapping)
  {
    "raw_image_path": ".../custome_dataset/000417/raw_image.jpg",              ← absolute, unchanged
    "seg_path":       ".../custome_dataset_seg_map/000417/000417_seg_map.png",
    "prompt":         "..."                                                    ← verbatim
  }
```

#### Quick reference

| Situation | Command |
|---|---|
| Your dataset (key = `"target"`, absolute paths) | `python seg_map_calculations.py --data_dir data/ --image_path target` |
| Same, but scan the folder for images | `... --dataset_dir /path/to/custome_dataset --data_dir data/ --image_path target` |
| JSONL key = `"source"` (default) | `python seg_map_calculations.py --data_dir data/` |
| Relative image paths in JSONL | add `--image_root /path/to/common/root` |
| Quick smoke-test | add `--dry_run_n 5` |

`--data_dir` and `--dataset_dir` both save to the sibling folder and both refuse to overwrite.

### 5.3b Dataset-scan mode (`--dataset_dir`) — save maps in a SIBLING, mirrored tree [VERIFIED]

Same saving behavior as `--data_dir` above (sibling `_seg_map` folder, mirrored, folder-named maps), with ONE difference: instead of trusting the JSONL paths to *find* images, the script **scans `--dataset_dir` recursively** for `raw_image.jpg`. Use it when you want disk to decide which images exist; use plain `--data_dir` when the JSONLs already list them. Either way the maps land in the same place, and `--data_dir` is still required (it supplies each image's prompt + split).

**Command:**
```bash
python seg_map_calculations.py \
  --dataset_dir /path/to/custome_dataset \
  --data_dir data/ \
  --image_path target
```

**Where the sibling folder is created:** automatically derived from `--dataset_dir`.
```
--dataset_dir  = /path/to/custome_dataset
sibling output = /path/to/custome_dataset_seg_map     (same parent, name + "_seg_map")
```

**What it produces (VERIFIED on a 914-folder dataset):**
```
/path/custome_dataset/000417/raw_image.jpg          ← input (found by scan, untouched)

/path/custome_dataset_seg_map/000417/000417_seg_map.png
                                                      ← NEW sibling tree, mirrors internal structure

data/seg_training/train.jsonl   (+ val, test)       ← rebuilt from the originals:
  {
    "raw_image_path": "/path/custome_dataset/000417/raw_image.jpg",           ← absolute, unchanged
    "seg_path":       "/path/custome_dataset_seg_map/000417/000417_seg_map.png",
    "prompt":         "..."                                                    ← from the split JSONL, verbatim
  }
```

**Arguments** are identical to depth's scan mode: `--dataset_dir` (scan root), `--data_dir` (split + prompt source, required), `--image_path` (JSONL key), `--image_name` (default `raw_image.jpg`).

**Verified guarantees (execution + two deliberate negative tests):**
- The source dataset folder is never modified — verified: `custome_dataset/000417/` contains only `raw_image.jpg` after a scan run, both depth and seg maps land in their own sibling trees.
- Running seg scan and depth scan back-to-back on the same dataset **does not interfere** — each writes to its own sibling folder (`custome_dataset_depth_map/` vs `custome_dataset_seg_map/`), and each scan still only picks up `raw_image.jpg`, ignoring the other pipeline's sibling folder entirely (verified: 6 seg maps from 6 folders that already had depth maps, not 12).
- The verifier (`_verify_scan_seg_training_jsonl`) was fed known-bad cases (map in-folder instead of sibling; wrong mirrored leaf-folder name) and correctly FAILed both while passing the valid entry.

**Dry run first:**
```bash
python seg_map_calculations.py --dataset_dir /path/custome_dataset --data_dir data/ --image_path target --dry_run_n 6
```

### 5.3c GPU resolution — never silently falls back to CPU [VERIFIED]

Identical mechanism to DEPTH.md 5.2c/5.2d (same `src/utils.py::resolve_device` / `auto_batch_size` functions, shared by both pipelines). `seg_map_calculations.py --device` now defaults to auto-detect-and-refuse (never silent CPU); `--batch_size` defaults to VRAM-based auto-scaling. See DEPTH.md 5.2c for the full verification table (4 tests: real GPU / no-GPU-auto / no-GPU-explicit-cpu / no-GPU-explicit-cuda, all PASS via a monkeypatched `torch.cuda.is_available`).

### 5.4 NEAREST Interpolation for Class IDs

When `SegJsonDataset` loads a saved seg PNG and resizes it, it uses `NEAREST` interpolation, not bilinear. This is critical:

- Bilinear interpolation would average adjacent class IDs (e.g., class 3 and class 7 blended → float 5.0, which rounds to class 5, a third wrong class).
- NEAREST interpolation selects the nearest pixel's class ID without mixing.

The resize happens in `_load_seg_colormap()` before `seg_colorize_ids()` is called.

### 5.5 `build_seg_square_preprocess()` — Single Source of Truth

The depth pipeline triplicates its preprocessing across three files. The seg pipeline fixes this with a single factory function:

```python
# src/data/transforms.py
def build_seg_square_preprocess(size):   # letterbox built in — stretch removed 2026-07-06
    ...
```

This function is imported by both:
- `seg_map_calculations.py` (Stage C offline)
- `seg_inference.py` (Stage D live inference)

Parity is guaranteed by construction — there is only one definition.

### 5.6 skip_encode — Same Pattern as Depth

```python
model.forward_easy(..., skip_encode=True)   # training:  uses pre-saved colour PNG
model.sample(...)                            # inference: calls SegFormer live
```

During training `batch["seg"]` contains the colour map loaded from the saved PNG via `SegJsonDataset`. `skip_encode=True` bypasses the SegFormer entirely. During inference, `SegmentationEncoder.forward()` runs live inside `model.sample()`.

### 5.7 Checkpoint Grid, val_steps/ckpt_steps, test.json

Identical to the depth pipeline — see DEPTH.md 5.4–5.6. The seg trainer (`seg_training.py`) is a direct mirror of `depth_training.py` with `batch["depth"]` replaced by `batch["seg"]` and depth-specific helpers renamed to their `_seg_*` equivalents.

### 5.7b Training Timing, Checkpoint Resume, and `training_params.txt` — see DEPTH.md 5.8–5.10

Same execution-over-assertion caveats, same self-measurement recipe (swap `depth_training.py` for `seg_training.py`), same verified checkpoint-resume limitation (weights reload correctly; LR schedule/global_step restart from 0), same `training_params.txt` snapshot written to `outputs/train/seg/runs/.../training_params.txt`. One seg-specific fact, measured not assumed: peak VRAM at `batch_size=4` was **identical to depth's measured value**, confirming `skip_encode=True` genuinely keeps the (much larger) SegFormer-b5 encoder out of the training forward pass; it isn't just architecturally true, it was checked with a real run.

### 5.7c 2026-07-05 fixes — see DEPTH.md 5.12–5.14 (applies to seg identically)

Three updates shared with depth, executed and verified on the seg side too:
1. **Manifest collision fixed + maps regenerated** (DEPTH.md 5.12): the seg manifests had the same all-rows-point-to-one-PNG bug; 913/913 seg maps were regenerated with the fixed flat-fill SquarePad, manifests rebuilt (639/137/137, verifier PASS), regenerated PNG IDs confirmed within 0..18 on real files. Since 2026-07-07 both `--data_dir` and `--dataset_dir` save collision-free maps to the sibling `_seg_map` folder — either command is safe for this dataset.
2. **`val/psnr_fixed` / `val/ssim_fixed` + `log_every_steps`** (DEPTH.md 5.13): identical implementation in `seg_training.py` (same `compute_psnr_ssim`, same fixed-scene protocol) — this is what makes the final depth-vs-seg comparison objective. Dead YAML keys (`use_empty_prompt_eval`, `n_samples`, `save_grid`, `log_cond`) removed from the seg configs too.
3. **Batch-shape kernel jitter** (DEPTH.md 5.14): seg's measured form is 2–5 argmax flips per 262k pixels between saved PNGs and live single-image output (boundary ties). Acceptance: ID-mismatch fraction ≤ 1e-4; palette-colourisation vs `forward()` must stay exactly 0.0 (it does — shared `_predict_ids` + shared palette).
4. **resize_mode: FINAL — letterbox only, stretch REMOVED** (2026-07-06, DEPTH.md 5.14a): the user evaluated stretch with real encoder previews (`outputs/viz/resize_mode_preview.png`) and rejected it (aspect distortion shifted seg classes — sky read as "building"). The former seg-side switch (`--resize_mode` flag, `inference.resize_mode` key, `build_seg_square_preprocess`'s `resize_mode` parameter) was deleted from the code so train and inference can never disagree by accident.

### 5.7d Best-model tracking + early stopping — see DEPTH.md 5.5 (applies to seg identically)

`seg_training.py` mirrors depth's `do_validation` exactly: `do_segmentation_validation` records **when** the best model was found (`best_epoch`/`best_step`/`best_epoch_frac`) and writes the same enriched `best_model/info.txt` (epoch, epoch_frac, global_step, val/loss, timestamp + appended `val/psnr_fixed`/`val/ssim_fixed`). Each validation logs the same `[seg val] … | best: epochX stepY | N epoch(s) since improvement` plateau line and a `*** NEW BEST (seg) ***` line when best_model/ is replaced.

`configs/train_seg.yaml` has the same `early_stop_patience: 3` key (`0` = off): at each epoch end, if val/loss produces no new best for that many full epochs, seg training stops itself (best_model/ already saved). The interrupted epoch's final weights + grid are still captured on the way out.

### 5.7e `val/miou_fixed` — segmentation controllability metric (per-model)

**What it measures.** psnr/ssim compare a generation to the one real source image (they punish legitimate creative variation). mIoU instead measures *controllability*: **did the generated image keep the class layout it was told to follow?** For every scored image it compares two class-ID maps and averages per-class IoU (`IoU(c) = |pred==c ∧ target==c| / |pred==c ∨ target==c|`, mean over classes present in either map). One image → one 0–1 number; the **mean over images is the model's controllability score** — the single number you rank seg models by. Implemented as `compute_miou` in `src/utils.py` (pure torch, no new dependency).

**The two maps compared (this is the key design):**
- **TARGET** = the seg map that *conditioned* the model, recovered by inverting the conditioning colour map back to IDs via `seg_ids_from_colormap` in `src/encoders/seg_encoder.py` (nearest palette colour). Because the map was colourised *from* `SEG_CITYSCAPES_PALETTE`, this recovery is **exact** — verified roundtrip `seg_colorize_ids → seg_ids_from_colormap` returns the original IDs. Zero extra model calls.
- **PREDICTION** = SegFormer re-run on the *generated* image (`model.encoders[0].label_ids`). One extra SegFormer forward per scored image.

Absent classes (in neither map) are excluded from the mean — the standard mIoU convention — so they can't deflate the score.

**Where it appears (rides the existing metric plumbing, no new wiring):**
- **Training** — `_save_checkpoint_segmentation_images` scores the **fixed** scenes (fixed seed → comparable checkpoint-to-checkpoint) and adds `val/miou_fixed` to the returned `metrics` dict. It then flows automatically into TensorBoard, the `[seg metric]` log line, `best_model/info.txt`, and the checkpoint trend — same path as psnr/ssim.
- **Inference** — `seg_inference.py` scores every entry, prints per-image mIoU, and writes `metrics.txt` (`mean mIoU (n=…)` + per-image lines) next to the results. That mean is the model's final controllability score.

**Reading it:** higher = better structural adherence; watch `val/miou_fixed` *rise* over training and plateau (a second opinion to `val/loss`, and the signal for when to stop). Note it depends on SegFormer as the judge — score every generation with the *same* SegFormer that conditioned it, or the comparison isn't apples-to-apples.

### 5.7f How much training does seg need?

**Honest status first (do not skip this):** a full seg training run has **never been completed** (see 12 item 5). So there is **no execution-verified answer** to "how many epochs/steps does seg need" — any number below is a *configured budget*, not a proven requirement. The correct number can only come from watching a real run's curves, and the pipeline is now set up to tell you that number automatically (see below).

**The configured budget (starting point, not a target).** `configs/experiment/train_seg.yaml` is sized for the real dataset (**59,766 train / 3,314 val / 3,327 test**):

| Quantity | Value | Where it comes from |
|---|---|---|
| effective batch | **16** | `batch_size 4 × gradient_accumulation_steps 4` |
| optimizer steps / epoch | **3,736** | `ceil(59,766 / 16)` |
| `epochs` | **5** | config default |
| total optimizer steps | **≈ 18,680** | `3,736 × 5` |
| LR schedule | `cosine`, warmup 500 | decays `1e-4 → ~0` across the full 18,680 steps |

5 epochs is chosen so the cosine schedule has room to decay to ~0 — it's an **upper budget you let run**, not a claim that 5 epochs is required. It may plateau earlier (then early-stop ends it) or still be improving at epoch 5 (then raise `epochs`).

**How the pipeline actually answers "how much" for you — empirically, no guessing.** Three mechanisms (all now in place) turn "guess a number" into "let it run and read off the answer":

1. **`val/loss`** (down) and **`val/miou_fixed`** (up, 5.7e) are logged every `val_steps`/checkpoint. When *both* stop improving, the model has learned what this data can teach it — that's "enough."
2. **Best-model tracking** — `best_model/` + `best_model/info.txt` record the exact **epoch/step/timestamp** of the best val/loss (5.7d). After the run, `info.txt` *is* the answer to "how much training gave the best model."
3. **Early stopping** — `early_stop_patience: 3` (config): if val/loss produces no new best for 3 full epochs, training **stops itself**. So a detached run finds its own stopping point; you don't have to predict it.

**Practical recipe.** Launch with the 5-epoch budget (`python seg_training.py experiment=train_seg`), let early-stop + best-model do the work, then read `best_model/info.txt` for the epoch that won. If early-stop never fires and both curves are still climbing at epoch 5, the data wants more — bump `epochs` and rerun. If it plateaus at (say) epoch 2, the honest answer for *this dataset* is "~2 epochs," and you now have it **from a real run**, not from a number written in a config.

**Timing (how long in wall-clock)** is hardware-dependent and only measured on a reference 12 GB GPU, never the real Linux/A2000 target — run the self-measurement command in 5.7b once on the real machine to get seconds/step, then multiply by 18,680 for the worst-case (no early-stop) wall-clock.

### 5.8 SegFormer-b5 vs b0 — Why b0 is Wrong

SegFormer-b0 is a small model designed for speed, not accuracy. On Cityscapes driving scenes:
- b0 misclassifies thin structures (pedestrian poles, traffic lights) that matter for structural conditioning.
- b5 achieves significantly higher mIoU on Cityscapes, especially on boundary-sensitive classes.

Two old experiment configs (`train_seg_12gb.yaml`, `train_seg_cluster.yaml`) incorrectly referenced b0 and had wrong JSON paths (missing `data/` prefix). **Deleted on 2026-06-30** — confirmed no other file referenced them (grepped the repo), confirmed `train_seg.yaml` fully supersedes them. Use `configs/experiment/train_seg.yaml` only.

---

## 6 · YAML Parameters Explained

### `configs/experiment/train_seg.yaml` (use this for all runs)

```yaml
size: 512
learning_rate: 1.0e-4             # IDENTICAL to depth — required for a valid comparison
lr_warmup_steps: 500
lr_scheduler: cosine
epochs: 5
val_steps: 500
ckpt_steps: 1000
val_batches: 64
n_grid_images: 10
grid_include_empty_prompt: false  # OFF, same as depth (true adds a 4th RAW SEG GEN panel)
bf16: true
gradient_checkpointing: true
gradient_accumulation_steps: 4
tag: seg
local_files_only: true
ignore_check: true
data:
  json_file:     data/seg_training/train.jsonl
  val_json_file: data/seg_training/val.jsonl   # NEVER test.json here
lora:
  struct:
    encoder:
      model: nvidia/segformer-b5-finetuned-cityscapes-1024-1024   # b5, not b0
```

### `configs/lora/encoder/segformer.yaml`

```yaml
_target_: src.encoders.seg_encoder.SegmentationEncoder
model: nvidia/segformer-b5-finetuned-cityscapes-1024-1024
size: ${size}
local_files_only: ${local_files_only}
```

### Dead keys — REMOVED (2026-07-05, same as depth pipeline)

`use_empty_prompt_eval`, `n_samples`, `save_grid`, `log_cond` were deleted from `configs/train_seg.yaml` + `configs/experiment/train_seg.yaml` (they were read by nothing in the seg training path). `log_every_steps: 50` added instead (see DEPTH.md 5.13).

### Broken configs (deleted)

`train_seg_12gb.yaml` and `train_seg_cluster.yaml` (b0 model, wrong JSON paths) were deleted on 2026-06-30 — `train_seg.yaml` is now the only seg experiment config and supersedes both use cases (12GB single-GPU and multi-GPU launch instructions are both documented inline in `train_seg.yaml`'s footer).

---

## 7 · Run Commands & Success Criteria

### Prerequisites

```
data/
  raw/             # original images
  seg_training/    # does NOT exist yet; Stage C creates it
checkpoints/local_models/
  stable-diffusion-v1-5/
  segformer-b5-cityscapes/   # required for local_files_only=true; see note below
```

**Model download note**: The b5 model is NOT bundled in the repo. On first run with `local_files_only: false` it auto-downloads from Hugging Face to the HF cache (`~/.cache/huggingface/`). To make it available offline, copy the HF cache to `checkpoints/local_models/segformer-b5-cityscapes/` and set `local_files_only: true`.

Activate the conda environment before any Python command:
```bash
conda activate loradapter
```

### Stage C — Precompute Segmentation Maps (run once)

Dry run on 1 image first:
```powershell
python seg_map_calculations.py --data_dir data/ --dry_run_n 1 --local_files_only False
```
Success: no errors; seg PNG written to the sibling `<dataset>_seg_map/` folder; class IDs in `[0, 18]`.

Full run (after b5 model is cached):
```powershell
python seg_map_calculations.py --data_dir data/ --image_path target
```
Success:
- `<dataset>_seg_map/` populated (one 8-bit PNG per image, mirrored structure)
- `data/seg_training/train.jsonl`, `val.jsonl`, `test.jsonl` written
- Verification output shows `0 failures`
- Dataset sizes: 639 train / 137 val / 137 test

### Stage D — Training

```powershell
python seg_training.py experiment=train_seg
```

Expected startup log:
```
[model] base = .../stable-diffusion-v1-5
Number params Mapper Network(s) 1,245,072
Number params all LoRAs(s) 29,999,104
Grid: 10 val scenes = 5 fixed (random per run) [...] + 5 re-randomized
start training
```

Per-checkpoint log pattern (every 1000 steps):
```
[val] step1000: val/loss = 0.XXXXXX
[grid] checkpoint-epoch1/step1000: 10 scene images -> .../checkpoint-epoch1/step1000
```

TensorBoard: `tensorboard --logdir outputs/train/seg/runs/`

Expected tags (same structure as depth, by design):
- Scalars: `train/loss`, `train/lr`, `train/grad_norm`, `train/epoch`, `val/loss`, `val/psnr_fixed`, `val/ssim_fixed`
- Images: `val/sample_00` … `val/sample_09`
- Tensors: `val/prompts/text_summary`

The tfevents hostname field will be `seg` (from `cfg.tag = "seg"`), not the machine hostname.

### Stage D — Inference

```bash
# single image + prompt:
python seg_inference.py \
  ckpt_path=outputs/train/seg/runs/YYYY-MM-DD/HH-MM-SS/best_model \
  "inference.images=[/path/to/raw_image.jpg]" \
  "inference.prompts=['your prompt here']"

# or a whole manifest:
python seg_inference.py \
  ckpt_path=outputs/train/seg/runs/YYYY-MM-DD/HH-MM-SS/best_model \
  inference.json_file=data/seg_training/test.jsonl
```

Success: 4-panel JPG grids written to `outputs/inference/seg/results/`.

---

## 8 · Known Limitations

1. **b5 model now in `checkpoints/local_models/segformer-b5-cityscapes/`**: Copied from HF cache on 2026-06-30. `local_files_only: true` is safe. If re-cloning to a new machine, copy the three files (`config.json`, `preprocessor_config.json`, `pytorch_model.bin`) from the HF cache (`~/.cache/huggingface/hub/models--nvidia--segformer-b5-finetuned-cityscapes-1024-1024/snapshots/<latest>/`) to `checkpoints/local_models/segformer-b5-cityscapes/`.

2. **`train_seg_12gb.yaml` and `train_seg_cluster.yaml` — DELETED** (2026-06-30, verified gone from `configs/experiment/`): they referenced b0 and had wrong JSON paths. The one and only seg experiment config is `configs/experiment/train_seg.yaml`.

3. **`max_train_steps` does not stop training**: Same limitation as the depth pipeline — see DEPTH.md 8, item 1.

4. **19-class Cityscapes only**: The palette and class count are hardcoded to Cityscapes. Adapting to a different segmentation taxonomy requires changing `SEG_CITYSCAPES_PALETTE` in `seg_encoder.py` (the SSOT) and rerunning Stage C.

5. **Seg training never verified by execution**: As of this guide's writing, a full seg training run has not been completed. Stage C (offline seg map generation) and Stage D (parity check) were both verified by execution. Stage D training itself — step-level logs, actual val/loss convergence curve, TensorBoard tags — remains UNVERIFIED. The code mirrors `train_depth.py` exactly (which was verified), so structural correctness is expected, but a real run is needed to confirm.

6. **NEAREST-resize is slow for large batches**: `SegJsonDataset` applies NEAREST resize per sample in the dataloader. For large datasets this can be a bottleneck. Pre-resizing the seg PNGs to 512x512 during Stage C (currently not done) would eliminate this.

7. **No test-time evaluation script**: Same limitation as the depth pipeline — see DEPTH.md 8, item 7.

---

## 9 · Audit Report — Segmentation Pipeline

**EXECUTION-OVER-ASSERTION**: items below are PASS only if the step was run with real data and output inspected. "Code reads correctly" is a separate claim.

**Audit date**: 2026-06-30  
**Machine**: Windows 11 (dev); final training will run on Ubuntu.

| Item | Description | Status | Evidence |
|------|-------------|--------|----------|
| S1 | Config files — correct experiment YAML | **FIXED** | `configs/experiment/train_seg.yaml` confirmed correct (b5 model, correct `data/seg_training/*.json` paths). The two stale b0 configs (`train_seg_12gb.yaml`, `train_seg_cluster.yaml`) were deleted on 2026-06-30 after grepping the repo to confirm nothing else referenced them. |
| S2 | b5 model availability | FIXED | Model was in HF cache (`~/.cache/huggingface/hub/models--nvidia--segformer-b5-finetuned-cityscapes-1024-1024/`). Copied to `checkpoints/local_models/segformer-b5-cityscapes/` on 2026-06-30. Dry run with `local_files_only=True` confirmed load succeeded (19 classes, 1172 weights loaded). |
| S3 | Stage C — offline seg map generation | **PASS** | `python seg_map_calculations.py --data_dir data/` completed with 0 errors. 913 PNG files written to `data/raw_seg/`. All 3 JSON splits verified by built-in checker: `train.json 639/639`, `val.json 137/137`, `test.json 137/137`. PNG inspection: shape `(512, 512)`, dtype `uint8`, values in `[0, 18]` — correct. |
| S4 | Stage D — training startup | PENDING FIRST RUN | `seg_training.py` mirrors `train_depth.py` exactly; training not yet executed. Next step: `python seg_training.py experiment=train_seg`. Success criterion: startup log shows `19 classes`, loss begins decreasing within first 100 steps. |
| S5 | Train/inference preprocessing parity | CODE READS CORRECT — NOT YET VERIFIED BY EXECUTION | `build_seg_square_preprocess()` SSOT confirmed imported by both `seg_map_calculations.py` and `seg_inference.py`. Parity is guaranteed by construction. Cannot be verified by execution until inference is run with a trained checkpoint. |
| S6 | Val loss + checkpoint grid | PENDING FIRST RUN | Blocked on S4 (training). Code mirrors train_depth.py's verified grid logic. |
| S7 | TensorBoard tags | PENDING FIRST RUN | Blocked on S4. Expected tags: `train/loss`, `train/lr`, `val/loss`, `val/sample_00`…`val/sample_09`. |
| S8 | Inference | PENDING FIRST RUN | `seg_inference.py` untested — no checkpoint available yet. Blocked on S4. |
| S9 | Known issues documented | DOCUMENTED | See 8 above. NEAREST-resize slow on large batches, no test-eval script. Stale configs (formerly S1) are now deleted, not just documented. |

### What is now unblocked

Stage C is verified. The single remaining blocker before training can start is **running `seg_training.py`** — there are no more missing models, no empty data directories, no broken JSON paths. All prerequisites are met.

```powershell
# Run on Ubuntu (final training machine):
python seg_training.py experiment=train_seg
```

Watch for these in the first 50 steps to confirm training is working:
- `[model] base = .../stable-diffusion-v1-5` in startup log
- `Number params Mapper Network(s) 1,245,072` (same as depth — encoder frozen)
- `val/loss` decreasing (not stuck at a constant)
- Checkpoint grid at step 1000 shows recognisable colour blobs per Cityscapes class

---

## 10 · Full Parameter Control

Everything you can tune, where it lives, and what changing it does. Mirrors DEPTH.md 9 with seg-specific values. Read 9 first for concepts, then use this section for seg-specific differences.

---

### 10.1 How Hydra overrides work

See DEPTH.md 9.1 — identical for seg. The key difference: the experiment config file is `configs/experiment/train_seg.yaml`. Resolved config is written to `outputs/train/seg/runs/YYYY-MM-DD/HH-MM-SS/.hydra/config.yaml` after each run.

```powershell
# Override any key without editing a file:
python seg_training.py experiment=train_seg epochs=3 val_steps=100 data.batch_size=2
```

---

### 10.2 Stage C — `seg_map_calculations.py` CLI flags

Run once before training to precompute segmentation-ID PNGs from raw images.

```powershell
python seg_map_calculations.py --data_dir data/ [flags]
```

| Flag | Default | What it does | When to change |
|------|---------|--------------|----------------|
| `--data_dir` | *(required)* | Folder containing `train.jsonl`, `val.jsonl`, `test.jsonl`. Non-default names like `my_train.jsonl` are found automatically if the stem contains "train"/"val"/"test". | Always set. |
| `--dry_run_n N` | off | Process only the first N images per split. Always run `--dry_run_n 2` first to check model loading and output format. | Use before every full run on a new machine. |
| `--size` | `512` | Square side for saved seg-ID PNGs. Must match `size` in training config. Changing this requires rerunning Stage C. | Keep 512 unless you change training resolution. |
| `--batch_size` | `4` | Images fed to SegFormer at once. | Lower if you get OOM. SegFormer-b5 is heavier than DPT so you may need `--batch_size 2`. |
| `--model` | `checkpoints/local_models/segformer-b5-cityscapes` | Local path to SegFormer-b5. **Locked — do not change to b0 or any other variant.** See 4.1 for why b5 is non-negotiable. | Only if you moved the model files. |
| `--local_files_only` | `True` | Offline-only loading from the local model path. | Keep `True` now that b5 is in `checkpoints/local_models/`. |
| `--device` | `cuda` if available | `cuda` or `cpu`. | `cpu` is very slow for SegFormer-b5 (~10x slower). |
| *(removed)* `--resize_mode` | — | This flag no longer exists (2026-07-06): letterbox squaring (flat local-mean fill, DEPTH.md 5.1) is built into `build_seg_square_preprocess()` after the stretch option was evaluated and rejected (DEPTH.md 5.14a). There is deliberately no knob to get preprocessing out of sync. | — |
| `--no_skip` | off | Recompute seg PNGs even if they already exist. | Add if you changed `--model` or `--size` and need to regenerate. |
| `--image_path` | `source` | JSONL key holding the image path. Your dataset uses `target`. | Always set `--image_path target`. |
| `--dataset_dir` | *(none)* | Optional: scan this folder for `raw_image.jpg` instead of trusting JSONL paths. Saves to the same sibling folder either way. | Add if you want disk (not the JSONL) to decide which images exist. |
| `--image_root` | *(none)* | Base prepended to **relative** JSONL image paths. Your paths are absolute, so it is not needed. | Only if your JSONL stores relative paths. |
| `--output_dir` | `<data_dir>/seg_training` | Where the output manifests are written. (The seg PNGs always go to the sibling `<dataset>_seg_map/` folder, derived automatically.) | Only to redirect manifest location. |

---

### 10.3 Stage D Training — `configs/experiment/train_seg.yaml`

Identical structure to depth (DEPTH.md 9.3). Differences are noted below; everything else is the same.

#### Resolution and hardware — identical to depth

Same keys (`size`, `bf16`, `gradient_checkpointing`, `gradient_accumulation_steps`, `data.batch_size`, `data.workers`) with the same defaults. See DEPTH.md 9.3.

#### Learning rate and schedule — identical to depth

Same keys and defaults. The learning rate is kept identical to depth so val/loss curves are directly comparable between the two pipelines.

#### When to save / validate — identical to depth

Same keys (`val_steps=500`, `ckpt_steps=1000`, `val_batches=64`). See DEPTH.md 9.3.

#### Checkpoint monitoring grid — one difference from depth

| Key | Default (seg) | Difference from depth |
|-----|--------------|----------------------|
| `n_grid_images` | `10` | Identical: 5 fixed + 5 fresh per checkpoint. Fixed scenes chosen once at startup with OS entropy; same scenes appear in every checkpoint grid. Files: `sample_00_fixed.jpg`…`sample_04_fixed.jpg`, `sample_05_new.jpg`…`sample_09_new.jpg`. |
| `grid_include_empty_prompt` | `false` | Depth default is also `false`. For seg the empty-prompt panel is called "RAW SEG GEN" (generation with empty text, pure seg conditioning). |

**Reading the seg checkpoint grid:** Each panel row is `ORIGINAL | SEG MAP | PREDICTED`. The SEG MAP column shows the 19-class Cityscapes colour palette — you should see distinct road (purple), sky (steel blue), vegetation (green), and car (deep blue) regions. If the SEG MAP looks like a uniform colour smear, the seg encoder is producing bad predictions; check the SegFormer model files.

#### Dataset and model paths — seg-specific

| Key | Default | What it does | When to change |
|-----|---------|--------------|----------------|
| `data.json_file` | `data/seg_training/train.jsonl` | Training manifest. Each line: `{raw_image_path, seg_path, prompt}`. | Change if your dataset is elsewhere. |
| `data.val_json_file` | `data/seg_training/val.jsonl` | Val manifest. **Never `test.jsonl` here.** | Only change to use a different val set. |
| `data.image_root` | `null` (= repo root) | Prepended to relative `raw_image_path` values. Set to `/mnt/dataset` when images are on a different drive. | **Required when training on Ubuntu with images at a different path.** Same rule as depth. |
| `seg_model_path` | `checkpoints/local_models/segformer-b5-cityscapes` | Local SegFormer-b5. Used at **inference only** (skip_encode=True means it's not loaded during training). | Only if you moved the model. |
| `lora.struct.ckpt_path` | `null` | Resume from checkpoint. | Same usage as depth. |

---

### 10.4 Stage D Inference — `configs/inference_seg.yaml`

```powershell
# From a JSONL manifest:
python seg_inference.py \
  ckpt_path=outputs/train/seg/runs/YYYY-MM-DD/HH-MM-SS/best_model \
  inference.json_file=data/seg_training/test.jsonl

# Single image:
python seg_inference.py \
  ckpt_path=outputs/train/seg/runs/YYYY-MM-DD/HH-MM-SS/best_model \
  "inference.images=[data/raw/000417/raw_image.jpg]" \
  "inference.prompts=['a driving scene at night in the rain']"
```

| Key | Default | What it does | When to change |
|-----|---------|--------------|----------------|
| `ckpt_path` | *(required)* | Path to a checkpoint folder. Usually `best_model/` or a specific `checkpoint-epoch1/step1000/`. | Always set. |
| `inference.json_file` | `null` | JSONL to run inference on. Entries need `seg_path` (required) + `raw_image_path` (optional, display only) + `prompt`. | Use `test.jsonl` for final evaluation. Never `val.jsonl` or `train.jsonl`. |
| `inference.seg_maps` | `[]` | Direct list of PRE-COMPUTED seg map paths. **REQUIRED** — inference never computes a map live. See §10.4b. | Quick single-map tests. |
| `inference.images` | `[]` | OPTIONAL, display-only raw photos, matched by index to `seg_maps`. No model ever runs on these. | Only if you want the ORIGINAL panel populated. |
| `inference.prompts` | `[]` | Prompts matching `inference.seg_maps`. | Required when using `inference.seg_maps`. |
| `inference.output_dir` | `outputs/inference/seg/results` | Where generated images are saved. Resolved from repo root. | Change per experiment. |
| `inference.save_generated_only` | `false` | `true` = save only the predicted image (no 4-panel grid). | Set `true` for clean batch evaluation. |
| *(removed)* `inference.resize_mode` | — | Key deleted 2026-07-06: letterbox is built into the shared preprocessing factory, so inference physically cannot use a different squaring than Stage C. | — |
| `inference.n_samples` | `1` | Images generated per input. | `2`–`4` for diversity. |
| `inference.num_inference_steps` | `50` | Diffusion denoising steps. | `20` preview, `50` quality, `80`+ max. |
| `inference.guidance_scale` | `7.5` | CFG scale. Higher = more prompt-driven. | `3`–`5` creative, `7.5` standard, `12`+ tight. |
| `inference.conditioning_kernel_size` | `0` | Box-blurs the seg colour map before the mapper, softening hard silhouette EDGES. `0` = off (no-op). | Odd int (`3`, `5`) if generated objects have unnaturally crisp/cut-out-looking boundaries. See §10.4a. |
| `inference.lora_scale_start` | `1.0` | Structure-conditioning strength for the early (layout) denoising steps. | Leave at `1.0` — full grip while layout is decided. |
| `inference.lora_scale_end` | `1.0` | Structure-conditioning strength for the late (detail) denoising steps. Equal to `lora_scale_start` = decay disabled. | Lower (e.g. `0.4`) to fix "fits the shape but doesn't know how it looks." See §10.4a. |
| `inference.lora_scale_decay_start_frac` | `0.3` | Fraction of steps held at `lora_scale_start` before decaying toward `lora_scale_end`. | Raise toward `1.0` for a later, more conservative handoff. |
| `local_files_only` | `true` | Offline mode. | Keep `true`. |

### 10.4a Fixing "fits the shape but doesn't know how it looks" [ADDED 2026-07-17]

**Root cause (traced in `src/lora.py` `NewStructLoRAConv.forward`):** the structure
conditioning is a per-pixel FiLM shift/scale applied through a **1×1 convolution**
(`self.beta`, `self.gamma`) — receptive field of exactly one pixel. A segmentation
map is a FLAT, constant colour inside any object (every "car" pixel is identical).
So the model receives the *exact same* instruction at every interior pixel: "car
here" — zero information about the object's actual appearance. This holds with
full force through EVERY denoising step, including the late steps where a
diffusion model normally decides fine detail/texture, starving the model of room
to use its own trained prior for what the object should look like. Confirmed via
a real generation comparison: a small/distant car rendered fine (its whole shape
fits inside the mapper's receptive field, richer context); a large/close-up truck
warped into a generic blob (deep interior pixels see nothing but flat colour in
every direction).

**Two independent inference-time knobs, no retraining required** (both in
`sample_easy`, `src/model.py`; wired through `seg_inference.py` above):

1. **`lora_scale_start`/`lora_scale_end`/`lora_scale_decay_start_frac`** — holds
   full conditioning strength for the early layout-deciding steps, then linearly
   decays it for the late detail-deciding steps (mirrors ControlNet's
   `control_guidance_start/end`). Implemented via a diffusers-native
   `callback_on_step_end` that mutates every struct-LoRA layer's `.lora_scale`
   live between steps (`ModelBase.make_lora_scale_callback`). The original scale
   is restored via `try/finally` after generation — required because these are
   the SAME module instances training and the monitoring-grid code (`sample_custom`)
   use; an unrestored decayed value would silently leak into the next call.
   **This is the primary lever** — it directly frees the late steps to paint
   realistic appearance instead of obeying a flat signal.

2. **`conditioning_kernel_size`** — box-blurs the seg colour map (`cond`, right
   after the encoder produces it — NOT the raw input photo) before the mapper.
   Verified by execution: with `kernel_size=0` output is byte-identical to no
   blur; with `kernel_size=5` a boundary pixel softens to an intermediate value
   while a DEEP interior pixel is mathematically unchanged (still exactly its
   original flat value). **This only fixes crisp/cut-out-looking edges — it
   cannot add information to a region's interior.** Pair it with the
   `lora_scale` decay above; it is not a substitute.

Both default to no-op values (`0`, and `lora_scale_start == lora_scale_end`), so
existing generations are byte-for-byte unchanged unless explicitly configured.
Architecture-level (lives in shared `model.py`), so it applies identically to
Grounded-SAM once its live encoder (Tier 2, see GROUNDED_SAM.md) exists.

### 10.4b Inference now ALWAYS uses a PROVIDED map — never computes one live [ADDED 2026-07-17]

**Change (user decision):** `seg_inference.py` no longer runs any segmentation
model live. Previously it ran SegFormer on the input photo TWICE — once via
`model.encoders[0](img_tensor)` to build the "SEG MAP" display panel, and again
inside `sample_easy` (which called `encoder(c)` unconditionally) to build the
actual conditioning signal. Both are gone. Now every entry supplies a
pre-computed `seg_path` (the exact same class-ID PNG format
`seg_map_calculations.py` saves and training already reads), which is loaded and
colourised directly — `raw_image_path`/`images` becomes OPTIONAL, used only to
populate the ORIGINAL display panel, never touched by any model.

**Mechanism:**
- `sample_easy` gained a `skip_encode: bool = False` parameter, mirroring the
  pattern `sample_custom` already had (`cond = c if skip_encode else
  encoder(c)`). `seg_inference.py` now calls `model.sample(..., cs=[seg_tensor],
  skip_encode=True)` for both the prompted and empty-prompt (RAW SEG GEN)
  generations.
- A new helper `_load_seg_map()` in `seg_inference.py` loads the raw class-ID PNG
  and colourises it, mirroring `src/data/local_seg.py`'s
  `_load_seg_colormap` EXACTLY (same NEAREST resize, same `seg_colorize_ids`
  call, same palette) — this is what guarantees a map loaded at inference
  produces the identical conditioning signal training saw for that file.
- The mIoU controllability metric (scores the GENERATED image's class layout
  against the requested map — a different use of the encoder than "computing
  the input map," since it runs AFTER generation) is now guarded by
  `encoder.live_available`, mirroring the guard already added to
  `seg_training.py`. It's skipped cleanly — not a crash — for encoders with no
  live path, e.g. Grounded-SAM's Tier-1 `GroundedSamEncoder`.

**Why this matters beyond SegFormer:** this makes inference identical for
SegFormer and Grounded-SAM. Grounded-SAM's Tier-1 encoder was never able to run
live at all — "always use a provided map" isn't a restriction added on top of a
working live path, it's the contract Grounded-SAM already required, now applied
uniformly. See GROUNDED_SAM.md §5.2 for what this newly unlocks.

**New/changed config keys:** `inference.seg_maps` (list mode, replaces the old
`inference.images` as the primary required input), `inference.images` (now
optional/display-only), manifest entries need `seg_path` (required) instead of
only `raw_image_path`. See the table above.

**Verified by execution (unit-level, not a live GPU run):** built a synthetic
class-ID PNG, ran `_load_seg_map` standalone — correct output shape/range
(`[1,3,size,size]` in `[0,1]`), and confirmed NEAREST resize introduces no
fabricated class ids (recovered ids are a strict subset of the original ids).
Full end-to-end generation was NOT run (no local SD weights/GPU in this
environment) — treat this as code-verified, pending a real smoke run on the
training machine.

---

### 10.5 Common scenarios — exact commands

#### Smoke test (verify pipeline works, ~5 minutes)
```powershell
python seg_map_calculations.py --data_dir data/ --image_path target --dry_run_n 3

python seg_training.py experiment=train_seg `
  epochs=1 val_steps=10 ckpt_steps=20 val_batches=4 n_grid_images=2 `
  "data.workers=0" ignore_check=true
```

#### Full training run (Ubuntu, 5 epochs)
```bash
python seg_training.py experiment=train_seg
```

#### Resume interrupted training
```bash
python seg_training.py experiment=train_seg \
  "lora.struct.ckpt_path=outputs/train/seg/runs/2026-07-01/00-47-30/checkpoint-epoch2/step4000"
```

#### Reduce memory (OOM on a 12 GB GPU)
```powershell
python seg_training.py experiment=train_seg data.batch_size=1 gradient_accumulation_steps=4
```

#### Images on a different drive (Ubuntu training with images at /mnt/data)
```bash
python seg_training.py experiment=train_seg data.image_root=/mnt/data
```

#### Watch 20 fixed scenes per checkpoint (strong convergence signal)
```powershell
python seg_training.py experiment=train_seg n_grid_images=40
# 20 fixed + 20 fresh — each checkpoint grid shows same 20 scenes for comparison
```

#### Add empty-prompt panel (see pure seg conditioning)
```powershell
python seg_training.py experiment=train_seg grid_include_empty_prompt=true
# 4th panel: "RAW SEG GEN" — no text, only the seg map drives generation
```

#### Evaluate on test split after training
```powershell
python seg_inference.py `
  ckpt_path=outputs/train/seg/runs/YYYY-MM-DD/HH-MM-SS/best_model `
  inference.json_file=data/seg_training/test.jsonl `
  inference.save_generated_only=true `
  inference.output_dir=outputs/inference/seg/test_eval
```

#### Generate comparison report (depth vs seg)
```powershell
python training_report.py
python training_report.py --markdown   # GitHub-flavoured Markdown
```

#### Why depth absorbs the letterbox band gracefully
```Depth is a continuous number per pixel (0.0…1.0, "how far"). When MiDaS sees the flat pad band, it outputs some smooth, unremarkable value — you saw this in the band check: the depth band is a soft gradient with no structure. A flat band in, a bland smooth region out. The conditioning tells the model something vague and weak about that zone — "roughly uniform distance, nothing detailed" — which is close to the truth ("this area is filler").

Why segmentation cannot mimic that
Segmentation is categorical: the encoder's final step is an argmax over exactly 19 Cityscapes classes. There is no "padding", "void", or "unknown" class in its vocabulary — every single pixel must be declared road, building, sky, person… one of the 19 real things. So when SegFormer sees our flat brown band, it's forced to pick its least-bad guess — you saw it label the band "building" in the examples. The encoder is architecturally incapable of saying "this is nothing."

That's the asymmetry: depth's band is a weak, honest-ish signal ("smooth region"), while seg's band is a confident false statement ("there is a building spanning the full width at the top of every image"). Categorical labels are inherently strong signals — there's no way to whisper in a language that only has 19 loud words.

Why this is contained, not broken
Two things keep it from being a real problem in practice:

The lie is perfectly consistent. The band gets the same deterministic label at training and at inference (same flat fill → same argmax), so the model just learns "band zone = render the band filler" as a stable rule. It never contaminates the real content region, and generated pad zones can be cropped off afterward using last_padding_fracs (stored exactly for this).
It's confined to a known, fixed region — top/bottom ~19% whose position is computable from the pad fractions, never overlapping actual scene content.
If it ever needs fixing, two options exist (neither implemented, both documented)
Mask the pad rows out of the training loss (the long-standing nice-to-have in references.md 5): the model is simply never graded on the band, so it never learns anything there. Cleanest fix, moderate effort.
Invent a 20th "padding" color: since the band's position is known exactly from last_padding_fracs, we could stamp a dedicated 20th palette color over the band in both the saved maps and live encoder output (post-argmax). The mapper network would then see "padding" as its own honest category. Works, but adds a synthetic class and more moving parts — only worth it if band conditioning measurably hurts results.
So, to state it as the one-liner you can repeat: depth's output space is continuous, so filler looks like filler; segmentation's output space is a closed list of 19 real-world objects, so filler is forced to impersonate one of them — we contain that with consistency and a known band position, rather than being able to eliminate it the way depth naturally does.
```