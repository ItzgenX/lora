# Depth Pipeline — Zero to Hero Guide

**Pipeline**: CTRLorALTer depth-conditioning arm (ECCV 2024, arXiv:2405.07913)  
**What it does**: Fine-tunes LoRA blocks inside a Stable Diffusion 1.5 UNet so the model generates images that respect a user-supplied depth map. The depth signal is injected via a learned mapper network (not a ControlNet adapter), keeping the base model frozen.

---

## 0 · Concepts From Zero — read this once and the rest of the doc makes sense

No prior diffusion knowledge assumed. Each concept below is (a) explained plainly, (b) tied to the exact place it lives in THIS repo, and (c) connected to a number you actually see when you run things. SEGMENTATION.md shares all of these concepts and adds its own §0 for seg-specific ones.

### 0.0 Plain-words glossary — every technical word used anywhere in these two docs

Read once; come back whenever a word feels new. Nothing below this table uses a term that isn't either here or explained where it first appears.

| Word | Plain meaning |
|---|---|
| **tensor** | A grid of numbers with a shape — a 512×512 RGB image is a `[3, 512, 512]` tensor (3 colour channels × height × width). Everything a neural network touches is a tensor. `[B, 3, 512, 512]` means a **B**atch of B such images processed together. |
| **uint8 / float32 / 8-bit** | Number formats. uint8 = whole numbers 0–255 (how PNGs store pixels — "8-bit" means the same). float32 = decimal numbers (how models compute). bf16 = half-size decimals (see §0.10). |
| **JSONL / manifest** | A plain text file with one JSON record per line. Our "manifests" are JSONLs where each line says: this image + this map + this prompt. Training reads them line by line. |
| **YAML** | A human-readable settings file format (`key: value`). Everything configurable in this project lives in `configs/*.yaml`. |
| **Hydra** | The library that reads our YAML configs, lets you override any value from the command line (`epochs=5`), and creates the dated run output folder automatically. |
| **conda env** | An isolated Python installation with pinned package versions (`loradapter` here) so the project always runs against the same libraries. |
| **accelerate** | Hugging Face's training launcher — handles GPU placement, mixed precision (bf16), and multi-GPU (`accelerate launch --num_processes=4`) without changing the training code. |
| **TensorBoard** | The curve viewer. Training writes numbers into `logs/tensorboard/`; `tensorboard --logdir ...` shows them as graphs in your browser. |
| **tqdm** | The progress bar in the terminal (`Steps: 50%\|#### \| 3.45s/it`). "s/it" = seconds per step. |
| **VRAM** | The GPU's own memory (12 GB on the dev card). Models + images must fit in it. |
| **OOM** | "Out of memory" — the crash when VRAM runs out. Fixed by lowering batch_size or enabling gradient checkpointing. |
| **gradient / backprop** | The direction-and-size of the weight change that would reduce the current error, computed by backpropagation ("backprop") after every forward pass. Training = repeat: forward → gradients → nudge weights. |
| **optimizer / AdamW** | The algorithm that applies those nudges. AdamW is the standard choice — it adapts the step size per weight. `optimizer.step()` in the code is one nudge. |
| **grad_norm** | The overall size of all gradients combined — the number logged as `train/grad_norm`. Spikes = instability; we clip it at 1.0. |
| **no_grad** | A "don't track gradients" mode used during validation/generation — faster and lighter because no training bookkeeping happens. |
| **RNG / seed / OS entropy** | RNG = random number generator. A **seed** (42 here) makes randomness repeatable — same seed, same "random" numbers. "OS entropy" = truly unseeded randomness (different every run). |
| **convolution / conv** | The basic image-network operation: a small filter slides across the image producing a new feature map. A "conv layer" is a layer of such filters. |
| **ResNet block** | A standard building unit inside the UNet: two conv layers plus a shortcut connection. The UNet is a stack of these — our LoRA adapters attach to the first conv of each block. |
| **cross-attention** | The layer where image features "look at" the text embedding — the mechanism that lets the prompt steer generation. |
| **logits** | A model's raw, unnormalised scores before picking a winner — for seg, 19 numbers per pixel (one per class). |
| **argmax** | "Pick the index of the biggest number." Turning 19 per-pixel scores into the one winning class ID is an argmax. |
| **bilinear / NEAREST (interpolation)** | Two ways to resize images. Bilinear blends neighbouring pixels smoothly (right for photos/depth). NEAREST copies the closest pixel unblended (required for class-ID maps — see SEGMENTATION.md §0.4). |
| **normalisation** | Rescaling numbers into a standard range, e.g. pixels [0,255] → [-1,1], or a depth map to [0,1] by its own min/max ("min-max"), or shifting by dataset statistics ("ImageNet normalisation" — the mean/std of the dataset SegFormer was trained on). |
| **checkpoint** | A saved snapshot of the trainable weights at some step, written to its own folder — you can generate from it or resume from it later. |
| **buffer** | A tensor stored inside a model that moves to the GPU with it but is never trained (e.g. the palette, the ImageNet mean/std). |
| **id2label** | The lookup table inside the SegFormer checkpoint mapping class ID → class name (0→"road", 11→"person"…). We verified our palette order against it. |
| **SSOT** | "Single source of truth" — the one place a value is defined; everything else imports it (e.g. the palette lives only in `seg_encoder.py`). |
| **PSNR / SSIM** | Peak Signal-to-Noise Ratio and Structural Similarity — two standard image-similarity scores (higher = more similar). Used per checkpoint, see §0.12. |
| **CLI flag** | A `--something` option passed on the command line (`--no_skip`, `--data_dir data/`). |
| **DPT / MiT-B5** | Just architecture names: DPT = the transformer design MiDaS uses for depth; MiT-B5 = the backbone size inside SegFormer-b5. |

### 0.1 What a diffusion model is

Take a photo and add a little random noise. Add more. After ~1000 rounds it's pure static. A diffusion model is a neural network trained to run that film **backwards**: shown a noisy image, it predicts what noise was added, so you can subtract it. Generation = start from pure static and denoise step by step until a clean image emerges. Think of a sculptor who "sees the statue inside the marble": each denoising step chips away a bit more noise toward a coherent image. At inference we use 50 steps (`num_inference_steps: 50` in the inference YAMLs) — a shortcut schedule through those ~1000 training noise levels.

### 0.2 Latent diffusion, the VAE, and why "512" appears everywhere

Denoising a full 512×512×3 image ~50 times is expensive. Stable Diffusion instead works on a compressed version: a **VAE** (variational autoencoder) squeezes the image into a 64×64×4 "latent" (a blueprint), diffusion happens on the blueprint, and the VAE decodes the final blueprint back to 512×512 pixels. That's why `size: 512` is fixed in our configs — it's SD1.5's native canvas (the VAE compresses it 8× to 64×64). In the code: `SD15.get_input()` in `src/model.py` does the VAE encoding.

### 0.3 The UNet and the loss number you watch

The denoiser network is a **UNet** — an hourglass-shaped conv network that takes (noisy latent, timestep, text) and predicts **the noise itself** (the "epsilon objective"). Training is disarmingly simple: take a training image, add a *known* random noise, ask the UNet to predict that noise, and score it with mean-squared error. That MSE **is** `train/loss` in TensorBoard. It never goes to 0 (predicting exact random noise is impossible); healthy values hover around 0.02–0.3 depending on the random timestep drawn, which is why the curve is jagged — watch the *trend*, not single points. `val/loss` is the same computation on held-out images with a fixed RNG seed, which is why it moves smoothly.

### 0.4 Text conditioning, the tokenizer, and classifier-free guidance

Your prompt is chopped into tokens by SD1.5's CLIP **tokenizer**, embedded by the CLIP text encoder, and fed to the UNet via cross-attention — that's how "a red car" steers denoising. Both our pipelines use this identical text path.
**Classifier-free guidance (CFG)**: during training, 5% of the time the prompt is replaced with "" (`c_dropout=0.05` in `src/model.py`), so the model also learns to denoise "blind". At inference we run it both with and without the prompt and push the result *away* from blind and *toward* prompted, scaled by `guidance_scale: 7.5`. The same 5% dropout is applied to the conditioning map — that's what makes the map's influence steerable too.

### 0.5 LoRA — why our checkpoints are 120 MB, not 4 GB

Fine-tuning all of SD1.5 (~860M params) is heavy and destroys its general knowledge. **LoRA** (Low-Rank Adaptation) freezes the original weight matrix `W` and learns only a small correction: two thin matrices `A` (down to rank 128) and `B` (back up), so the effective weight is `W + B·A`. Sticky notes on a textbook: the book is untouched; the notes carry your changes. `B` starts as all-zeros (`nn.init.zeros_` in `src/lora.py`), so at step 0 the correction is exactly nothing and training starts from vanilla SD1.5 behaviour. Only A, B and the mapper are saved — hence ~120 MB checkpoints (`lora-checkpoint.pt` + `mapper-checkpoint.pt`).

### 0.6 The CTRLorALTer trick — conditional LoRA via FiLM

Plain LoRA is static — the same correction for every image. This paper's idea: make the correction **depend on the conditioning map**. A small **mapper network** (`FixedStructureMapper15`) reads the 512×512 map and produces, per UNet block, a pair of vectors; inside each LoRA the down-projected activation gets multiplied by `gamma(c) + 1.0` and shifted by `beta(c)` — a technique called **FiLM** (feature-wise linear modulation). The `+1.0` matters: when gamma's output starts near 0, the scale starts near 1 = "change nothing", the same safe-start principle as B's zero-init. Injection points: the **first conv of every ResNet block** in the UNet (`adaption_mode: only_res_conv`, matching `0.conv1`/`1.conv1` in `src/model.py::add_lora_to_unet`). So: map → mapper → per-block FiLM signals → LoRA corrections → the generated image follows the map's layout.

### 0.7 Structure conditioning: depth maps and the two encoders

A **depth map** is a grayscale image where brightness = relative distance (bright = near, dark = far, per-image min-max normalised to [0,1]). We get it from **MiDaS** (`Intel/dpt-hybrid-midas`, `src/annotators/midas.py` — untouched stock code, confirmed by git history). A **seg map** labels every pixel with a class (road/person/car/... — SEGMENTATION.md §0). Both are "structure": they say WHERE things go, while the prompt says WHAT they look like.

### 0.8 Our big deliberate divergence from the stock repo (and why)

Stock LoRAdapter computes depth **live** from the raw image at every training step (`cond = encoder(img)` inside the forward pass). We instead:
- **Training: pre-saved maps.** Stage A/C compute every map once, offline (`depth_map_calculations.py` / `seg_map_calculations.py`), and training loads PNGs (`skip_encode=True` bypasses the encoder — §5.3). Why: the encoder (especially SegFormer-b5, 82M params) would otherwise run on every step of every epoch, again and again, for identical outputs — measured to keep VRAM identical between depth and seg training precisely because neither encoder runs during training.
- **Inference: live encoder.** You hand the script a raw photo, it computes the map on the fly (`model.sample()` → `encoder(c)`), because at inference each image is seen once and convenience wins.
- **The parity rule (the single most important rule in this repo)**: saved-for-training maps and live-at-inference maps MUST come from the same model, same preprocessing, same value range — otherwise the model trains on one "dialect" of maps and is quizzed in another. Everything in §5 (letterbox, shared preprocessing, parity checks) exists to enforce this.

### 0.9 Letterbox in one paragraph (full story: §5.1, §5.14a)

Our images are 1280×800; the model needs squares. Cropping cuts content (rejected), stretching distorts shapes and even changes what the segmenter sees (evaluated with real previews and rejected — §5.14a). So we **letterbox**: pad top/bottom with a flat local-mean colour, then resize. The pad-fill used to be a stretched copy of the edge row, which produced a striped band that the model *learned to draw* (see §5.1's 2026-07-05 fix — the flat fill removed it). Extra reason the square matters: MiDaS silently center-crops any non-square input internally, so feeding it a square is a correctness requirement, not a style choice (§5.1).

### 0.10 Training mechanics decoded (every number in train_depth.yaml)

- **batch_size: 4** — images per forward pass; the most a 12 GB GPU fits (measured ~90.6% VRAM).
- **gradient_accumulation_steps: 4** — gradients from 4 mini-batches are summed before one weight update, simulating a batch of 4×4=**16** (the "effective batch") without the memory cost. One weight update = one "optimizer step" = one tick of `global_step`.
- **learning_rate: 1e-4, lr_warmup_steps: 500, lr_scheduler: cosine** — how big each weight nudge is; ramped up gently for 500 steps (a cold start with full LR can wreck the zero-init'd adapters), then decayed along a cosine curve to ~0 so training "settles".
- **bf16: true** — 16-bit floats: half the memory, GPU-native, numerically safe for training.
- **gradient_checkpointing: true** — trades ~20% speed for ~800 MB VRAM by recomputing activations during backprop instead of storing them. Required on 12 GB.
- **epochs: 5** — full passes over the training set; steps/epoch = ceil(N_images / 16).
- **seed: 42** — fixes the random generator for reproducible sampling/validation.

### 0.11 train / val / test — three jobs, never mixed

- **train** (639 local / 59,766 real): the model learns from these.
- **val** (137 / 3,314): never trained on; used for `val/loss` (overfitting detector — if train/loss falls while val/loss rises, the model is memorising), for picking `best_model/`, and as the source of checkpoint monitoring images.
- **test** (137 / 3,327): touched by NOTHING during training — kept clean as the final honest benchmark. If we peeked at it for decisions, its verdict would be worthless.

### 0.12 What a checkpoint gives you and how we judge it

Every `ckpt_steps` (and every epoch end) a checkpoint folder is written: LoRA+mapper weights, N=10 labeled sample images (ORIGINAL | DEPTH MAP | PREDICTED), and prompts.txt. Half the scenes are **fixed** (same every checkpoint → watch the same scenes improve), half **fresh** (re-drawn → generalisation peek). Two numbers accompany every save: `val/psnr_fixed` and `val/ssim_fixed` — pixel- and structure-similarity between each fixed scene's generation and its real photo. Because the scenes and the sampling seed are fixed, these numbers are comparable across checkpoints *and* between the depth run and the seg run at matched steps — this is the objective evidence for the final "depth vs seg: which conditions better" verdict (§5.13).

### 0.13 Self-test — if you can answer these, you're the hero

1. Why does `train/loss` never reach 0, and why is it so jagged while `val/loss` is smooth? *(0.3: predicting random noise exactly is impossible; val uses a fixed seed.)*
2. Why are checkpoints 120 MB when SD1.5 is ~4 GB? *(0.5: only LoRA A/B + mapper are saved; the base model is frozen.)*
3. What does `skip_encode=True` do and where is it used vs not used? *(0.8/§5.3: bypasses the live encoder — training/validation use it with pre-saved maps; live inference never does.)*
4. Why must training maps and inference maps use identical preprocessing? *(0.8: otherwise train/inference distribution mismatch — the model is trained on one dialect and quizzed in another.)*
5. Why letterbox and not crop or stretch? *(0.9/§5.14a: crop loses driving-scene content; stretch distorts geometry and shifted seg classes in a real preview.)*
6. Why did the old padding produce a striped band, and why did generated images show it too? *(§5.1: stretched 1-px edge row → each busy pixel became a vertical stripe; the model learned the pattern from training data.)*
7. What was the manifest-collision bug and what now prevents it? *(§5.12: all rows pointed at one overwritten map file because every image is raw_image.jpg. Three guards now: both `--data_dir` and `--dataset_dir` name maps after the per-image folder, a collision guard aborts before any GPU work if two images would share a PNG, and verifier check 5 fails loudly on any duplicate map path.)*
8. What's the effective batch and why does it matter more than batch_size? *(0.10: batch_size × grad_accum = 16 — it's what learning dynamics (and the LR choice) actually depend on.)*
9. Why is `lora.struct.ckpt_path` resume only a "weight warm-start"? *(§5.9: weights reload, but optimizer state/LR schedule/global_step restart from 0 — verified by execution.)*
10. Why do the depth and seg configs have to be identical except the conditioning block? *(§0.12/§9 of the task: otherwise the depth-vs-seg comparison measures config differences, not conditioning quality — verified 29/29 params identical.)*
11. Why does `gamma(c) + 1.0` have the `+1.0`? *(0.6: scale starts at ≈1 = "change nothing" — a safe identity start, like B's zero-init.)*
12. Why must the image be square BEFORE it reaches MiDaS? *(§5.1: MiDaS internally center-crops non-square input — you'd silently lose the frame edges.)*

---

## 1 · What & Why

Standard T2I diffusion ignores structure — the same prompt generates different spatial layouts each run. This pipeline injects a depth map as a structural conditioning signal through LoRA rank-128 adapters that sit inside the UNet's residual-conv layers. A lightweight mapper network translates the depth embedding into LoRA delta-weights; the base UNet weights are never updated.

The result: the model generates images that follow the supplied depth map's spatial structure while still following the text prompt.

---

## 2 · Files in Order

| Stage | File | When |
|-------|------|------|
| A — offline preprocessing | `depth_map_calculations.py` | **Once**, before any training |
| B — training | `depth_training.py` | Iterative; resumes from checkpoint |
| B — inference | `depth_inference.py` | After training; per-image generation |

Supporting files:

| File | Role |
|------|------|
| `src/annotators/midas.py` | `DepthEstimator` class wrapping Intel/dpt-hybrid-midas |
| `src/data/local.py` | `DepthJsonDataset`, `DepthJsonDataModule` |
| `src/data/transforms.py` | `SquarePad` (flat local-mean fill padding) |
| `configs/train_depth.yaml` | Base training config (required keys + defaults) |
| `configs/experiment/train_depth.yaml` | Unified experiment overrides (use this one) |
| `configs/inference_depth.yaml` | Inference config |
| `configs/data/local_depth.yaml` | Data module + transform chain config |

---

## 3 · Data Flow Diagram

```
RAW IMAGE (arbitrary aspect ratio)
        |
        v
  [Stage A -- depth_map_calculations.py]  (run ONCE offline)
        |
        +-- SquarePad (flat local-mean fill) --> square image (no distortion)
        +-- Resize to 512x512
        +-- ToTensor + Normalize [-1,1]
        +-- DepthEstimator (DPT-hybrid-midas)  -->  per-image min-max --> [0,1] depth
        +-- Scale x255, save as 8-bit grayscale PNG
               --> <dataset>_depth_map/000417/000417_depth_map.png  (sibling folder)
        +-- Write data/depth_training/{train,val,test}.jsonl
               (keys: raw_image_path, depth_path, prompt)

        |
        v
  [Stage B -- depth_training.py]  (per training step)
        |
        +-- DepthJsonDataset.__getitem__:
        |     read depth PNG (L mode) --> replicate to 3 channels --> Tensor [0,1]
        |     read raw image --> SquarePad-->Resize-->ToTensor-->Normalize [-1,1]
        |     (NO SquarePad on depth PNG -- it's already square from Stage A)
        |
        +-- batch["depth"]  --->  StructMapper  --->  LoRA delta-weights
        |   (skip_encode=True: DepthEstimator NOT called during training)
        |
        +-- model.forward_easy(imgs, prompts, [depth_maps], skip_encode=True)
        |
        +-- val/loss logged every val_steps; checkpoint+grid every ckpt_steps

        |
        v
  [Stage B -- depth_inference.py]
        |
        +-- Raw image --> SquarePad-->Resize-->ToTensor-->Normalize [-1,1]
        |   (same chain, applied LIVE here -- no pre-saved depth PNG needed)
        |
        +-- LIVE encoder call: DepthEstimator(image) --> depth map
        |   (skip_encode=False: encoder runs in the model.sample() path)
        |
        +-- Generates 4-panel grid: ORIGINAL | DEPTH MAP | PREDICTED | RAW DEPTH GEN
```

---

## 4 · MiDaS / DepthEstimator

**Model**: `Intel/dpt-hybrid-midas` (DPT-Hybrid architecture, `model_size=384`)  
**Class**: `src.annotators.midas.DepthEstimator`

Key behaviours:

- **Per-image min-max normalisation**: output is scaled to `[0, 1]` based on each image's own min/max depth. There is no cross-image calibration. This means depth values from different images are not on the same scale — only spatial structure is meaningful.
- **`better_resize()` internal crop**: MiDaS internally centre-crops non-square inputs before running. If you feed a 16:9 image directly you lose the left and right edges. **`SquarePad` must be applied before any encoder call to prevent this.**
- **Output shape**: `[B, 1, H, W]` float32 in `[0, 1]` (after normalisation).
- **3-channel replication**: `DepthJsonDataset` replicates the single-channel PNG to 3 channels so the mapper network sees the same tensor shape it would see from a colour-image encoder.

### 4.1 The 384 question — why `better_resize(384)` and not 512

You will see `self.model_size = 384` in `src/annotators/midas.py` and wonder: our whole pipeline uses 512×512, so why is there a 384 here?

**Answer: 384 is DPT-Hybrid-MiDaS's native training resolution. The depth model expects 384×384 input. 512 is the LoRAdapter pipeline canvas size. They serve different purposes and the two sizes correctly coexist.**

Here is the full image journey through the depth pipeline:

```
RAW IMAGE (e.g. 1280×800)
    │
    ▼ [Stage A preprocessing — depth_map_calculations.py]
    SquarePad()             → 1280×1280  (flat local-mean fill, no distortion)
    Resize(512, 512)        → 512×512    (our pipeline canvas)
    ToTensor + Normalize    → [-1, 1]
    │
    ▼ [Inside DepthEstimator.forward() — midas.py]
    (imgs + 1.0) / 2.0      → [0, 1]    (convert from pipeline format)
    better_resize(384)      → 384×384   ← MiDaS's REQUIRED input size
    │   (center_crop to square → already square, so no-op)
    │   (avg_pool2d if needed → downscale factor = 512//384 = 1, so no-op)
    │   (bilinear interpolate to 384 → actual resize 512→384)
    depth_estimator(pixel_values)  →  depth logits 384×384
    F.interpolate(self.size=512)   →  depth map 512×512  (back to our canvas)
    min-max normalize per image    →  depth in [0, 1]
    cat([depth]*3)                 →  [B, 3, 512, 512]   (3-channel for mapper)
```

**The two 512s bookend the 384:**
- The image ENTERS DepthEstimator at 512×512 (our canvas, after SquarePad+Resize)
- `better_resize(384)` shrinks it to 384×384 for the model to run on
- `F.interpolate(512)` expands the depth map BACK to 512×512 (our canvas)

**Why 384 specifically?** DPT-Hybrid-MiDaS was fine-tuned on MiX-6 at 384×384 (that is the checkpoint's training size). Running it at exactly 384 gives the sharpest depth predictions. Running it larger or smaller changes the patch embedding stride resolution and degrades results.

**The `center_crop` in `better_resize` — why it is not dangerous here:**

```python
# src/annotators/util.py — better_resize() full implementation
def better_resize(imgs, image_size):
    H, W = imgs.shape[-2:]
    side  = min(H, W)             # e.g. min(512, 512) = 512
    imgs  = center_crop(imgs, [side, side])  # 512×512 → 512×512: NO-OP (already square)
    factor = side // image_size   # 512 // 384 = 1 → NO avg_pool
    if factor > 1:
        imgs = avg_pool2d(imgs, factor)
    imgs = interpolate(imgs, [image_size, image_size], mode="bilinear")
    return imgs
```

Because SquarePad already made the image square BEFORE it reaches `better_resize`, the `center_crop` step is a geometric no-op — cropping a 512×512 square to min(512,512)=512 changes nothing. Without SquarePad, a 1280×800 landscape image would be center-cropped to 800×800, silently discarding the left and right 240 px of content.

---

### 4.2 Q&A — Do we use `DPTImageProcessor`? Is this the original code?

**Q: The HuggingFace DPT example uses `DPTImageProcessor` + `DPTForDepthEstimation`. Do we use both?**

**A: `DPTForDepthEstimation` yes. `DPTImageProcessor` no — it is imported but commented out. This is the ORIGINAL CompVis code, not something we changed.**

**Verified by:**
- `git diff HEAD -- src/annotators/midas.py` returns empty — zero changes since the initial commit
- Fetching `https://raw.githubusercontent.com/CompVis/LoRAdapter/main/src/annotators/midas.py` directly and comparing line by line — files are byte-for-byte identical

The original authors shipped `midas.py` with the processor commented out and manual preprocessing in its place. We inherited this code unchanged.

#### What the original code does (lines 6–50 of `src/annotators/midas.py`)

```python
from transformers import (
    DPTImageProcessor,        # imported but NEVER used
    DPTForDepthEstimation,    # this IS used
)

class DepthEstimator(nn.Module):
    def __init__(self, size, model, local_files_only):
        self.depth_estimator = DPTForDepthEstimation.from_pretrained(model, ...)
        # self.feature_extractor = DPTImageProcessor.from_pretrained(...)  # COMMENTED OUT

    def forward(self, imgs):
        imgs = (imgs + 1.0) / 2.0          # [-1,1] -> [0,1]  (manual, no processor)
        imgs = better_resize(imgs, 384)     # resize to 384     (manual, no processor)
        # depth_dict = self.feature_extractor(...)              # COMMENTED OUT
        depth_map = self.depth_estimator(pixel_values=imgs).predicted_depth
        # ... min-max normalise, replicate to 3 channels
```

#### Why the original authors skipped the processor

`DPTImageProcessor` resizes the image and normalises pixel values. But `DepthEstimator` already does both steps manually in two lines before calling the model:
1. `(imgs + 1.0) / 2.0` — convert from the pipeline's `[-1, 1]` to `[0, 1]`
2. `better_resize(imgs, 384)` — resize to the model's input size

Using the processor on top of this would double-process the image and break the input contract. The manual path is also faster (no CPU round-trip, no dict wrapping) and keeps the preprocessing visible in the code rather than hidden inside a HuggingFace object.

Our `SegmentationEncoder` in `src/encoders/seg_encoder.py` was deliberately designed to mirror this exact pattern — see SEGMENTATION.md §4.2 and §4.3 for the full comparison.

---

## 5 · Key Design Concepts

### 5.1 SquarePad (flat local-mean fill padding)

`src.data.transforms.SquarePad` pads the shorter dimension to make the image square using a **single flat fill colour per pad region** — the mean colour of a thin (8 px) strip just inside that edge (not zero-padding, not centre-crop, and since 2026-07-05 no longer edge-replication). This preserves spatial content at borders.

**Flat-fill fix (2026-07-05, VERIFIED)**: the previous version stretched a 1-px boundary strip to fill the pad region. That is correct edge-replicate padding by definition — but every pixel of real horizontal detail in that single boundary row (sky/building edges, power lines, sensor noise) survived exactly and was repeated straight down the pad band, producing a smeared multicolour striped band on busy edge rows. The model then **learned** that band from the training data (it appeared in generated output too, not just conditioning panels). The flat local-mean fill has zero internal variation by construction, so it cannot band regardless of how busy the edge row is. A `fill_mode="mode"` option (most-frequent colour, never a blend) exists for the case where `SquarePad` is ever applied directly to a categorical/palette map — not needed by any current code path. Verified by the module's own self-check (`python -m src.data.transforms`, both PASS lines) and by numeric + visual inspection of regenerated maps. **All cached maps computed before this fix carry the old artifact — regenerated 2026-07-05 with `--no_skip` (913/913 depth + seg). Checkpoints trained before 2026-07-05 will keep showing the band — that's the old learned pattern, not a fix failure.** Note: the vehicle hood visible at the *bottom* of real driving frames is real sensor content, not a padding artifact — it needs no fixing. `last_padding_fracs` keeps its exact (left, top, right, bottom)-fractions contract.

**Why it's required**: MiDaS's `better_resize()` internally crops to a square before running. Without `SquarePad`, a landscape image would lose its left and right portions inside the DPT encoder. The padding is applied **in all three preprocessing sites** to ensure consistency:

1. `depth_map_calculations.py` — Stage A offline preprocess
2. `depth_inference.py` — Stage B live inference preprocess
3. `configs/data/local_depth.yaml` — Stage B training transform chain (for raw images)

**Triplication risk**: These three sites must stay byte-identical. If you change the preprocessing (e.g., add a normalisation step), update all three.

**Cross-platform fix (2026-07-01, VERIFIED)**: `SquarePad` used to call `torchvision.transforms.functional.pad(img, ..., padding_mode='edge')`. This routes through torchvision's internal PIL/numpy conversion for non-`"constant"` padding modes, which is **not guaranteed identical across torchvision versions** — this pipeline hit exactly that: worked in a Windows conda env, threw inside `preprocess()` on a valid image in a separate Linux conda env with a different torchvision build (misleadingly reported as "could not load image" because the old error handling didn't distinguish a Pillow load failure from a preprocess failure — now split into two separate try/except blocks with distinct messages). Fixed by rewriting edge-replication using only plain PIL `crop()`/`resize()`/`paste()` calls — no `torchvision.functional.pad`, no numpy conversion, no version dependency. Verified bit-identical (`max abs pixel diff = 0`) against the old implementation on 3 real images before replacing it. Also added `PIL.ImageFile.LOAD_TRUNCATED_IMAGES = True` (set once in `src/data/transforms.py`, which all 4 entrypoints import) to tolerate minor JPEG defects that some decoders accept and Pillow otherwise rejects with "image file is truncated".

### 5.2 Image Discovery and Path Control — how the pipeline finds your images

This section answers: where does the pipeline look for images, what key in the JSONL tells it where the image is, and what do you change if your images live somewhere else?

#### The three JSONL file types and their key names

```
data/train.jsonl                          ← source split (created by dataset prep)
  {"source": "data/raw/000417/raw_image.jpg", "prompt": "..."}
   ▲ default key = "source"

data/depth_training/train.jsonl           ← depth training manifest (output of Stage A)
  {"raw_image_path": "data/raw/000417/raw_image.jpg", "depth_path": "...", "prompt": "..."}
   ▲ key = "raw_image_path"
```

Image filename is always `raw_image.jpg` — the folder name (`000417/`) is the scene identifier.

#### `--image_path` — tell the script which key holds the image path

The CLI arg `--image_path` sets which key the script reads from each JSONL entry. Default is `"source"`. If your JSONL uses `"target"` (or any other name), pass it explicitly:

```bash
# Your JSONL has {"source": "...", "prompt": "..."}  → default, no flag needed
python depth_map_calculations.py --data_dir data/

# Your JSONL has {"target": "...", "prompt": "..."}
python depth_map_calculations.py --data_dir data/ --image_path target
```

The key name flows from CLI → `_get_image_path(entry, image_path)` → file open. Nothing else in the script hard-codes a key name.

#### Where the depth maps are saved — a SIBLING folder, derived automatically (2026-07-07)

You do NOT choose an output folder and you cannot cause an overwrite. `--data_dir` looks at the image paths in your JSONLs, finds the folder common to all of them (the **dataset root**, e.g. `.../custome_dataset`), and saves each map into a **sibling folder next to it** — `.../custome_dataset_depth_map/` — mirroring the internal structure. The map file is named after the image's **folder**, so every image gets a unique map:

```
.../custome_dataset/000417/raw_image.jpg  →  .../custome_dataset_depth_map/000417/000417_depth_map.png
.../custome_dataset/000420/raw_image.jpg  →  .../custome_dataset_depth_map/000420/000420_depth_map.png
```

This is IDENTICAL to what `--dataset_dir` scan mode does (§5.2b) — the two modes were unified on 2026-07-07 so they always agree, and both were proven to produce byte-identical manifest entries for the same image.

**Why the old design was replaced:** `--data_dir` used to save maps into `data/raw_depth/` named after the image *filename*. Because every image in this dataset is named `raw_image.jpg`, all 913 maps overwrote each other into one file (the 2026-07 bug). Two safeguards now make that impossible:
1. Maps are named after the per-image **folder**, not the filename.
2. A **collision guard** runs before any GPU work — if two images would ever write the same PNG, the script aborts loudly and prints both image paths.

`--image_root` still exists, but only to resolve **relative** image paths in the JSONL (prepend this base). Your `data/*.jsonl` store **absolute** paths, so you don't need it.

#### The full discovery → depth → output flow (VERIFIED by execution)

```
data/train.jsonl                               ← --data_dir
  {"target": ".../custome_dataset/000417/raw_image.jpg", "prompt": "..."}
       │
       │ --image_path target
       ▼
  _get_image_path(entry, "target")
       │  → ".../custome_dataset/000417/raw_image.jpg"   (absolute, used as-is)
       ▼
  dataset root = folder common to ALL images  →  .../custome_dataset
  sibling      = dataset root + "_depth_map"   →  .../custome_dataset_depth_map
       ▼
  DepthEstimator.forward(image)   → depth tensor [1, 512, 512] in [0,1]
       │
       ▼
  _sibling_map_path: mirror structure, name after the FOLDER
       │  → .../custome_dataset_depth_map/000417/000417_depth_map.png
       ▼
  data/depth_training/train.jsonl  (output — VERIFIED mapping)
  {
    "raw_image_path": ".../custome_dataset/000417/raw_image.jpg",              ← absolute, unchanged
    "depth_path":     ".../custome_dataset_depth_map/000417/000417_depth_map.png",
    "prompt":         "this is a close up of a person holding a map..."         ← verbatim
  }
```

#### Quick reference

| Situation | Command |
|---|---|
| Your dataset (key = `"target"`, absolute paths) | `python depth_map_calculations.py --data_dir data/ --image_path target` |
| Same, but scan the folder for images instead of trusting JSONL paths | `... --dataset_dir /path/to/custome_dataset --data_dir data/ --image_path target` |
| JSONL key = `"source"` (default) | `python depth_map_calculations.py --data_dir data/` |
| Relative image paths in JSONL | add `--image_root /path/to/common/root` |
| Quick smoke-test | add `--dry_run_n 5` |

`--data_dir` and `--dataset_dir` both save to the sibling folder and both refuse to overwrite — you cannot corrupt your data by picking the "wrong" one.

### 5.2b Dataset-scan mode (`--dataset_dir`) — save maps in a SIBLING, mirrored tree [VERIFIED]

Same saving behavior as `--data_dir` above (sibling folder, mirrored, folder-named maps), with ONE difference: instead of trusting the JSONL paths to *find* images, the script **scans `--dataset_dir` recursively** for `raw_image.jpg` and uses whatever is on disk. Use it when you want disk to be the source of truth for which images exist; use plain `--data_dir` when the JSONLs already list exactly the images you want. Either way the maps land in the same place. The `--data_dir` argument is still required (it supplies each image's prompt + split).

**Command:**
```bash
python depth_map_calculations.py \
  --dataset_dir /path/to/custome_dataset \
  --data_dir data/ \
  --image_path target
```

**What each argument does:**
- `--dataset_dir` — the folder to scan recursively for `raw_image.jpg`. Activates scan mode.
- `--data_dir` — folder holding `train/val/test.jsonl`. Used ONLY to recover each image's **prompt** and which **split** it belongs to (matched by absolute image path). Required.
- `--image_path` — the JSONL key holding the image path (`target` here).
- `--image_name` — the exact filename to process (default `raw_image.jpg`). Every other file in the folder is ignored, so the source dataset can freely contain other images per folder.

**Where the sibling folder is created:** automatically derived from `--dataset_dir` — no separate flag needed.
```
--dataset_dir  = /path/to/custome_dataset
sibling output = /path/to/custome_dataset_depth_map     (same parent, name + "_depth_map")
```

**What it produces (VERIFIED on a 914-folder dataset):**
```
/path/custome_dataset/000417/raw_image.jpg              ← input (found by scan, untouched)

/path/custome_dataset_depth_map/000417/000417_depth_map.png
                                                          ← NEW sibling tree, mirrors internal structure,
                                                            file named after the leaf folder

data/depth_training/train.jsonl   (+ val, test)          ← rebuilt from the originals:
  {
    "raw_image_path": "/path/custome_dataset/000417/raw_image.jpg",               ← absolute, unchanged
    "depth_path":     "/path/custome_dataset_depth_map/000417/000417_depth_map.png",
    "prompt":         "..."                                                        ← from the split JSONL, verbatim
  }
```

**Key guarantees (all verified by execution, including two deliberate negative tests):**
- The source dataset folder is **never modified** — verified: after a scan run, `custome_dataset/000417/` contains only `raw_image.jpg`, nothing else.
- Only `raw_image.jpg` is processed — other files/images in the folder are ignored.
- The prompt + split for each image are preserved exactly (matched by absolute path against the original `data/*.jsonl`).
- Output JSONL paths are absolute (the data lives outside the repo).
- Re-runs skip already-computed maps unless you pass `--no_skip`.
- The verifier (`_verify_scan_training_jsonl`) was fed known-bad cases (map written in-folder instead of sibling; map under the wrong mirrored leaf-folder name) and correctly FAILed both, while passing the valid entry — so a FAIL from this checker is a real signal, not an unproven check.

**Dry run first:**
```bash
python depth_map_calculations.py --dataset_dir /path/custome_dataset --data_dir data/ --image_path target --dry_run_n 6
```
In a dry run the scan is capped to the first N images; split JSONLs contain only the entries whose images were in that capped set (the rest are reported as "not in the scanned subset — expected").

### 5.2c GPU resolution — never silently falls back to CPU [VERIFIED]

**Problem this fixes**: `depth_map_calculations.py` (and every other pipeline entrypoint) used to pick a device with `"cuda" if torch.cuda.is_available() else "cpu"`. If CUDA wasn't detected for ANY reason — CPU-only torch build, driver mismatch, wrong conda env, `CUDA_VISIBLE_DEVICES` unset/empty — the script silently ran on CPU. Nothing printed a warning; the run just looked normal but never touched the GPU. This was hit for real on a Linux env that worked fine on Windows.

**Fix**: all 6 pipeline entrypoints (both calc scripts, both training scripts, both inference scripts) now call `resolve_device()` / the equivalent `accelerator.device` check from `src/utils.py`. Behavior:

- Prints full GPU diagnostics **every run**: `torch.cuda.is_available()`, device count, `CUDA_VISIBLE_DEVICES`, and each visible GPU's name + VRAM.
- If no device was explicitly requested and no GPU is visible → **raises `RuntimeError`** with a concrete fix checklist (check `nvidia-smi`, check `torch.version.cuda`, check `CUDA_VISIBLE_DEVICES`). It does **not** silently run on CPU.
- `--device cpu` is still honoured as an **explicit opt-in** (prints a loud reminder it'll be slow), for real CPU testing.
- `--device cuda:N` pins a specific GPU index; errors if that index isn't visible.

**Verified by execution** (`src/utils.py::resolve_device`, isolated test with monkeypatched `torch.cuda.is_available`):
| Case | Result |
|---|---|
| Real GPU present, no `--device` passed | returns `"cuda"` — PASS |
| No GPU (simulated), no `--device` passed | raises `RuntimeError` — PASS |
| No GPU (simulated), `--device cpu` | returns `"cpu"`, allowed — PASS |
| No GPU (simulated), `--device cuda` | raises `RuntimeError` — PASS |

### 5.2d Batch size auto-scales to GPU VRAM (12GB local vs cluster GPUs)

`--batch_size` now defaults to `None` (was `4`). When left unset, `auto_batch_size()` (`src/utils.py`) scales it from the detected GPU's total VRAM relative to a 12GB baseline: `max(4, round(4 * gpu_gb / 12))`. On a 12GB GPU this returns `4` (unchanged from the old hardcoded default). Pass `--batch_size N` explicitly to disable auto-scaling entirely.

**Verification status — be precise about what's actually been checked**: VERIFIED on a real ~12GB GPU (returns `4`, matching the previously-hardcoded, already-proven-safe value). The scale-up behavior for much larger GPUs (e.g. a 98GB cluster GPU → ~32) is a **reasoned extrapolation, not executed** — no such hardware was available to test against. If a run on a larger GPU ever OOMs, override with `--batch_size` explicitly.

### 5.3 skip_encode — Training vs Inference Path

```python
model.forward_easy(..., skip_encode=True)   # training:   uses pre-saved depth PNG
model.sample(...)                            # inference:  calls DepthEstimator live
```

During training the depth map is loaded from `batch["depth"]` (the pre-computed PNG from Stage A). `skip_encode=True` tells `SD15.forward()` to skip the encoder entirely and feed the depth tensor directly to the mapper. This is fast and avoids running the GPU-heavy DPT model on every training step.

During inference the raw image is fed to the live `DepthEstimator` inside `model.sample()`. The preprocessing chain must be identical to Stage A to ensure the mapper sees the same distribution of depth maps.

### 5.3b Conditional LoRA (StructLoRA / NewStructLoRAConv)

LoRA adapters are inserted into the UNet's residual-conv layers only (`adaption_mode: only_res_conv`). Rank = 128. The mapper network takes a 128-dim depth embedding and outputs per-layer delta-weights. The base UNet is frozen throughout.

Parameter counts (verified by execution):
- Mapper Network: 1,245,072
- Encoder Network: 0 (DepthEstimator is frozen)
- LoRA adapters: 29,999,104

### 5.4 Checkpoint Grid (50/50 Fixed/Fresh Split)

Every `ckpt_steps` steps the trainer saves weights **and** monitoring images together in the same folder (`checkpoint-epochN/stepM/`). Images are:

- **Fixed half**: `n_grid_images // 2` val scenes chosen randomly ONCE per run (using OS entropy, not the training seed). These exact scenes recur at every checkpoint so you can watch the same scene improve over time. The chosen indices are logged at startup.
- **Fresh half**: re-drawn randomly from the remaining val indices at each checkpoint — a quick generalization peek.

Source is **validation set only** — never train or test. This is enforced by `dm.val_dataset`.

### 5.5 val_steps vs ckpt_steps (Decoupled)

```yaml
val_steps:  500    # cheap: compute val/loss, possibly update best_model
ckpt_steps: 1000   # heavy: save weights + N monitoring images
```

These are **independent**. You can validate every 50 steps and checkpoint every 500. Both can fire at the same step — the code handles this correctly by calling `do_validation` then `save_ckpt_and_grid` sequentially.

`best_model/` is written by `do_validation` (when val/loss improves), not by `save_ckpt_and_grid`. `best_model/info.txt` records the step and val/loss it came from.

### 5.6 Why test.jsonl Is Never Touched During Training

`data/depth_training/test.jsonl` is written by `depth_map_calculations.py` alongside `train.jsonl` and `val.jsonl`, but **no training or validation config ever reads it**. It exists so you can run a final held-out evaluation after training is complete, using `depth_inference.py`.

Training configs (`configs/experiment/train_depth.yaml`) reference only:
```yaml
json_file:     data/depth_training/train.jsonl
val_json_file: data/depth_training/val.jsonl
```

The test split stays clean.

### 5.7 Hostname Override in TensorBoard

TensorBoard embeds `socket.gethostname()` in the tfevents filename. Without intervention this would produce `events.out.tfevents.<ts>.<your-machine-hostname>.<pid>.0`, leaking the machine name into shareable log files.

The code overrides it before calling `accelerator.init_trackers`:
```python
import socket as _socket
_socket.gethostname = lambda: str(cfg.get("tag", "loradapter"))
```

Verified result: `events.out.tfevents.1782770112.depth.13012.0` — the field is `depth` (from `cfg.tag = "depth"`).

### 5.8 Training Timing — Measured Numbers and Self-Measurement Recipe

VRAM and disk numbers below are hardware-independent (a property of the model/batch/tensors) and apply directly to your training machine. **Timing (seconds/step, epoch time, 5-epoch ETA) is GPU-speed-dependent and must be measured on your own training machine** — the self-measurement command below takes ~3-5 minutes and gives you the real number.

**Measured (12GB dev GPU, 639 real local images, batch_size=4, grad_accum=4, gradient_checkpointing=True, bf16=True):**
- Peak VRAM: **~90.6% of a 12GB card** at batch_size=4 — identical for depth and seg (confirms `skip_encode=True` keeps the encoder out of the training forward pass, measured not assumed). Do not raise batch_size on a 12GB card.
- Steady-state optimizer-step time (after ~20-step warmup settles): **~3.44-3.47 s/step** on the dev GPU — reference only, re-measure on your own machine.
- One-time first-step warmup (CUDA/cuDNN compilation): ~29s, a one-off cost per process launch, not per-step.
- Validation (256 images, no_grad): ~31-35s per `val_steps` trigger (dev-GPU reference).
- Checkpoint grid generation (10 images, full 50-step diffusion sampling): ~52s per event — doubles to ~104s when `best_model/` AND the regular `checkpoint-epochN/` both fire on the same trigger (as happens whenever val/loss improves). Dev-GPU reference; re-measure for your own ETA.

**Get your real number** — run this for ~3-5 minutes on your own training machine (it will not disrupt anything — it's a normal short training run):
```bash
python depth_training.py experiment=train_depth epochs=1 val_steps=999999 ckpt_steps=999999
```
Let it run past the first ~20 steps (the first step includes one-time warmup and is not representative). Read the stabilized `s/it` value from the tqdm progress line, e.g. `Steps: 50%|##### | 20/N [01:34<01:08, 3.45s/it, ...]` → `3.45` is your real per-step measurement.

Then compute your real epoch/5-epoch time:
```
steps_per_epoch = ceil(TRAIN_IMAGES / (batch_size * gradient_accumulation_steps))
pure_training_time_per_epoch = steps_per_epoch * measured_seconds_per_step
checkpoint_overhead_per_epoch ≈ (val triggers * ~33s) + (ckpt triggers * ~52-104s)  [also hardware-dependent, same caveat]
total_epoch_time ≈ pure_training_time_per_epoch + checkpoint_overhead_per_epoch
5_epoch_time ≈ total_epoch_time * 5
```
With `59,766` train images, `batch_size=4`, `grad_accum=4`: `steps_per_epoch = ceil(59766/16) = 3736`.

### 5.9 Checkpoint Resume — Verified Limitation

`lora.struct.ckpt_path=<path>` (see `add_lora_from_config()` in `src/utils.py`) **does** correctly reload the mapper + LoRA weight tensors — verified by executing a real resume: the log printed `loaded checkpoint for lora struct` and the resumed run's loss reflected the already-trained weights, not a cold random init.

**It does NOT restore**: `global_step`, optimizer momentum/variance state, or the LR scheduler's position. Verified by execution: a resumed run's step 1 showed `lr=2.00e-07` — **identical** to a completely fresh run's step 1 — proving the warmup/cosine schedule restarts from scratch on every resume, regardless of how far the original run had progressed.

**Practical implication**: if training is interrupted mid-run (e.g. across sessions on a multi-hour full-data run) and you resume via `ckpt_path`, the resumed run's LR curve does NOT continue from where the original left off — it warms up from `lr_warmup_steps` again. This is a weight warm-start, not a true training-state checkpoint/resume. Plan session boundaries around this (e.g. prefer to let one epoch fully complete before stopping, since `epochs`/checkpoint folders are epoch-aligned) rather than assuming an interrupted run picks back up exactly where it left off.

### 5.10 `training_params.txt` — Know What Parameters Produced a Given Run

Every training run writes `outputs/train/depth/runs/YYYY-MM-DD/HH-MM-SS/training_params.txt` — a plain-text snapshot of the parameters that run used (GPU, batch_size, gradient_accumulation_steps, effective batch, epochs, learning_rate, val_steps/ckpt_steps, dataset paths + real image counts, model paths, resume path).

**Why not just use Hydra's own `.hydra/config.yaml`?** Hydra's snapshot is a raw config dump (harder to skim) and doesn't include derived facts like the real dataset image counts. This file is a single, human-readable summary of exactly what produced the checkpoints sitting next to it.

Implementation: `write_training_params_txt()` in `src/utils.py`, called from both `depth_training.py`/`seg_training.py` on the main process only. **Verified by execution**: an initial version read the dataset manifest paths using the process's current working directory, which under Hydra's `chdir=true` is the run's own output folder, not the repo root — silently producing "unknown images" for both train/val counts. Fixed by resolving relative paths against `get_original_cwd()` (the same pattern already used elsewhere in these two files); re-verified the fix produces the correct real image counts (639/137 on the local test dataset) for both depth and seg.

### 5.11 No Runtime Auto-Scaling — `recommend_training_params.py` Instead

`depth_training.py`/`seg_training.py` used to auto-scale `data.batch_size`/`gradient_accumulation_steps` to the detected GPU at runtime (a function called `auto_scale_training_hardware()`). **This was removed entirely** — for a project whose whole point is a defensible, reproducible comparison between depth and seg conditioning, a value that silently depends on which GPU happened to run the job undermines being able to say "these exact numbers produced this exact model." `configs/experiment/train_depth.yaml`/`train_seg.yaml` are now plain, fixed values — what's written is exactly what runs, full stop.

In its place: `recommend_training_params.py` (repo root) — a standalone advisor that detects your GPU and reads your real dataset manifests, then **prints** a recommended set of hyperparameters. It never writes or modifies any YAML file; you review the output and paste the values in yourself.

```bash
python recommend_training_params.py
python recommend_training_params.py --data_dir data/depth_training --epochs 10
```

**Verified by execution**: run against the real local dataset (639 images) on a 12GB dev GPU, correctly recommended `batch_size=4`, `gradient_accumulation_steps=4`, `gradient_checkpointing=true` (matching the independently-measured-safe values from §5.8) and `steps_per_epoch=40`. Also verified against a synthetic 59,766-line manifest (matching this project's real target dataset size) — correctly computed `steps_per_epoch=3736`, `total_steps=18680`, matching the manual calculation in §5.8 exactly. The batch/accum-scaling formula for GPUs other than ~12GB was separately verified across 4 simulated sizes (12/24/48/98GB) to still hold the effective batch at exactly 16 in every case — same capping logic that was bug-fixed in the now-removed runtime version, re-verified intact in this standalone tool.

### 5.12 Manifest-Collision Bug Fixed (2026-07-05), then Mode-Unified (2026-07-07) [VERIFIED]

**The bug**: the old `--data_dir` mode wrote each map as `<stem>.png` into one flat folder (`data/raw_depth/`). Every image in this dataset is named `raw_image.jpg`, so all 913 maps overwrote each other into ONE file, and every manifest row pointed at that single survivor — training would have run *silently* with the same conditioning map for every image. It first surfaced on 2026-07-05 and was hit again on Linux on 2026-07-07 (running `--data_dir` without `--dataset_dir`).

**The permanent fix (2026-07-07, verified by execution)**: `--data_dir` and `--dataset_dir` were **unified** — both now derive the dataset root from the image paths and save collision-free `<folder>_depth_map.png` names into the mirrored **sibling** tree `<dataset>_depth_map/` (§5.2, §5.2b). Both were proven to produce byte-identical manifest entries for the same image, and both write manifests the real training dataset loaders consume (checked by loading a produced manifest through `DepthJsonDataset` → correct `jpg`/`depth`/`caption` tensors). Three independent guards now make the overwrite impossible:
1. Maps are named after each image's **folder**, never the filename.
2. A **collision guard** runs before any GPU work and aborts loudly (naming both images) if two would share a PNG. Negative-tested: it fires.
3. Verifier **check 5** fails on any duplicate map path in the finished manifest.

**Earlier regeneration**: both calc scripts were re-run with `--no_skip` on 2026-07-05 — 913/913 depth + 913/913 seg maps regenerated (with the fixed flat-fill SquarePad, so this also purged the striped-band artifact). Any checkpoint trained before 2026-07-05 was trained on collided conditioning — treat it as invalid for quality judgements.

### 5.13 Per-Checkpoint Quantitative Metric + Per-Step File Logging (2026-07-05) [VERIFIED]

- **`val/psnr_fixed` / `val/ssim_fixed`**: at every checkpoint-image event, PSNR + SSIM are computed between each FIXED scene's generation and its real validation image (`src/utils.py::compute_psnr_ssim`, pure numpy), averaged, and logged to TensorBoard and the run's `.log`. Fixed scenes + fixed seed make the trend comparable across checkpoints *and* across the depth-vs-seg runs at matched steps — this is the objective signal for the final comparison. (FID is meaningless at 10 samples per point; CLIP would need weights not present in `checkpoints/local_models/`. `torch-fidelity` remains available for a one-off final FID over a full val-set generation pass if wanted.)
- **`log_every_steps`** (YAML, default 50): every N optimizer steps a `[train] step/loss/lr/grad_norm/epoch` line is written into the run's own `depth_training.log` (Hydra job log — already timestamped and file-based), so a run can be read back without opening TensorBoard.
- **Removed dead YAML keys** (read by nothing in the training path, verified by grep): `use_empty_prompt_eval`, `n_samples`, `save_grid`, `log_cond` — deleted from all four depth/seg training configs rather than left as silent no-ops. `ignore_check` stays (it is read by `add_lora_from_config`).

### 5.14a resize_mode: FINAL DECISION — letterbox only; stretch REMOVED (2026-07-06)

**The user evaluated stretch with real side-by-side encoder previews and rejected it.** Evidence used for the decision (kept for the record): `outputs/viz/resize_mode_preview.png` (letterbox vs stretch: original + depth map + seg map, real encoders, same scene) and `outputs/viz/letterbox_vs_stretch.png`. Deciding observations: stretch's aspect distortion (1280×800 → 512×512 compresses width 2.5× vs height 1.6×) visibly squeezed people/buildings AND shifted segmentation classes in the preview (part of the sky read as "building") — SegFormer was trained on undistorted photos. Letterbox keeps true geometry; its flat pad bars are clean (§5.1) and identical at training and inference.

**Consequence — the stretch option was REMOVED from the code entirely** (not just left unselected), so training and inference can never be run in different modes by accident:
- `src/data/transforms.py` `build_seg_square_preprocess(size)` — no `resize_mode` parameter anymore; letterbox is built in.
- `seg_map_calculations.py` — `--resize_mode` CLI flag removed.
- `configs/inference_seg.yaml` — `inference.resize_mode` key removed.
- `seg_inference.py` — no longer reads a mode; calls the fixed builder.
- Depth never had a switch (its three preprocess sites are letterbox inline).

Also remember §5.14 above: whichever preprocessing produced the training maps is the preprocessing inference must use forever for that checkpoint — mode-mixing (e.g. train letterbox, infer stretch) runs without error but silently degrades quality, because the model learns the bar layout and the map geometry from training data. This is why the toggle was removed rather than documented.

### 5.14 Parity Tolerance — Batch-Shape Kernel Jitter (2026-07-05) [MEASURED]

Saved training PNGs and live single-image encoder output are *not bit-identical*: measured up to `0.0086` max abs diff (depth, ≈2.2/255) and ~1e-5 of pixels (seg argmax flips at region boundaries). Root cause isolated by execution: the same image through the same weights in the same process differs between batch-of-1 and batch-of-4 (up to `0.0036` depth) — GPU kernels are selected per batch shape. The calc scripts run batched; live inference runs single-image. This is inherent GPU numerics (and exists across different GPUs anyway), ~100× below the 5% conditioning dropout, and **not** a preprocessing bug — same-shape float-vs-float parity is exactly `0.0`. Acceptance criteria: same-shape diff `= 0.0`; PNG-vs-live `≤ 0.015` (depth); ID-mismatch fraction `≤ 1e-4` (seg).

---

## 6 · YAML Parameters Explained

### `configs/train_depth.yaml` (base — required keys and defaults)

| Key | Default | Notes |
|-----|---------|-------|
| `size` | `???` | **Required.** Image size; experiment sets 512 |
| `learning_rate` | `1e-4` | AdamW learning rate; experiment keeps `1.0e-4` (identical to seg — required for the comparison) |
| `lr_scheduler` | `constant` | Experiment overrides to `cosine` |
| `lr_warmup_steps` | `0` | Experiment overrides to `500` |
| `epochs` | `10` | Experiment overrides to `5` |
| `val_steps` | `1000` | How often to run validation |
| `ckpt_steps` | `1000` | How often to save checkpoint + images |
| `val_batches` | `4` | Number of val batches per validation pass |
| `seed` | `42` | Training seed (not the fixed-image selection seed) |
| `bf16` | `false` | Experiment overrides to `true` |
| `gradient_checkpointing` | `false` | Experiment overrides to `true` |
| `gradient_accumulation_steps` | `1` | Experiment overrides to `4` |
| `tag` | `''` | Experiment sets `'depth'`; used as hostname in tfevents |
| `local_files_only` | `false` | Set `true` once models are downloaded |
| `ignore_check` | `false` | Suppress the checkpoint-key completeness assert in `add_lora_from_config` |
| `prompt` | `null` | If set, overrides all per-sample captions |
| `log_every_steps` | `50` | Write a `[train] step/loss/lr` line into the run's `.log` every N optimizer steps |

**Dead keys — REMOVED (2026-07-05)**: `use_empty_prompt_eval`, `n_samples`, `save_grid`, `log_cond` used to sit in the depth/seg training configs unread (they belong to the stock `train.py`/`sample.py` path). They were deleted from all four depth/seg training YAMLs — a key you can set with zero effect silently lies about what the run did. They still exist in the untouched stock configs (`configs/train.yaml`, `configs/sample*.yaml`), which is correct for stock scripts.

### `configs/experiment/train_depth.yaml` (use this for all runs)

```yaml
size: 512
learning_rate: 1.0e-4
lr_warmup_steps: 500
lr_scheduler: cosine
epochs: 5
val_steps: 500
ckpt_steps: 1000
val_batches: 64
n_grid_images: 10
grid_include_empty_prompt: false
bf16: true
gradient_checkpointing: true
gradient_accumulation_steps: 4
tag: depth
local_files_only: true
ignore_check: true
data:
  json_file:     data/depth_training/train.jsonl
  val_json_file: data/depth_training/val.jsonl
```

### `configs/inference_depth.yaml`

| Key | Value | Notes |
|-----|-------|-------|
| `ckpt_path` | `???` | **Required** — path to a checkpoint folder |
| `size` | `512` | Must match training size |
| `seed` | `42` | Reproducible generation |
| `local_files_only` | `true` | Offline mode |
| `inference.n_samples` | `1` | Images generated per input |
| `inference.num_inference_steps` | `50` | Denoising steps |
| `inference.guidance_scale` | `7.5` | CFG scale |

---

## 7 · Run Commands & Success Criteria

### Prerequisites

```
data/
  raw/             # original images
  depth_training/  # does NOT exist yet; Stage A creates it
checkpoints/local_models/
  stable-diffusion-v1-5/
  dpt-hybrid-midas/
```

Activate the conda environment before any Python command:
```bash
conda activate loradapter
```

### Stage A — Precompute Depth Maps (run once)

Dry run on 3 images first:
```powershell
python depth_map_calculations.py --data_dir data/ --image_path target --dry_run_n 3
```
Success: no errors; depth PNGs in the sibling folder `<dataset>_depth_map/`.

Full run:
```powershell
python depth_map_calculations.py --data_dir data/ --image_path target
```
Success:
- `<dataset>_depth_map/` populated (one PNG per image, mirrored structure)
- `data/depth_training/train.jsonl`, `val.jsonl`, `test.jsonl` written
- Verification output shows `0 failures`
- Dataset sizes: 639 train / 137 val / 137 test

### Stage B — Training

```powershell
python depth_training.py experiment=train_depth
```

Expected startup log:
```
[model] base  = .../stable-diffusion-v1-5
[model] depth = .../dpt-hybrid-midas
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

TensorBoard: `tensorboard --logdir outputs/train/depth/runs/`

Expected tags (confirmed by execution):
- Scalars: `train/loss`, `train/lr`, `val/loss`
- Images: `val/sample_00` … `val/sample_09`
- Tensors: `val/prompts/text_summary`

### Stage B — Inference

```powershell
python depth_inference.py \
  ckpt_path=outputs/train/depth/runs/YYYY-MM-DD/HH-MM-SS/best_model \
  inference.input_dir=data/raw
```

Success: 4-panel JPG grids written to `outputs/inference/depth/`.

---

## 8 · Known Limitations

1. **`max_train_steps` does not stop training**: Setting `+max_train_steps=N` via Hydra affects the progress-bar total and LR scheduler warmup calculation only. The batch loop runs for the full number of epochs. To stop early, send SIGINT (Ctrl+C); the signal handler finishes the current step and saves a final checkpoint.

2. **Triplication risk in preprocessing**: The depth preprocessing transform chain (SquarePad → Resize → ToTensor → Normalize) is defined in three separate places (`depth_map_calculations.py`, `depth_inference.py`, `configs/data/local_depth.yaml`). Any future change must be applied to all three simultaneously or parity will break.

3. **Per-image min-max depth**: Depth values are normalised per image. You cannot meaningfully compare depth magnitudes across images. The pipeline learns a structural prior, not a metric depth prior.

4. **Float→uint8 round-trip quantisation**: Stage A saves depth as 8-bit PNG, introducing up to 1/255 ≈ 0.004 error per pixel. The observed max diff between freshly-computed depth and saved PNG was 0.00837 (~2 uint8 levels), caused by floating-point non-determinism between sessions. This is not a preprocessing bug — it is inherent to the round-trip.

5. **`references.md §6` incorrect file reference**: Lists `src/data/local_depth.py` as a separate file. This file does not exist. `DepthJsonDataset` and `DepthJsonDataModule` live in `src/data/local.py`.

6. **OOM with experiment defaults on consumer GPU**: Experiment config uses `batch_size=4`, `gradient_accumulation_steps=4`, `gradient_checkpointing=True`, `bf16=True`. On a GPU with less than 16 GB VRAM this may OOM during backward. Use `data.batch_size=1 gradient_accumulation_steps=1` for single-GPU runs.

7. **No test-time evaluation script**: `test.jsonl` exists but there is no built-in script to run batch inference on the test split and compute quantitative metrics. `depth_inference.py` with `save_generated_only=true` produces images but not scores.

---

## 9 · Full Parameter Control

Everything you can tune, where it lives, and what changing it does. One place to look up any flag.

---

### 9.1 How Hydra overrides work

`depth_training.py` and `depth_inference.py` use [Hydra](https://hydra.cc) for configuration. The base config is `configs/train_depth.yaml`; the experiment overrides live in `configs/experiment/train_depth.yaml`. You can override ANY key at the command line without editing a file:

```powershell
# Single override
python depth_training.py experiment=train_depth epochs=3

# Multiple overrides
python depth_training.py experiment=train_depth epochs=3 val_steps=100 data.batch_size=2

# Nested keys use dot notation
python depth_training.py experiment=train_depth data.batch_size=1 lora.struct.ckpt_path=outputs/train/.../step1000

# Adding a new key that is not in any config (use + prefix)
python depth_training.py experiment=train_depth +max_train_steps=500
```

Hydra writes its own log + a copy of the resolved config to:
- `outputs/train/depth/runs/YYYY-MM-DD/HH-MM-SS/.hydra/config.yaml` — what actually ran
- `outputs/train/depth/runs/YYYY-MM-DD/HH-MM-SS/.hydra/overrides.yaml` — what you passed on the CLI

If something behaves unexpectedly, read `config.yaml` — it shows every key's final resolved value.

---

### 9.2 Stage A — `depth_map_calculations.py` CLI flags

Run once before training to precompute depth PNGs from raw images.

```powershell
python depth_map_calculations.py --data_dir data/ [flags]
```

| Flag | Default | What it does | When to change |
|------|---------|--------------|----------------|
| `--data_dir` | *(required)* | Folder containing `train.jsonl`, `val.jsonl`, `test.jsonl`. The script scans for files whose names contain "train"/"val"/"test" — non-default names like `my_train.jsonl` are found automatically. | Always set this. |
| `--dry_run_n N` | off | Process only the first N images per split. Run with `--dry_run_n 3` first, check output, then run without it. | Always use before first full run. |
| `--size` | `512` | Square side for saved depth PNGs. Must match `size` in your training config. If you change this you must rerun Stage A. | Keep 512 unless you change training resolution. |
| `--batch_size` | `4` | Images fed to the depth model at once. Higher = faster but uses more GPU memory. | Lower if you get OOM during Stage A. |
| `--image_path` | `source` | JSONL key holding the image path. Your dataset uses `target`. | Always set `--image_path target` for this dataset. |
| `--dataset_dir` | *(none)* | Optional: scan this folder for `raw_image.jpg` instead of trusting the JSONL paths to find images. Saves maps to the same sibling folder either way. | Add it if you want disk (not the JSONL) to decide which images exist. |
| `--model` | `checkpoints/local_models/dpt-hybrid-midas` | Path to the local DPT model folder OR a HuggingFace repo ID. Default is the local offline copy. | Only change if you switch depth models (breaks parity with existing depth PNGs — delete `<dataset>_depth_map/` and rerun). |
| `--local_files_only` | `True` | `True` = load model from local disk only (offline). `False` = allow HF download if model is not cached. | Set to `False` if you need to download the model for the first time. |
| `--device` | `cuda` if available | `cuda` or `cpu`. | Set to `cpu` if no GPU available (very slow). |
| `--no_skip` | off | Re-compute depth maps even if PNG already exists. By default existing PNGs are reused (fast). | Add this flag if you changed `--model` or `--size` and need to regenerate. |
| `--image_root` | *(none)* | Base prepended to **relative** image paths in the JSONL. Your JSONLs use absolute paths, so it is not needed. | Only if your JSONL stores relative image paths. |
| `--output_dir` | `<data_dir>/depth_training` | Where the output `train.jsonl`, `val.jsonl`, `test.jsonl` are written. (The depth PNGs themselves always go to the sibling `<dataset>_depth_map/` folder, derived automatically.) | Only to redirect manifest location. |

---

### 9.3 Stage B Training — `configs/experiment/train_depth.yaml`

All training behaviour is controlled from here. Override any key on the command line (see §9.1).

#### Resolution and hardware

| Key | Default (experiment) | What it does | Impact of changing |
|-----|---------------------|--------------|-------------------|
| `size` | `512` | Square canvas size. Must match the size used in Stage A. | Changing requires rerunning Stage A AND deletes training parity. Do not change mid-training. |
| `bf16` | `true` | Brain-float16 mixed precision. Cuts VRAM by ~40%, minimal quality loss on modern GPUs. | Set `false` if your GPU does not support bf16 (older cards). Training becomes slower and heavier. |
| `gradient_checkpointing` | `true` | Trade compute for memory: recomputes activations during backward instead of storing them. Saves ~1.5 GB VRAM, slows backward ~20%. | Set `false` if you have plenty of VRAM and want faster training. Required for 12 GB GPUs. |
| `gradient_accumulation_steps` | `4` | How many micro-batches to accumulate before one optimizer step. Effective batch = `data.batch_size x gradient_accumulation_steps`. | Reduce if training is too slow and you have VRAM to spare. Increase if you want a larger effective batch without more VRAM. |
| `data.batch_size` | `4` | Per-GPU micro-batch size. | Reduce to `1` or `2` if you OOM. On a 24 GB GPU you can use `8`. |
| `data.workers` | `4` | DataLoader worker processes. | Set `0` on Windows if you get multiprocessing errors. |

#### Learning rate and schedule

| Key | Default | What it does | Impact of changing |
|-----|---------|--------------|-------------------|
| `learning_rate` | `1e-4` | Peak AdamW learning rate. | Too high (>5e-4): loss spikes and diverges. Too low (<1e-5): very slow convergence. |
| `lr_scheduler` | `cosine` | LR decay schedule after warmup. `cosine` decays smoothly to ~0; `constant` holds the peak LR throughout. | Use `constant` for a quick test; `cosine` for production runs (better final quality). |
| `lr_warmup_steps` | `500` | Steps where LR ramps from 0 up to `learning_rate`. Prevents early instability. | Reduce if your dataset is small and 500 steps is a large fraction of total training. Set `0` to disable. |

#### When to save / validate

| Key | Default | What it does | Impact of changing |
|-----|---------|--------------|-------------------|
| `epochs` | `5` | Total training passes over the dataset. | More epochs = more training time and potentially better quality, but also risk of overfitting. Watch val/loss — if it rises while train/loss falls, stop earlier. |
| `val_steps` | `500` | Every N optimizer steps: compute val/loss on held-out data + update `best_model/` if improved. Cheap (no image generation). | Lower = more frequent val/loss updates in TensorBoard. Higher = faster training throughput. |
| `ckpt_steps` | `1000` | Every N optimizer steps: save weights + generate N monitoring images. Heavy (runs the diffusion model). | Lower = more disk usage + slower overall training. Reasonable range: 500–2000 for a 5-epoch run. |
| `val_batches` | `64` | How many val batches to average for val/loss. More = more accurate estimate but slower. | Reduce to `4`–`8` for smoke tests. Keep `64` for real runs. |

#### Checkpoint monitoring grid (the 50/50 split)

| Key | Default | What it does | Impact of changing |
|-----|---------|--------------|-------------------|
| `n_grid_images` | `10` | Total images in each checkpoint's monitoring grid. Split 50/50: `n_grid_images // 2` are **fixed** (same scenes every checkpoint), the remaining half are **fresh** (re-randomized each checkpoint). | More = more disk use and slower checkpoint saves. Even number recommended (clean 50/50 split). Minimum 2. |
| `grid_include_empty_prompt` | `false` | When `true`, each monitoring image gets a 4th panel: generation with **empty prompt** (pure depth conditioning, no text). Useful for seeing how much the model leans on the depth map vs the prompt. | `true` doubles image generation time per checkpoint. Set `true` for diagnostic runs; keep `false` for speed. |

The fixed half: chosen **once at training start** using OS entropy (different each run). Stored in `_fixed_val_idxs`. Every checkpoint at `step1000`, `step2000`, `step3000` shows these **same scenes** so you can directly compare how the model improves. Files are named `sample_00_fixed.jpg` … `sample_04_fixed.jpg`.

The fresh half: re-drawn at each checkpoint from the remaining val indices (never overlaps the fixed half). Files are named `sample_05_new.jpg` … `sample_09_new.jpg`. Quickly checks generalization to unseen scenes.

#### Dataset and model paths

| Key | Default | What it does | When to change |
|-----|---------|--------------|----------------|
| `data.json_file` | `data/depth_training/train.jsonl` | Training manifest. Each line: `{raw_image_path, depth_path, prompt}`. | Change to point at a different JSONL if your dataset is elsewhere. |
| `data.val_json_file` | `data/depth_training/val.jsonl` | Validation manifest. **Never use `test.jsonl` here** — test split must stay uncontaminated. | Only change to use a different val set. |
| `data.image_root` | `null` (= repo root) | Base path prepended to relative `raw_image_path` values in the JSONL. Set to `/mnt/dataset` when images are on a different drive or machine. | **Required when training on Ubuntu with images at a different path from where you generated the JSONL.** |
| `local_files_only` | `true` | `true` = load all models from local disk (fully offline). `false` = allow HF downloads. | Keep `true` for training. |
| `base_model_path` | `checkpoints/local_models/stable-diffusion-v1-5` | Local path to SD1.5. | Only if you moved the model. |
| `depth_model_path` | `checkpoints/local_models/dpt-hybrid-midas` | Local path to DPT. | Only if you moved the model. |
| `lora.struct.ckpt_path` | `null` | Resume from a previous checkpoint. Set to e.g. `outputs/train/depth/runs/.../step2000`. | Use when continuing an interrupted training run. |
| `seed` | `42` | Random seed for training noise. Does **not** affect the fixed monitoring image selection (that uses OS entropy). | Change if you want to run multiple experiments with different initialization. |
| `prompt` | `null` | If set, overrides every image's caption with this single string. | Use for single-concept fine-tuning where all images share one prompt. |
| `tag` | `depth` | Written into the TensorBoard event filename and used as the output subfolder name. | Change if running multiple experiments to keep outputs separated. |
| `ignore_check` | `true` | Skip the startup data-integrity pre-check (verifies every JSONL entry exists on disk). Skipping saves ~30 seconds. | Set `false` if you suspect your JSONL has stale paths. |

---

### 9.4 Stage B Inference — `configs/inference_depth.yaml`

```powershell
# From a JSONL manifest (recommended for batch runs):
python depth_inference.py \
  ckpt_path=outputs/train/depth/runs/YYYY-MM-DD/HH-MM-SS/best_model \
  inference.json_file=data/depth_training/test.jsonl

# Single image:
python depth_inference.py \
  ckpt_path=outputs/train/depth/runs/YYYY-MM-DD/HH-MM-SS/best_model \
  "inference.images=[data/raw/000417/raw_image.jpg]" \
  "inference.prompts=['a driving scene at night in the rain']"
```

| Key | Default | What it does | When to change |
|-----|---------|--------------|----------------|
| `ckpt_path` | *(required)* | Path to a checkpoint folder containing `struct/lora-checkpoint.pt` + `struct/mapper-checkpoint.pt`. Usually `best_model/` or a specific `checkpoint-epoch1/step1000/`. | Always set. Use `best_model/` for final evaluation. Use step checkpoints to compare across training. |
| `inference.json_file` | `null` | JSONL file to run inference on. Each line needs `raw_image_path` and `prompt`. | Use `test.jsonl` for final held-out evaluation. Never use `val.jsonl` or `train.jsonl` here (contamination). |
| `inference.images` | `[]` | Direct list of image paths (alternative to json_file). | For quick single-image tests. |
| `inference.prompts` | `[]` | Prompts for the images list. Must match length of `inference.images`. | Required when using `inference.images`. |
| `inference.output_dir` | `outputs/inference/depth/results` | Where to save generated images. Resolved from repo root (absolute paths also accepted). | Change to organize results per experiment. |
| `inference.save_generated_only` | `false` | `false` = save 4-panel grid + individual panels (original, depth, predicted, raw-depth-gen). `true` = save only the predicted image, preserving the folder structure from the JSONL. | Set `true` for clean batch evaluation where you only need the generated images. |
| `inference.n_samples` | `1` | How many images to generate per input. | `2`–`4` for diversity comparison. Multiplies inference time. |
| `inference.num_inference_steps` | `50` | Diffusion denoising steps. More = slower but sharper. | `20` for fast preview, `50` for quality, `100` for maximum quality (diminishing returns above 80). |
| `inference.guidance_scale` | `7.5` | CFG scale: how strictly the model follows the text prompt. Higher = more prompt-driven, less variation. | `3`–`5` for creative / more variation. `7.5` standard. `12`–`15` for very tight prompt adherence. |
| `size` | `512` | Must match the size used during training. | Do not change. |
| `local_files_only` | `true` | Offline model loading. | Keep `true` after first download. |

---

### 9.5 Common scenarios — exact commands

#### Smoke test (verify pipeline works, ~5 minutes)
```powershell
python depth_map_calculations.py --data_dir data/ --dry_run_n 3

python depth_training.py experiment=train_depth `
  epochs=1 val_steps=10 ckpt_steps=20 val_batches=4 n_grid_images=2 `
  "data.workers=0" ignore_check=true
```

#### Full training run (Ubuntu, 5 epochs)
```bash
python depth_training.py experiment=train_depth
```

#### Resume interrupted training
```powershell
python depth_training.py experiment=train_depth `
  "lora.struct.ckpt_path=outputs/train/depth/runs/2026-07-01/00-41-13/checkpoint-epoch2/step4000"
```

#### Reduce memory (OOM on a 12 GB GPU)
```powershell
python depth_training.py experiment=train_depth data.batch_size=1 gradient_accumulation_steps=4
```

#### Images on a different drive (Ubuntu training with images at /mnt/data)
```bash
python depth_training.py experiment=train_depth data.image_root=/mnt/data
```
The JSONL has paths like `data/raw/000417/raw_image.jpg`. With `image_root=/mnt/data`, the dataset resolves each path to `/mnt/data/data/raw/000417/raw_image.jpg`. So keep the JSONL relative paths as-is and just point `image_root` at the drive root that makes them correct.

#### More monitoring images per checkpoint (watch 10 fixed scenes)
```powershell
python depth_training.py experiment=train_depth n_grid_images=20
# → 10 fixed + 10 fresh scenes per checkpoint grid
```

#### Add empty-prompt panel to see pure depth conditioning
```powershell
python depth_training.py experiment=train_depth grid_include_empty_prompt=true
# Each monitoring image gets a 4th panel: generated with empty prompt
```

#### Evaluate on test split after training
```powershell
python depth_inference.py `
  ckpt_path=outputs/train/depth/runs/YYYY-MM-DD/HH-MM-SS/best_model `
  inference.json_file=data/depth_training/test.jsonl `
  inference.save_generated_only=true `
  inference.output_dir=outputs/inference/depth/test_eval
```

#### Generate comparison report
```powershell
python training_report.py
python training_report.py --markdown   # GitHub-flavoured Markdown
```

