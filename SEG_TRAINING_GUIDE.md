# Segmentation Pipeline — Zero-to-Hero Training Parameter Guide

**Who this file is for:** Someone who has never trained a diffusion model and wants to
understand every parameter in `configs/experiment/train_seg.yaml` from first principles.

**Read first:** `DEPTH_TRAINING_GUIDE.md` — concepts like loss function, dataset math,
gradient accumulation, bf16, and TensorBoard reading are explained there in full.
This file documents only what is **different** for segmentation, plus a complete
parameter reference with seg-specific values.

**Companion files:**
- `SEGMENTATION.md` — architecture and pipeline explanation
- `configs/experiment/train_seg.yaml` — the config you edit
- `segformer_training.py` — the training script

---

## 0. Key difference from depth — what changes in the conditioning signal

Both pipelines use the same LoRA + mapper architecture. The only differences are:

| | Depth | Segmentation |
|---|---|---|
| Conditioning signal | Grayscale depth map `[0,1]` per-image normalized | 19-class Cityscapes colour map `[0,1]` fixed palette |
| Offline calc script | `depth_map_calculations.py` → `data/raw_depth/` | `seg_map_calculations.py` → `data/raw_seg/` |
| Training manifest | `data/depth_training/*.jsonl` | `data/seg_training/*.jsonl` |
| Encoder (Stage C/live) | DPT-Hybrid-MiDaS (regression) | SegFormer-b5-Cityscapes (classifier) |
| Encoder during training | NOT called (`skip_encode=True`) | NOT called (`skip_encode=True`) |
| Mapper input | depth map → mapper | colour seg map → mapper |
| TensorBoard tag | `depth` | `seg` |
| Output panels in grid | `ORIGINAL \| DEPTH MAP \| PREDICTED` | `ORIGINAL \| SEG MAP \| PREDICTED` |

**Why `skip_encode=True` during training for both:**
The conditioning encoder (MiDaS or SegFormer) runs only at inference. During training
we use pre-saved maps from disk. This makes training ~3× faster than running the encoder
every step, and it lets the encoder stay frozen without ever touching GPU memory during training.

---

## 1. Dataset math — identical to depth, using the same numbers

```python
# 60K train / 4K val / 4K test — same as depth for fair comparison
N_TRAIN = 60_000
N_VAL   =  4_000

# Same hardware config as depth:
BATCH_SIZE          = 4     # data.batch_size
GRAD_ACCUM          = 4     # gradient_accumulation_steps
EFFECTIVE_BATCH     = 16

STEPS_PER_EPOCH     = 3_750     # ceil(60000 / 16)
EPOCHS              = 5
TOTAL_STEPS         = 18_750    # 5 × 3750

# val/loss checks: 3750 / 500 = 7–8 per epoch
# step checkpoints: 3750 / 1000 = 3–4 per epoch
```

**Why we keep the math identical to depth:**
The whole purpose of this project is to **compare** depth conditioning vs segmentation
conditioning under identical conditions. If the datasets, batch sizes, or training lengths
differ, the comparison becomes unfair. Keep these numbers the same as depth and only
change what segmentation genuinely requires (encoder, data paths).

**Don't hand-compute this table for your own run — use the advisor script.**
The numbers above are a worked example, not a value to copy blindly.
`recommend_training_params.py` (repo root) counts your REAL manifest line counts
and detects your GPU's VRAM, then prints the matching `batch_size` /
`gradient_accumulation_steps` / `steps_per_epoch` / `val_steps` / `ckpt_steps`
for the dataset that's actually on the machine you run it on — see §10 for the
exact command. `--data_dir` is required (no default) because the segmentation
dataset (real-world photos) and the Grounded-SAM dataset (CARLA renders) are
different datasets with different counts — run it once per pipeline.

**`epochs` is an upper bound, not a target — early stopping already decides
the real stop point.** `early_stop_patience: 3` is set in the shared base
config `configs/train_seg.yaml` (used by both `experiment=train_seg` and
`experiment=train_grounded_sam`, since both run through `segformer_training.py`):
if val/loss produces no new best for 3 full epochs, training stops itself
(`segformer_training.py`, the "EARLY STOPPING" block near the end of the epoch
loop) — `best_model/` is already saved at that point, so nothing is lost.
Neither pipeline has an empirical convergence curve yet at real (60K-image)
scale, so don't try to predict the exact right epoch count — set `epochs` to
a generous ceiling (e.g. 15-20) and let `early_stop_patience` do the actual
work of stopping when val/loss plateaus.

---

## 1a. The loss, from scratch — what the model is actually learning [ADDED 2026-07-19]

You cannot tune a hyperparameter sensibly without knowing what number you're
reacting to. This section explains the loss itself, line by line from the
real code, then how to read it to make decisions — no formula tells you "the
right epoch count" in advance; you read the loss curve from a real run.

### 1a.1 What training actually does (the one-paragraph version)

Take a real image. Add a random amount of noise to it. Ask the model to guess
**exactly what noise was added**. Train it to guess correctly, at every
possible noise strength (a little noise, up to almost-pure static). If a
model can always correctly identify "what noise is this," it can also run the
process **backward**: start from pure noise and repeatedly subtract its best
guess, and a real image emerges. That backward process is generation.
Training only ever does the easy, forward direction — we always know the true
noise, because we added it ourselves. That's the entire trick.

### 1a.2 The real code, step by step

`segformer_training.py:697` calls:
```python
model_pred, loss, x0, _ = model.forward_easy(imgs, prompts, cs, skip_encode=True, ...)
```
`forward_easy` (`src/model.py:571-603`) sets up one training step:
```python
latents, c = self.get_input(imgs, prompts)          # image -> compressed latent, text -> embedding
noise = torch.randn_like(latents)                    # random static, same shape as latents
timesteps = torch.randint(0, num_train_timesteps, (bsz,), device=latents.device)  # random noise LEVEL per image
```
- `get_input` (`src/model.py:429-471`): `latents = self.vae.encode(imgs).latent_dist.sample()`
  compresses your 512×512 image into a small 64×64×4 latent (a VAE, trained
  separately, ships with SD1.5). Everything below operates on this
  compressed representation, not raw pixels — purely a compute-saving trick
  from the original Stable Diffusion paper.
- `noise`: literally `torch.randn_like(latents)` — pure Gaussian noise. This
  is "the static" from the analogy above.
- `timesteps`: a random integer 0–999 per image. This is "how much noise."
  0 ≈ barely noisy (easy). 999 ≈ almost pure noise (hard — this is where
  real generation actually starts from). A **different random level every
  single training step**, so across many steps the model learns to denoise
  at every level, not just one.

Then `forward` (`src/model.py:473-569`) does the actual work:
```python
noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)   # :484
```
Blend `noise` into the real `latents` at the strength `timesteps` dictates.

```python
for i, (encoder, dp, mapper, lora_c) in enumerate(zip(encoders, self.dps, mappers, cs)):
    cond = lora_c if skip_encode else encoder(lora_c)   # :526-535 — your seg map, unchanged here
    mapped_cond = mapper(cond)                            # FixedStructureMapper15
    dp.set_batch(mapped_cond)                              # :537 — handed to the shared DataProvider
```
Your segmentation map gets pushed into the `DataProvider` here. Every
`NewStructLoRAConv` layer inside the UNet reads it from there on the very
next line — see [LORA_ARCHITECTURE.md](LORA_ARCHITECTURE.md) for the full
trace of what happens to it once it's inside the UNet.

```python
model_pred = self.unet(noisy_latents, timesteps, encoder_hidden_states=prompt_embeds).sample   # :540-542
```
**The "guess the noise" step.** The UNet sees the noisy latent, the noise
level, the text, and — via every LoRA layer reading the DataProvider — your
segmentation map. `model_pred` is its guess at what `noise` was added.

```python
target = noise                                        # :560-561 — the REAL noise (we made it, we know it exactly)
loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")   # :567
```
**This is the loss**: mean squared error between the model's guess and the
real noise — `mean((model_pred - noise)^2)`. Lower loss = better
noise-guesses = better denoising = better generation. `loss.backward()`
(`segformer_training.py:704`) uses this single number to compute gradients and
update every trainable weight (the LoRA `A`/`B`/`beta`/`gamma` parameters —
the frozen UNet weights never change, per `add_lora_to_unet`).

### 1a.3 Which loss to actually watch: `train/loss` vs `val/loss`

Two different losses get computed, answering different questions:

- **`train/loss`** (`segformer_training.py:718,724`) — the MSE above, computed on
  the batch you just trained on, WITH gradients on, logged every step.
  Inherently noisy (small batches, random noise/timestep each time) — watch
  the trend over dozens of steps, not any single value.
- **`val/loss`** (`_segmentation_validation_loss`, `segformer_training.py:269-332`)
  — the SAME formula, computed on held-out images the model never trains on,
  with `torch.no_grad()` (no learning happens), averaged over several
  batches, only at `val_steps` intervals. Its own docstring states exactly
  why it exists: *"(a) if it diverges from train/loss, you're overfitting.
  (b) an objective best_model criterion — not a biased training-loss
  average."*

**Watch `val/loss` to make decisions.** It's what `best_model`/early-stopping
(`best_loss` tracking at `segformer_training.py:509`, the early-stop check at
`:778-793`) already use automatically. `train/loss` is only a sanity check —
is it decreasing, is it NaN — not a decision signal.

### 1a.4 Tuning hyperparameters *by watching the loss curve*

This is read-after-a-real-run, not computed in advance — there's no formula
for "the right epoch count" that doesn't require actually training:

| What you observe | What it means | What to change |
|---|---|---|
| `train/loss` is `NaN` or explodes | Numerical instability / LR too aggressive | Lower `learning_rate`; gradients are already clipped to `max_norm=1.0` (`segformer_training.py:711`) as a safety net |
| `train/grad_norm` stays pinned at 1.0 for a long time | Gradients are being clipped constantly — the optimizer wants bigger steps than allowed | Normal early on; if it never relaxes, LR may be too high for this effective batch |
| `val/loss` decreases then **flattens** | Model has converged on this data — more epochs teach it nothing new | Nothing to do — `early_stop_patience` (§1) catches this automatically |
| `val/loss` **increases** while `train/loss` keeps falling | Overfitting — memorizing training images instead of generalizing | More epochs make this WORSE. Real fix is more/more-diverse data, not a hyperparameter |
| `val/loss` still steadily falling when `epochs` ceiling is hit | Ceiling was too low, model hadn't finished learning | Rerun with a higher `epochs`, resuming via `lora.struct.ckpt_path` — no need to restart from step 0 |
| `val/loss` jumps around a lot between checks | Too few `val_batches` averaged, or a small/noisy val set | Increase `val_batches` — this is a measurement-noise artifact, not a real problem to fix with training hyperparameters |

### 1a.5 The actual workflow, for however many images YOU have

Say you have `N_train` training images, `N_val` validation images (any
numbers — this generalizes, it isn't specific to 60K/4K). Nothing below
requires you to already know the answer:

1. **Run the advisor script on the real data, on the real machine**:
   ```
   python recommend_training_params.py --data_dir data/seg_training --epochs 15
   ```
   It reads your actual `N_train`/`N_val`/`N_test` by counting JSONL lines
   (`_count_images`, `recommend_training_params.py:40-45` — no guessing) and
   detects your actual GPU's VRAM (`torch.cuda.get_device_properties`,
   `:104-106`). From those two REAL numbers it computes, as plain arithmetic:
   - `batch_size`/`gradient_accumulation_steps` sized to your VRAM, holding
     the SAME validated effective batch (16) this project's `learning_rate`
     was tuned at (`recommend_batch_and_accum`, `:48-73`) — LR is never
     auto-scaled here because no scaling behaviour has been verified on this
     codebase; changing effective batch without also re-validating LR is an
     experiment, not a safe default.
   - `steps_per_epoch = ceil(N_train / effective_batch)` and
     `total_steps = steps_per_epoch * epochs` — pure arithmetic on YOUR
     `N_train`, not an assumed 60K.
   - `val_steps`/`ckpt_steps` sized to check ~7 times and checkpoint ~3-4
     times per epoch, scaled to your `steps_per_epoch`.
   `--epochs 15` here is **your chosen ceiling**, not a computed answer — see
   below for why that number can't be computed.
2. **Paste the printed block into `configs/experiment/train_seg.yaml`.**
3. **Leave `early_stop_patience` at its default (3)** unless you have a
   specific reason to change it — e.g. if `val/loss` looks genuinely noisy
   (small val set, per the table above) rather than truly plateaued, a
   larger patience avoids stopping on a temporary blip.
4. **Launch training**, optionally watching
   `tensorboard --logdir outputs/train/seg/runs/` live (§5) for `val/loss`.
5. **After it finishes** (either the epoch ceiling, or early-stop firing),
   read `best_model/info.txt` (`segformer_training.py:580-588` writes it: epoch,
   step, and the psnr/ssim/miou metrics for the checkpoint that won). **This
   is your empirically-discovered right epoch count for THIS dataset** —
   discovered by running, not predicted in advance. No dataset size or
   formula can tell you this before you actually train; task difficulty and
   data diversity matter just as much as `N_train`, and neither is knowable
   without a real run.
6. **If early-stop never fired** (the run hit your `epochs` ceiling still
   improving), rerun with a higher ceiling, resuming from the last
   checkpoint (`lora.struct.ckpt_path=<path to checkpoint-epochN>`) rather
   than restarting from scratch.

---

## 2. Every parameter in `configs/experiment/train_seg.yaml`

Parameters identical to depth are not repeated here — see `DEPTH_TRAINING_GUIDE.md §2`
for full explanations of `size`, `learning_rate`, `lr_scheduler`, `lr_warmup_steps`,
`epochs`, `gradient_accumulation_steps`, `data.batch_size`, `bf16`,
`gradient_checkpointing`, `val_steps`, `ckpt_steps`, `val_batches`, `seed`, `prompt`,
`local_files_only`, `ignore_check`, `lora.struct.ckpt_path`.

Below are seg-specific parameters and values that differ.

---

### 2.1 `seg_model_path` and `seg_model_name` — the locked SegFormer-b5 model

```yaml
seg_model_name: nvidia/segformer-b5-finetuned-cityscapes-1024-1024
seg_model_path: checkpoints/local_models/segformer-b5-cityscapes
```

**What it is:** the frozen SegFormer-b5 encoder that produces segmentation maps.
Used at inference (`segformer_inference.py`) and in Stage C (`seg_map_calculations.py`).
NOT used during training (skip_encode=True).

**Why b5 is locked (never use b0):**
SegFormer-b5 has 82M parameters in its MiT-B5 backbone. SegFormer-b0 has 3.7M.
For driving scenes (pedestrians, poles, traffic lights), b0 misclassifies thin structures
that matter for structural conditioning. See `SEGMENTATION.md §4` for full evidence.

**This model is NOT loaded during training.** Only at Stage C and inference.
The config key exists so `segformer_inference.py` knows where to find it.

---

### 2.2 `data.json_file` and `data.val_json_file` — seg manifests

```yaml
data:
  json_file:     data/seg_training/train.jsonl
  val_json_file: data/seg_training/val.jsonl   # NEVER test.jsonl here
```

**What it is:** paths to the training and validation manifests built by `seg_map_calculations.py`.
Each line has three fields:
```json
{"raw_image_path": "data/raw/000417/raw_image.jpg",
 "seg_path":       "data/raw_seg/000417/raw_image.png",
 "prompt":         "a driving scene with pedestrians"}
```

**`seg_path` points to the raw class-ID PNG** (8-bit grayscale, values 0–18).
The dataset class (`SegJsonDataset`) reads this PNG, applies NEAREST resize, then
colourises with `SEG_CITYSCAPES_PALETTE` at load time. It does NOT call SegFormer.

**Warning — NEAREST resize is required for class-ID maps:**
```python
# src/data/local_seg.py — _load_seg_colormap()
ids_pil = ids_pil.resize((self.size, self.size), Image.NEAREST)
# NOT Image.BILINEAR — bilinear would blend class IDs and invent new classes
```

**Why test.jsonl must never appear here:**
`test.jsonl` is the final held-out evaluation set. Reading it during training would
let the val/loss metric see test data → contamination → your final evaluation is worthless.
The config enforces this: only `train.jsonl` and `val.jsonl` are ever listed here.

---

### 2.2a Getting a manifest without building one by hand — `seg_map_calculations.py`'s ad-hoc modes [ADDED 2026-07-20]

Beyond the `--data_dir`/`--dataset_dir` full-dataset modes described elsewhere in this
repo, the calc script has two smaller modes for a handful of images — useful when you
just want to try inference on a new photo, or build a small manifest without hand-editing
JSONL:

```bash
# ONE image -> map saved BESIDE it as <stem>_seg_map.png, same folder.
# Prints a ready-to-paste segformer_inference.py command for the pair.
python seg_map_calculations.py --image path/to/frame.jpg

# ONE manifest (only needs raw_image_path + optional prompt per line) ->
# maps saved into a SIBLING <images_root>_seg_map/ folder (mirrored
# structure, <stem>_seg_map.png each) -> writes <stem>_seg.jsonl BESIDE
# the input, with the project-standard keys, self-verified before you
# can trust it for training or inference.
python seg_map_calculations.py --json_file data/my_frames.jsonl
```

Both modes, and every other mode of this script, now default `--image_path` to
`raw_image_path` (previously `source`) — the project-standard key used everywhere,
training and inference, both pipelines. Pass `--image_path target` (or whatever key
your manifest actually uses) to override.

`--json_file` mode reads with `utf-8-sig`, so a manifest saved by a Windows editor
(which may prepend a BOM) parses correctly instead of crashing on line 1.

---

### 2.3 `data.image_root` — images on a different drive (seg version)

```yaml
data:
  image_root: null   # null = repo root
```

Same as depth. For Ubuntu training with images at `/mnt/dataset`:
```yaml
data:
  image_root: /mnt/dataset
  json_file:     data/seg_training/train.jsonl   # JSONL stays in repo
  val_json_file: data/seg_training/val.jsonl
```

The dataset class resolves `data/raw/000417/raw_image.jpg` as
`/mnt/dataset/data/raw/000417/raw_image.jpg`. The `seg_path` field in the JSONL
(`data/raw_seg/000417/raw_image.png`) is resolved the same way — so seg PNGs
must also be at `/mnt/dataset/data/raw_seg/`.

---

### 2.4 `n_grid_images` and `grid_include_empty_prompt` — seg grid settings

```yaml
n_grid_images: 10                # 5 fixed + 5 fresh (same as depth)
grid_include_empty_prompt: false # default off; enable to see "RAW SEG GEN" panel
```

**The 4th panel for segmentation — "RAW SEG GEN":**
When `grid_include_empty_prompt: true`, each monitoring image gets a 4th panel:
```
[ORIGINAL | SEG MAP | PREDICTED (with prompt) | RAW SEG GEN (empty prompt)]
```

`RAW SEG GEN` shows what the model generates with an empty text prompt — pure
segmentation conditioning with no text influence. This is useful for answering:
"is the model actually following the segmentation map, or is it relying on the text?"

If `RAW SEG GEN` shows similar spatial structure to `PREDICTED`, the segmentation
conditioning is working — text is enhancing but not overriding the structural signal.

**Reading the SEG MAP panel:**
The segmentation colour map uses the Cityscapes palette:
```
Purple  (128, 64, 128)  = road          ← should cover most of the lower frame
Cyan    (70, 130, 180)  = sky           ← should cover most of the upper frame  
Green   (107, 142, 35)  = vegetation
Deep blue (0, 0, 142)  = car
Red     (220, 20, 60)   = person
```
If the SEG MAP looks like a uniform blob instead of distinct coloured regions,
the SegFormer encoder is not producing correct segmentation — check the model files.

---

### 2.5 `tag` — separates seg outputs from depth outputs

```yaml
tag: seg
```

**What it does:**
- TensorBoard events go to `outputs/train/seg/runs/` (separate from depth's `outputs/train/depth/`)
- The tfevents hostname is set to `"seg"` (not the machine hostname)
- You can run `tensorboard --logdir outputs/train/` and see BOTH pipelines labeled

**Never change this** when running seg training. If you run multiple seg experiments,
use CLI overrides to add a sub-tag:
```powershell
python segformer_training.py experiment=train_seg tag=seg_lr3e4
# outputs go to outputs/train/seg_lr3e4/runs/...
```

---

### 2.6 `lora.struct.ckpt_path` — resume a seg training run

```yaml
lora:
  struct:
    ckpt_path: null
```

Identical to depth. CLI usage:
```powershell
python segformer_training.py experiment=train_seg \
  "lora.struct.ckpt_path=outputs/train/seg/runs/2026-07-01/00-47-30/checkpoint-epoch2/step7500"
```

---

## 3. The seg-specific preprocessing chain — what makes it different from depth

During training, the raw image goes through this exact chain
(`configs/data/local_seg.yaml` + `src/data/local_seg.py`):

```python
# ── RGB IMAGE (same as depth) ──────────────────────────────────────────────── #
# configs/data/local_seg.yaml — transform list:
SquarePad()                           # pad to square (flat local-mean fill, no crop)
Resize((512, 512))                    # square → 512×512
ToTensor()                            # uint8 [0,255] → float [0,1]
Normalize(mean=[0.5]*3, std=[0.5]*3)  # [0,1] → [-1,1]
# → batch["jpg"]: [B, 3, 512, 512] in [-1,1]

# ── SEG MAP (unique to segmentation) ──────────────────────────────────────── #
# src/data/local_seg.py — SegJsonDataset._load_seg_colormap():
Image.open(seg_path).convert("L")         # 8-bit class-ID PNG (values 0–18)
ids_pil.resize((512, 512), Image.NEAREST) # NEAREST resize — never bilinear
ids = torch.from_numpy(np.asarray(ids_pil, dtype=np.int64))  # [512, 512] long
colour = seg_colorize_ids(ids, palette)   # palette lookup → [3, 512, 512] in [0,1]
# → batch["seg"]: [B, 3, 512, 512] in [0,1]
```

**Key difference from depth:**
Depth preprocessing applies a `Resize` that can use bilinear (fine for continuous values).
Seg preprocessing MUST use `NEAREST` — bilinear on class IDs produces non-existent classes.
This is enforced in `_load_seg_colormap()` and cannot be overridden via config.

---

## 4. Recommended configs for different GPU sizes — seg edition

### 4.1 12 GB GPU (default — matches depth exactly for fair comparison)

```yaml
# This IS configs/experiment/train_seg.yaml — shown here for clarity
gradient_checkpointing: true
gradient_accumulation_steps: 4   # effective batch 16
size: 512
data:
  batch_size: 4
bf16: true
learning_rate: 1.0e-4
lr_warmup_steps: 500
epochs: 5
val_steps: 500
ckpt_steps: 1000
val_batches: 64
n_grid_images: 10
grid_include_empty_prompt: false
tag: seg
```

### 4.2 8 GB GPU

```yaml
gradient_checkpointing: true
gradient_accumulation_steps: 8
data:
  batch_size: 1
bf16: true
learning_rate: 7.07e-5   # sqrt(8/16) × 1e-4
lr_warmup_steps: 500
epochs: 5
val_steps: 500
ckpt_steps: 1000
tag: seg
```

### 4.3 24 GB GPU

```yaml
gradient_checkpointing: false
gradient_accumulation_steps: 2
data:
  batch_size: 8
bf16: true
learning_rate: 1.0e-4   # effective batch still 16
lr_warmup_steps: 500
epochs: 5
val_steps: 500
ckpt_steps: 1000
tag: seg
```

---

## 5. Reading TensorBoard for segmentation

```powershell
tensorboard --logdir outputs/train/seg/runs/
# or both pipelines together:
tensorboard --logdir outputs/train/
```

Expected scalars (same tags as depth, different values):

| Scalar | What to watch |
|---|---|
| `train/loss` | Should decrease from ~0.15 to ~0.12–0.13 over 5 epochs |
| `val/loss` | True quality signal — should track train/loss without diverging |
| `train/grad_norm` | Should stay near 0.002, occasionally spikes to 0.003 |
| `train/epoch` | Fractional epoch number for context |
| `train/lr` | Should show cosine decay from 1e-4 to ~0 |

**Expected val/loss for segmentation vs depth:**
Both pipelines should reach similar val/loss ranges (~0.13–0.15 after 5 epochs)
because the loss function is the same (epsilon-prediction MSE). The comparison between
the two pipelines is done via:
1. Final val/loss (lower = better conditioning)
2. `python training_report.py` — generates the side-by-side comparison table
3. Visual inspection of checkpoint grid images

---

## 6. Reading the segmentation checkpoint monitoring grid

Each checkpoint-grid image shows three panels side by side:

```
┌──────────────┬──────────────┬──────────────┐
│  ORIGINAL    │   SEG MAP    │  PREDICTED   │
│              │  (palette    │  (generated  │
│  raw input   │   colours)   │   image)     │
└──────────────┴──────────────┴──────────────┘
```

**What to look for at different training stages:**

**Step 500–1000 (early):**
- PREDICTED will look like a rough driving scene but without specific structure
- The segmentation map may have limited influence — road/sky broad regions may start aligning
- The colour palette regions in SEG MAP should be distinct and recognisable

**Step 3000–7000 (mid):**
- PREDICTED should show road-sky boundary alignment with SEG MAP
- Large classes (road = purple, sky = blue) should be reproduced in PREDICTED
- Person/car colours in SEG MAP should appear in correct spatial locations

**Step 10000+ (late):**
- PREDICTED should closely follow the SEG MAP class regions
- Scene category (city street, highway, etc.) should match the original
- Fine details (individual pedestrian shapes) may still be blurry — this is acceptable

**Comparing to depth grids:**
At the same training step, compare depth's grid vs seg's grid on the same scene.
Depth conditioning preserves 3D spatial structure (foreground/background).
Segmentation conditioning preserves semantic layout (which class is where).
Neither is "better" — they condition on different aspects of the scene.

---

## 7. Segmentation-specific problems and fixes

| Problem | Symptom | Fix |
|---|---|---|
| SEG MAP is uniform colour | All pixels same class in grid | Check segformer-b5 model files in `checkpoints/local_models/segformer-b5-cityscapes/` |
| NEAREST interpolation error | `AttributeError: NEAREST` | Pillow version issue; run `pip install -U Pillow` |
| OOM on Stage C (seg calc) | CUDA OOM during `seg_map_calculations.py` | Reduce `--batch_size 2` or `--batch_size 1` during calc |
| Wrong class IDs in PNG | `ValueError: class id 255` in verifier | Wrong seg model (b5 vs b0 mismatch); rerun Stage C with correct model |
| Seg path missing from manifest | `KeyError: seg_path` on training start | Rerun `seg_map_calculations.py --data_dir data/` to rebuild manifests |
| Colour palette mismatch | Generated image has wrong class colours | `SEG_CITYSCAPES_PALETTE` in `seg_encoder.py` was modified; restore from git |
| Val/loss worse than depth | Seg val/loss stuck above depth's | Normal at early steps; both should converge to similar range by epoch 3 |
| PREDICTED ignores seg map | Generated image looks random | `skip_encode` may be False; check `segformer_training.py` batch["seg"] path |

---

## 8. Comparing depth vs segmentation results

After training both pipelines to completion, run:

```powershell
python training_report.py
```

This generates a side-by-side table from both runs' TensorBoard logs:

```
  Metric                                  DEPTH           SEGMENTATION
  ────────────────────────────────────    ──────────────  ──────────────
  Best val/loss (best_model/)             0.132           0.134
  Train loss at epoch 5                   0.089           0.093
  Runtime                                 11h 24m         11h 31m
  Checkpoints saved                       22              22
  Grid images generated                   220             220
  Inference status                        1 grid generated  1 grid generated
```

**Interpreting the comparison:**
- Lower val/loss = the conditioning signal is being used more effectively
- Similar val/loss = both conditionings are roughly equally useful
- Look at the grid images: depth preserves 3D structure; segmentation preserves semantic layout

---

## 9. Quick reference — config values for 60K / 4K / 4K

```yaml
# WORKED EXAMPLE for 60,000 train / 4,000 val / 4,000 test on a 12 GB GPU
# (effective batch = 16). This is illustrative, not a value to paste blindly —
# see §10 for the command that computes YOUR actual numbers from YOUR real
# manifests once they're in place at data/seg_training/*.jsonl.
# Run: python segformer_training.py experiment=train_seg

size: 512
learning_rate: 1.0e-4
lr_scheduler: cosine
lr_warmup_steps: 500               # 13% of first epoch (500 / 3750)
epochs: 15                         # UPPER BOUND — early_stop_patience:3 (below)
                                    # decides the real stop point, see §1
gradient_checkpointing: true       # required for 12 GB GPU
gradient_accumulation_steps: 4     # 4 bsz × 4 accum = 16 effective batch
early_stop_patience: 3             # inherited from configs/train_seg.yaml base —
                                    # no new best val/loss for 3 epochs -> stop
data:
  batch_size: 4                    # max for 12 GB with checkpointing + bf16
  val_batch_size: 4
  workers: 4                       # set 0 on Windows if multiprocessing crashes
  json_file:     data/seg_training/train.jsonl   # seg manifests (not depth)
  val_json_file: data/seg_training/val.jsonl     # NEVER test.jsonl here
val_steps: 500                     # 7-8 val/loss checks per epoch
ckpt_steps: 1000                   # 3-4 checkpoints per epoch
val_batches: 64                    # 64 × 4 = 256 val images per check
n_grid_images: 10                  # 5 fixed + 5 fresh scenes per checkpoint
grid_include_empty_prompt: false   # true → add "RAW SEG GEN" panel (doubles gen time)
bf16: true
seed: 42
prompt: null                       # use per-image captions from JSONL
local_files_only: true
ignore_check: true                 # false → verify all seg_path PNGs exist on disk
tag: seg                           # DO NOT change — separates seg from depth in TensorBoard

# Model paths (locked — do not change):
seg_model_name: nvidia/segformer-b5-finetuned-cityscapes-1024-1024
seg_model_path: checkpoints/local_models/segformer-b5-cityscapes   # b5, NOT b0

# GPU scaling:
# 8 GB:  batch_size=1, gradient_accumulation_steps=16, learning_rate=7e-5
# 24 GB: batch_size=8, gradient_accumulation_steps=2, gradient_checkpointing=false
# 4×GPU: batch_size=4, gradient_accumulation_steps=1, learning_rate=2e-4
```

---

## 10. The full segmentation training checklist

Before running `python segformer_training.py experiment=train_seg` **at real,
full-capacity scale** (not the local 913-image smoke-test set):

```
[ ] Stage C complete: seg_map_calculations.py ran without errors
      → data/raw_seg/ exists with your real PNG count
      → data/seg_training/train.jsonl, val.jsonl, test.jsonl all show [PASS]
      → these are the REAL manifests, placed at data/seg_training/ on the
        machine that will actually train (this is DIFFERENT from the local
        913-image data/train.jsonl/val.jsonl/test.jsonl test set, which is
        not real training data and needs no action)

[ ] Model present: checkpoints/local_models/segformer-b5-cityscapes/ exists
      → contains config.json, preprocessor_config.json, pytorch_model.bin
      → NOT segformer-b0-cityscapes

[ ] Base model present: checkpoints/local_models/stable-diffusion-v1-5/ exists

[ ] Config correct: configs/experiment/train_seg.yaml has
      → json_file: data/seg_training/train.jsonl   (not depth_training)
      → val_json_file: data/seg_training/val.jsonl  (not test.jsonl)
      → tag: seg

[ ] Hyperparameters sized for YOUR real dataset — run the advisor script,
    on the machine where the real manifests live (it only reads local
    files + detects the local GPU, so it must run there, not here):
      python recommend_training_params.py --data_dir data/seg_training --epochs 15
    Paste the printed batch_size / gradient_accumulation_steps / val_steps /
    ckpt_steps into configs/experiment/train_seg.yaml. `--epochs` here is
    the ceiling you're choosing to allow (§1) — early_stop_patience:3
    (already set in the shared base config, no action needed) decides when
    training actually stops.

[ ] (Optional) Smoke test passes:
    python segformer_training.py experiment=train_seg \
      epochs=1 val_steps=10 ckpt_steps=20 n_grid_images=2 "data.workers=0"
```

(Grounded-SAM has its own copy of this checklist on the `grounded_sam` branch
of this repo — a separate pipeline with a different real dataset, real-world
photos here vs. CARLA renders there, so its hyperparameters are sized
separately and shouldn't be reused across the two.)

---

## 11. Running inference — output layout and the run recipe [ADDED 2026-07-20]

`segformer_inference.py` never computes a seg map — every call must supply a
`seg_path` (single-image via `inference.seg_maps=[...]`, or a manifest via
`inference.json_file=...`; `raw_image_path`/`inference.images` is optional,
display-only). This is unconditional — there's no live-SegFormer inference
mode any more (see the file's own module docstring, "CHANGED 2026-07-17").

**Every run gets its own timestamped folder.** `inference.output_dir` is a
*base* path — the actual outputs land in
`<output_dir>/<YYYY-MM-DD_HH-MM-SS>/`, a new one each run, so two runs (even
with the same `output_dir`) can never overwrite or mix each other's files:

```bash
python segformer_inference.py \
    ckpt_path=outputs/train/seg/runs/2026-01-01/00-00-00/best_model \
    "inference.seg_maps=[data/raw_seg/000888/000888_seg_map.png]" \
    "inference.images=[data/raw/000888/raw_image.jpg]" \
    "inference.prompts=['two windows on a brick building with vines']" \
    inference.output_dir=outputs/inference/seg/results
# -> outputs/inference/seg/results/2026-07-20_14-32-05/  (this run's files)
```

**`run_params.txt`** is written into that folder *before* generation starts
(so even a crashed run leaves it behind), recording the complete recipe:
checkpoint path, seed, size, `num_inference_steps`, `guidance_scale`,
`conditioning_kernel_size` (softening kernel), `lora_scale_start`/`_end`/
`_decay_start_frac` (conditioning-strength decay — see `model.py`'s
`sample_easy` docstring), the resolved base model, and every
`seg_path | raw_image_path | prompt` triplet processed. Any result folder is
therefore self-documenting weeks later, and the exact command can be
reconstructed from it without guessing what settings produced a given image.
