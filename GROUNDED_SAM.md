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

For exactly where and how the LoRA blocks sit inside the frozen SD1.5 UNet —
which layers get wrapped, the FiLM math, how a conditioning map fans out
across the UNet's resolutions — see
**[LORA_ARCHITECTURE.md](LORA_ARCHITECTURE.md)**. That mechanism (`src/lora.py`,
`src/model.py`) is identical for Grounded-SAM and SegFormer; only the encoder
that *produces* the colour map (§5.1 below vs. live SegFormer) differs.

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
  the class-definition file. **LOCKED FINAL (2026-07-16, user decision):** CARLA's
  official 29-class semantic-segmentation taxonomy (IDs 0–28: `Unlabeled` +
  the 19 Cityscapes classes shifted +1 + 9 CARLA extras — Static, Dynamic,
  Other, Water, RoadLine, Ground, Bridge, RailTrack, GuardRail), with CARLA's
  official RGB colours, taken verbatim from the
  [CARLA sensor reference](https://carla.readthedocs.io/en/latest/ref_sensors/#semantic-segmentation-camera)
  (CARLA 0.9.14+). Not to be changed.

- **[configs/experiment/train_grounded_sam.yaml](configs/experiment/train_grounded_sam.yaml)**,
  **[configs/data/local_grounded_sam.yaml](configs/data/local_grounded_sam.yaml)**,
  **[configs/lora/encoder/grounded_sam.yaml](configs/lora/encoder/grounded_sam.yaml)**
  — the config that points training at your class set + manifests.

- Shared loader **[src/data/local_seg.py](src/data/local_seg.py)** gained a
  `classes_file` (build palette from your classes) and manifest-key flags
  `image_key` / `seg_key` / `prompt_key` (so your JSONL can use any key names).

### 5.0a The REAL mask format, and how the loader handles it [ADDED 2026-07-20]

The actual masks this pipeline trains on (user-confirmed 2026-07-20 via a
`check_seg_map_format.py` scan of a real file) are:

| Property | Value |
|---|---|
| File format | PNG (lossless — good, class ids survive exactly) |
| PIL mode / dtype | `I;16` / uint16 (16-bit single channel) |
| Size | **1280 x 800 — NOT square** |
| Pixel values | raw CARLA class ids (e.g. 0,1,3,6,9,11,14,15,24 in the scanned file) |

Two loader behaviours exist specifically because of this format (both in
`SegJsonDataset._load_seg_colormap` and mirrored in
`seg_inference._load_seg_map`, kept byte-identical — verified by execution
2026-07-20 with the real dataset class, the real CARLA palette, a real
1280x800 RGB, and a synthetic mask saved in this exact I;16 format):

1. **Raw pixel read, no `.convert("L")`.** `np.asarray(Image.open(path))`
   reads I;16 and 8-bit L masks identically. `I;16 -> L` conversion happens
   to be exact on this machine's Pillow (tested), but the behaviour has
   differed across Pillow versions — reading raw removes that risk on the
   training machine entirely.
2. **Non-square maps are LETTERBOXED, never stretched.** The RGB image is
   squared by `SquarePad` (letterbox, pad top+bottom). Before this fix the
   ID map was NEAREST-*stretched* to 512x512 — a geometry mismatch that
   misaligned conditioning vs target by up to **96 px (18.8% of the frame)**
   at the top and bottom of the image (0 px only at mid-frame; measured by
   execution 2026-07-20). Now the map is padded with `pad_id`
   (config `pad_id` in `local_grounded_sam.yaml`, `seg_pad_id` in
   `inference_grounded_sam.yaml`; default **0 = CARLA Unlabeled**) using
   SquarePad's exact rounding, so map and image share one square grid.
   Training pairs are pixel-aligned; a boundary at mask row 400/800 lands at
   square row 256 on both paths (asserted in the verification run).

Consequence for the model: the pad band is now consistently `Unlabeled` in
the conditioning and a flat fill colour in the target — a learnable,
consistent mapping (previously the map had hallucination-free content
stretched over the band while the target showed a flat fill: an impossible
lesson that degraded everything near the top/bottom edges).

### 5.1a Generation-quality fix for "fits the shape but doesn't know how it looks"
This exact symptom was root-caused (flat segmentation regions carry no
appearance information through a 1×1-conv FiLM conditioning path — see full
writeup in [LORA_ARCHITECTURE.md §8](LORA_ARCHITECTURE.md#8--why-this-design-produces-fits-the-shape-but-doesnt-know-how-it-looks))
and fixed with two inference-time knobs (`lora_scale_start/end` decay +
`conditioning_kernel_size` edge-softening) added to `sample_easy` in
`src/model.py`. Both are **architecture-level**, not SegFormer-specific, so they
apply identically to Grounded-SAM once its live encoder (Tier 2, below) exists —
nothing extra to build here when that day comes.

### 5.1b Inference now works TODAY for a Grounded-SAM-produced map [ADDED 2026-07-17]
Separate change, same day: `seg_inference.py` was rewritten (user decision) so
inference **always uses a provided segmentation map and never computes one
live** — see `seg_inference.py`'s own module docstring for the full mechanism
(this is also the config `configs/inference_grounded_sam.yaml` on this branch
uses by default). This directly unlocks something that wasn't possible
before: **you can generate from a Grounded-SAM map right now**, via
`inference.seg_maps=[...]`, once you have a trained `train_grounded_sam`
checkpoint — no Tier 2 required for that. What Tier 2 (below) still gates is
narrower than it looked before this change: only (a) producing a map for a
BRAND-NEW image that has no pre-computed map yet, and (b) the mIoU metric
(scoring a generation against a live re-segmentation) — both correctly
auto-skip/require Tier 2 rather than crashing (`encoder.live_available=False`
is checked before either runs).

### 5.1c The loss, from scratch — what the model is actually learning [ADDED 2026-07-19]

You cannot tune a hyperparameter sensibly without knowing what number you're
reacting to. This walks the real training code line by line, then how to
read the resulting loss curve to make decisions — no formula tells you "the
right epoch count" in advance; you read the curve from a real run.

**What training actually does, in one paragraph:** take a real image, add a
random amount of noise to it, ask the model to guess **exactly what noise
was added**. Train it to guess correctly at every possible noise strength (a
little noise, up to almost-pure static). If a model can always correctly
identify "what noise is this," it can also run the process **backward**:
start from pure noise and repeatedly subtract its best guess, and a real
image emerges. That backward process is generation. Training only ever does
the easy, forward direction — we always know the true noise, because we
added it ourselves.

**The real code.** `seg_training.py:698` calls:
```python
model_pred, loss, x0, _ = model.forward_easy(imgs, prompts, cs, skip_encode=True, ...)
```
`skip_encode=True` is not a training-time shortcut here the way it can look
on the SegFormer side — `GroundedSamEncoder` (§5.1) has **no live forward
path at all** (`forward()` raises `NotImplementedError`, Tier 1 by design),
so `skip_encode=True` is the only value that ever works for this pipeline.
The pre-saved colour map goes straight to the mapper; the encoder slot exists
only so `accelerate` has an `nn.Module` to `.prepare()`.

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
    cond = lora_c if skip_encode else encoder(lora_c)   # :526-535 — your Grounded-SAM map, unchanged (always True here)
    mapped_cond = mapper(cond)                            # FixedStructureMapper15
    dp.set_batch(mapped_cond)                              # :537 — handed to the shared DataProvider
```
Your Grounded-SAM segmentation map gets pushed into the `DataProvider` here.
Every `NewStructLoRAConv` layer inside the UNet reads it from there on the
very next line — see [LORA_ARCHITECTURE.md](LORA_ARCHITECTURE.md) for the
full trace of what happens to it once it's inside the UNet.

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
(`seg_training.py:705`) uses this single number to compute gradients and
update every trainable weight (the LoRA `A`/`B`/`beta`/`gamma` parameters —
the frozen UNet weights never change, per `add_lora_to_unet`).

**Which loss to actually watch: `train/loss` vs `val/loss`.** Two different
losses get computed:
- **`train/loss`** (`seg_training.py:719,725`) — the MSE above, computed on
  the batch you just trained on, WITH gradients on, logged every step.
  Inherently noisy (small batches, random noise/timestep each time) — watch
  the trend over dozens of steps, not any single value.
- **`val/loss`** (`_segmentation_validation_loss`, `seg_training.py:270-333`)
  — the SAME formula, computed on held-out images the model never trains on,
  with `torch.no_grad()` (no learning happens), averaged over several
  batches, only at `val_steps` intervals. Its own docstring states exactly
  why: *"(a) if it diverges from train/loss, you're overfitting. (b) an
  objective best_model criterion — not a biased training-loss average."*

**Watch `val/loss` to make decisions.** It's what `best_model`/early-stopping
(`best_loss` tracking at `seg_training.py:510`, the early-stop check at
`:779-794`) already use automatically. `train/loss` is only a sanity check —
is it decreasing, is it NaN — not a decision signal.

**Tuning hyperparameters by watching the loss curve** — read-after-a-real-run,
not computed in advance:

| What you observe | What it means | What to change |
|---|---|---|
| `train/loss` is `NaN` or explodes | Numerical instability / LR too aggressive | Lower `learning_rate`; gradients are already clipped to `max_norm=1.0` (`seg_training.py:712`) as a safety net |
| `train/grad_norm` stays pinned at 1.0 for a long time | Gradients are being clipped constantly — the optimizer wants bigger steps than allowed | Normal early on; if it never relaxes, LR may be too high for this effective batch |
| `val/loss` decreases then **flattens** | Model has converged on this data — more epochs teach it nothing new | Nothing to do — `early_stop_patience` (§6) catches this automatically |
| `val/loss` **increases** while `train/loss` keeps falling | Overfitting — memorizing training images instead of generalizing | More epochs make this WORSE. Real fix is more/more-diverse data, not a hyperparameter |
| `val/loss` still steadily falling when `epochs` ceiling is hit | Ceiling was too low, model hadn't finished learning | Rerun with a higher `epochs`, resuming via `lora.struct.ckpt_path` — no need to restart from step 0 |
| `val/loss` jumps around a lot between checks | Too few `val_batches` averaged, or a small/noisy val set | Increase `val_batches` — this is a measurement-noise artifact, not a real problem to fix with training hyperparameters |

**The actual workflow, for however many images YOU have.** Say you have
`N_train` training images, `N_val` validation images (any numbers — this
generalizes, it isn't specific to any particular dataset size). Nothing
below requires you to already know the answer:

1. **Run the advisor script on the real data, on the real machine**:
   ```
   python recommend_training_params.py --data_dir data/grounded_sam --epochs 15
   ```
   It reads your actual `N_train`/`N_val`/`N_test` by counting JSONL lines
   (no guessing) and detects your actual GPU's VRAM (no guessing). From
   those two REAL numbers it computes, as plain arithmetic: `batch_size`/
   `gradient_accumulation_steps` sized to your VRAM while holding the SAME
   validated effective batch (16) this project's `learning_rate` was tuned
   at — LR is never auto-scaled, because no scaling behaviour has been
   verified on this codebase; `steps_per_epoch = ceil(N_train /
   effective_batch)` and `total_steps = steps_per_epoch * epochs`, pure
   arithmetic on YOUR `N_train`; `val_steps`/`ckpt_steps` sized to check
   ~7 times and checkpoint ~3-4 times per epoch. `--epochs 15` here is
   **your chosen ceiling**, not a computed answer.
2. **Paste the printed block into `configs/experiment/train_grounded_sam.yaml`.**
3. **Leave `early_stop_patience` at its default (3)** unless your `val/loss`
   looks genuinely noisy (small val set) rather than truly plateaued.
4. **Launch training**, optionally watching
   `tensorboard --logdir outputs/train/grounded_sam/runs/` live for `val/loss`.
5. **After it finishes**, read `best_model/info.txt`
   (`seg_training.py:581-589` writes it: epoch, step, psnr/ssim — no mIoU
   here, `live_available=False`). **This is your empirically-discovered
   right epoch count for THIS dataset** — discovered by running, not
   predicted; task difficulty and data diversity matter as much as
   `N_train`, and neither is knowable without a real run.
6. **If early-stop never fired**, rerun with a higher ceiling, resuming from
   the last checkpoint rather than restarting from scratch.

### 5.1d Real mask format handling, class taxonomy verification, and the inference run recipe [ADDED 2026-07-20]

**Class taxonomy verified against the official CARLA docs.** The user posted
the full CARLA `instance-segmentation-camera` tag table (0–28, names + RGB)
directly from the CARLA docs; every id/name/colour in
`configs/grounded_sam_classes.json` was checked against it line by line —
all 29 entries match exactly. (CARLA's semantic- and instance-segmentation
camera pages share one underlying tag table, so citing either is correct;
this file's `__README__` cites the semantic-segmentation page.) If your saved
masks ever encode *instance* IDs rather than plain semantic tags, that's a
different question this file doesn't answer — confirm your `class_map.png`
pixel values are the plain 0–28 semantic tag before trusting this palette.

**Inference run layout — same change as the segformer branch, same day.**
`seg_inference.py` now writes every run into its own timestamped subfolder of
`inference.output_dir` (`<output_dir>/<YYYY-MM-DD_HH-MM-SS>/`), so two runs
never overwrite each other, and saves a `run_params.txt` into that folder
*before* generating — checkpoint, seed, size, `num_inference_steps`,
`guidance_scale`, `conditioning_kernel_size`, `lora_scale_start`/`_end`/
`_decay_start_frac`, base model, `classes_file`, `seg_pad_id`, and every
`seg_path | raw_image_path | prompt` triplet processed. Any result folder is
self-documenting.

**`seg_pad_id`** (new inference config key, `configs/inference_grounded_sam.yaml`)
must match training's `pad_id` (`configs/data/local_grounded_sam.yaml`, both
default 0 = CARLA `Unlabeled`) — see §5.0a for what it controls (letterbox
fill for non-square maps).

### 5.2 What was NOT built — "Tier 2": live map generation
A live `GroundedSamEncoder` that actually runs GroundingDINO + SAM to make a map
for a brand-new image. Needed for **inference on new frames without a
pre-computed map** and for the **mIoU metric** (which segments the generated
image to score layout adherence — auto-skipped for now). This requires the
packages, checkpoints, and your class-name
prompts, none of which are set up yet. Deliberately deferred.

---

## 6 · Before you train — a checklist grounded in what we found

1. **Class list is locked** (§5.1) — CARLA's 29-class taxonomy, IDs 0–28. No
   further action needed on the class file itself.

2. **Confirm the map format.** Run
   `python check_seg_map_format.py --seg_map <one_real_map>`. Your real masks
   (user-confirmed 2026-07-20, `class_map.png`) are mode `I;16` (16-bit), PNG
   (lossless — good), **1280x800, non-square**, clean CARLA ids. The loader
   handles this format directly as of `5db3624` (§5.0a) — raw pixel read
   (no lossy conversion) + letterbox to square (not stretch), matching the
   paired RGB's geometry. (An earlier, now-superseded scan of a *different,
   unused* file showed 8-bit JPEG — ignore that finding; it does not describe
   what this pipeline actually trains on.)

3. **Recommended (not required): sanity-check the pixel range against the locked
   list.** Run
   `python scan_seg_map_classes.py --json_file data/grounded_sam/train.jsonl`.
   The class file assumes pixel values are CARLA's own tag IDs (0–28); this scan
   just tells you the true range actually present in your files, for your own
   awareness — it doesn't change the (locked) class file.

4. **Size the hyperparameters to your real dataset.** `recommend_training_params.py`
   (repo root, `--data_dir` required, no default) counts your real manifest line
   counts and detects the local GPU, then prints `batch_size` /
   `gradient_accumulation_steps` / `val_steps` / `ckpt_steps` for THIS pipeline:
   ```
   python recommend_training_params.py --data_dir data/grounded_sam --epochs 15
   ```
   Run it on the machine that will actually train (it only reads local files).
   Paste the printed values into `configs/experiment/train_grounded_sam.yaml`.
   **Grounded-SAM's dataset (CARLA renders) and SegFormer's dataset (real-world
   photos) are different image sets with different counts** — SegFormer has
   its own copy of this checklist on the `segformer` branch of this repo; the
   two are separate runs, don't reuse one pipeline's numbers for the other.

   `--epochs 15` here is an upper-bound ceiling, not a prediction: early
   stopping is already active for this pipeline too —
   `early_stop_patience: 3` lives in the shared base `configs/train_seg.yaml`
   (confusingly named — it's `seg_training.py`'s base config on BOTH the
   `segformer` and `grounded_sam` branches, hard-wired via
   `@hydra.main(config_name="train_seg")`; `experiment=train_grounded_sam`
   just layers overrides on top of it here), so training stops itself once
   val/loss stops improving for 3 epochs — `best_model/` is already saved at
   that point. Neither pipeline
   has an empirical convergence curve yet at real scale, so treat `epochs`
   as "how long am I willing to let it run," not a number to get exactly right.

5. `python seg_training.py experiment=train_grounded_sam`

6. **Do a short smoke run first.** Training here is code- and config-verified but
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
| [configs/grounded_sam_classes.json](configs/grounded_sam_classes.json) | **LOCKED FINAL**: CARLA's official 29-class taxonomy + colours. |
| [configs/experiment/train_grounded_sam.yaml](configs/experiment/train_grounded_sam.yaml) | The training experiment: wires encoder + data + your manifests. |
| [configs/data/local_grounded_sam.yaml](configs/data/local_grounded_sam.yaml) | Data config: class file + manifest key names. |
| [configs/lora/encoder/grounded_sam.yaml](configs/lora/encoder/grounded_sam.yaml) | Encoder slot config (no HF model to load). |
| [configs/inference_grounded_sam.yaml](configs/inference_grounded_sam.yaml) | Base inference config — `seg_inference.py`'s default on this branch. |
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
| [src/lora.py](src/lora.py) / [src/model.py](src/model.py) | The LoRA classes and UNet-wrapping logic itself — identical code path for both pipelines. See [LORA_ARCHITECTURE.md](LORA_ARCHITECTURE.md) for the full trace. |
| [recommend_training_params.py](recommend_training_params.py) | GPU + dataset-sized hyperparameter advisor — run once per pipeline (`--data_dir` required, no shared default). |

### 7.3 Diagnostics built for this investigation (keep — model-agnostic)
| File | Purpose |
|---|---|
| [scan_seg_map_classes.py](scan_seg_map_classes.py) | Global class count + JPEG-noise check across a manifest. |
| [check_seg_map_format.py](check_seg_map_format.py) | Inspect one map file's format (mode, dtype, value range). |
| [analyze_car_coverage.py](analyze_car_coverage.py) | Car-class coverage histogram (found the CARLA-truck data gap). |
| [check_seg_coverage.py](check_seg_coverage.py) | Per-image, per-class coverage vs the training distribution. |

### 7.4 SegFormer-only (removed from this branch — see the `segformer` branch)
Earlier, both approaches lived side by side on one branch. That changed
(2026-07-19, user decision): **this repo now has two separate branches**,
`segformer` and `grounded_sam`, each keeping only its own pipeline's files on
top of the shared engine (§7.2). The following were removed from this branch
because they're SegFormer-only and not imported by any Grounded-SAM code —
they still exist on the `segformer` branch:
| File (on the `segformer` branch) | What it does (SegFormer) |
|---|---|
| configs/lora/encoder/segformer.yaml | SegFormer encoder config. |
| configs/experiment/train_seg.yaml | SegFormer training experiment. |
| configs/data/local_seg.yaml | SegFormer data config (this branch uses local_grounded_sam.yaml instead). |
| seg_map_calculations.py | Computes SegFormer maps offline (Grounded-SAM maps are made externally). |
| seg_finetune.py / seg_make_draft_masks.py | Earlier SegFormer-on-CARLA finetuning exploration. |
| SEGMENTATION.md / SEG_TRAINING_GUIDE.md | SegFormer pipeline docs. |

`seg_inference.py` and `configs/inference_grounded_sam.yaml` are **not** in
that list — `seg_inference.py` is shared engine (§7.2) and stays on both
branches; only its default *config* differs per branch
(`inference_grounded_sam.yaml` here, `inference_seg.yaml` there), since
inference never runs the encoder live on either branch anymore.

Note: `seg_encoder.py` also stays here (§7.2) — the Grounded-SAM path imports
its palette functions. Its `SegmentationEncoder` class is SegFormer-specific
and unused on this branch (dead code, harmless, not instantiated by any
config here) — left in place rather than stripped, since no config on this
branch references it and removing it has no functional benefit.

---

## 8 · Bottom line

Grounded-SAM is the right tool when you need **your own object vocabulary** and
**crisp object masks**, and you accept weaker "stuff" coverage, a non-
deterministic and heavier live path, and more setup. For dense structure
conditioning on driving scenes it is a legitimate alternative to SegFormer —
worth testing on its own merits. Just keep two facts in view: it does **not**
address the separate CARLA-truck data-coverage problem, and the live-inference
half (Tier 2) is real additional work still ahead.
