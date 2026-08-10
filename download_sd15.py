"""
download_sd15.py
----------------
FIRST-TIME SETUP: download every model this project needs and save it to
checkpoints/local_models/ for fully offline training and inference.

Run this ONCE on a machine with internet access, then copy
checkpoints/local_models/ to your training machine and set
local_files_only: true in all configs.

SKIP-EXISTING: each model is checked BEFORE downloading -- if its folder
already has the right marker file (model_index.json for a diffusers
pipeline, config.json for a bare transformers/AutoencoderKL model), it is
SKIPPED, not re-downloaded. Safe to re-run this script any time you add a
new model to the list below; only the new one actually downloads. Pass
--force to ignore this and re-download everything anyway.

Models downloaded:
  1. stable-diffusion-v1-5       -- stock SD1.5 (kept for comparison / fallback)
  2. epiCRealism                 -- photorealistic SD1.5-architecture finetune,
                                     THE BASE MODEL this project actually trains
                                     against now (2026-08 decision: stock SD1.5's
                                     output looked messy/unnatural; epiCRealism
                                     chosen over Realistic Vision V6.0 because it
                                     ships in ready-to-use diffusers format --
                                     V6.0 is safetensors-only and would need
                                     from_single_file() support this repo doesn't
                                     have)
  3. sd-vae-ft-mse                -- improved SD1.5 VAE (stabilityai/sd-vae-ft-mse).
                                     Paired in via the model.vae_path config
                                     (src/model.py) -- needed for any "noVAE"-style
                                     community checkpoint, harmless to also have
                                     on hand for epiCRealism.
  4. dpt-hybrid-midas            -- stock upstream depth encoder (src/annotators/midas.py)
  5. taesd                       -- Tiny AutoEncoder (fast VAE preview, optional)

  Training ALWAYS reads pre-saved class_map.png masks (skip_encode=True), so
  the two models below are OPTIONAL and only downloaded with --with-live-gsam:
  6. grounding-dino-base         -- open-vocabulary detector for the LIVE
                                     Grounded-SAM encoder (Tier 2, added
                                     2026-08, src/encoders/grounded_sam_encoder.py).
                                     Needed for inference on a brand-new image
                                     with no pre-computed mask, or the mIoU
                                     metric. Uses transformers' NATIVE support
                                     (AutoModelForZeroShotObjectDetection) --
                                     not the standalone GroundingDINO package,
                                     so no compiled CUDA extension to build.
  7. sam-vit-huge                 -- SAM checkpoint for the same live encoder
                                     (also via transformers' native SamModel).

NOTE — DPTImageProcessor for the MiDaS model is saved but not used at runtime
  (DepthEstimator does manual preprocessing; the processor call is commented out
  in src/annotators/midas.py). Saving it keeps the folder complete.

Usage:
  python download_sd15.py
  python download_sd15.py --force              # re-download even if already present
  python download_sd15.py --with-live-gsam      # also fetch grounding-dino-base + sam-vit-huge
                                                  # (large: ~1GB + ~2.4GB) for the LIVE Grounded-SAM
                                                  # encoder (lora.struct.encoder.live=true)

  With a HF token (required for gated models, optional here since all models
  below are public):
    Set HF_TOKEN below, or export HF_TOKEN=your_token before running.
"""

import argparse
import os
from diffusers import StableDiffusionPipeline, AutoencoderTiny, AutoencoderKL
from transformers import (
    DPTForDepthEstimation,
    DPTImageProcessor,
)

# --- Hugging Face authentication -------------------------------------------- #
# Paste your HF READ token here, or leave empty for public models.
#
# SECURITY: never commit this file to a public repo with a token in it.
# Safer: export HF_TOKEN=hf_xxx in your shell, then use token=os.environ.get("HF_TOKEN")
HF_TOKEN = os.environ.get("HF_TOKEN", "")

_parser = argparse.ArgumentParser(description=__doc__)
_parser.add_argument("--force", action="store_true",
                      help="Re-download every model even if already present locally.")
_parser.add_argument("--with-live-gsam", action="store_true",
                      help="Also download grounding-dino-base + sam-vit-huge for the "
                           "LIVE Grounded-SAM encoder (large, ~3.4GB combined). Skip "
                           "this if you only train/infer on pre-saved class_map.png masks.")
_args = _parser.parse_args()
FORCE = _args.force
WITH_LIVE_GSAM = _args.with_live_gsam

LOCAL_MODEL_DIR = "checkpoints/local_models"
os.makedirs(LOCAL_MODEL_DIR, exist_ok=True)


# Helper: print a section banner
def banner(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def already_downloaded(local_path: str, marker: str = "model_index.json") -> bool:
    """
    True if `local_path` already contains `marker` (i.e. this model was
    already fully saved by a previous run of this script) AND --force was
    not passed. This is the skip-existing check every model below runs
    before doing any network I/O.

    marker: "model_index.json" for a full diffusers pipeline (SD1.5,
      epiCRealism); "config.json" for a bare single model (MiDaS, TAESD, the
      VAE) -- diffusers pipelines don't write a config.json at their own
      root, only inside subfolders, so the two marker files are how this
      distinguishes "a whole pipeline was saved here" from "just one component."
    """
    if FORCE:
        return False
    return os.path.isfile(os.path.join(local_path, marker))


# ── 1. Stable Diffusion 1.5 (kept for comparison / fallback, no longer the
#       default training base -- see epiCRealism below) ─────────────────── #
banner("1/5  Stable Diffusion 1.5")
sd_path = os.path.join(LOCAL_MODEL_DIR, "stable-diffusion-v1-5")
if already_downloaded(sd_path):
    print(f"  Already present -> {sd_path}  (skipped; pass --force to re-download)")
else:
    sd_pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        token=HF_TOKEN or None,
    )
    sd_pipe.save_pretrained(sd_path)
    print(f"  Saved -> {sd_path}")


# ── 2. epiCRealism (photorealistic SD1.5-architecture finetune) ──────────── #
#
# THE ACTUAL TRAINING BASE now (2026-08 decision, see docstring). Same UNet
# architecture/state-dict keys as stock SD1.5, so add_lora_to_unet works on
# it completely unchanged -- only base_model_name/base_model_path in the
# experiment/inference configs need to point here instead of stable-diffusion-v1-5.
# Ships as a full diffusers pipeline (unlike Realistic Vision, which is
# safetensors-only) -- from_pretrained works exactly like SD1.5 above.
banner("2/5  epiCRealism (photorealistic base checkpoint)")
epic_path = os.path.join(LOCAL_MODEL_DIR, "epicrealism")
if already_downloaded(epic_path):
    print(f"  Already present -> {epic_path}  (skipped; pass --force to re-download)")
else:
    epic_pipe = StableDiffusionPipeline.from_pretrained(
        "emilianJR/epiCRealism",
        token=HF_TOKEN or None,
    )
    epic_pipe.save_pretrained(epic_path)
    print(f"  Saved -> {epic_path}")


# ── 3. sd-vae-ft-mse (improved SD1.5 VAE) ─────────────────────────────────── #
#
# Paired in via the model.vae_path config (src/model.py ModelBase.__init__,
# mirrors the existing tiny_vae pattern) -- REQUIRED for any "noVAE"-style
# community checkpoint, and a safe quality upgrade over epiCRealism's own
# baked-in VAE too. Not auto-applied: model.vae_path stays null (= use
# whatever VAE the base checkpoint ships with) until set in a config.
banner("3/5  sd-vae-ft-mse (VAE)")
vae_ft_path = os.path.join(LOCAL_MODEL_DIR, "sd-vae-ft-mse")
if already_downloaded(vae_ft_path, marker="config.json"):
    print(f"  Already present -> {vae_ft_path}  (skipped; pass --force to re-download)")
else:
    vae_ft = AutoencoderKL.from_pretrained(
        "stabilityai/sd-vae-ft-mse",
        token=HF_TOKEN or None,
    )
    vae_ft.save_pretrained(vae_ft_path)
    print(f"  Saved -> {vae_ft_path}")


# ── 4. MiDaS / DPT-Hybrid (stock upstream depth encoder, src/annotators/midas.py) ─ #
#
# NOTE on DPTImageProcessor:
#   DepthEstimator (src/annotators/midas.py) does NOT use DPTImageProcessor at
#   runtime — the processor call is commented out and replaced with manual
#   preprocessing ((x+1)/2 -> better_resize -> direct model call).
#   We still save the processor here so the checkpoint folder is complete and
#   no tool ever complains about a missing preprocessor_config.json.
banner("4/5  MiDaS DPT-Hybrid (depth encoder)")
midas_path = os.path.join(LOCAL_MODEL_DIR, "dpt-hybrid-midas")
if already_downloaded(midas_path, marker="config.json"):
    print(f"  Already present -> {midas_path}  (skipped; pass --force to re-download)")
else:
    midas_model = DPTForDepthEstimation.from_pretrained(
        "Intel/dpt-hybrid-midas",
        token=HF_TOKEN or None,
    )
    midas_processor = DPTImageProcessor.from_pretrained(
        "Intel/dpt-hybrid-midas",
        token=HF_TOKEN or None,
    )
    midas_model.save_pretrained(midas_path)
    midas_processor.save_pretrained(midas_path)
    print(f"  Saved -> {midas_path}")
    print(f"  (DPTImageProcessor saved for completeness — not used at runtime)")


# ── 5. Tiny VAE / TAESD (fast VAE preview — optional) ────────────────────── #
banner("5/5  Tiny VAE (TAESD)")
taesd_path = os.path.join(LOCAL_MODEL_DIR, "taesd")
if already_downloaded(taesd_path, marker="config.json"):
    print(f"  Already present -> {taesd_path}  (skipped; pass --force to re-download)")
else:
    tiny_vae = AutoencoderTiny.from_pretrained(
        "madebyollin/taesd",
        token=HF_TOKEN or None,
    )
    tiny_vae.save_pretrained(taesd_path)
    print(f"  Saved -> {taesd_path}")


# ── 6/7. LIVE Grounded-SAM models (OPTIONAL, --with-live-gsam only) ──────── #
#
# Training never needs these (skip_encode=True always reads pre-saved
# class_map.png masks). Only needed for the LIVE encoder (Tier 2,
# src/encoders/grounded_sam_encoder.py, lora.struct.encoder.live=true) --
# inference on a brand-new image with no pre-computed mask, or the mIoU
# metric. Both use transformers' native from_pretrained() -- same pattern as
# every other model above, no separate package/compiled extension needed.
if WITH_LIVE_GSAM:
    from transformers import (
        AutoModelForZeroShotObjectDetection, AutoProcessor as _AutoProcessor,
        SamModel, SamProcessor,
    )

    banner("6/7  GroundingDINO-base (live Grounded-SAM detector)")
    gdino_path = os.path.join(LOCAL_MODEL_DIR, "grounding-dino-base")
    if already_downloaded(gdino_path, marker="config.json"):
        print(f"  Already present -> {gdino_path}  (skipped; pass --force to re-download)")
    else:
        gdino_processor = _AutoProcessor.from_pretrained(
            "IDEA-Research/grounding-dino-base", token=HF_TOKEN or None)
        gdino_model = AutoModelForZeroShotObjectDetection.from_pretrained(
            "IDEA-Research/grounding-dino-base", token=HF_TOKEN or None)
        gdino_processor.save_pretrained(gdino_path)
        gdino_model.save_pretrained(gdino_path)
        print(f"  Saved -> {gdino_path}")

    banner("7/7  SAM ViT-Huge (live Grounded-SAM segmenter)")
    sam_path = os.path.join(LOCAL_MODEL_DIR, "sam-vit-huge")
    if already_downloaded(sam_path, marker="config.json"):
        print(f"  Already present -> {sam_path}  (skipped; pass --force to re-download)")
    else:
        sam_processor = SamProcessor.from_pretrained(
            "facebook/sam-vit-huge", token=HF_TOKEN or None)
        sam_model_dl = SamModel.from_pretrained(
            "facebook/sam-vit-huge", token=HF_TOKEN or None)
        sam_processor.save_pretrained(sam_path)
        sam_model_dl.save_pretrained(sam_path)
        print(f"  Saved -> {sam_path}")
else:
    print("\n[SKIPPED] Live Grounded-SAM models (grounding-dino-base, sam-vit-huge) -- "
          "pass --with-live-gsam to fetch them (only needed for lora.struct.encoder.live=true).")


# ── Summary ──────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("  All models ready (downloaded now, or already present and skipped).")
print(f"  Location: {os.path.abspath(LOCAL_MODEL_DIR)}/")
print()
print("  Folder structure:")
_all_models = [
    ("stable-diffusion-v1-5", "model_index.json"),
    ("epicrealism", "model_index.json"),
    ("sd-vae-ft-mse", "config.json"),
    ("dpt-hybrid-midas", "config.json"),
    ("taesd", "config.json"),
]
if WITH_LIVE_GSAM:
    _all_models += [
        ("grounding-dino-base", "config.json"),
        ("sam-vit-huge", "config.json"),
    ]
for name, marker in _all_models:
    path = os.path.join(LOCAL_MODEL_DIR, name)
    status = "OK" if os.path.isfile(os.path.join(path, marker)) else "MISSING"
    print(f"    [{status}]  {name}")
print()
print("  Next step: point base_model_path at epicrealism/ in the experiment/")
print("  inference configs (instead of stable-diffusion-v1-5/), and set")
print("  local_files_only: true everywhere.")
print(f"{'='*60}\n")
