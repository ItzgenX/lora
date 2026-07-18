"""
download_sd15.py
----------------
FIRST-TIME SETUP: download every model this project needs and save it to
checkpoints/local_models/ for fully offline training and inference.

Run this ONCE on a machine with internet access, then copy
checkpoints/local_models/ to your training machine and set
local_files_only: true in all configs.

Models downloaded:
  1. stable-diffusion-v1-5       -- base diffusion model (backbone, always frozen)
  2. dpt-hybrid-midas            -- stock upstream depth encoder (src/annotators/midas.py)
  3. segformer-b5-cityscapes     -- live segmentation encoder (src/encoders/seg_encoder.py)
  4. taesd                       -- Tiny AutoEncoder (fast VAE preview, optional)

NOTE — DPTImageProcessor for the MiDaS model is saved but not used at runtime
  (DepthEstimator does manual preprocessing; the processor call is commented out
  in src/annotators/midas.py). Saving it keeps the folder complete.

Usage:
  python download_sd15.py

  With a HF token (required for gated models, optional here since all models
  below are public):
    Set HF_TOKEN below, or export HF_TOKEN=your_token before running.
"""

import os
from diffusers import StableDiffusionPipeline, AutoencoderTiny
from transformers import (
    DPTForDepthEstimation,
    DPTImageProcessor,
    SegformerForSemanticSegmentation,
)

# --- Hugging Face authentication -------------------------------------------- #
# Paste your HF READ token here, or leave empty for public models.
#
# SECURITY: never commit this file to a public repo with a token in it.
# Safer: export HF_TOKEN=hf_xxx in your shell, then use token=os.environ.get("HF_TOKEN")
HF_TOKEN = os.environ.get("HF_TOKEN", "")

LOCAL_MODEL_DIR = "checkpoints/local_models"
os.makedirs(LOCAL_MODEL_DIR, exist_ok=True)

# Helper: print a section banner
def banner(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


# ── 1. Stable Diffusion 1.5 (base model, always frozen) ──────────────────── #
banner("1/5  Stable Diffusion 1.5")
sd_pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    token=HF_TOKEN or None,
)
sd_path = os.path.join(LOCAL_MODEL_DIR, "stable-diffusion-v1-5")
sd_pipe.save_pretrained(sd_path)
print(f"  Saved -> {sd_path}")


# ── 2. MiDaS / DPT-Hybrid (stock upstream depth encoder, src/annotators/midas.py) ─ #
#
# NOTE on DPTImageProcessor:
#   DepthEstimator (src/annotators/midas.py) does NOT use DPTImageProcessor at
#   runtime — the processor call is commented out and replaced with manual
#   preprocessing ((x+1)/2 -> better_resize -> direct model call).
#   We still save the processor here so the checkpoint folder is complete and
#   no tool ever complains about a missing preprocessor_config.json.
banner("2/5  MiDaS DPT-Hybrid (depth encoder)")
midas_model = DPTForDepthEstimation.from_pretrained(
    "Intel/dpt-hybrid-midas",
    token=HF_TOKEN or None,
)
midas_processor = DPTImageProcessor.from_pretrained(
    "Intel/dpt-hybrid-midas",
    token=HF_TOKEN or None,
)
midas_path = os.path.join(LOCAL_MODEL_DIR, "dpt-hybrid-midas")
midas_model.save_pretrained(midas_path)
midas_processor.save_pretrained(midas_path)
print(f"  Saved -> {midas_path}")
print(f"  (DPTImageProcessor saved for completeness — not used at runtime)")


# ── 3. SegFormer-b5-Cityscapes (live segmentation encoder) ──────────────── #
#
# LOCKED MODEL: b5, NOT b0 (references.md §9 / SEGMENTATION.md). No separate
# image processor is downloaded — SegmentationEncoder (src/encoders/seg_encoder.py)
# does not load one at runtime, only SegformerForSemanticSegmentation itself.
banner("3/5  SegFormer-b5-Cityscapes (segmentation encoder)")
seg_model = SegformerForSemanticSegmentation.from_pretrained(
    "nvidia/segformer-b5-finetuned-cityscapes-1024-1024",
    token=HF_TOKEN or None,
)
seg_path = os.path.join(LOCAL_MODEL_DIR, "segformer-b5-cityscapes")
seg_model.save_pretrained(seg_path)
print(f"  Saved -> {seg_path}")


# ── 4. Tiny VAE / TAESD (fast VAE preview — optional) ────────────────────── #
banner("4/5  Tiny VAE (TAESD)")
tiny_vae = AutoencoderTiny.from_pretrained(
    "madebyollin/taesd",
    token=HF_TOKEN or None,
)
vae_path = os.path.join(LOCAL_MODEL_DIR, "taesd")
tiny_vae.save_pretrained(vae_path)
print(f"  Saved -> {vae_path}")


# ── Summary ──────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("  All models downloaded.")
print(f"  Location: {os.path.abspath(LOCAL_MODEL_DIR)}/")
print()
print("  Folder structure:")
for name in [
    "stable-diffusion-v1-5",
    "dpt-hybrid-midas",
    "segformer-b5-cityscapes",
    "taesd",
]:
    path = os.path.join(LOCAL_MODEL_DIR, name)
    status = "OK" if os.path.isdir(path) else "MISSING"
    print(f"    [{status}]  {name}")
print()
print("  Next step: set local_files_only: true in all configs.")
print(f"{'='*60}\n")
