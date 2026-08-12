"""
grounded_sam_inference.py
----------------
Run inference with a trained segmentation-conditioned LoRAdapter.

Inference ALWAYS uses a PROVIDED segmentation
map — it never computes one live from a raw photo. You must supply `seg_path`
(a pre-computed class-ID PNG), exactly like training does via skip_encode=True.
This is a design choice, not a limitation of GroundedSamEncoder: a live Tier 2
path (real GroundingDINO+SAM, live=true) does exist on this branch, but this
script never uses it for the conditioning input — see the module docstring
at src/encoders/grounded_sam_encoder.py for what Tier 2 is actually for
(fresh maps on new images, or the mIoU metric below).

This script:
  1. Loads the SD 1.5 base model + trained LoRA/mapper from a checkpoint.
  2. For each entry, LOADS the pre-computed seg map (`seg_path`) and colourises
     it with the configured palette — no live segmentation model runs for this.
     `raw_image_path` is now OPTIONAL and used ONLY for the "ORIGINAL" display
     panel (no computation happens on it).
  3. Saves a 4-panel grid (ORIGINAL | SEG MAP | PREDICTED | RAW SEG GEN) per image
     so you can visually evaluate quality and pick the best checkpoint.
  4. mIoU (controllability metric) is scored ONLY if the loaded encoder can
     itself segment the GENERATED image (`encoder.live_available`). Grounded-SAM's
     Tier-1 encoder cannot, so this is skipped automatically — not a crash.

INPUT OPTIONS — SAME SCHEMA training's manifests already use:
  a) JSON manifest file (recommended) — each entry needs "seg_path" (required),
     "raw_image_path" (optional, display only), "prompt" (optional):
       inference.json_file=data/grounded_sam/test.jsonl
  b) Direct lists:
       "inference.seg_maps=[data/grounded_sam_raw/000001/000001_seg_map.png]"
       "inference.images=[data/raw/000001/raw_image.jpg]"   # optional, display only
       "inference.prompts=['a car driving down a rainy street']"

OUTPUT MODES:
  Default (save_generated_only=false) — saves 4 files per image:
    <stem>_grid.jpg         ← 4 panels: ORIGINAL | SEG MAP | PREDICTED | RAW SEG GEN
    <stem>_original.jpg     ← input image if provided, else a blank placeholder
    <stem>_seg.jpg          ← the loaded + colourised seg map
    <stem>_predicted.jpg    ← the generated image

  Batch eval (save_generated_only=true, json_file required):
    Saves ONLY the generated image, mirroring folder structure from the JSON.

RESIZE_MODE (see GROUNDED_SAM.md §5.0b): pass
  resize_mode=letterbox (default) or resize_mode=CenterCrop -- MUST MATCH
  what the loaded checkpoint was trained with (a mismatch is warned about
  loudly, reading the stamp grounded_sam_training.py writes to best_model/info.txt).
  Also names the output folder: outputs/inference/grounded_sam_<mode>/results/.

USAGE (this branch's default config is inference_grounded_sam.yaml):
  # Standard inference from a manifest (seg_path required, raw_image_path optional):
  python grounded_sam_inference.py \\
      ckpt_path=outputs/train/grounded_sam_letterbox/runs/YYYY-MM-DD/HH-MM-SS/best_model \\
      resize_mode=letterbox \\
      inference.json_file=data/grounded_sam/test.jsonl

  # Direct seg map + optional raw image for the display panel:
  python grounded_sam_inference.py \\
      ckpt_path=outputs/train/grounded_sam_letterbox/runs/YYYY-MM-DD/HH-MM-SS/best_model \\
      resize_mode=letterbox \\
      "inference.seg_maps=[data/grounded_sam_raw/000001/000001_seg_map.png]" \\
      "inference.images=[data/raw/000001/raw_image.jpg]" \\
      "inference.prompts=['a car driving down a rainy street']"

QUICK COMMANDS (run from repo root with conda loradapter env active):
  # --- Single map dry run (replace YYYY-MM-DD/HH-MM-SS with actual run folder) ---
  python grounded_sam_inference.py ckpt_path=outputs/train/grounded_sam_letterbox/runs/YYYY-MM-DD/HH-MM-SS/best_model resize_mode=letterbox "inference.seg_maps=[data/grounded_sam_raw/000001/000001_seg_map.png]" "inference.prompts=['a car driving down a rainy street']"

  # --- Batch test-set inference ---
  python grounded_sam_inference.py ckpt_path=outputs/train/grounded_sam_letterbox/runs/YYYY-MM-DD/HH-MM-SS/best_model resize_mode=letterbox inference.json_file=data/grounded_sam/test.jsonl

  # --- Same, but the checkpoint was trained with CenterCrop ---
  python grounded_sam_inference.py ckpt_path=outputs/train/grounded_sam_CenterCrop/runs/YYYY-MM-DD/HH-MM-SS/best_model resize_mode=CenterCrop inference.json_file=data/grounded_sam/test.jsonl
"""

import hydra
import os
import json
from datetime import datetime
import torch
import numpy as np
from PIL import Image, ImageDraw
import torchvision.transforms.functional as TF
from pathlib import Path
from tqdm import tqdm

from hydra.utils import get_original_cwd
from src.model import ModelBase
from src.utils import add_lora_from_config, resolve_device, compute_miou
from src.data.transforms import build_seg_display_preprocess, square_id_map, normalize_size
from src.data.seg_palette import seg_palette_tensor, seg_ids_from_colormap, seg_colorize_ids

torch.set_float32_matmul_precision("high")


# ===================================================================== #
#  VISUALIZATION HELPERS                                                  #
# ===================================================================== #

def _seg_label_bar(width: int, text: str, bar_h: int = 28) -> np.ndarray:
    """
    Dark banner bar with centered text. Returns [bar_h, width, 3] uint8.
    """
    bar  = Image.new("RGB", (width, bar_h), color=(25, 25, 25))
    draw = ImageDraw.Draw(bar)
    bbox = draw.textbbox((0, 0), text)
    tx   = max(0, (width - (bbox[2] - bbox[0])) // 2)
    draw.text((tx, 5), text, fill=(255, 230, 80))
    return np.asarray(bar)


def make_seg_inference_grid(
    orig_pil:     Image.Image,
    seg_pil:      Image.Image,
    pred_pil:     Image.Image,
    raw_pred_pil: Image.Image,
    size,
) -> Image.Image:
    """
    Create a single image with 4 panels SIDE BY SIDE:

        ┌──────────┬──────────┬──────────┬─────────────┐
        │ ORIGINAL │  SEG MAP │PREDICTED │ RAW SEG GEN │  ← label bars (28 px)
        ├──────────┼──────────┼──────────┼─────────────┤
        │          │          │          │             │
        │ size_w x │ size_w x │ size_w x │  size_w x   │  ← image panels
        │ size_h   │ size_h   │ size_h   │  size_h     │
        └──────────┴──────────┴──────────┴─────────────┘

    RAW SEG GEN: same seg conditioning but empty prompt — shows pure
    segmentation adherence with no text influence. Mirrors training val grid.
    Total image: (4*size_w) wide, (size_h + 28) tall.

    `size`: int (square) or (width, height) pair — see normalize_size.
    """
    size_w, size_h = normalize_size(size)
    target = (size_w, size_h)   # PIL .resize() order: (width, height)
    orig     = orig_pil.resize(target).convert("RGB")
    seg      = seg_pil.resize(target).convert("RGB")
    pred     = pred_pil.resize(target).convert("RGB")
    raw_pred = raw_pred_pil.resize(target).convert("RGB")

    imgs_row  = np.concatenate([np.asarray(orig), np.asarray(seg),
                                 np.asarray(pred), np.asarray(raw_pred)], axis=1)

    label_row = np.concatenate([
        _seg_label_bar(size_w, "ORIGINAL"),
        _seg_label_bar(size_w, "SEG MAP"),
        _seg_label_bar(size_w, "PREDICTED"),
        _seg_label_bar(size_w, "RAW SEG GEN"),
    ], axis=1)

    grid = np.concatenate([label_row, imgs_row], axis=0)
    return Image.fromarray(grid)


def _load_seg_map(seg_path: Path, size, palette: torch.Tensor, device,
                  pad_id: int = 0, resize_mode: str = "letterbox") -> torch.Tensor:
    """
    Load a PRE-COMPUTED segmentation map (raw class-ID PNG) and colourise it.

    Mirrors src/data/local_seg.py's SegJsonDataset._load_seg_colormap EXACTLY
    (same raw read, same square_id_map geometry, same seg_colorize_ids call,
    same palette) — this is what guarantees a map loaded here produces the
    identical conditioning signal training saw for the same file, AS LONG AS
    resize_mode matches what that checkpoint was trained with (see the
    resize_mode-vs-checkpoint check in main()).

      1. RAW pixel read via np.asarray, NO .convert("L") — the real
         Grounded-SAM/CARLA masks are 16-bit PNGs (PIL mode I;16, ids 0..28)
         and I;16 -> L behaviour differs between Pillow versions; reading raw
         works for both I;16 and the 8-bit "L" maps identically everywhere.
      2. Squared via square_id_map() (src/data/transforms.py) using the SAME
         `resize_mode` the RGB display panel uses — "letterbox" (SquarePad
         geometry, pad_id fill) or "CenterCrop" (torchvision Resize+CenterCrop
         geometry). NEAREST-only throughout — averaging class ids would
         fabricate classes that don't exist.

    Returns [1, 3, size, size] float tensor in [0, 1].
    """
    ids_np = np.asarray(Image.open(seg_path)).astype(np.int64)
    if ids_np.max() > 255:
        raise ValueError(
            f"{seg_path}: max pixel value {ids_np.max()} is not a class id "
            "(expected 0..255). Is this really a raw class-ID map?"
        )
    ids_pil = Image.fromarray(ids_np.astype(np.uint8), mode="L")
    ids_pil = square_id_map(ids_pil, size, resize_mode, pad_id)

    ids = torch.from_numpy(np.asarray(ids_pil, dtype=np.int64)).unsqueeze(0)   # [1, H, W]
    colour = seg_colorize_ids(ids, palette)                                    # [1, 3, size, size] in [0,1]
    return colour.to(device)


def _find_trained_resize_mode(ckpt_path: Path) -> str | None:
    """
    Determine which resize_mode the checkpoint at ckpt_path was TRAINED with.
    Checks two sources, in order:

      1. <ckpt_path>/info.txt -- written only for best_model/ saves
         (grounded_sam_training.py's save_seg_ckpt_and_grid, is_best=True
         only). Absent for other checkpoint folders (checkpoint-epochN/stepM/,
         checkpoint-epochN/checkpoint-epochN/).
      2. training_params.txt, walked up from ckpt_path -- written once per
         run, at the run's root folder. Covers checkpoints info.txt misses.

    A found file with no resize_mode line (a run predating this feature) is
    treated as 'letterbox' -- the only mode that existed before.

    Returns the mode string, or None if neither file could be found.
    """
    candidates = []
    info = ckpt_path / "info.txt"
    if info.exists():
        candidates.append(info)
    p = ckpt_path
    for _ in range(4):   # best_model/ is 1 level below the run root;
        p = p.parent     # checkpoint-epochN/stepM/ is 2 levels below -- walk
        tp = p / "training_params.txt"   # up a few levels to cover both shapes.
        if tp.exists():
            candidates.append(tp)
            break

    for c in candidates:
        for line in c.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith("resize_mode"):
                # training_params.txt has trailing comment text after the
                # value ("letterbox  (letterbox=SquarePad, ...)") -- take
                # only the first token.
                return line.split(":", 1)[1].strip().split()[0]

    return "letterbox" if candidates else None


# ===================================================================== #
#  MAIN                                                                   #
# ===================================================================== #

@hydra.main(config_path="configs", config_name="inference_grounded_sam")
def main(cfg):
    # Resolve device LOUDLY: prints full GPU diagnostics and raises a clear
    # error (instead of silently running on CPU) unless device=cpu was
    # explicitly passed. See src/utils.py resolve_device() for why this exists.
    device = resolve_device(cfg.device)

    # Resolve output_dir from the original repo root (not Hydra's run dir),
    # then make it UNIQUE PER RUN (every inference run
    # stores its results in its own folder — two runs can never overwrite or
    # mix outputs). A timestamped subfolder is appended to the configured
    # base path; Hydra's own run dir already embeds the same date/time format,
    # so the two are easy to correlate when debugging a specific run.
    _root = get_original_cwd()
    _out = Path(cfg.inference.output_dir)
    _out = _out if _out.is_absolute() else Path(_root) / _out
    output_dir = _out / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  grounded_sam_inference.py")
    print(f"  Device     : {device}")
    print(f"  Output dir : {output_dir}")
    print(f"{'='*60}\n")

    # ------------------------------------------------------------------ #
    # resize_mode — "letterbox" or "CenterCrop". MUST match the technique  #
    # the loaded checkpoint was TRAINED with, or the seg map fed to the    #
    # model here won't align with what the LoRA/mapper learned. Checked    #
    # before any model loading. Three outcomes, each prints a distinct     #
    # message (verified / mismatch / unverified).                          #
    # ------------------------------------------------------------------ #
    size        = cfg.size
    size_w, size_h = normalize_size(size)   # (width, height); square unless resize_mode=aspect
    resize_mode = cfg.get("resize_mode", "letterbox")
    # Resolve ckpt_path against _root (the ORIGINAL repo-root cwd): hydra:
    # job: chdir: true has already moved the process cwd to the Hydra run
    # dir by the time this line runs. Same resolution rule
    # add_lora_from_config (src/utils.py) uses.
    _ckpt_path_resolved = Path(cfg.ckpt_path)
    if not _ckpt_path_resolved.is_absolute():
        _ckpt_path_resolved = Path(_root) / _ckpt_path_resolved
    _trained_mode = _find_trained_resize_mode(_ckpt_path_resolved)
    if _trained_mode is None:
        print(f"[WARN] could not determine which resize_mode this checkpoint was "
              f"trained with (no info.txt or training_params.txt found near "
              f"{cfg.ckpt_path}) -- proceeding with resize_mode='{resize_mode}' "
              f"UNVERIFIED against the checkpoint.")
    elif _trained_mode != resize_mode:
        print(f"[WARN] resize_mode mismatch: this checkpoint was TRAINED with "
              f"'{_trained_mode}' but inference.resize_mode='{resize_mode}' -- "
              f"the seg map's geometry will NOT match what this model learned. "
              f"Pass resize_mode={_trained_mode} to fix, or use it intentionally "
              f"as an experiment.")
    else:
        print(f"[resize_mode] {resize_mode} (verified: matches this checkpoint's training run)")

    # ── Pick LOCAL model folders vs HUB ids from the local_files_only flag ──────
    # Same logic as training, so inference uses the SAME model (parity rule).
    # Local paths are made absolute from the repo root; when offline we also
    # export HF_HUB_OFFLINE so nothing can touch the network.
    # The Grounded-SAM encoder has no HF `model` to load (it's a training/
    # inference-only slot filler — see configs/lora/encoder/grounded_sam.yaml),
    # so only rewrite encoder.model when the encoder config actually has that
    # key. Mirrors the same guard in grounded_sam_training.py.
    _enc_has_model = "model" in cfg.lora.struct.encoder
    # Tier 2 (live Grounded-SAM) keys -- see the matching guard in
    # grounded_sam_training.py for the full reasoning.
    _enc_has_live_models = "grounding_dino_model" in cfg.lora.struct.encoder
    if cfg.local_files_only:
        os.environ["HF_HUB_OFFLINE"]      = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        cfg.model.model_name = os.path.join(_root, cfg.base_model_path)
        if _enc_has_model:
            cfg.lora.struct.encoder.model = os.path.join(_root, cfg.seg_model_path)
        if _enc_has_live_models:
            cfg.lora.struct.encoder.grounding_dino_model = os.path.join(
                _root, "checkpoints/local_models/grounding-dino-base")
            cfg.lora.struct.encoder.sam_model = os.path.join(
                _root, "checkpoints/local_models/sam-vit-huge")
        # vae_path: same relative-path-under-Hydra's-chdir problem as
        # base_model_path/seg_model_path above -- make absolute here too, but
        # only when actually set (default null = use the base model's own VAE).
        if cfg.model.get("vae_path"):
            cfg.model.vae_path = os.path.join(_root, cfg.model.vae_path)
    else:
        cfg.model.model_name = cfg.base_model_name
        if _enc_has_model:
            cfg.lora.struct.encoder.model = cfg.seg_model_name
    _vae_path_display = cfg.model.get("vae_path") or "(none -- using base model's own VAE)"
    print(f"[model] base = {cfg.model.model_name}")
    print(f"[model] vae_path = {_vae_path_display}")
    print(f"[model] seg encoder = {cfg.lora.struct.encoder.model if _enc_has_model else cfg.lora.struct.encoder._target_}")
    print(f"[model] local_files_only = {cfg.local_files_only}")

    # ------------------------------------------------------------------ #
    # Build model from Hydra config (SD15 + LoRA structure)               #
    # ------------------------------------------------------------------ #
    # Capture the resolved model-name STRING before instantiate() replaces
    # cfg.model with the built SD15 object (run_params.txt needs the string).
    _base_model_name = str(cfg.model.model_name)
    cfg = hydra.utils.instantiate(cfg)
    model: ModelBase = cfg.model
    model = model.to(device)
    model.pipe.to(device)
    model.unet.requires_grad_(False)
    model.unet.eval()

    # ------------------------------------------------------------------ #
    # Load trained LoRA + mapper weights from checkpoint                  #
    # add_lora_from_config reads cfg.ckpt_path and loads:                 #
    #   <ckpt_path>/struct/lora-checkpoint.pt                             #
    #   <ckpt_path>/struct/mapper-checkpoint.pt                           #
    # ------------------------------------------------------------------ #
    cfg_mask = add_lora_from_config(model, cfg, device, dtype=torch.float32)
    print(f"Loaded checkpoint. cfg_mask = {cfg_mask}\n")

    for e in model.encoders: e.eval()
    for m in model.mappers:  m.eval()

    # ------------------------------------------------------------------ #
    # Collect input seg maps (REQUIRED) + optional raw images + prompts    #
    # from JSON or direct lists. seg_path is required on every entry --   #
    # this script never computes a map live, only loads a provided one.   #
    # ------------------------------------------------------------------ #
    entries = []

    if cfg.inference.get("json_file") and cfg.inference.json_file:
        json_path = Path(cfg.inference.json_file)
        if not json_path.is_absolute():
            json_path = Path(_root) / json_path   # _root = original cwd (repo root), not Hydra's run dir
        with open(json_path, "r", encoding="utf-8") as f:
            data = [json.loads(line) for line in f if line.strip()]
        skipped = 0
        for item in data:
            if not item.get("seg_path"):
                skipped += 1
                continue
            entries.append({
                "seg_path":   item["seg_path"],
                "image_path": item.get("raw_image_path"),   # optional, display only
                "prompt":     item.get("prompt", ""),
            })
        if skipped:
            print(f"[WARN] {skipped} entries had no 'seg_path' -- skipped (this script requires a provided map).")
        print(f"Loaded {len(entries)} entries from JSON: {json_path}")

    elif cfg.inference.get("seg_maps") and cfg.inference.seg_maps:
        seg_maps = cfg.inference.seg_maps
        images   = list(cfg.inference.get("images") or [])
        prompts  = list(cfg.inference.get("prompts") or [])
        for i, seg_path in enumerate(seg_maps):
            entries.append({
                "seg_path":   seg_path,
                "image_path": images[i] if i < len(images) else None,
                "prompt":     prompts[i] if i < len(prompts) else "",
            })
        print(f"Processing {len(entries)} seg maps from config.")

    if not entries:
        print("[ERROR] No input seg maps provided (this script requires a PROVIDED map, not a raw photo).")
        print("  Set inference.json_file=data/seg_training/test.jsonl  (entries need 'seg_path')")
        print("  or  \"inference.seg_maps=[data/raw_seg/000417/000417_seg_map.png]\"")
        return

    # Raw-image preprocessing — DISPLAY ONLY (never fed to any model): squares
    # an optional raw image with the SAME geometry as the seg map, so the
    # ORIGINAL and SEG MAP grid panels visually align. Output: PIL.Image.
    display_preprocess = build_seg_display_preprocess(size=size, resize_mode=resize_mode)

    generator = torch.Generator(device=device).manual_seed(cfg.seed)

    # ── Controllability metric (mIoU) accumulator ───────────────────────────
    # One mIoU per generated image (structure asked for vs. structure produced);
    # the mean over all entries is this model's controllability score, written to
    # metrics.txt at the end. See src/utils.py compute_miou.
    # ONLY meaningful if the loaded encoder can itself segment the GENERATED
    # image (scoring output quality, unrelated to "computing the input map
    # live" -- that's what this file no longer does). The Grounded-SAM Tier-1
    # slot filler (src/encoders/grounded_sam_encoder.py) cannot do this
    # (live_available=False), so it's skipped automatically -- no crash.
    #
    # Palette must match whatever palette the seg maps were saved with, or
    # _load_seg_map below raises "class id >= palette size". Required on
    # this branch, no fallback palette (Cityscapes has 19 colours, CARLA
    # has 29). Mirrors local_seg.py's SegJsonDataModule.
    _classes_file = cfg.get("classes_file", None)
    if _classes_file is None:
        raise ValueError(
            "classes_file is required (configs/inference_grounded_sam.yaml "
            "sets it by default -- did you override it away?). No fallback "
            "palette exists on this branch."
        )
    from src.encoders.grounded_sam_encoder import load_grounded_sam_palette
    _cf = Path(_root) / _classes_file
    _class_names, _palette_list = load_grounded_sam_palette(_cf)
    _palette = seg_palette_tensor(_palette_list).to(device)
    print(f"[palette] loaded {len(_palette_list)} classes from {_classes_file}")
    _enc0 = getattr(model.encoders[0], "module", model.encoders[0])
    _live_seg_available = getattr(_enc0, "live_available", True)
    if not _live_seg_available:
        print("[INFO] Encoder has no live segmentation path (live_available=False) "
              "-- mIoU controllability metric will be skipped for this run.")
    miou_lines = []                                # "stem: 0.6123" per image
    mious      = []                                # values, for the mean

    # ------------------------------------------------------------------ #
    # run_params.txt — the full recipe of THIS run, saved with its outputs #
    # Every generation-affecting setting is                                #
    # recorded so any result folder is self-documenting: you can look at   #
    # an image weeks later and know exactly how it was made, or re-run the #
    # identical command. Written BEFORE generating, so even a crashed run  #
    # leaves its recipe behind.                                            #
    # ------------------------------------------------------------------ #
    _inf = cfg.inference
    _params_lines = [
        "grounded_sam_inference.py run parameters (grounded_sam pipeline)",
        f"timestamp                    : {datetime.now().isoformat(timespec='seconds')}",
        f"ckpt_path                    : {cfg.ckpt_path}",
        f"seed                         : {cfg.seed}",
        f"size                         : {cfg.size}",
        f"resize_mode                  : {resize_mode}  (letterbox=SquarePad, CenterCrop=original repo recipe)",
        f"num_inference_steps          : {_inf.get('num_inference_steps', 50)}",
        f"guidance_scale               : {_inf.get('guidance_scale', 7.5)}",
        f"n_samples                    : {_inf.get('n_samples', 1)}",
        f"conditioning_kernel_size     : {_inf.get('conditioning_kernel_size', 0)}  (softening kernel; 0 = off)",
        f"lora_scale_start             : {_inf.get('lora_scale_start', 1.0)}  (conditioning scale, early steps)",
        f"lora_scale_end               : {_inf.get('lora_scale_end', 1.0)}  (conditioning scale, late steps)",
        f"lora_scale_decay_start_frac  : {_inf.get('lora_scale_decay_start_frac', 0.3)}",
        f"base model                   : {_base_model_name}",
        f"classes_file                 : {_classes_file or '(none — Cityscapes fallback palette)'}",
        f"seg_pad_id                   : {cfg.get('seg_pad_id', 0)}  (letterbox fill class for non-square maps)",
        f"input mode                   : {'json_file: ' + str(_inf.json_file) if _inf.get('json_file') else 'direct seg_maps list'}",
        f"entries                      : {len(entries)}",
        "",
        "inputs (seg_path | raw_image_path | prompt):",
    ]
    for e_ in entries:
        _params_lines.append(
            f"  {e_['seg_path']} | {e_.get('image_path') or '-'} | {e_.get('prompt', '')!r}"
        )
    (output_dir / "run_params.txt").write_text(
        "\n".join(_params_lines) + "\n", encoding="utf-8"
    )
    print(f"[params] wrote {output_dir / 'run_params.txt'}")

    # ------------------------------------------------------------------ #
    # Inference loop                                                       #
    # ------------------------------------------------------------------ #
    for entry in tqdm(entries, desc="Generating"):
        seg_path = Path(entry["seg_path"])
        if not seg_path.is_absolute():
            seg_path = Path(_root) / seg_path
        prompt = entry["prompt"]
        stem   = seg_path.stem

        if not seg_path.exists():
            print(f"[WARN] seg_path not found: {seg_path} — skipping.")
            continue

        print(f"\n  Seg map: {seg_path.name}")
        print(f"  Prompt : {prompt!r}")

        # ---- Optional raw image, DISPLAY ONLY -- no model ever sees it ----
        img_path = entry.get("image_path")
        if img_path:
            img_path = Path(img_path)
            if not img_path.is_absolute():
                img_path = Path(_root) / img_path
            if img_path.exists():
                # Square with the SAME technique as the seg map (resize_mode),
                # so the ORIGINAL panel visually matches what SEG MAP shows
                # (previously this panel used a plain stretch-resize while the
                # seg map was letterboxed -- the two panels didn't align).
                orig_pil = display_preprocess(Image.open(img_path).convert("RGB"))
            else:
                print(f"[WARN] raw_image_path not found: {img_path} — using blank placeholder.")
                orig_pil = Image.new("RGB", (size_w, size_h), color=(40, 40, 40))
        else:
            orig_pil = Image.new("RGB", (size_w, size_h), color=(40, 40, 40))

        with torch.no_grad():

            # ---- Step 1: Load + colourise the PROVIDED seg map ----
            # No model runs here -- this is a file load + palette lookup, not a
            # live computation. See _load_seg_map (mirrors local_seg.py exactly).
            # pad_id: letterbox fill class for non-square maps (base config
            # `seg_pad_id`, default 0 = CARLA Unlabeled) — must match training.
            seg_tensor = _load_seg_map(seg_path, size, _palette, device,
                                       pad_id=int(cfg.get("seg_pad_id", 0)),
                                       resize_mode=resize_mode)   # [1,3,size,size] in [0,1]
            seg_pil    = TF.to_pil_image(seg_tensor[0].cpu().float().clamp(0, 1))

            # ---- Step 2: Generate image (prompt-conditioned) ----
            # cs=[seg_tensor], skip_encode=True: the PROVIDED map is used AS-IS
            # as the conditioning signal -- no encoder runs, matching exactly
            # how training consumes pre-saved maps (skip_encode=True there too).
            preds = model.sample(
                prompt=[prompt],
                num_images_per_prompt=cfg.inference.n_samples,
                cs=[seg_tensor],
                skip_encode=True,
                generator=generator,
                cfg_mask=cfg_mask,
                # Explicit height/width: sample_easy forwards **kwargs straight
                # to the underlying diffusers pipeline, which otherwise DEFAULTS
                # to unet.config.sample_size * vae_scale_factor for BOTH
                # dimensions (i.e. always square) when they're not passed. For
                # resize_mode="aspect" (non-square), omitting these would
                # silently generate a square image misaligned with the
                # non-square conditioning map.
                height=size_h,
                width=size_w,
                num_inference_steps=cfg.inference.get("num_inference_steps", 50),
                guidance_scale=cfg.inference.get("guidance_scale", 7.5),
                # negative_prompt: forwarded via **kwargs straight to the underlying
                # diffusers pipeline (sample_easy has no explicit param for it, but
                # it passes **kwargs through to self.pipe(...), which natively
                # supports negative_prompt). null = diffusers' own CFG default.
                negative_prompt=cfg.inference.get("negative_prompt", None),
                # Generation-quality knobs (default = no-op; see model.py
                # sample_easy docstring). conditioning_kernel_size softens hard
                # seg-map edges; lora_scale_start/end decays structure-conditioning
                # strength over the denoising trajectory so late steps can lean on
                # SD's own prior for object appearance instead of a flat map.
                conditioning_kernel_size=cfg.inference.get("conditioning_kernel_size", 0),
                lora_scale_start=cfg.inference.get("lora_scale_start", 1.0),
                lora_scale_end=cfg.inference.get("lora_scale_end", 1.0),
                lora_scale_decay_start_frac=cfg.inference.get("lora_scale_decay_start_frac", 0.3),
            )

            # ---- Step 2b: RAW SEG GEN (empty prompt) ----
            # Same seg conditioning, empty text — shows pure segmentation adherence
            # with no text influence. Matches the 4th panel of the training grid.
            raw_preds = model.sample(
                prompt=[""],
                num_images_per_prompt=cfg.inference.n_samples,
                cs=[seg_tensor],
                skip_encode=True,
                generator=torch.Generator(device=device).manual_seed(cfg.seed),
                cfg_mask=cfg_mask,
                height=size_h,
                width=size_w,
                num_inference_steps=cfg.inference.get("num_inference_steps", 50),
                guidance_scale=cfg.inference.get("guidance_scale", 7.5),
                conditioning_kernel_size=cfg.inference.get("conditioning_kernel_size", 0),
                lora_scale_start=cfg.inference.get("lora_scale_start", 1.0),
                lora_scale_end=cfg.inference.get("lora_scale_end", 1.0),
                lora_scale_decay_start_frac=cfg.inference.get("lora_scale_decay_start_frac", 0.3),
            )

        # ---- Step 2c: mIoU (controllability) -- ONLY if the encoder can score it --
        # TARGET ids = the PROVIDED seg map (palette-inverted, exact, no extra
        # model call). PREDICTION ids = the encoder re-run on the GENERATED image
        # (first sample) -- this is scoring the OUTPUT, unrelated to "computing
        # the input map live" (which this script no longer does at all). Skipped
        # entirely when the encoder has no live path (e.g. Grounded-SAM Tier 1).
        if _live_seg_available:
            target_ids = seg_ids_from_colormap(seg_tensor[0], _palette)   # [size_h,size_w]
            gen_t = (TF.to_tensor(preds[0].resize((size_w, size_h)).convert("RGB"))
                     .unsqueeze(0).to(device) * 2.0 - 1.0)                # [1,3,H,W] [-1,1]
            with torch.no_grad():
                pred_ids = _enc0.label_ids(gen_t)[0]                      # [size_h,size_w]
            miou = compute_miou(pred_ids, target_ids, num_classes=int(_palette.shape[0]))
            mious.append(miou)
            miou_lines.append(f"{stem}: {miou:.4f}")
            print(f"  mIoU  : {miou:.4f}")

        # ---- Step 3: Save outputs ----------------------------------------
        # orig_pil is already (size_w,size_h) via display_preprocess above.
        orig_display = orig_pil.convert("RGB")

        save_generated_only = cfg.inference.get("save_generated_only", False)
        _target_wh = (size_w, size_h)   # PIL .resize() order: (width, height)

        for k, (pred_pil, raw_pred_pil) in enumerate(zip(preds, raw_preds)):
            suffix = f"_{k}" if len(preds) > 1 else ""

            if save_generated_only:
                # Mirror the exact path from the JSON so folder structure is preserved.
                rel = Path(entry["seg_path"])
                out_path = output_dir / rel.parent / f"{rel.stem}{suffix}{rel.suffix}"
                out_path.parent.mkdir(parents=True, exist_ok=True)
                pred_pil.resize(_target_wh).save(out_path, quality=95)
                print(f"  -> {out_path}")
            else:
                # Full debug output: 4-panel grid + individual panels.
                grid = make_seg_inference_grid(orig_display, seg_pil, pred_pil, raw_pred_pil, size)
                grid_path = output_dir / f"{stem}{suffix}_grid.jpg"
                grid.save(grid_path, quality=95)

                orig_display.save(
                    output_dir / f"{stem}_original.jpg"
                )
                seg_pil.resize(_target_wh).convert("RGB").save(
                    output_dir / f"{stem}_seg.jpg"
                )
                pred_pil.resize(_target_wh).save(
                    output_dir / f"{stem}{suffix}_predicted.jpg", quality=95
                )
                raw_pred_pil.resize(_target_wh).save(
                    output_dir / f"{stem}{suffix}_raw_seg_gen.jpg", quality=95
                )
                print(f"  -> {grid_path}")

    # ── Per-model controllability summary ───────────────────────────────────
    # The mean mIoU over every scored image IS this model's controllability
    # score — the single number you rank models by. Written next to the results
    # so a run's quality is readable without re-opening the images.
    if mious:
        mean_miou = float(np.mean(mious))
        summary = (f"mean mIoU (n={len(mious)}): {mean_miou:.4f}\n\n"
                   + "\n".join(miou_lines) + "\n")
        (output_dir / "metrics.txt").write_text(summary, encoding="utf-8")
        print(f"\nmean mIoU over {len(mious)} images: {mean_miou:.4f}")
        print(f"  -> {output_dir / 'metrics.txt'}")

    print(f"\nDone. All results saved to: {output_dir}")


if __name__ == "__main__":
    main()
