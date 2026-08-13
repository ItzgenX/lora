"""
grounded_sam_fid_eval.py
--------------------------
Compute FID (Frechet Inception Distance) between real photos and this
checkpoint's generated images, over a large sample -- the Grounded-SAM twin
of seg_fid_eval.py (segformer branch). See that file's docstring for the
full reasoning; only the encoder-specific bits differ here (29-class CARLA
palette via classes_file, pad_id for non-square maps).

Reuses the SAME inference_grounded_sam Hydra config (base model, VAE,
checkpoint format, resize_mode, classes_file) grounded_sam_inference.py
uses via configs/fid_grounded_sam.yaml's `defaults: - inference_grounded_sam`.

ONE-TIME SETUP: torch_fidelity downloads Inception-v3 weights from GitHub on
first use, cached at ~/.cache/torch/hub/checkpoints/ afterward.

USAGE:
  python grounded_sam_fid_eval.py \\
      ckpt_path=outputs/train/grounded_sam_aspect/runs/YYYY-MM-DD/HH-MM-SS/best_model \\
      resize_mode=aspect \\
      fid.json_file=data/grounded_sam/test.jsonl \\
      fid.n_samples=1000
"""

import hydra
import os
import json
from datetime import datetime
from pathlib import Path

import torch
from PIL import Image

from hydra.utils import get_original_cwd
from src.model import ModelBase
from src.utils import add_lora_from_config, resolve_device
from src.data.transforms import build_seg_display_preprocess, normalize_size
from src.data.seg_palette import seg_palette_tensor
from src.encoders.grounded_sam_encoder import load_grounded_sam_palette
# _load_seg_map is the exact same raw-class-ID-PNG -> colourised-map loader
# grounded_sam_inference.py uses -- imported, not reimplemented.
from grounded_sam_inference import _load_seg_map


@hydra.main(config_path="configs", config_name="fid_grounded_sam")
def main(cfg):
    device = resolve_device(cfg.device)
    _root = get_original_cwd()

    _out = Path(cfg.fid.output_dir)
    _out = _out if _out.is_absolute() else Path(_root) / _out
    output_dir = _out / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    real_dir = output_dir / "real"
    gen_dir = output_dir / "generated"
    real_dir.mkdir(parents=True, exist_ok=True)
    gen_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  grounded_sam_fid_eval.py")
    print(f"  Device     : {device}")
    print(f"  Output dir : {output_dir}")
    print(f"{'='*60}\n")

    n_samples = int(cfg.fid.n_samples)
    if n_samples < 1000:
        print(f"[WARNING] fid.n_samples={n_samples} is below the ~1000 floor FID needs to be "
              f"stable. Treat the printed score as a pipeline smoke check only.")

    size = cfg.size
    size_w, size_h = normalize_size(size)
    resize_mode = cfg.get("resize_mode", "aspect")
    seg_pad_id = cfg.get("seg_pad_id", 0)

    # ── Palette: required, no fallback on this branch (29 CARLA classes) ────
    _classes_file = cfg.get("classes_file", None)
    if _classes_file is None:
        raise ValueError("classes_file is required (configs/inference_grounded_sam.yaml sets "
                          "it by default -- did you override it away?).")
    _cf = Path(_root) / _classes_file
    _class_names, _palette_list = load_grounded_sam_palette(_cf)
    print(f"[palette] loaded {len(_palette_list)} classes from {_classes_file}")

    # ── Pick LOCAL model folders vs HUB ids from local_files_only ────────────
    if cfg.local_files_only:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        cfg.model.model_name = os.path.join(_root, cfg.base_model_path)
        if cfg.model.get("vae_path"):
            cfg.model.vae_path = os.path.join(_root, cfg.model.vae_path)
    else:
        cfg.model.model_name = cfg.base_model_name
    print(f"[model] base = {cfg.model.model_name}")
    print(f"[model] local_files_only = {cfg.local_files_only}")

    cfg = hydra.utils.instantiate(cfg)
    model: ModelBase = cfg.model
    model = model.to(device)
    model.pipe.to(device)
    model.unet.requires_grad_(False)
    model.unet.eval()

    cfg_mask = add_lora_from_config(model, cfg, device, dtype=torch.float32)
    print(f"Loaded checkpoint. cfg_mask = {cfg_mask}\n")
    for e in model.encoders: e.eval()
    for m in model.mappers: m.eval()

    # ── Load entries: raw_image_path is REQUIRED here for FID's real-image pool ──
    json_path = Path(cfg.fid.json_file)
    if not json_path.is_absolute():
        json_path = Path(_root) / json_path
    with open(json_path, "r", encoding="utf-8-sig") as f:
        all_entries = [json.loads(line) for line in f if line.strip()]

    entries = []
    skipped = 0
    for item in all_entries:
        if not item.get("seg_path") or not item.get("raw_image_path"):
            skipped += 1
            continue
        entries.append(item)
        if len(entries) >= n_samples:
            break
    if skipped:
        print(f"[WARNING] skipped {skipped} entries missing seg_path/raw_image_path")
    if len(entries) < n_samples:
        print(f"[WARNING] manifest only yielded {len(entries)} usable entries "
              f"(requested n_samples={n_samples})")
    print(f"Generating {len(entries)} images for FID...\n")

    display_preprocess = build_seg_display_preprocess(size=size, resize_mode=resize_mode)
    generator = torch.Generator(device=device).manual_seed(cfg.seed)
    _palette = seg_palette_tensor(_palette_list).to(device)

    for i, item in enumerate(entries):
        seg_path = Path(item["seg_path"])
        if not seg_path.is_absolute():
            seg_path = Path(_root) / seg_path
        img_path = Path(item["raw_image_path"])
        if not img_path.is_absolute():
            img_path = Path(_root) / img_path

        # [1,3,size_h,size_w] in [0,1] -- same loader, same palette, same
        # geometry grounded_sam_inference.py uses for the same file.
        seg_tensor = _load_seg_map(seg_path, size, _palette, device,
                                    pad_id=seg_pad_id, resize_mode=resize_mode)

        with torch.no_grad():
            preds = model.sample(
                prompt=[item.get("prompt", "")], num_images_per_prompt=1,
                cs=[seg_tensor], skip_encode=True, generator=generator,
                cfg_mask=cfg_mask,
                num_inference_steps=cfg.inference.get("num_inference_steps", 50),
                guidance_scale=cfg.inference.get("guidance_scale", 7.5),
                negative_prompt=cfg.inference.get("negative_prompt", None),
                height=size_h, width=size_w,
            )
        preds[0].save(gen_dir / f"{i:06d}.png")

        real_pil = Image.open(img_path).convert("RGB")
        real_pil = display_preprocess(real_pil)
        real_pil.save(real_dir / f"{i:06d}.png")

        if (i + 1) % 50 == 0:
            print(f"  ...{i + 1}/{len(entries)}")

    print(f"\nGenerated {len(entries)} pairs -> {gen_dir} / {real_dir}\n")

    import torch_fidelity
    print("Computing FID (downloads Inception-v3 weights on first run)...")
    metrics = torch_fidelity.calculate_metrics(
        input1=str(real_dir), input2=str(gen_dir),
        cuda=(device.type == "cuda"), fid=True, verbose=False,
    )
    fid_score = metrics["frechet_inception_distance"]
    print(f"\n{'='*60}")
    print(f"  FID = {fid_score:.4f}   (n_samples={len(entries)}, lower = closer to real)")
    print(f"{'='*60}\n")

    (output_dir / "fid_result.txt").write_text(
        f"FID = {fid_score:.4f}\n"
        f"n_samples requested = {n_samples}\n"
        f"n_samples used      = {len(entries)}\n"
        f"ckpt_path            = {cfg.ckpt_path}\n"
        f"resize_mode          = {resize_mode}\n"
        f"size                 = {list(size)}\n"
        f"classes_file         = {_classes_file}\n"
        f"timestamp            = {datetime.now().isoformat(timespec='seconds')}\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
