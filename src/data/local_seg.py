"""
src/data/local_seg.py
---------------------
Dataset: load (RGB image, pre-computed segmentation map, prompt) triplets for
training the segmentation-conditioned LoRAdapter. The maps are generated OFFLINE
(on this branch, by Grounded-SAM — see GROUNDED_SAM.md) and read here directly.

Two segmentation-specific points, both critical:

  1. The saved map is a RAW CLASS-ID PNG (8-bit, values 0..N-1), NOT a colour
     image. Colourised at load time with the Grounded-SAM/CARLA class
     palette (configs/grounded_sam_classes.json, loaded via `classes_file` —
     REQUIRED, no fallback: CARLA class ids go up to 28, Cityscapes only has
     19 colours, so a Cityscapes palette cannot cover this branch's data),
     producing a 3-channel RGB map in [0,1] via the shared seg_colorize_ids.

  2. Resizing a class-ID map MUST use NEAREST interpolation. Averaging categorical
     class ids is meaningless: the mean of "road"=0 and "car"=6 is 3, a DIFFERENT
     class that isn't in the image. NEAREST preserves exact labels.

Returned by __getitem__:
  {
    "jpg"    : RGB tensor   [3, H, W] in [-1, 1]   (standard training format)
    "seg"    : colour map   [3, H, W] in [0, 1]    (palette RGB)
    "caption": prompt string
  }
grounded_sam_training.py reads batch["seg"].
"""

import json
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader

from src.data.seg_palette import seg_palette_tensor, seg_colorize_ids
from src.data.transforms import square_id_map, build_seg_preprocess


class SegJsonDataset(Dataset):
    """
    Load image + pre-computed segmentation-ID map pairs from a JSON manifest.

    JSONL format — one JSON object per line (key names are configurable via
    image_key / seg_key / prompt_key; defaults shown):
        {"raw_image_path": "...jpg", "seg_path": "...png", "prompt": "..."}
        {"raw_image_path": "...jpg", "seg_path": "...png", "prompt": "..."}

    Paths resolve: absolute as-is, else relative to project_root, else to json_dir.
    """

    def __init__(
        self,
        json_file: Path,
        image_transform,           # torchvision Compose for the RGB image (-> [-1,1])
        size: int = 512,           # square side for the conditioning colour map
        project_root: Path = None,
        palette: list = None,      # class-id -> RGB; REQUIRED (no fallback —
                                   # see module docstring), loaded from your
                                   # real classes_file by SegJsonDataModule
        image_root: Path = None,
        image_key: str = "raw_image_path",   # JSONL key for the source RGB image
        seg_key: str = "seg_path",           # JSONL key for the class-ID seg map
        prompt_key: str = "prompt",          # JSONL key for the text caption
        pad_id: int = 0,                     # class id used to letterbox non-square
                                             # maps (CARLA 0 = Unlabeled; see
                                             # _load_seg_colormap step 2)
        resize_mode: str = "letterbox",      # "letterbox" or "CenterCrop" — MUST
                                             # match the RGB image_transform's mode
                                             # (SegJsonDataModule builds both from
                                             # the same key, so this can't drift).
    ):
        self.json_file    = Path(json_file)
        self.pad_id       = pad_id
        self.resize_mode  = resize_mode
        self.json_dir     = self.json_file.parent
        self.project_root = Path(project_root) if project_root else self.json_dir
        self.image_root   = Path(image_root) if image_root else None
        self.size         = size
        self.image_transform = image_transform
        # Configurable manifest keys: default to the project-standard names
        # (raw_image_path/seg_path/prompt), override via the data config if a
        # manifest ever used different key names.
        self.image_key    = image_key
        self.seg_key      = seg_key
        self.prompt_key   = prompt_key

        # palette is REQUIRED -- no fallback (see module docstring). Fail
        # loudly here rather than deep inside a DataLoader worker.
        if palette is None:
            raise ValueError(
                "SegJsonDataset requires a palette (no default/fallback exists "
                "on this branch) -- pass classes_file to SegJsonDataModule so "
                "it can load your real Grounded-SAM/CARLA class set."
            )
        self.seg_palette  = seg_palette_tensor(palette)
        self.num_classes  = self.seg_palette.shape[0]   # 29 for the locked CARLA taxonomy

        with open(self.json_file, "r", encoding="utf-8") as f:
            self.items = [json.loads(line) for line in f if line.strip()]

        # Early, loud missing-file check — catch a bad manifest before the
        # dataloader workers surface a confusing deep stack trace mid-training.
        missing_img = missing_seg = 0
        for item in self.items:
            if not self._seg_resolve(item[self.image_key]).exists():
                print(f"[SegJsonDataset] WARN image not found: {item[self.image_key]}")
                missing_img += 1
            if not item.get(self.seg_key, ""):
                missing_seg += 1
            elif not self._seg_resolve(item[self.seg_key]).exists():
                print(f"[SegJsonDataset] WARN seg not found: {item[self.seg_key]}")
                missing_seg += 1
        if missing_seg:
            print(
                f"[SegJsonDataset] {missing_seg}/{len(self.items)} entries missing "
                f"'{self.seg_key}'."
            )

    def _seg_resolve(self, p: str) -> Path:
        """
        Resolve a path: absolute as-is; else project_root relative; else json_dir.
        """
        p = Path(p)
        if p.is_absolute():
            return p
        if self.image_root:
            abs_p = self.image_root / p
            if abs_p.exists():
                return abs_p
        abs_p = self.project_root / p
        return abs_p if abs_p.exists() else self.json_dir / p

    def __len__(self):
        return len(self.items)

    def _load_seg_colormap(self, seg_path: Path) -> torch.Tensor:
        """
        Load a raw class-ID PNG and return a colourised map [3, size, size] in [0,1].

        Steps:
          1. Read RAW pixel values as class ids. Grounded-SAM/CARLA masks are
             16-bit PNGs (PIL mode I;16, dtype uint16, ids 0..28); the
             SegFormer pipeline saves 8-bit "L" PNGs. np.asarray on the
             opened image handles both without a .convert("L") -- I;16 -> L
             conversion behaviour is Pillow-version-dependent, reading raw
             values is not.
          2. Square the map (real masks are 1280x800) using the SAME
             `resize_mode` ("letterbox" or "CenterCrop") the paired RGB
             image_transform used -- square_id_map() (src/data/transforms.py)
             is the single shared geometry, so image and map can't drift
             apart. NEAREST-only: a bilinear resize/stretch here misaligns
             conditioning vs target by up to ~19% of the frame (letterbox mode).
          3. seg_colorize_ids() with the shared palette -> [1,3,size,size] in [0,1].

        Returns [3, size, size] float tensor in [0, 1].
        """
        ids_np = np.asarray(Image.open(seg_path)).astype(np.int64)   # [H, W], raw ids

        # Class ids must fit 8-bit for the PIL "L" round-trip below. Both live
        # taxonomies do (Cityscapes max 18, CARLA max 28); fail loudly if not.
        if ids_np.max() > 255:
            raise ValueError(
                f"{seg_path}: max pixel value {ids_np.max()} is not a class id "
                "(expected 0..255). Is this really a raw class-ID map?"
            )
        ids_pil = Image.fromarray(ids_np.astype(np.uint8), mode="L")

        ids_pil = square_id_map(ids_pil, self.size, self.resize_mode, self.pad_id)

        ids = torch.from_numpy(np.asarray(ids_pil, dtype=np.int64))   # [size, size]

        # seg_colorize_ids expects [B, H, W]; add/remove the batch dim around it.
        colour = seg_colorize_ids(ids.unsqueeze(0), self.seg_palette)   # [1,3,size,size]
        return colour[0]                                                  # [3,size,size]

    def __getitem__(self, idx: int):
        item    = self.items[idx]
        caption = item.get(self.prompt_key, "")

        # ---- RGB image -> [-1, 1] -------------------------------------------
        image = Image.open(self._seg_resolve(item[self.image_key])).convert("RGB")
        if self.image_transform:
            image = self.image_transform(image)

        # ---- Segmentation colour map -> [0,1], 3-channel -------------------
        # Fail loudly rather than propagating a KeyError deep in a worker.
        if self.seg_key not in item:
            raise KeyError(
                f"Entry {idx} in {self.json_file.name} has no '{self.seg_key}' key."
            )
        seg = self._load_seg_colormap(self._seg_resolve(item[self.seg_key]))

        return {"jpg": image, "seg": seg, "caption": caption}


class SegJsonDataModule:
    """
    Data module reading a JSON manifest for train + optional validation.

    Exposes train_dataloader()/val_dataloader() and
    .train_dataset / .val_dataset (grounded_sam_training.py indexes val_dataset directly for
    the fixed-scene monitoring images). val_json_file MUST point at the real
    validation set — NEVER test.json (references.md §8).
    """

    def __init__(
        self,
        json_file: str,
        size: int = 512,
        val_json_file: str = None,
        batch_size: int = 8,
        val_batch_size: int = 4,       # 4 = safe under no_grad; matches depth default
        workers: int = 4,
        val_workers: int = 2,
        palette: list = None,
        image_root: str = None,
        classes_file: str = None,      # Grounded-SAM class-definition JSON (id->name/colour)
        image_key: str = "raw_image_path",
        seg_key: str = "seg_path",
        prompt_key: str = "prompt",
        pad_id: int = 0,               # letterbox fill class for non-square maps
                                       # (0 = Unlabeled in the CARLA taxonomy)
        resize_mode: str = "letterbox",  # "letterbox" or "CenterCrop" — drives
                                       # both the RGB transform (build_seg_preprocess)
                                       # and the seg map geometry (square_id_map).
    ):
        # project_root: three levels up from this file (src/data/ -> src/ -> root).
        project_root = Path(os.path.abspath(__file__)).parent.parent.parent
        image_tfm    = build_seg_preprocess(size=size, resize_mode=resize_mode)
        _img_root    = Path(project_root, image_root) if image_root else project_root

        self.batch_size     = batch_size
        self.val_batch_size = val_batch_size
        self.workers        = workers
        self.val_workers    = val_workers

        # Palette source, exactly one wins:
        #   1. classes_file (Grounded-SAM class set -> palette).
        #   2. palette arg (explicit list).
        #   3. neither -> SegJsonDataset raises (no fallback on this branch).
        # Loaded once here so train and val use the identical palette.
        if classes_file is not None:
            from src.encoders.grounded_sam_encoder import load_grounded_sam_palette
            _cf = Path(project_root, classes_file)
            self.class_names, palette = load_grounded_sam_palette(_cf)
            print(f"[SegJsonDataModule] loaded {len(palette)} classes from {classes_file}")
        else:
            self.class_names = None
        self.palette = palette

        _keys = dict(image_key=image_key, seg_key=seg_key, prompt_key=prompt_key,
                     pad_id=pad_id, resize_mode=resize_mode)

        self.train_dataset = SegJsonDataset(
            json_file=Path(project_root, json_file),
            image_transform=image_tfm, size=size,
            project_root=_img_root, palette=palette, **_keys,
        )

        if val_json_file:
            self.val_dataset = SegJsonDataset(
                json_file=Path(project_root, val_json_file),
                image_transform=image_tfm, size=size,
                project_root=_img_root, palette=palette, **_keys,
            )
        else:
            # No val set provided -> fall back to train set.
            # Prefer always providing val_json_file; this fallback is only for quick
            # debugging runs where no val split is available.
            print("[SegJsonDataModule] WARNING: no val_json_file — using train set as val.")
            self.val_dataset = self.train_dataset

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size,
            shuffle=True, num_workers=self.workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, batch_size=self.val_batch_size,
            shuffle=False, num_workers=self.val_workers,
        )
