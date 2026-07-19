"""
src/data/local_seg.py
---------------------
Dataset: load (RGB image, pre-computed segmentation map, prompt) triplets for
training the segmentation-conditioned LoRAdapter. The maps are generated OFFLINE
(on this branch, by Grounded-SAM — see GROUNDED_SAM.md) and read here directly.

Two segmentation-specific points, both critical:

  1. The saved map is a RAW CLASS-ID PNG (8-bit, values 0..N-1), NOT a colour
     image. We COLOURISE it at load time with the configured palette (the
     Grounded-SAM class palette when a classes_file is set, otherwise the
     Cityscapes SEG_CITYSCAPES_PALETTE fallback from src/encoders/seg_encoder.py),
     producing a 3-channel RGB map in [0,1] via the shared seg_colorize_ids.

  2. Resizing a class-ID map MUST use NEAREST interpolation. Averaging categorical
     class ids is meaningless: the mean of "road"=0 and "car"=6 is 3, a DIFFERENT
     class that isn't in the image. NEAREST preserves exact labels. In practice the
     PNG is already at the right size, so this resize is usually a no-op alignment
     step — but we still force NEAREST so it never silently corrupts labels.

Returned by __getitem__:
  {
    "jpg"    : RGB tensor   [3, H, W] in [-1, 1]   (standard training format)
    "seg"    : colour map   [3, H, W] in [0, 1]    (palette RGB)
    "caption": prompt string
  }
seg_training.py reads batch["seg"].
"""

import json
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from src.encoders.seg_encoder import SEG_CITYSCAPES_PALETTE, seg_palette_tensor, seg_colorize_ids


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
        palette: list = None,      # class-id -> RGB; defaults to Cityscapes SSOT
        image_root: Path = None,
        image_key: str = "raw_image_path",   # JSONL key for the source RGB image
        seg_key: str = "seg_path",           # JSONL key for the class-ID seg map
        prompt_key: str = "prompt",          # JSONL key for the text caption
        pad_id: int = 0,                     # class id used to letterbox non-square
                                             # maps (CARLA 0 = Unlabeled; see
                                             # _load_seg_colormap step 2)
    ):
        self.json_file    = Path(json_file)
        self.pad_id       = pad_id
        self.json_dir     = self.json_file.parent
        self.project_root = Path(project_root) if project_root else self.json_dir
        self.image_root   = Path(image_root) if image_root else None
        self.size         = size
        self.image_transform = image_transform
        # Configurable manifest keys: default to the SegFormer pipeline's names,
        # override (e.g. for a Grounded-SAM manifest that used "image"/"mask")
        # via the data config so no JSONL renaming is needed.
        self.image_key    = image_key
        self.seg_key      = seg_key
        self.prompt_key   = prompt_key

        # Build the palette tensor ONCE here from the shared constant so every
        # sample colourises identically, and identically to the live encoder.
        # Using seg_palette_tensor keeps the [0,1] conversion in one function.
        self.seg_palette  = seg_palette_tensor(
            palette if palette is not None else SEG_CITYSCAPES_PALETTE
        )
        self.num_classes  = self.seg_palette.shape[0]   # 19 for Cityscapes

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

        The 'seg_' prefix marks this as segmentation-pipeline code (naming rule).
        Same resolution logic as depth's dataset _resolve.
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

        Steps (the seg-specific points live here):
          1. Open and read RAW pixel values — they ARE class ids. The real
             Grounded-SAM/CARLA masks are 16-bit PNGs (PIL mode I;16, dtype
             uint16, ids 0..28, confirmed from the user's own format scan);
             the SegFormer pipeline saves 8-bit "L" PNGs. np.asarray on the
             opened image handles BOTH without a .convert("L") — important
             because I;16 -> L conversion behaviour differs between Pillow
             versions (verified exact on THIS env 2026-07-20, but the
             training machine may run a different Pillow; reading raw is
             version-proof).
          2. If the map is NOT square (the real masks are 1280x800), LETTERBOX
             it to a square with `pad_id` fill — the SAME geometry SquarePad
             applies to the paired RGB image (pad the shorter side, extra
             pixel to the bottom/right on odd totals). A plain resize here
             would STRETCH the map while the RGB is letterboxed, vertically
             misaligning conditioning vs target by up to ~19% of the frame
             at the top/bottom (measured 2026-07-20). Alignment is the whole
             point of structure conditioning, so the two paths MUST match.
          3. NEAREST resize to (size, size) — label-preserving (NEVER bilinear).
          4. seg_colorize_ids() with the shared palette -> [1,3,size,size] in [0,1].

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

        # ---- Letterbox a non-square map (mirror SquarePad's geometry) ------ #
        # SquarePad pads the SHORTER side: pad_before = pad_total // 2, the
        # remainder goes after (bottom/right). Identical rounding here, so a
        # 1280x800 map and its 1280x800 RGB land on the same square grid.
        w, h = ids_pil.size
        if w != h:
            side = max(w, h)
            pad_before = (side - min(w, h)) // 2
            canvas = Image.new("L", (side, side), color=self.pad_id)
            # landscape -> pad top+bottom (paste at y offset); portrait -> left+right
            canvas.paste(ids_pil, (0, pad_before) if w > h else (pad_before, 0))
            ids_pil = canvas

        # NEAREST is REQUIRED for a class-ID map: bilinear would average class
        # ids and produce fabricated class values. PIL.Image.NEAREST is the flag.
        if ids_pil.size != (self.size, self.size):
            ids_pil = ids_pil.resize((self.size, self.size), Image.NEAREST)

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
    .train_dataset / .val_dataset (seg_training.py indexes val_dataset directly for
    the fixed-scene monitoring images). val_json_file MUST point at the real
    validation set — NEVER test.json (references.md §8).
    """

    def __init__(
        self,
        json_file: str,
        transform: list,               # Hydra-instantiated image transforms (-> [-1,1])
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
    ):
        # project_root: three levels up from this file (src/data/ -> src/ -> root).
        project_root = Path(os.path.abspath(__file__)).parent.parent.parent
        image_tfm    = transforms.Compose(transform)
        _img_root    = Path(project_root, image_root) if image_root else project_root

        self.batch_size     = batch_size
        self.val_batch_size = val_batch_size
        self.workers        = workers
        self.val_workers    = val_workers

        # PALETTE SOURCE — exactly one wins, in this order:
        #   1. classes_file (Grounded-SAM): load the user's class set -> palette.
        #   2. palette arg (explicit list).
        #   3. neither -> SegJsonDataset falls back to the Cityscapes SSOT.
        # Loading from classes_file here (once) guarantees train and val use the
        # IDENTICAL palette, and lets seg_training.py resolve the same one.
        if classes_file is not None:
            from src.encoders.grounded_sam_encoder import load_grounded_sam_palette
            _cf = Path(project_root, classes_file)
            self.class_names, palette = load_grounded_sam_palette(_cf)
            print(f"[SegJsonDataModule] loaded {len(palette)} classes from {classes_file}")
        else:
            self.class_names = None
        self.palette = palette

        _keys = dict(image_key=image_key, seg_key=seg_key, prompt_key=prompt_key,
                     pad_id=pad_id)

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
