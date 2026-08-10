"""
src/data/local_seg.py
---------------------
Dataset: load (RGB image, pre-computed segmentation map, prompt) triplets for
training the segmentation-conditioned LoRAdapter. The maps are generated OFFLINE
by seg_map_calculations.py and read here directly.

Two segmentation-specific points, both critical:

  1. The saved map is a RAW CLASS-ID PNG (8-bit, values 0..N-1), NOT a colour
     image. We COLOURISE it at load time with the configured palette (the
     Cityscapes SEG_CITYSCAPES_PALETTE from src/encoders/seg_encoder.py, unless
     an explicit `palette` list is passed in), producing a 3-channel RGB map
     in [0,1] via the shared seg_colorize_ids.

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
segformer_training.py reads batch["seg"].
"""

import json
import os
import re
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader

from src.encoders.seg_encoder import SEG_CITYSCAPES_PALETTE, seg_palette_tensor, seg_colorize_ids
from src.data.transforms import build_seg_preprocess, RESIZE_MODES, normalize_size


# Matches the mode stamp seg_map_calculations.py always bakes into its output
# path (folder name for scan/data_dir/json_file modes: "..._seg_map_letterbox/...";
# filename for single-image mode: "<stem>_seg_map_letterbox.png") -- reused here
# (not a separate marker file) since it's already guaranteed present by every
# calc-mode's own naming convention.
_SEG_MAP_MODE_RE = re.compile(r"_seg_map_(" + "|".join(RESIZE_MODES) + r")\b")


def _detect_seg_map_resize_mode(seg_path: str) -> str | None:
    """Return the resize_mode baked into a calc-produced seg_path, or None if
    the path doesn't carry the stamp (e.g. a hand-placed/legacy map)."""
    m = _SEG_MAP_MODE_RE.search(str(seg_path))
    return m.group(1) if m else None


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
        size: int | tuple = 512,   # square side, or (width, height) for resize_mode="aspect"
        project_root: Path = None,
        palette: list = None,      # class-id -> RGB; defaults to Cityscapes SSOT
        image_root: Path = None,
        image_key: str = "raw_image_path",   # JSONL key for the source RGB image
        seg_key: str = "seg_path",           # JSONL key for the class-ID seg map
        prompt_key: str = "prompt",          # JSONL key for the text caption
        resize_mode: str = "letterbox",      # MUST match what seg_map_calculations.py
                                             # used to COMPUTE these maps (baked in at
                                             # calc time, unlike the grounded_sam branch)
                                             # -- used ONLY to cross-check seg_path's own
                                             # mode stamp below, catching a real silent
                                             # training-data-misalignment risk found by
                                             # self-review 2026-07-20: this dataset's own
                                             # image_transform (built from resize_mode)
                                             # squares the RGB HERE, live, while the
                                             # paired seg map was squared PERMANENTLY at
                                             # calc time -- if the two modes ever
                                             # disagree, image and map geometrically
                                             # diverge with no other check catching it.
    ):
        self.json_file    = Path(json_file)
        self.json_dir     = self.json_file.parent
        self.project_root = Path(project_root) if project_root else self.json_dir
        self.image_root   = Path(image_root) if image_root else None
        # (width, height) — normalized once here so every downstream PIL resize
        # call uses PIL's own (width, height) order explicitly, never the bare
        # `size` value (see _load_seg_colormap: PIL.Image.resize() takes
        # (width, height), the OPPOSITE axis order from torchvision's Resize).
        self.size_w, self.size_h = normalize_size(size)
        self.resize_mode  = resize_mode
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
        # resize_mode cross-check (2026-07-20): seg_map_calculations.py always
        # stamps its output path with the mode used to compute it (folder or
        # filename, see _detect_seg_map_resize_mode) -- collect what modes are
        # ACTUALLY present across this manifest's seg_path entries, so a single
        # loud warning can catch (a) this dataset's resize_mode disagreeing
        # with what the maps were really computed with, and (b) a manifest
        # that accidentally mixes maps from two different calc runs.
        _detected_modes = set()
        _unstamped = 0
        for item in self.items:
            if not self._seg_resolve(item[self.image_key]).exists():
                print(f"[SegJsonDataset] WARN image not found: {item[self.image_key]}")
                missing_img += 1
            if not item.get(self.seg_key, ""):
                missing_seg += 1
            elif not self._seg_resolve(item[self.seg_key]).exists():
                print(f"[SegJsonDataset] WARN seg not found: {item[self.seg_key]}")
                missing_seg += 1
            if item.get(self.seg_key, ""):
                _mode = _detect_seg_map_resize_mode(item[self.seg_key])
                if _mode is not None:
                    _detected_modes.add(_mode)
                else:
                    _unstamped += 1
        if missing_seg:
            print(
                f"[SegJsonDataset] {missing_seg}/{len(self.items)} entries missing "
                f"'{self.seg_key}'."
            )
        if len(_detected_modes) > 1:
            print(
                f"[SegJsonDataset] WARN: this manifest MIXES seg maps computed with "
                f"different resize_mode values: {sorted(_detected_modes)} -- image "
                f"and map geometry will disagree for whichever entries don't match "
                f"this dataset's resize_mode={self.resize_mode!r}. Rebuild the "
                f"manifest from a single seg_map_calculations.py run."
            )
        elif _detected_modes and self.resize_mode not in _detected_modes:
            print(
                f"[SegJsonDataset] WARN resize_mode mismatch: this dataset is "
                f"configured with resize_mode={self.resize_mode!r}, but every "
                f"seg_path in {self.json_file.name} was computed with "
                f"resize_mode={sorted(_detected_modes)[0]!r} -- the RGB image "
                f"(squared HERE, live, with this dataset's resize_mode) and its "
                f"paired seg map (squared PERMANENTLY at calc time with the OTHER "
                f"mode) will NOT geometrically align. Pass "
                f"resize_mode={sorted(_detected_modes)[0]!r} to fix, or point "
                f"data.json_file at a manifest built with resize_mode="
                f"{self.resize_mode!r}."
            )
        if _unstamped and _unstamped < len(self.items):
            print(
                f"[SegJsonDataset] NOTE: {_unstamped}/{len(self.items)} seg_path "
                f"entries have no recognisable resize_mode stamp in their path "
                f"(hand-placed or pre-2026-07-20 maps) -- not checked against "
                f"resize_mode={self.resize_mode!r}."
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
        Load a raw class-ID PNG and return a colourised map [3, H, W] in [0,1],
        where (W, H) = (self.size_w, self.size_h) — square unless resize_mode
        is "aspect".

        Steps (the two seg-specific points live here):
          1. Open as "L" (8-bit single channel) — pixel values ARE class ids.
          2. NEAREST resize to (size_w, size_h) — label-preserving (NEVER bilinear).
          3. seg_colorize_ids() with the shared palette -> [1,3,H,W] in [0,1].

        Returns [3, H, W] float tensor in [0, 1].
        """
        ids_pil = Image.open(seg_path).convert("L")

        # NEAREST is REQUIRED for a class-ID map: bilinear would average class
        # ids and produce fabricated class values. PIL.Image.NEAREST is the flag.
        # PIL's .size and .resize() both use (width, height) order — matches
        # (self.size_w, self.size_h) directly, no swap needed here (unlike the
        # torchvision.transforms.Resize / F.interpolate call sites elsewhere,
        # which take (height, width) — do not copy this ordering there).
        target_wh = (self.size_w, self.size_h)
        if ids_pil.size != target_wh:
            ids_pil = ids_pil.resize(target_wh, Image.NEAREST)

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
    .train_dataset / .val_dataset (segformer_training.py indexes val_dataset directly for
    the fixed-scene monitoring images). val_json_file MUST point at the real
    validation set — NEVER test.json (references.md §8).
    """

    def __init__(
        self,
        json_file: str,
        size: int | tuple = 512,   # square side, or (width, height) for resize_mode="aspect"
        val_json_file: str = None,
        batch_size: int = 8,
        val_batch_size: int = 4,       # 4 = safe under no_grad; matches depth default
        workers: int = 4,
        val_workers: int = 2,
        palette: list = None,
        image_root: str = None,
        image_key: str = "raw_image_path",
        seg_key: str = "seg_path",
        prompt_key: str = "prompt",
        resize_mode: str = "letterbox",  # "letterbox" or "CenterCrop" (user decision
                                       # 2026-07-20) — built ONCE here from
                                       # build_seg_preprocess so train/val use the
                                       # identical RGB transform. UNLIKE the
                                       # grounded_sam branch, this does NOT re-square
                                       # the seg map (already squared at calc time by
                                       # seg_map_calculations.py) — it MUST match
                                       # whatever mode that script used, and IS cross-
                                       # checked against each seg_path's own mode
                                       # stamp in SegJsonDataset (loud warning on
                                       # mismatch, found by self-review 2026-07-20).
    ):
        # project_root: three levels up from this file (src/data/ -> src/ -> root).
        project_root = Path(os.path.abspath(__file__)).parent.parent.parent
        image_tfm    = build_seg_preprocess(size=size, resize_mode=resize_mode)
        _img_root    = Path(project_root, image_root) if image_root else project_root

        self.batch_size     = batch_size
        self.val_batch_size = val_batch_size
        self.workers        = workers
        self.val_workers    = val_workers

        # PALETTE SOURCE — explicit `palette` arg if given, else SegJsonDataset
        # falls back to the Cityscapes SSOT (SEG_CITYSCAPES_PALETTE).
        self.class_names = None
        self.palette = palette

        _keys = dict(image_key=image_key, seg_key=seg_key, prompt_key=prompt_key,
                     resize_mode=resize_mode)

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
