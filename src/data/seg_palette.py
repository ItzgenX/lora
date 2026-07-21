"""
src/data/seg_palette.py
------------------------
Taxonomy-agnostic class-ID <-> RGB-colour math, shared by every place that
reads or writes a segmentation colour map on this branch: local_seg.py
(training dataset), grounded_sam_training.py (mIoU controllability metric),
grounded_sam_inference.py (loading + scoring a provided map).

WHY THIS FILE EXISTS (2026-07-20, replacing src/encoders/seg_encoder.py):
  This branch trains on Grounded-SAM/CARLA maps (29 classes, see
  configs/grounded_sam_classes.json) and never runs SegFormer or any
  Cityscapes taxonomy. The functions below never assumed any PARTICULAR
  palette -- they always took one in as an argument -- so they belong here,
  not in a file whose name implies a SegFormer-specific live encoder. There
  is deliberately NO default/fallback palette: every caller on this branch
  loads the real CARLA classes_file (src/encoders/grounded_sam_encoder.py
  load_grounded_sam_palette) and must pass it in explicitly. A silent
  Cityscapes fallback previously lived here and was actively broken for this
  branch's own data -- Cityscapes only has 19 colours, but CARLA class ids
  go up to 28, so the fallback would raise "class id >= palette size 19" on
  nearly any real mask the moment it was ever actually used.
"""

import torch


def seg_palette_tensor(palette: list[tuple[int, int, int]]) -> torch.Tensor:
    """
    Convert an integer RGB palette into a lookup tensor in [0, 1].

    WHY: colourising a class-ID map is a gather/index operation; having the
    palette as a [num_classes, 3] float tensor in [0,1] lets us do
    `palette[ids]` to produce a [..., 3] colour image directly in the output
    range the mapper expects.

    palette: REQUIRED, no default -- pass the real class-set palette (e.g.
      from grounded_sam_encoder.load_grounded_sam_palette).

    Returns: FloatTensor [num_classes, 3] in [0, 1].
    """
    return torch.tensor(palette, dtype=torch.float32) / 255.0


def seg_colorize_ids(
    ids: torch.Tensor,          # [B, H, W] long — class IDs
    palette: torch.Tensor,      # [K, 3] float in [0,1] — from seg_palette_tensor()
) -> torch.Tensor:              # [B, 3, H, W] float in [0,1]
    """
    Map a class-ID map to a 3-channel RGB colour image in [0, 1].

    WHY IT EXISTS IN THE PIPELINE:
      This is the SINGLE colourisation step shared by the training dataset
      (src/data/local_seg.py colourises the saved raw-ID PNG at load time)
      and inference visualisation. Sharing it guarantees a class always gets
      the identical colour everywhere — the parity guarantee that training
      and inference conditioning match.

    Inputs:
      ids     : Long tensor [B, H, W] of class ids in [0, num_classes-1].
      palette : Float tensor [num_classes, 3] in [0,1] (from seg_palette_tensor()).
    Output:
      Float tensor [B, 3, H, W] in [0, 1].

    Fails loudly if any id would index outside the palette — a silent
    out-of-range gather is exactly the kind of quiet corruption to avoid.
    """
    max_id = int(ids.max()) if ids.numel() else 0
    if max_id >= palette.shape[0]:
        raise ValueError(
            f"class id {max_id} >= palette size {palette.shape[0]}. "
            f"The ID map and the loaded classes_file disagree on class count."
        )
    palette = palette.to(ids.device)
    # palette[ids] -> [B, H, W, 3]; permute to channels-first [B, 3, H, W].
    colour = palette[ids.long()]          # [B, H, W, 3] in [0,1]
    return colour.permute(0, 3, 1, 2).contiguous()


def seg_ids_from_colormap(
    colour: torch.Tensor,       # [3, H, W] or [B, 3, H, W] float in [0,1]
    palette: torch.Tensor,      # [K, 3] float in [0,1] — from seg_palette_tensor()
) -> torch.Tensor:              # [H, W] or [B, H, W] long — class IDs
    """
    Inverse of seg_colorize_ids: map an RGB colour seg map back to class IDs.

    WHY IT EXISTS: the conditioning map fed to the model is stored as COLOUR
    (colourised from IDs via seg_colorize_ids). To score mIoU we need the IDs
    behind it as the TARGET. Because the map was colourised FROM this exact
    palette, nearest-colour recovery returns the original IDs exactly.

    Each pixel is assigned the class whose palette colour is nearest (L2 in RGB);
    for maps produced by seg_colorize_ids the nearest colour is an exact match.

    Inputs:
      colour  : Float tensor [3,H,W] or [B,3,H,W] in [0,1].
      palette : Float tensor [K,3] in [0,1] (from seg_palette_tensor()).
    Output:
      Long tensor [H,W] or [B,H,W] of ids in [0, K-1] (batch dim preserved).
    """
    squeeze = colour.dim() == 3
    if squeeze:
        colour = colour.unsqueeze(0)                 # [1,3,H,W]
    palette = palette.to(colour.device)              # [K,3]

    B, _, H, W = colour.shape
    pixels = colour.permute(0, 2, 3, 1).reshape(-1, 3)          # [B*H*W, 3]
    # squared L2 distance to every palette colour, then pick the closest.
    dists = ((pixels[:, None, :] - palette[None, :, :]) ** 2).sum(-1)   # [B*H*W, K]
    ids = dists.argmin(dim=1).reshape(B, H, W)                  # [B,H,W] long

    return ids[0] if squeeze else ids
