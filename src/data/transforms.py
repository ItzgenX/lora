from PIL import Image, ImageFile, ImageStat

# Tolerate minor JPEG defects (e.g. a missing/odd end-of-image marker) instead
# of raising "image file is truncated". Pillow is intentionally strict here;
# most other viewers/decoders (Windows Photo Viewer, browsers, libjpeg-turbo
# used elsewhere) silently accept these same files. This flag is process-
# global; it lives here because src/data/transforms.py is imported by the
# segmentation pipeline entrypoints (seg_map_calculations.py, segformer_inference.py),
# so setting it once here covers every place an image gets loaded — a single
# source of truth.
ImageFile.LOAD_TRUNCATED_IMAGES = True


class SquarePad:
    """
    Pad the shorter dimension of a PIL image to produce a square, using a
    single FLAT fill color per padded region — the average (or, for
    categorical maps, the most frequent) color of a thin strip just inside
    that edge — rather than a stretched copy of the boundary row.

    Works as a plain callable so it is compatible with both v1 and v2
    torchvision.transforms.Compose chains and with Hydra instantiation.

    Input:  PIL.Image of any size (H x W)
    Output: PIL.Image of size (max(H,W) x max(H,W))

    After each call, `last_padding_fracs` holds the padding amounts as
    fractions of the ORIGINAL image dimensions in (left, top, right, bottom)
    order.  Storing fractions rather than pixel counts means the values
    remain correct even after the image is later resized to 512 x 512.
    Use them to identify or mask the padded region in generated outputs.

    CHANGED FROM THE PREVIOUS VERSION — WHY:
      The previous implementation cropped a 1px-wide/tall strip from the
      image boundary and `.resize()`-ed it to fill the pad region. That
      resize only changes the strip's HEIGHT (or WIDTH, for left/right
      padding); the OTHER axis is left unchanged, so every pixel of real
      horizontal (or vertical) detail in that single boundary row/column —
      sky-vs-building edges, power lines, antennas, sensor/JPEG noise right
      at the frame edge — is carried through exactly and simply repeated
      across the pad region. That is correct edge-replicate padding by
      definition; it is also exactly why it produced a noisy, multi-colored
      band once stretched, on real driving-scene frames whose top/bottom
      boundary row is rarely a flat color. Because this fed real training
      data, the model learned to reproduce that same banded pattern at
      generation time too — visible in generated ("PREDICTED") output, not
      only in the raw conditioning image.
      A FLAT fill (mean color of a thin edge-adjacent strip, collapsed to
      one value) has zero internal variation by construction, so it cannot
      produce that banding regardless of how busy the real edge pixels are.

    Two fill strategies, chosen explicitly via `fill_mode` — no
    auto-detection:
      - "mean" (default): average color of the sampled strip. Correct for
        continuous-tone images — the raw RGB photo fed to either encoder
        (MiDaS or the segmentation encoder) in this project today.
      - "mode": the single most frequent color in the sampled strip. Use
        this if `SquarePad` is ever applied directly to an already-
        categorical/palette map (e.g. a class-color segmentation map)
        instead of to the raw photo — averaging two different classes'
        palette RGB values would invent a color matching no real class;
        the mode guarantees a real, valid class color instead. Not needed
        for anything in this file today (both current uses below are raw
        photos), kept available for that case.

    ORIGINAL rationale for a smooth boundary (still applies — why not solid
    black/zero):
      A solid-colour boundary looks like a real depth discontinuity to the
      DPT model and produces a sharp artefact ring in the depth map at the
      pad boundary. A locally-averaged flat colour keeps the transition
      smooth for the same reason, without the banding the stretched-strip
      approach introduced on busy rows.

    IMPLEMENTATION NOTE (cross-platform bug fix, still applies): this class
    performs its fill with plain PIL crop / paste / ImageStat calls only —
    no torchvision.transforms.functional.pad, no numpy conversion — which
    avoids the torchvision-version-dependent PIL/numpy conversion path that
    previously caused a Windows-vs-Linux conda env discrepancy (this pipeline
    hit that exactly once already; see git history on this file).
    """

    # Height/width (in source pixels) of the strip sampled just inside the
    # edge being padded, used to compute the flat fill colour. Small and
    # deliberate: capture "the general tone right at the boundary," not
    # drift into unrelated content further into the frame.
    EDGE_SAMPLE_PX = 8

    def __init__(self, fill_mode: str = "mean"):
        assert fill_mode in ("mean", "mode"), f"unknown fill_mode: {fill_mode!r}"
        self.fill_mode = fill_mode
        self.last_padding_fracs = (0.0, 0.0, 0.0, 0.0)

    def __call__(self, img):
        w, h = img.size          # PIL uses (width, height)

        if h == w:
            self.last_padding_fracs = (0.0, 0.0, 0.0, 0.0)
            return img

        if w > h:                # landscape (e.g. 1280 x 800): pad top + bottom
            pad_total  = w - h
            pad_top    = pad_total // 2
            pad_bottom = pad_total - pad_top
            pad_left = pad_right = 0
        else:                    # portrait: pad left + right
            pad_total  = h - w
            pad_left   = pad_total // 2
            pad_right  = pad_total - pad_left
            pad_top = pad_bottom = 0

        # Record as fractions so the values stay valid after the downstream Resize.
        self.last_padding_fracs = (
            pad_left   / w,
            pad_top    / h,
            pad_right  / w,
            pad_bottom / h,
        )

        new_w = w + pad_left + pad_right
        new_h = h + pad_top + pad_bottom
        out = Image.new(img.mode, (new_w, new_h))
        out.paste(img, (pad_left, pad_top))

        # Flat fill: sample a thin strip just inside each padded edge, reduce
        # it to a single colour (mean or mode, per fill_mode), and paste a
        # solid block of that colour into the corresponding pad region.
        # Only one pair (top/bottom) or the other (left/right) is ever
        # non-zero at once, since only the shorter dimension is padded.
        sample = self.EDGE_SAMPLE_PX
        if pad_top > 0:
            strip = img.crop((0, 0, w, min(sample, h)))
            fill = self._fill_color(strip)
            out.paste(Image.new(img.mode, (w, pad_top), fill), (pad_left, 0))
        if pad_bottom > 0:
            strip = img.crop((0, max(h - sample, 0), w, h))
            fill = self._fill_color(strip)
            out.paste(Image.new(img.mode, (w, pad_bottom), fill),
                      (pad_left, pad_top + h))
        if pad_left > 0:
            strip = img.crop((0, 0, min(sample, w), h))
            fill = self._fill_color(strip)
            out.paste(Image.new(img.mode, (pad_left, h), fill), (0, pad_top))
        if pad_right > 0:
            strip = img.crop((max(w - sample, 0), 0, w, h))
            fill = self._fill_color(strip)
            out.paste(Image.new(img.mode, (pad_right, h), fill),
                      (pad_left + w, pad_top))

        return out

    def _fill_color(self, strip):
        """Return a single flat fill value/tuple matching strip.mode."""
        if self.fill_mode == "mean":
            means = ImageStat.Stat(strip).mean
            if len(means) == 1:
                return int(round(means[0]))
            return tuple(int(round(m)) for m in means)

        # fill_mode == "mode": the single most frequent colour in the strip.
        rgb_strip = strip.convert("RGB")
        colors = rgb_strip.getcolors(maxcolors=rgb_strip.width * rgb_strip.height)
        return max(colors, key=lambda entry: entry[0])[1]


# ============================================================================ #
#  SEGMENTATION PIPELINE — shared preprocessing builder                        #
# ============================================================================ #

from torchvision import transforms as _tv   # local alias: avoids shadowing outer scope


RESIZE_MODES = ("letterbox", "CenterCrop")


def _square_rgb_steps(size: int, resize_mode: str) -> list:
    """
    The GEOMETRIC steps (no ToTensor/Normalize) that square a PIL RGB image,
    for the chosen resize_mode. Shared by build_seg_preprocess (adds tensor
    conversion, for the model) and build_seg_display_preprocess (stays PIL,
    for on-screen panels) so the two can never geometrically disagree.

      "letterbox"  : SquarePad (flat-fill pad, current default) then Resize.
                     Keeps 100% of the scene; adds a flat-colour pad band.
      "CenterCrop" : the ORIGINAL stock LoRAdapter recipe (configs/data/local.yaml)
                     — torchvision Resize(size) [shorter edge -> size, aspect kept]
                     then CenterCrop(size). No pad band, but crops the longer
                     edge's overhang off the left/right (or top/bottom).
    """
    assert resize_mode in RESIZE_MODES, f"unknown resize_mode: {resize_mode!r}"
    if resize_mode == "letterbox":
        return [SquarePad(), _tv.Resize((size, size))]
    return [_tv.Resize(size), _tv.CenterCrop(size)]


def build_seg_preprocess(size: int, resize_mode: str = "letterbox"):
    """
    Build the ONE canonical RGB preprocessing pipeline for the SEGMENTATION pipeline.

    WHY THIS EXISTS (and why it belongs here, not in a seg-specific file):
      Both seg_map_calculations.py (offline calc) and segformer_inference.py (live
      inference) must apply byte-for-byte identical preprocessing so the seg map the
      network sees at inference exactly matches what was saved for training. The only
      way to guarantee that is to import the SAME function in both places — this is it.
      (Depth triplicated this preprocessing and references.md flags that as a drift
      risk; segmentation fixes it with this single source of truth.)

    The "seg" prefix distinguishes it from any hypothetical depth equivalent and
    satisfies the user's visual-identity rule: segmentation code has "seg" in name.

    INPUT  (of the returned callable): PIL.Image of any size, any aspect ratio.
    OUTPUT (of the returned callable): float tensor [3, size, size] in [-1, 1].
      This is the range every encoder slot in src/model.py expects
      (asserted by SegmentationEncoder._predict_ids).

    Args:
        size: final square side in pixels (e.g. 512). Must match cfg.size
              and the size used when the offline seg PNGs were computed.
        resize_mode: "letterbox" (default) or "CenterCrop" — see _square_rgb_steps.
          User decision 2026-07-20 (mirrored from the grounded_sam branch):
          both techniques are selectable via the `resize_mode` config key so
          results can be compared by training/calculating twice, once per mode.
          UNLIKE the grounded_sam branch, this mode must also match what
          seg_map_calculations.py used to COMPUTE the saved maps (the map's
          squaring is baked in at calc time here, not re-applied live at load
          time) — see seg_map_calculations.py's own resize_mode docs.

    Correctness note: the image is a (size, size) square before the ToTensor
    step, so SegmentationEncoder always receives exactly (size, size) input
    and never triggers the kind of internal forced-crop that MiDaS does.
    """
    return _tv.Compose(_square_rgb_steps(size, resize_mode) + [
        _tv.ToTensor(),                 # PIL [0,255] -> tensor [0,1]
        # mean=std=0.5 maps [0,1] linearly to [-1,1] (the SD1.5 VAE / encoder range).
        _tv.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])


def build_seg_display_preprocess(size: int, resize_mode: str = "letterbox"):
    """
    Same geometry as build_seg_preprocess, but stops after squaring — returns a
    PIL.Image, not a tensor. For DISPLAY-ONLY panels (e.g. the inference grid's
    ORIGINAL panel) that must visually match what the model actually saw,
    without needing the [-1,1] tensor conversion.
    """
    return _tv.Compose(_square_rgb_steps(size, resize_mode))


if __name__ == "__main__":
    # Self-check, meant to be run in this project's real environment (needs
    # torchvision) so the fix is confirmed by execution here, not just read.
    # Run: python -m src.data.transforms   (or however this repo's other
    # entrypoints invoke local modules)
    import random

    print("=== SquarePad self-check ===")

    # 1) mean-fill: a busy top row (random per-pixel noise) must NOT bleed
    #    into the pad region — the pad must come out perfectly flat.
    w, h = 1280, 800
    noisy_img = Image.new("RGB", (w, h), (30, 60, 90))
    px = noisy_img.load()
    for x in range(w):
        px[x, 0] = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

    pad = SquarePad(fill_mode="mean")
    result = pad(noisy_img)
    side = result.size[1]
    pad_top = (side - h) // 2
    top_region = result.crop((0, 0, w, pad_top))
    colors_in_pad = top_region.convert("RGB").getcolors(maxcolors=w * pad_top + 1)
    print(f"pad_top_px={pad_top}  distinct colors in pad band={len(colors_in_pad)} (expect 1)")
    assert len(colors_in_pad) == 1, "REGRESSION: pad region is not flat"
    print("PASS: mean-fill pad region is flat regardless of a noisy real edge row.")

    # 2) mode-fill: a strip with two colors present must fill with the
    #    MORE FREQUENT one exactly, never a blended color.
    class_a, class_b = (70, 70, 70), (128, 64, 128)
    seg_img = Image.new("RGB", (w, h), class_a)
    px2 = seg_img.load()
    for y in range(SquarePad.EDGE_SAMPLE_PX):
        for x in range(0, w, 3):
            px2[x, y] = class_b
            px2[x + 1, y] = class_b
    seg_pad = SquarePad(fill_mode="mode")
    seg_result = seg_pad(seg_img)
    fill_seen = seg_result.crop((0, 0, w, pad_top)).getpixel((0, 0))
    print(f"mode-fill picked: {fill_seen} (must be class_b {class_b}, the majority)")
    assert fill_seen == class_b, "REGRESSION: mode fill did not pick the majority class"
    print("PASS: mode-fill always picks a real class color, never a blend.")

    print("=== all SquarePad checks passed ===")