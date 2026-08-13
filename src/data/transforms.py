from PIL import Image, ImageFile, ImageStat

# Tolerate minor JPEG defects (e.g. a missing/odd end-of-image marker) instead
# of raising "image file is truncated". Pillow is intentionally strict here;
# most other viewers/decoders (Windows Photo Viewer, browsers, libjpeg-turbo
# used elsewhere) accept these same files. This flag is process-global; it
# lives here because src/data/transforms.py is imported by every place an
# image gets loaded on this branch (local_seg.py, grounded_sam_inference.py).
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

    Uses a FLAT fill rather than a stretched copy of the boundary row: a
    stretched 1px-wide/tall boundary strip only changes one axis' extent, so
    real edge detail (sky/building edges, power lines, sensor/JPEG noise at
    the frame edge) repeats across the pad region and produces a noisy,
    multi-colored band on real driving-scene frames — visible in generated
    ("PREDICTED") output, not only the raw conditioning image, since it's
    part of the real training data. A flat fill (mean color of a thin
    edge-adjacent strip, collapsed to one value) has zero internal variation
    by construction, so it cannot produce that banding.

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

    A solid-colour boundary (e.g. black/zero) looks like a real depth
    discontinuity to the DPT model and produces a sharp artefact ring in the
    depth map at the pad boundary; a locally-averaged flat colour keeps the
    transition smooth without that artefact.

    Fill uses plain PIL crop / paste / ImageStat calls only — no
    torchvision.transforms.functional.pad, no numpy conversion — avoiding a
    torchvision-version-dependent PIL/numpy conversion path that differs
    between Windows and Linux conda environments.
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


RESIZE_MODES = ("aspect",)


def normalize_size(size) -> tuple:
    """
    Normalize a config `size` value to an explicit (width, height) pair.

    Accepts a single int (square) or a (width, height) list/tuple.
    """
    if isinstance(size, int):
        return (size, size)
    w, h = size
    return (int(w), int(h))


def _square_rgb_steps(size, resize_mode: str) -> list:
    """
    The GEOMETRIC steps (no ToTensor/Normalize) that resize a PIL RGB image.
    Shared by build_seg_preprocess (adds tensor conversion, for the model)
    and build_seg_display_preprocess (stays PIL, for on-screen panels) so the
    two can never geometrically disagree.

    Only "aspect" is supported: a direct resize to an explicit (width,
    height) target, no pad, no crop. Distortion is negligible when the
    target ratio is chosen close to the source ratio (e.g. 512x320 for a
    1280x800 source: 1280/800 = 1.6 = 512/320 exactly, both divisible by
    64). `size` may be an int (square target) or a (width, height) pair.
    """
    assert resize_mode in RESIZE_MODES, f"unknown resize_mode: {resize_mode!r}"
    # torchvision.Resize's tuple form is (height, width) -- the OPPOSITE axis
    # order from PIL's (width, height) used in square_id_map below; spelled
    # out explicitly to avoid silently swapping width/height for a
    # non-square target.
    w, h = normalize_size(size)
    return [_tv.Resize((h, w))]


def build_seg_preprocess(size, resize_mode: str = "aspect"):
    """
    Build the ONE canonical RGB preprocessing pipeline for the SEGMENTATION pipeline.

    Every stage that reads a raw photo (offline calc, training's dataset,
    inference) applies this identical preprocessing, imported from this
    single source, so the map the network sees at inference exactly matches
    what training used.

    INPUT  (of the returned callable): PIL.Image of any size, any aspect ratio.
    OUTPUT (of the returned callable): float tensor [3, H, W] in [-1, 1],
      where (W, H) = normalize_size(size). This is the range every encoder
      slot in src/model.py expects.

    Args:
        size: an int (square side, e.g. 512) or a (width, height) pair
          (e.g. (512, 320)). Must match cfg.size.
        resize_mode: "aspect" (only mode supported) — see _square_rgb_steps.
          The seg-ID map MUST use the identical size (see square_id_map
          below) or conditioning and image misalign.
    """
    return _tv.Compose(_square_rgb_steps(size, resize_mode) + [
        _tv.ToTensor(),                 # PIL [0,255] -> tensor [0,1]
        # mean=std=0.5 maps [0,1] linearly to [-1,1] (the SD1.5 VAE / encoder range).
        _tv.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])


def build_seg_display_preprocess(size, resize_mode: str = "aspect"):
    """
    Same geometry as build_seg_preprocess, but stops after resizing — returns
    a PIL.Image, not a tensor. For DISPLAY-ONLY panels (e.g. the inference
    grid's ORIGINAL panel) that must visually match what the model actually
    saw, without needing the [-1,1] tensor conversion.
    """
    return _tv.Compose(_square_rgb_steps(size, resize_mode))


def square_id_map(ids_pil, size, resize_mode: str = "aspect", pad_id: int = 0):
    """
    Resize a raw class-ID PIL image (mode "L", one integer id per pixel) to
    the target size, using the SAME geometric technique as build_seg_preprocess
    applies to the paired RGB image — so image and map stay pixel-aligned.
    NEAREST-only: averaging or blending class ids fabricates classes that
    don't exist.

    Shared by src/data/local_seg.py (training/val dataset) and grounded_sam_inference.py
    (_load_seg_map) so both read a saved map identically — the parity rule
    this project follows everywhere else.

    Only "aspect" is supported: NO pad, NO crop — direct NEAREST resize to an
    explicit (width, height) target (see _square_rgb_steps for the
    rationale). `pad_id` is unused (nothing is padded, kept as a parameter
    for call-site compatibility). `size` may be an int or a (width, height) pair.
    """
    assert resize_mode in RESIZE_MODES, f"unknown resize_mode: {resize_mode!r}"
    target_wh = normalize_size(size)   # PIL order: (width, height)
    if ids_pil.size != target_wh:
        ids_pil = ids_pil.resize(target_wh, Image.NEAREST)
    return ids_pil


if __name__ == "__main__":
    # SquarePad self-check. Needs torchvision.
    # Run: python -m src.data.transforms
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