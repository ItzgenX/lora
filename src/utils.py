from typing import Any, Literal
import torch
from accelerate import Accelerator
from pathlib import Path
from torch.nn.utils import clip_grad_norm_
from functools import reduce
import os

# from src.model import ModelBase

MODE = Literal[
    "train",
    "val",
    "always",
]


# ============================================================================ #
#  GPU RESOLUTION — single source of truth for all 6 pipeline entrypoints      #
#  (depth_map_calculations.py, seg_map_calculations.py, depth_training.py,      #
#  seg_training.py, depth_inference.py, seg_inference.py)                      #
#                                                                                #
#  WHY THIS EXISTS: the calc/inference scripts used to pick a device with      #
#  `"cuda" if torch.cuda.is_available() else "cpu"` and silently run on CPU     #
#  if CUDA wasn't detected for ANY reason (CPU-only torch build, driver        #
#  mismatch, wrong conda env, CUDA_VISIBLE_DEVICES unset/empty). Nothing        #
#  printed a warning — the run just looked "normal" but was extremely slow      #
#  and never touched the GPU. print_gpu_diagnostics() always prints what's     #
#  visible; resolve_device() refuses to silently fall back to CPU unless the   #
#  caller explicitly opts in.                                                  #
# ============================================================================ #

def print_gpu_diagnostics() -> int:
    """
    Print full GPU visibility diagnostics UNCONDITIONALLY (every run, every
    entrypoint) so "is the GPU actually being used" is answered by the log
    itself instead of something you have to infer from how slow it feels.

    Returns the number of visible CUDA devices (0 if none).
    """
    n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    print(f"\n{'=' * 60}")
    print(f"GPU DIAGNOSTICS")
    print(f"{'=' * 60}")
    print(f"  torch.cuda.is_available() : {torch.cuda.is_available()}")
    print(f"  torch.cuda.device_count() : {n_gpus}")
    print(f"  CUDA_VISIBLE_DEVICES      : {os.environ.get('CUDA_VISIBLE_DEVICES', '<not set>')}")
    for i in range(n_gpus):
        p = torch.cuda.get_device_properties(i)
        print(f"  cuda:{i} = {p.name}  ({p.total_memory / 1024**3:.1f} GB)")
    print(f"{'=' * 60}\n")
    return n_gpus


def resolve_device(requested: str | None = None, allow_cpu_fallback: bool = False) -> str:
    """
    Resolve which device to actually run on, LOUDLY. Never silently choose CPU.

    Args:
      requested: what the caller/CLI asked for.
        None        -> auto-pick: "cuda" if any GPU is visible, else error
                       (unless allow_cpu_fallback=True).
        "cuda"      -> use cuda:0; error if no GPU is visible.
        "cuda:N"    -> use that exact GPU index; error if not visible.
        "cpu"       -> explicit opt-in to CPU. Always honoured (this is a real
                       request, not a silent fallback), but prints a loud
                       reminder that this will be slow.
      allow_cpu_fallback: only affects the `requested=None` auto-pick path.
        When True, falls back to CPU with a printed warning if no GPU is
        visible, instead of raising. Off by default — the whole point of this
        function is that "no GPU" should stop you, not quietly downgrade you.

    Raises RuntimeError with a concrete fix checklist if a GPU was required
    but none is visible.
    """
    n_gpus = print_gpu_diagnostics()

    if requested == "cpu":
        print("[INFO] --device cpu explicitly requested -- running on CPU. This will be MUCH slower than GPU.")
        return "cpu"

    fix_checklist = (
        "  Fix checklist:\n"
        "    1. Run `nvidia-smi` in this exact shell/container -- does it show a GPU?\n"
        "       If not, the driver/GPU isn't visible here (wrong container, no GPU passthrough, etc.).\n"
        "    2. Confirm the ACTIVE Python env has a CUDA-enabled torch build:\n"
        "         python -c \"import torch; print(torch.__version__, torch.version.cuda)\"\n"
        "       If torch.version.cuda prints None, this is a CPU-ONLY torch install --\n"
        "       reinstall torch with a CUDA build matching this machine's driver.\n"
        "    3. Check CUDA_VISIBLE_DEVICES isn't set to an empty string or an invalid index.\n"
        "  If you genuinely want to run on CPU, pass --device cpu explicitly."
    )

    if requested is not None and requested != "cpu":
        # Validate the string is actually "cuda" or "cuda:N" BEFORE anything else.
        # A typo like --device gpu or --device cuda:abc must fail HERE with a
        # clear message, not silently pass through and blow up later inside
        # some .to(device) call with a confusing raw torch/ValueError.
        if requested != "cuda" and not requested.startswith("cuda:"):
            raise RuntimeError(
                f"--device {requested!r} is not a recognised device string. "
                f"Use 'cuda', 'cuda:N', or 'cpu'."
            )
        if requested.startswith("cuda:"):
            idx_str = requested.split(":", 1)[1]
            if not idx_str.isdigit():
                raise RuntimeError(
                    f"--device {requested!r} has an invalid GPU index {idx_str!r} "
                    f"(expected a non-negative integer, e.g. 'cuda:0')."
                )
            idx = int(idx_str)
        else:
            idx = 0

        if n_gpus == 0:
            raise RuntimeError(
                f"--device {requested!r} was requested but no CUDA GPU is visible "
                f"(torch.cuda.is_available() is False).\n{fix_checklist}"
            )
        if idx >= n_gpus:
            raise RuntimeError(
                f"--device {requested!r} requested but only {n_gpus} GPU(s) are visible "
                f"(valid indices: 0..{n_gpus - 1})."
            )
        return requested

    # requested is None: auto-pick.
    if n_gpus > 0:
        return "cuda"
    if allow_cpu_fallback:
        print("[WARN] No GPU detected -- falling back to CPU (allow_cpu_fallback=True). This will be VERY slow.")
        return "cpu"
    raise RuntimeError(
        "No CUDA GPU detected and no device was explicitly requested. Refusing to "
        f"silently run on CPU.\n{fix_checklist}"
    )


def auto_batch_size(default: int, baseline_gpu_gb: float = 12.0, device: str = "cuda") -> int:
    """
    Scale a batch size to the ACTUAL GPU's memory, so the same command adapts
    whether it's run on a 12GB local GPU or a cluster node with far more VRAM
    (e.g. 4x98GB), instead of a value hand-tuned for one machine silently
    under-using or over-committing another.

    Formula (linear, deliberately conservative): scale `default` by the ratio
    of this GPU's total memory to `baseline_gpu_gb`, never going BELOW
    `default` (so behaviour on a <=baseline_gpu_gb GPU is unchanged from today).

        auto = max(default, round(default * (this_gpu_gb / baseline_gpu_gb)))

    VERIFICATION STATUS (be honest about what's actually been checked):
      - baseline_gpu_gb=12 -> returns `default` unchanged. This is the path
        that has been run hundreds of times already in this project (the
        existing batch_size=4 default on a ~12GB GPU) -- VERIFIED safe.
      - Scaling UP for a much larger GPU (e.g. 98GB) is a REASONED
        extrapolation, not something that has been executed on that hardware
        (no such GPU was available to test against). Treat the returned
        number as a strong starting point, not a guarantee -- if a run OOMs,
        pass --batch_size explicitly to override this auto-scaling entirely.

    Only called when the CLI's --batch_size was left at its "unset" sentinel;
    an explicit --batch_size always wins and this function is never consulted.
    """
    if device == "cpu" or not torch.cuda.is_available():
        return default  # no GPU memory to scale against; leave batch size alone
    idx = int(device.split(":")[1]) if ":" in device else 0
    total_gb = torch.cuda.get_device_properties(idx).total_memory / 1024**3
    scaled = max(default, round(default * (total_gb / baseline_gpu_gb)))
    print(f"[INFO] auto_batch_size: detected {total_gb:.1f} GB on {device} "
          f"(baseline {baseline_gpu_gb:.0f} GB) -> batch_size={scaled} "
          f"(pass --batch_size explicitly to override)")
    return scaled


def write_training_params_txt(cfg, output_path: Path, device: str, original_cwd: str | Path = None) -> Path:
    """
    Write a plain, human-readable snapshot of the parameters a training run
    used, into <output_path>/training_params.txt.

    WHY THIS EXISTS (not just relying on Hydra's own .hydra/config.yaml):
    Hydra's snapshot is a raw config dump (harder to skim) and does NOT
    include derived facts like the real dataset image counts. This file is a
    single, human-readable summary of exactly what produced the checkpoints
    sitting next to it -- GPU used, effective batch, real train/val image
    counts, schedule, model paths, resume path.

    Only call this on the main process (checked by the caller via
    accelerator.is_main_process) -- writing a plain text file doesn't need to
    happen once per GPU in a multi-process run.

    original_cwd: pass hydra.utils.get_original_cwd() here. Hydra's chdir=true
    changes the process CWD to the run's OWN output folder before this runs,
    so a relative manifest path like "data/depth_training/train.jsonl" would
    silently fail to resolve (caught below, showing "unknown") without this --
    verified by execution: omitting it produced "unknown images" instead of
    the real count. If omitted, falls back to the current process CWD, which
    will be wrong under Hydra's chdir -- always pass it from the caller.
    """
    output_path.mkdir(parents=True, exist_ok=True)
    txt_path = output_path / "training_params.txt"
    _root = Path(original_cwd) if original_cwd is not None else Path.cwd()

    # Image counts: read directly from the manifest files (cheap -- just a
    # line count) rather than requiring the dataloaders to already be built.
    # Resolved against _root (the REPO root, not Hydra's run-dir CWD) so
    # relative paths in the YAML (e.g. "data/depth_training/train.jsonl")
    # are found regardless of Hydra's chdir=true.
    def _count_lines(path):
        try:
            p = Path(path)
            if not p.is_absolute():
                p = _root / p
            with open(p, "r", encoding="utf-8") as f:
                return sum(1 for line in f if line.strip())
        except Exception:
            return "unknown"

    train_n = _count_lines(cfg.data.json_file)
    val_n = _count_lines(cfg.data.val_json_file) if cfg.data.get("val_json_file") else "unknown"

    batch_size = cfg.data.batch_size
    accum = cfg.gradient_accumulation_steps
    effective_batch = batch_size * accum

    gpu_line = "cpu"
    if device != "cpu" and torch.cuda.is_available():
        idx = int(device.split(":")[1]) if ":" in device else 0
        p = torch.cuda.get_device_properties(idx)
        gpu_line = f"{p.name} ({p.total_memory / 1024**3:.1f} GB) [{device}]"

    lines = [
        f"Training run started : {__import__('datetime').datetime.now().isoformat(timespec='seconds')}",
        f"Output directory     : {output_path}",
        f"Tag                  : {cfg.get('tag', '?')}",
        f"GPU                  : {gpu_line}",
        "",
        "-- Effective batch --",
        f"data.batch_size                : {batch_size}",
        f"gradient_accumulation_steps     : {accum}",
        f"effective batch (batch*accum)   : {effective_batch}",
        f"gradient_checkpointing          : {cfg.get('gradient_checkpointing', False)}",
        f"bf16                            : {cfg.get('bf16', False)}",
        "",
        "-- Dataset --",
        f"train manifest   : {cfg.data.json_file}  ({train_n} images)",
        f"val manifest     : {cfg.data.get('val_json_file', 'null')}  ({val_n} images)",
        f"image size       : {cfg.get('size', '?')}",
        "",
        "-- Schedule --",
        f"epochs           : {cfg.get('epochs', '?')}",
        f"learning_rate    : {cfg.get('learning_rate', '?')}",
        f"lr_scheduler     : {cfg.get('lr_scheduler', '?')}",
        f"lr_warmup_steps  : {cfg.get('lr_warmup_steps', '?')}",
        f"seed             : {cfg.get('seed', '?')}",
        "",
        "-- Validation / checkpoint cadence --",
        f"val_steps        : {cfg.get('val_steps', '?')}",
        f"ckpt_steps       : {cfg.get('ckpt_steps', '?')}",
        f"val_batches      : {cfg.get('val_batches', '?')}  (x val_batch_size={cfg.data.get('val_batch_size', '?')})",
        f"n_grid_images    : {cfg.get('n_grid_images', '?')}",
        f"grid_include_empty_prompt : {cfg.get('grid_include_empty_prompt', '?')}",
        "",
        "-- Model --",
        f"base model       : {cfg.get('base_model_path') if cfg.get('local_files_only') else cfg.get('base_model_name')}",
        f"conditioning enc.: {cfg.lora.struct.encoder.get('model', '?')}",
        f"local_files_only : {cfg.get('local_files_only', '?')}",
        f"lora rank/c_dim  : {cfg.lora.struct.config.get('rank', '?')} / {cfg.lora.struct.config.get('c_dim', '?')}",
        f"resume ckpt_path : {cfg.lora.struct.get('ckpt_path') or '(fresh run, no resume)'}",
    ]

    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[INFO] Training parameters written to: {txt_path}")
    return txt_path


class DataProvider:
    def __init__(self):
        self.batch = None

    def set_batch(self, batch):
        if self.batch is not None:
            if isinstance(self.batch, torch.Tensor):
                assert self.batch.shape[1:] == batch.shape[1:], "Check: shapes probably should not change during training"

        self.batch = batch

    def get_batch(self):
        assert self.batch is not None, "Error: need to set a batch first"

        return self.batch

    def reset(self):
        self.batch = None


def getattr_recursive(obj: Any, path: str) -> Any:
    parts = path.split(".")
    for part in parts:
        if part.isnumeric():
            obj = obj[int(part)]
        else:
            obj = getattr(obj, part)
    return obj


def add_lora_from_config(model, cfg: Any, device: torch.device, dtype: torch.dtype = torch.float32) -> list[bool]:
    total_dict_keys: list[str] = []
    cfg_mask: list[bool] = []

    global_ckpt_path = cfg.get("ckpt_path", None)
    project_root = Path(os.path.abspath(__file__)).parent.parent

    for name, l in cfg.lora.items():
        if l.get("enable", "always") == "never":
            continue

        optimize = l.get("optimize", False)
        lora_cfg = l.config
        print(f"Adding {name} lora! Optimize: {optimize}")

        dp = DataProvider()
        mapper_network = l.mapper_network.to(device, dtype)
        encoder = l.encoder.to(device, dtype)
        local_ckpt_path = l.get("ckpt_path", None)

        model.add_lora_to_unet(
            lora_cfg,
            name=name,
            data_provider=dp,
            mapper=mapper_network,
            encoder=encoder,
            optimize=optimize,
            transforms=l.get("transforms", []),
        )

        cfg_mask.append(l.get("cfg", True))

        p = None
        if global_ckpt_path is not None:
            p = Path(project_root, global_ckpt_path) / name

        # local checkpoints path always override global ones
        if local_ckpt_path is not None:
            p = Path(project_root, local_ckpt_path) / name

        if p is not None:
            print("loaded checkpoint for lora", name)
            mapper_sd = torch.load(p / "mapper-checkpoint.pt", map_location=device)
            lora_sd = torch.load(p / "lora-checkpoint.pt", map_location=device)

            if os.path.isfile(p / "encoder-checkpoint.pt"):
                encoder_sd = torch.load(p / "encoder-checkpoint.pt", map_location=device)
                encoder.load_state_dict(encoder_sd)

            mapper_network.load_state_dict(mapper_sd)

            if not optimize:
                mapper_network.requires_grad_(False)
                mapper_network.eval()

            model.unet.load_state_dict(lora_sd, strict=False)
            model.unet.to(device, dtype)
            total_dict_keys += list(lora_sd.keys())

    if len(total_dict_keys) > 0 and not cfg.get("ignore_check", False):
        assert set([v for vs in model.lora_state_dict_keys.values() for v in vs]) == set(
            total_dict_keys
        ), "Probably missing or incorrect checkpoint file path. Otherwise set ignore_check=true in config."

    return cfg_mask


def toggle_loras(model, cfg: Any, mode: MODE):
    for name, l in cfg.lora.items():
        if l.get("enable", "always") in [mode, "always"]:
            for layer in model.lora_layers[name]:
                layer.lora_scale = l.config.get("lora_scale", 1.0)
        else:
            try:
                for layer in model.lora_layers[name]:
                    layer.lora_scale = 0.0
            except:
                print(f"LoRA {name} is disabled. Ignoring...")


def global_gradient_norm(model):
    mappers_params = list(filter(lambda p: p.requires_grad, reduce(lambda x, y: x + list(y.parameters()), model.mappers, [])))
    encoder_params = list(filter(lambda p: p.requires_grad, reduce(lambda x, y: x + list(y.parameters()), model.encoders, [])))

    total_norm = clip_grad_norm_(model.params_to_optimize + mappers_params + encoder_params, 1e9)
    return total_norm.item()


def save_checkpoint(unet_sds: dict[str, dict[str, torch.Tensor]], mapper_network_sd: list[dict[str, torch.Tensor]], encoder_sd: list[dict[str, torch.Tensor]] | None, path: Path):
    for i, (name, sd) in enumerate(unet_sds.items()):
        p = path / name
        p.mkdir(parents=True, exist_ok=True)

        torch.save(sd, p / "lora-checkpoint.pt")
        torch.save(mapper_network_sd[i], p / f"mapper-checkpoint.pt")
        if encoder_sd is not None and len(encoder_sd[i]) > 0:
            torch.save(encoder_sd[i], p / f"encoder-checkpoint.pt")


def roll_list(l, n):
    # consistent with torch.roll
    return l[-n:] + l[:-n]


# ============================================================================ #
#  PER-CHECKPOINT QUANTITATIVE METRIC — shared by depth_training.py and         #
#  seg_training.py                                                              #
#                                                                               #
#  WHY THIS EXISTS: the project's end goal is an OBJECTIVE depth-vs-seg          #
#  comparison. Eyeballing checkpoint grids alone can't decide "which             #
#  conditioning is better"; a number logged per checkpoint can. PSNR + SSIM     #
#  between each FIXED validation scene's generation and its real image are     #
#  used because:                                                                #
#    • the FIXED scenes + the fixed generation seed make the value comparable   #
#      across checkpoints of one run AND across the depth run vs the seg run    #
#      (identical protocol, same val set, same seed);                           #
#    • FID needs thousands of samples per point to be meaningful — useless at   #
#      10 images per checkpoint;                                                #
#    • CLIP similarity would need CLIP weights, which are not among the local   #
#      offline models (local_files_only must stay true end-to-end).             #
#  These are structural-fidelity proxies, not absolute quality scores: watch    #
#  the TREND across checkpoints, and compare depth vs seg at MATCHED steps.     #
# ============================================================================ #

def compute_psnr_ssim(img_a, img_b):
    """
    PSNR (dB) + SSIM between two uint8 RGB images of identical shape.

    img_a, img_b: numpy uint8 arrays [H, W, 3] (e.g. np.asarray(pil_img)).
    Returns (psnr: float, ssim: float). Pure numpy — no new dependencies.

    SSIM follows Wang et al. 2004 on the GRAYSCALE image with the standard
    constants (K1=0.01, K2=0.03, L=255) and an 8x8 non-overlapping uniform
    window — a standard simplification that is fully adequate for TREND
    comparison across checkpoints, which is the only use here.
    """
    import numpy as np

    a = np.asarray(img_a, dtype=np.float64)
    b = np.asarray(img_b, dtype=np.float64)
    assert a.shape == b.shape, f"shape mismatch: {a.shape} vs {b.shape}"

    # ---- PSNR on RGB ----
    mse = np.mean((a - b) ** 2)
    psnr = 99.0 if mse == 0 else 10.0 * np.log10(255.0 ** 2 / mse)

    # ---- SSIM on grayscale, 8x8 block statistics (uniform window) ----
    def to_gray(x):
        return 0.299 * x[..., 0] + 0.587 * x[..., 1] + 0.114 * x[..., 2]

    ga, gb = to_gray(a), to_gray(b)
    win = 8
    H, W = ga.shape
    H8, W8 = H - H % win, W - W % win

    def blocks(g):
        return (g[:H8, :W8]
                .reshape(H8 // win, win, W8 // win, win)
                .transpose(0, 2, 1, 3)
                .reshape(-1, win * win))

    A, B = blocks(ga), blocks(gb)
    mu_a, mu_b = A.mean(1), B.mean(1)
    var_a, var_b = A.var(1), B.var(1)
    cov = ((A - mu_a[:, None]) * (B - mu_b[:, None])).mean(1)
    C1, C2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    ssim_map = (((2 * mu_a * mu_b + C1) * (2 * cov + C2)) /
                ((mu_a ** 2 + mu_b ** 2 + C1) * (var_a + var_b + C2)))
    return float(psnr), float(ssim_map.mean())
