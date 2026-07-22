#!/bin/bash -x
# =============================================================================
# train_grounded_sam_jupiter.sbatch — JUPITER Booster (4x GH200, single node)
# GPU training job, grounded_sam branch (experiment=train_grounded_sam).
#
# DIFFERENT HARDWARE THAN slurm/train_grounded_sam_jusuf.sbatch:
#   JUSUF   = 1x NVIDIA V100 16GB per node   (that script: 1 GPU, bare `srun python`)
#   JUPITER = 4x NVIDIA GH200 96GB per node  (THIS script: 4 GPUs, one node,
#             `accelerate launch --multi_gpu --num_processes=4`)
# GH200's Grace CPU is ARM64 (aarch64), NOT x86_64 like JUSUF/most clusters.
#
# ======================= UNVERIFIED — YOU MUST FILL IN ======================
# The following were NOT confirmed against your actual JUPITER allocation
# (public docs did not list them at the time this script was written):
#   - SLURM partition name for the Booster module (--partition below is a
#     placeholder)
#   - Max walltime for your project on this partition (--time below is a
#     placeholder — check your allocation's limits before relying on it)
#   - Module names / software stage for ARM64 (module load lines below are
#     placeholders — DO NOT assume the JUSUF script's x86 module names exist
#     here; check JSC's JUPITER user documentation or support for the
#     Booster's own Stages system)
#   - Your existing conda `loradapter` env was built for x86_64 and will NOT
#     run on this ARM64 node as-is — you need an aarch64-native environment
#     (check whether JSC provides a pre-built module stack for this before
#     attempting to rebuild your own from scratch)
# ==============================================================================
#
# NO CALC JOB EXISTS FOR THIS PIPELINE, AND THAT IS CORRECT, NOT A GAP:
# your CARLA masks (class_map.png, I;16, 1280x800) are produced BY CARLA
# ITSELF when you capture with the instance-segmentation-camera enabled —
# the "calculation" already happened at capture time on your simulation
# machine. This repo's grounded_sam branch never runs a live segmentation
# model (GroundedSamEncoder.forward() intentionally raises NotImplementedError
# — see src/encoders/grounded_sam_encoder.py's own docstring, "Tier 2" was
# deliberately never built). There is nothing to submit to this cluster to
# "compute" a grounded_sam map — just stage your already-captured masks (see
# step 2 below) and train directly.
# =============================================================================

#SBATCH --job-name=loradapter-gs-jupiter
#SBATCH --account=<YOUR_BUDGET_ACCOUNT>
#SBATCH --partition=<FILL IN — Booster partition name, unconfirmed>
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=<FILL IN — size to your allocation; GH200 nodes have
#                          many ARM cores across 4 superchips, but the exact
#                          schedulable count for your project is unconfirmed>
#SBATCH --time=<FILL IN — check your project's walltime limit on this partition>
#SBATCH --signal=B:USR1@300                    # graceful-stop mechanism (grounded_sam_training.py:90-93)
#SBATCH --output=logs/train-%j.out
#SBATCH --error=logs/train-%j.err

set -euo pipefail
mkdir -p logs

module purge
# FILL IN: JUPITER Booster's own ARM64 software stage — do NOT reuse the
# JUSUF script's x86 module names (Stages/2024, GCC, OpenMPI, CUDA, cuDNN,
# PyTorch built for x86 will not exist/work here). Check JSC's JUPITER user
# docs or support for the Booster's module system before filling this in.
# module load <Booster-specific stage>
# module load <Booster-specific GCC/CUDA/cuDNN/PyTorch, ARM64 builds>
source $PROJECT/<your_project>/sc_venv_template/activate.sh   # must point at an ARM64-native env

cd $PROJECT/<your_project>/LoRAdapter

# ---- Stage your CARLA captures onto $SCRATCH first (from a LOGIN node) ----
# $DATA is not visible to compute nodes. One-time / per refresh, e.g.:
#   rsync -a $DATA/<your_project>/grounded_sam/ $SCRATCH/<your_project>/grounded_sam/
# Each manifest line needs: raw_image_path, seg_path (your class_map.png),
# prompt — the loader (commit 5db3624) reads the real I;16, 1280x800 format
# directly and letterboxes it correctly; no conversion needed on your part.
DATA_DIR=$SCRATCH/<your_project>/grounded_sam

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

# RESIZE_MODE (see GROUNDED_SAM.md §5.0b): "letterbox"
# (SquarePad, default) or "CenterCrop" (original stock LoRAdapter recipe).
# Also names the output folder: outputs/train/grounded_sam_${RESIZE_MODE}/...
# -- submit this script twice, once per mode, to compare both by trained
# generation quality. No calc step needed for either (see banner above).
RESIZE_MODE="letterbox"

RESUME_CKPT=""   # e.g. "outputs/train/grounded_sam_letterbox/runs/2026-01-01/00-00-00/checkpoint-epoch3/step9000"

# (Optional, no GPU needed, cheap CPU sanity check before spending GPU time —
# run this on a LOGIN node beforehand, not inside this GPU job:
#   python check_seg_map_format.py --seg_map <one_real_class_map.png>
#   python scan_seg_map_classes.py --json_file "${DATA_DIR}/train.jsonl"  )

# ---- Size batch_size / gradient_accumulation_steps to THIS GPU ------------
# recommend_batch_and_accum() extrapolates LINEARLY from the only MEASURED
# point this project has (12GB GPU: batch_size=4, accum=4). At 96GB this is a
# REASONED EXTRAPOLATION, NOT independently verified at this scale — sanity
# check with a short dry run (see grounded_sam_training.py's docstring) before
# committing to the full run. Detected PER-GPU here; each of the 4 accelerate
# processes below uses this same per-GPU batch_size (NOT divided further).
read -r REC_BATCH REC_ACCUM <<< "$(python -c "
import torch
from recommend_training_params import recommend_batch_and_accum
total_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
b, a, _, _ = recommend_batch_and_accum(total_gb)
print(b, a)
")"
echo "[INFO] GPU-sized (per GPU): data.batch_size=${REC_BATCH}  gradient_accumulation_steps=${REC_ACCUM}"
echo "[INFO] TOTAL effective batch across 4 GPUs = $((REC_BATCH * REC_ACCUM * 4))"

# ---- Learning rate: UNSCALED, deliberately -------------------------------
# Effective batch here (~64, 4x this project's validated 16) is untested.
# sqrt(4)-scaling (~2e-4) is a plausible follow-up experiment, but stacking
# an unverified LR on top of unverified hardware (GH200/ARM64) and an
# unverified multi-GPU batch size risks wasting a long cluster run
# diagnosing which change caused an instability. Keep the ONE number this
# project actually has evidence for; try scaled LR as a deliberate, separate
# follow-up once this baseline is confirmed stable.
LEARNING_RATE="1.0e-4"

# ---- Epoch / monitoring schedule, sized to YOUR REAL dataset --------------
# 30,000 train / 10,000 val / 10,000 test images.
# steps/epoch = ceil(30000 / (REC_BATCH * REC_ACCUM * 4)) -- recompute if
# your real image count differs from 30k. At batch=16,accum=1 (4 GPUs):
#   steps/epoch = ceil(30000 / 64) = 469
#   epochs=10 -> total steps = 4,690
#   val_steps  = 70   (469/7  ~ 67, rounded  -- ~7 val/loss checks per epoch)
#   ckpt_steps = 130  (469/3.5 ~ 134, rounded -- ~3-4 checkpoint saves per epoch)
#   lr_warmup  = 60   (~13% of epoch 1; NOT overridden before this fix -- the
#     job was silently inheriting the YAML's 500-step warmup, which is MORE
#     than a full epoch at this batch size. Recompute all FOUR of these if
#     REC_BATCH/REC_ACCUM differ from the 16/1 this was sized for.)
EPOCHS=10
VAL_STEPS=70
CKPT_STEPS=130
LR_WARMUP_STEPS=60

# ---- Launch: single node, 4 GPUs, one accelerate process per GPU ----------
# NOT `srun python` (that's the JUSUF single-GPU pattern) -- accelerate
# spawns and coordinates the 4 worker processes itself within this one
# Slurm task. mixed_precision=bf16 matches configs/experiment/train_grounded_sam.yaml's
# bf16: true (kept explicit here so it's correct even if that default ever changes).
accelerate launch --multi_gpu --num_processes=4 --mixed_precision=bf16 \
    grounded_sam_training.py experiment=train_grounded_sam \
    resize_mode=${RESIZE_MODE} \
    data.json_file="${DATA_DIR}/train.jsonl" \
    data.val_json_file="${DATA_DIR}/val.jsonl" \
    data.batch_size=${REC_BATCH} \
    gradient_accumulation_steps=${REC_ACCUM} \
    learning_rate=${LEARNING_RATE} \
    lr_warmup_steps=${LR_WARMUP_STEPS} \
    epochs=${EPOCHS} \
    val_steps=${VAL_STEPS} \
    ckpt_steps=${CKPT_STEPS} \
    ${RESUME_CKPT:+lora.struct.ckpt_path="$RESUME_CKPT"}
