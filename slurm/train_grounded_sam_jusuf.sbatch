#!/bin/bash -x
# =============================================================================
# train_grounded_sam_jusuf.sbatch — JUSUF GPU training job, grounded_sam
# branch (experiment=train_grounded_sam).
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

#SBATCH --job-name=loradapter-gs
#SBATCH --account=<YOUR_BUDGET_ACCOUNT>
#SBATCH --partition=gpus
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --time=24:00:00
#SBATCH --signal=B:USR1@300                    # same graceful-stop mechanism as train_seg (seg_training.py:90-93,
                                                # byte-identical on this branch, verified)
#SBATCH --output=logs/train-%j.out
#SBATCH --error=logs/train-%j.err

set -euo pipefail
mkdir -p logs

module purge
module load Stages/2024
module load GCC OpenMPI CUDA cuDNN PyTorch
source $PROJECT/<your_project>/sc_venv_template/activate.sh

cd $PROJECT/<your_project>/LoRAdapter

# ---- Stage your CARLA captures onto $SCRATCH first (from a LOGIN node) ----
# $DATA is not visible to compute nodes. One-time / per refresh, e.g.:
#   rsync -a $DATA/<your_project>/grounded_sam/ $SCRATCH/<your_project>/grounded_sam/
# Each manifest line needs: raw_image_path, seg_path (your class_map.png),
# prompt — the loader (fixed 2026-07-20, commit 5db3624) reads the real
# I;16, 1280x800 format directly and letterboxes it correctly; no conversion
# needed on your part.
DATA_DIR=$SCRATCH/<your_project>/grounded_sam

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

RESUME_CKPT=""   # e.g. "outputs/train/grounded_sam/runs/2026-01-01/00-00-00/checkpoint-epoch3/step9000"

# (Optional, no GPU needed, cheap CPU sanity check before spending GPU time —
# run this on a LOGIN node beforehand, not inside this GPU job:
#   python check_seg_map_format.py --seg_map <one_real_class_map.png>
#   python scan_seg_map_classes.py --json_file "${DATA_DIR}/train.jsonl"  )

# ---- Size batch_size / gradient_accumulation_steps to THIS GPU ------------
# Same situation as the segformer branch's train_seg_jusuf.sbatch: this
# branch's configs/experiment/train_grounded_sam.yaml also hardcodes
# batch_size=4, gradient_accumulation_steps=4 (confirmed: "4 batch x 4 accum
# = 16 effective batch", same 12GB-reference baseline) and seg_training.py
# never auto-scales it. Bare python (no srun -- this script already executes
# ON the allocated compute node) reuses this repo's OWN validated formula
# (recommend_training_params.py's recommend_batch_and_accum) against THIS
# job's real GPU. Effective batch held at 16 (what learning_rate=1e-4 was
# validated at) -- only the batch_size/accum SPLIT changes, never silently
# the learning rate.
read -r REC_BATCH REC_ACCUM <<< "$(python -c "
import torch
from recommend_training_params import recommend_batch_and_accum
total_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
b, a, _, _ = recommend_batch_and_accum(total_gb)
print(b, a)
")"
echo "[INFO] GPU-sized: data.batch_size=${REC_BATCH}  gradient_accumulation_steps=${REC_ACCUM}  (effective batch = $((REC_BATCH * REC_ACCUM)))"

srun python seg_training.py experiment=train_grounded_sam \
    data.json_file="${DATA_DIR}/train.jsonl" \
    data.val_json_file="${DATA_DIR}/val.jsonl" \
    data.batch_size=${REC_BATCH} \
    gradient_accumulation_steps=${REC_ACCUM} \
    ${RESUME_CKPT:+lora.struct.ckpt_path="$RESUME_CKPT"}
