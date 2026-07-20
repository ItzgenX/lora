#!/bin/bash -x
# =============================================================================
# calc_seg_segformer_jusuf.sbatch — JUSUF GPU job: compute SegFormer seg maps
# for the segformer branch, dataset-SCAN mode (matches your established local
# workflow: --dataset_dir scans your custome_dataset for raw_image.jpg, saves
# maps to a SIBLING folder, mirrored structure — NEVER plain --data_dir mode,
# that mode collided all maps into one file the last time it was tried).
# =============================================================================

#SBATCH --job-name=loradapter-segcalc
#SBATCH --account=<YOUR_BUDGET_ACCOUNT>
#SBATCH --partition=develgpus                  # calc is I/O-bound, not compute-heavy — the smaller/dev partition
                                                # is usually enough and has internet if you ever need it (see the
                                                # 0-to-hero guide for gpus vs develgpus)
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --time=04:00:00                        # calc is one forward pass per image, much faster than training —
                                                # raise this if your real image count makes 4h too tight
#SBATCH --output=logs/segcalc-%j.out
#SBATCH --error=logs/segcalc-%j.err

set -euo pipefail
mkdir -p logs

# ---- 1. Modules + venv (identical to the training script — see the 0-to-hero
#         guide §5/§6 for what these two blocks do and why) -------------------
module purge
module load Stages/2024                       # verify with `module load Stages` / `module avail`
module load GCC OpenMPI CUDA cuDNN PyTorch     # verify exact names with `module spider PyTorch`
source $PROJECT/<your_project>/sc_venv_template/activate.sh

cd $PROJECT/<your_project>/LoRAdapter

# ---- 2. Stage your RAW dataset onto $SCRATCH first (from a LOGIN node, not
#         here) -- $DATA is not visible to compute nodes. One-time / per
#         refresh, e.g.:
#   rsync -a $DATA/<your_project>/custome_dataset/ $SCRATCH/<your_project>/custome_dataset/
RAW_DATASET_DIR=$SCRATCH/<your_project>/custome_dataset
MANIFEST_DIR=$SCRATCH/<your_project>/data          # folder holding your train/val/test.jsonl (prompt + split source)

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

# ---- 3. Compute the maps ---------------------------------------------------
# Saves seg maps to a SIBLING folder next to RAW_DATASET_DIR (mirrored
# structure), named <RAW_DATASET_DIR>_seg_map/ — matches your local workflow.
srun python seg_map_calculations.py \
    --dataset_dir "${RAW_DATASET_DIR}" \
    --data_dir "${MANIFEST_DIR}" \
    --image_path raw_image_path

# Alternative, simpler mode if you'd rather work from ONE manifest instead of
# a scanned folder (produces the sibling _seg_map folder + a self-verified
# <stem>_seg.jsonl beside your input — see seg_map_calculations.py's own
# --json_file help text):
#   srun python seg_map_calculations.py --json_file "${MANIFEST_DIR}/train.jsonl"
