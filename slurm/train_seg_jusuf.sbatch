#!/bin/bash -x
# =============================================================================
# train_seg_jusuf.sbatch — JUSUF GPU training job, segformer branch
# (experiment=train_seg). Run calc_seg_segformer_jusuf.sbatch FIRST — this
# script only trains on maps that already exist.
# =============================================================================

#SBATCH --job-name=loradapter-seg
#SBATCH --account=<YOUR_BUDGET_ACCOUNT>
#SBATCH --partition=gpus                       # production partition (no internet); use develgpus for a quick smoke test
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --time=24:00:00                        # 24h = hard max on 'gpus' — see §9 (resume) in the 0-to-hero guide
#SBATCH --signal=B:USR1@300                    # warns the job 300s before the wall; seg_training.py already
                                                # handles SIGUSR1 (seg_training.py:90-93) by saving a checkpoint
                                                # and exiting cleanly instead of being killed mid-step
#SBATCH --output=logs/train-%j.out
#SBATCH --error=logs/train-%j.err

set -euo pipefail
mkdir -p logs

module purge
module load Stages/2024
module load GCC OpenMPI CUDA cuDNN PyTorch
source $PROJECT/<your_project>/sc_venv_template/activate.sh

cd $PROJECT/<your_project>/LoRAdapter

# Must match where calc_seg_segformer_jusuf.sbatch wrote its output.
DATA_DIR=$SCRATCH/<your_project>/data/seg_training

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

# Leave empty for a fresh run; set after a timeout/requeue to resume from the
# last checkpoint (printed at the end of the previous job's .out log).
RESUME_CKPT=""   # e.g. "outputs/train/seg/runs/2026-01-01/00-00-00/checkpoint-epoch3/step9000"

# ---- Size batch_size / gradient_accumulation_steps to THIS GPU ------------
# seg_training.py itself does NOT auto-scale batch_size -- it just reads the
# static value in configs/experiment/train_seg.yaml (batch_size=4, tuned for
# a ~12GB reference GPU: "peak VRAM ~11.1GB/12GB at batch_size=4"). Left as-is
# on JUSUF's 16GB V100 nodes, that under-uses ~4GB of VRAM every step -- not
# wrong, just not "appropriate" for this hardware. This block runs BARE
# (no srun needed -- this whole script already executes ON the allocated
# compute node, not the login node) on this job's real GPU and reuses this
# repo's OWN validated formula (recommend_training_params.py's
# recommend_batch_and_accum -- the same function that already prints these
# numbers for a manual --data_dir run) so the numbers are correct on THIS GPU
# specifically, not a guess. Effective batch (batch_size*accum) is held at
# 16 -- the value learning_rate=1e-4 was validated at -- never silently
# changed, only how it's SPLIT between batch_size and accumulation.
read -r REC_BATCH REC_ACCUM <<< "$(python -c "
import torch
from recommend_training_params import recommend_batch_and_accum
total_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
b, a, _, _ = recommend_batch_and_accum(total_gb)
print(b, a)
")"
echo "[INFO] GPU-sized: data.batch_size=${REC_BATCH}  gradient_accumulation_steps=${REC_ACCUM}  (effective batch = $((REC_BATCH * REC_ACCUM)))"

srun python seg_training.py experiment=train_seg \
    data.json_file="${DATA_DIR}/train.jsonl" \
    data.val_json_file="${DATA_DIR}/val.jsonl" \
    data.batch_size=${REC_BATCH} \
    gradient_accumulation_steps=${REC_ACCUM} \
    ${RESUME_CKPT:+lora.struct.ckpt_path="$RESUME_CKPT"}
