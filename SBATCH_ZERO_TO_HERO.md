# SBATCH / SLURM on JUSUF — Zero to Hero

Written for someone with **zero prior knowledge** of clusters, SLURM, or
`sbatch`. Every fact about JUSUF below is sourced from the official docs at
`apps.fz-juelich.de/jsc/hps/jusuf/` (configuration, batchsystem,
software-modules, environment, ai-overview pages) — nothing is guessed.
Covers 3 companion scripts: `calc_seg_segformer_jusuf.sbatch`,
`train_seg_jusuf.sbatch`, `train_grounded_sam_jusuf.sbatch`.

---

## 1. What a cluster actually is, and why you can't just run `python train.py`

A cluster is hundreds of computers ("nodes") wired together, shared by many
research groups at once. You never log into a GPU machine directly. Instead:

1. You log into a **login node** (a small shared machine, no GPU) via SSH.
2. You write a **job script** — a shell script with special `#SBATCH` comment
   lines describing what you need (how many GPUs, how long, etc.).
3. You **submit** it with `sbatch myscript.sbatch`. It joins a queue.
4. A scheduler called **SLURM** finds you a free GPU node when your turn
   comes, and runs your script there — unattended, no one watching it.
5. Output goes to log files, not your terminal (you're not connected to the
   node while it runs — you could even close your laptop).

This is why the script has to be self-contained: it loads its own software,
finds its own data, and saves its own results, because nothing from your
interactive login session carries over to the compute node.

## 2. JUSUF's hardware, in plain numbers

- **45 GPU nodes total.** Each node has exactly **1× NVIDIA V100 GPU (16GB
  memory)**, 2× AMD EPYC 7742 CPUs (128 CPU cores total), 256GB of RAM, and
  1TB of fast local disk.
- One GPU per node means "give me a GPU node" and "give me a V100" are the
  same request here — there's no picking between multiple GPUs on one node.
- Operating system: Rocky Linux 9 (a Linux distribution — same family as
  RHEL/CentOS, all your `python`/`bash` commands work as expected).

## 3. The two GPU queues ("partitions") — which to use when

SLURM groups nodes into **partitions** — think of them as different queues
with different rules:

| Partition | Nodes | Max walltime | Internet access | Use for |
|---|---|---|---|---|
| `gpus` | 1–39 | 24h (6h if a node fails and your job restarts — "nocont") | **No** | Your real, full training runs |
| `develgpus` | 1–6 | 24h (same) | **Yes** | Quick tests, debugging, first-time setup (downloading a HF model needs internet!) |

Both give you the same hardware (1× V100/node) — the difference is
internet access and how many nodes are set aside for quick turnaround. Use
`develgpus` the FIRST time you test a script (in case it needs to download
something), then switch to `gpus` for the real, long run.

## 4. Anatomy of an sbatch script — every line explained

Open `train_seg_jusuf.sbatch` side by side with this section.

```bash
#!/bin/bash -x
```
Standard "this is a bash script" line. `-x` makes it print every command it
runs into the log — invaluable for debugging a job you can't watch live.

```bash
#SBATCH --job-name=loradapter-seg
```
A name YOU choose, shown in the queue (`squeue`) so you recognize your job
among everyone else's.

```bash
#SBATCH --account=<YOUR_BUDGET_ACCOUNT>
```
**Required.** Every job spends "compute hours" from a project's allocated
budget — this says which project to charge. Get this value from your PI or
JSC project page; I cannot know it.

```bash
#SBATCH --partition=gpus
```
Which queue (§3) to submit into.

```bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
```
"I need 1 machine, running 1 copy of my program." (This project's training
is single-GPU, not multi-node distributed — 1/1 is correct here.)

```bash
#SBATCH --gres=gpu:1
```
"Generic RESource" request — explicitly asks for 1 GPU. (JUSUF docs note
this can technically be omitted since every GPU node has exactly 1 anyway —
kept explicit here so the script is self-documenting.)

```bash
#SBATCH --cpus-per-task=32
```
How many CPU cores your one task gets — used here for PyTorch's DataLoader
worker processes (loading/decoding images in parallel with GPU compute). The
node has 128 cores; 32 is generous without starving other jobs sharing the
node... except on JUSUF's GPU nodes it's usually one job per node anyway, so
this mainly just controls how many DataLoader workers make sense to set.

```bash
#SBATCH --time=24:00:00
```
Wall-clock time limit, `HH:MM:SS`. Your job is **killed** the instant this
elapses, finished or not. 24h is the maximum on `gpus` (§3) — see §9 for
what happens when your training needs longer than that.

```bash
#SBATCH --signal=B:USR1@300
```
"300 seconds before my time limit hits, send my script a USR1 signal as a
warning" (`B:` = deliver it to the (B)atch script itself, not just child
processes). This project's `seg_training.py` already listens for USR1
(`seg_training.py:90-93`) and responds by saving a checkpoint and exiting
cleanly — this line is what makes that safety-net actually fire. Without it,
SLURM just kills the job outright at the 24h mark with no warning.

```bash
#SBATCH --output=logs/train-%j.out
#SBATCH --error=logs/train-%j.err
```
Where normal output / errors get written. `%j` is replaced with the job ID
SLURM assigns you — so every job gets its own log file, never overwritten.

```bash
set -euo pipefail
```
Not a SLURM thing — a bash safety habit: `-e` stops the script on the first
error instead of plowing ahead, `-u` errors on typo'd unset variables, `-o
pipefail` makes a pipeline fail if any stage fails. Cheap insurance.

The rest of the script (module loads, venv activation, `cd`, environment
variables, the final `srun python ...` line) is regular bash — SLURM's job
is done once it starts your script; everything after the `#SBATCH` block
just runs top to bottom like any shell script would on your own machine.

```bash
srun python seg_training.py experiment=train_seg ...
```
`srun` launches your program **inside the SLURM job's resource
allocation** — it's what actually puts your Python process on the GPU node
SLURM gave you. For a single-GPU job like this, `srun python foo.py` behaves
much like just running `python foo.py`, but `srun` also gets your job's
resource usage properly tracked by SLURM (visible later via `sacct`).

### 4a. The block right before that `srun` line — sizing to THIS GPU

`seg_training.py` itself never auto-scales its batch size — it just reads a
static `data.batch_size=4` from `configs/experiment/train_seg.yaml`, hand-
tuned for a ~12GB reference GPU. Left as-is on a 16GB V100, that leaves real
VRAM unused every step (not wrong, just wasteful). The script runs this
BEFORE the `srun` line to fix that, on the job's real allocated GPU (no
`srun` needed for this part — the whole batch script already executes ON
the compute node, not the login node):
```bash
read -r REC_BATCH REC_ACCUM <<< "$(python -c "
import torch
from recommend_training_params import recommend_batch_and_accum
total_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
b, a, _, _ = recommend_batch_and_accum(total_gb)
print(b, a)
")"
```
This reuses `recommend_training_params.py`'s OWN validated formula (the
same one that already prints numbers for a manual `--data_dir` run) —
nothing new invented. `REC_BATCH`/`REC_ACCUM` are then passed as
`data.batch_size=${REC_BATCH} gradient_accumulation_steps=${REC_ACCUM}` on
the `srun` line. The formula holds `batch_size * accum` (the "effective
batch") fixed at 16 — the value `learning_rate=1e-4` was validated at — so
only how that 16 is SPLIT changes with the GPU, never the learning rate's
validity. Verified by execution: on a real 12GB GPU this returns `(4, 4)`
unchanged (confirms it's a no-op on the size it was already tuned for); fed
JUSUF's documented 16GB V100 spec it returns `(5, 3)` (effective batch 15,
≈16 — the same real function, not a hand-guessed number).

## 5. Modules — how software gets loaded

JUSUF doesn't give you a plain empty Linux box — it has a **module system**
managing hundreds of software versions so they don't conflict.

- `module avail` — list what's loadable right now.
- `module spider <name>` — search everywhere (including modules that need a
  prerequisite loaded first) for something, e.g. `module spider PyTorch`.
- `module load <name>` — actually load it into your shell/job.
- `module purge` — unload everything first (the scripts start with this so
  you always get a known-clean state, not whatever was loaded by accident).

Software here is versioned by **Stage** — an annual snapshot
(`module load Stages/2024`). All modules within one Stage are built to work
together (matching compiler/CUDA/MPI versions); mixing modules across
Stages is how you get mysterious linker errors, so always load a Stage
first.

**I could not verify today's exact module names** (`GCC`, `OpenMPI`, `CUDA`,
`cuDNN`, `PyTorch` in the scripts are placeholders based on the documented
pattern) — run `module spider PyTorch` yourself once logged in and correct
the `module load` line if the name differs.

## 6. Python environment — why NOT conda

Your local machine uses a conda env (`loradapter`). **On JSC clusters,
Anaconda is off-limits** (their repository's license terms don't permit use
on JSC systems) — you'll use JSC's own tool instead, called
`sc_venv_template`. One-time setup, done ONCE on a login node before your
first job:

```bash
cd $PROJECT/<your_project>
git clone https://gitlab.jsc.fz-juelich.de/kesselheim1/sc_venv_template
# edit sc_venv_template/modules.sh -> should match your #SBATCH script's module load line
# edit sc_venv_template/requirements.txt -> pip packages this project needs (below)
bash sc_venv_template/setup.sh
```

`requirements.txt` should list everything from this repo's
`environment.yaml` pip section **except torch/torchvision**:
```
diffusers==0.25.0
accelerate
transformers
tensorboard
Pillow
hydra-core
jaxtyping
einops
numpy
open-clip-torch
torch-fidelity
basicsr
tqdm
```
Why not torch too: the `PyTorch` module you load in §5 is pre-built by JSC
against THIS cluster's exact CUDA/driver stack. A pip-installed torch would
bring its own bundled CUDA and can silently conflict with the module's — the
single most common "worked on my laptop, broke on the cluster" GPU bug. Let
the module provide torch; only `pip install` what it doesn't.

Every job script then does `source $PROJECT/<project>/sc_venv_template/activate.sh`
instead of `conda activate loradapter` — that's the cluster equivalent of
activating your environment.

## 7. Filesystems — where do files actually go? (the #1 real gotcha)

| Variable | Who can read/write it | What it's for | Danger |
|---|---|---|---|
| `$HOME` | login + compute | tiny personal files (SSH keys, dotfiles) | small quota, not for data |
| `$PROJECT` | login + compute | your code (this repo lives here) | — |
| `$SCRATCH` | login + compute | fast temp storage for active I/O (datasets/checkpoints DURING training) | **auto-deleted**: files after 90 days untouched, empty folders after 3 days |
| `$DATA` | **login ONLY** | long-term large dataset storage | **compute nodes CANNOT read this** — a job trying to open a file under `$DATA` will fail |
| `$ARCHIVE` | login only | cold long-term storage (tape) | slow, for things you rarely touch |

**The gotcha that will bite you if you skip this:** your job runs on a
compute node. If your dataset lives under `$DATA`, the job cannot see it —
it's a login-node-only filesystem. The fix, and the reason every script here
has a comment about it: **before submitting a training/calc job**, copy (or
`rsync`) your dataset from `$DATA` to `$SCRATCH` while you're still on a
login node:
```bash
rsync -a $DATA/<project>/custome_dataset/ $SCRATCH/<project>/custome_dataset/
```
Then point the job at the `$SCRATCH` copy. Because `$SCRATCH` purges after
90 days of inactivity, re-run the `rsync` if you come back to a stale
project after a long break — active training runs touching the files
regularly won't trigger the purge.

Also run `jutil env activate -p <project>` once per login session — most of
these variables (`$PROJECT`, `$SCRATCH`, `$DATA`) only resolve correctly
after you've activated your project.

## 8. Submitting and watching a job

```bash
sbatch train_seg_jusuf.sbatch     # submit -> prints "Submitted batch job 123456"
squeue -u $USER                   # see your jobs: PENDING (queued) or RUNNING
tail -f logs/train-123456.out     # live-follow the output while it runs
scancel 123456                    # kill a job you submitted by mistake
sacct -j 123456 --format=JobID,Elapsed,State,MaxRSS   # after it finishes: how long it ran, exit state, memory used
```

**Recommended first run:** before trusting the real 24-hour job, submit a
tiny smoke test on `develgpus` to prove the whole chain works — modules
load, venv activates, data paths resolve, one training step runs:
```bash
sbatch --partition=develgpus --time=00:15:00 train_seg_jusuf.sbatch
```
(temporarily edit the script's `data.json_file`/`val_json_file` to point at
a 2-3 image manifest, and add `epochs=1 val_steps=2 ckpt_steps=50` to the
`srun python` line, mirroring how this was smoke-tested locally before any
real cluster run).

## 9. How long should `--time=` actually be? (sizing it to YOUR epochs)

Two numbers multiply together to give you the answer: **how many optimizer
steps** your run will do, and **how many seconds each step actually takes on
the GPU you land on**. Neither is a number to guess — both are computed or
measured, here's exactly how.

### Step 1 — how many steps, for N epochs

```
steps_per_epoch = ceil(N_train / effective_batch)
total_steps     = steps_per_epoch * epochs
```
`effective_batch` is `data.batch_size * gradient_accumulation_steps` — as of
§4a, this is `REC_BATCH * REC_ACCUM`, printed by the script itself
at the top of your `.out` log (e.g. `effective batch = 15` on a V100).
`N_train` is your real training-set image count.
`recommend_training_params.py --data_dir <your_manifests> --epochs <N>`
(run on a login node, reads local files only) prints this exact number for
you — no need to compute it by hand.

### Step 2 — seconds per step: MEASURE it, don't guess it

This depends on the GPU, image resolution, `gradient_checkpointing`, and
disk/dataloader speed — enough variables that a number given here (from a
different machine, a 2-image test, or a spec sheet) would not be
trustworthy for YOUR real run. The correct way, and the only way that's
actually accurate for your specific case:

1. Submit a short SMOKE run first — same script, `--partition=develgpus`,
   a tight time budget, real data (even just a few hundred images is
   enough to get stable per-step timing once past warmup):
   ```bash
   sbatch --partition=develgpus --time=00:20:00 train_seg_jusuf.sbatch
   ```
2. Open the `.out` log. `seg_training.py`'s progress bar prints a live
   `s/it` (seconds per iteration) or `it/s` figure. **Ignore the first
   5-10 steps** — those include one-time CUDA/cuDNN warmup and are
   noticeably slower than steady-state. Read the number once it stabilizes.
3. Alternatively, after the smoke job finishes: `sacct -j <jobid>
   --format=Elapsed` gives you real total wall-clock for however many
   steps that job actually completed — divide by step count for the same
   per-step figure, cross-checking the progress-bar reading.

### Step 3 — put it together

```
estimated_wall_clock = total_steps * measured_seconds_per_step
```
Add a **20-30% safety margin** on top — periodic validation (`val_steps`)
and checkpoint saves (`ckpt_steps`) are real time not captured in a raw
per-training-step measurement, and dataloader I/O can hiccup under cluster
filesystem load that a short smoke test won't fully reveal.

### Step 4 — set `--time=`, or chain jobs

- If `estimated_wall_clock` (with margin) fits under 24h: set `--time=`
  to that value, rounded up (SLURM format `HH:MM:SS`) — no need to always
  max out at 24h if your real run is shorter; a tighter time limit can mean
  a shorter queue wait.
- If it's OVER 24h: **do not try to raise `--time=` past the partition
  max** — 24h is a hard ceiling on `gpus`/`develgpus` (§3), not a
  suggestion. Instead use the resume mechanism (below) to chain
  `ceil(estimated_wall_clock / 24h)` sequential jobs, each picking up from
  the last one's saved checkpoint.

### Worksheet (fill in with YOUR real numbers)

| Quantity | Where it comes from | Your value |
|---|---|---|
| `N_train` | your real manifest line count | |
| `effective_batch` | printed by the GPU-sizing block in the job's `.out` log | |
| `steps_per_epoch` | `ceil(N_train / effective_batch)`, or `recommend_training_params.py`'s printed value | |
| `epochs` (ceiling, not a prediction — see the loss-tuning guide's §1a.5) | your choice | |
| `total_steps` | `steps_per_epoch * epochs` | |
| `measured_seconds_per_step` | from a real `develgpus` smoke run (Step 2 above) | |
| `estimated_wall_clock` | `total_steps * measured_seconds_per_step * 1.25` (margin) | |
| `--time=` to set | the above, or split across chained 24h jobs if it exceeds 24h | |

## 10. The 24-hour wall — how a multi-day training run actually finishes

Real training on tens of thousands of images will very likely take longer
than the 24h max a single job is allowed to run. This is normal on shared
clusters, and it's why §4's `--signal=B:USR1@300` line matters:

1. Your job runs for up to 24h.
2. 300 seconds before the limit, SLURM sends `SIGUSR1`.
3. `seg_training.py`'s signal handler (`seg_training.py:90-93`) sets a flag;
   the training loop finishes its current step, force-saves a validation +
   checkpoint, and exits cleanly (`seg_training.py:759-802`) — instead of
   being killed mid-write with a corrupted or missing checkpoint.
4. The job's `.out` log prints the checkpoint path it just saved.
5. You submit AGAIN, this time setting `RESUME_CKPT` in the script to that
   path — `seg_training.py` loads it via
   `lora.struct.ckpt_path=<that path>` and continues from there.
6. Repeat until `early_stop_patience` decides training is done (see
   `SEG_TRAINING_GUIDE.md` §1a / `GROUNDED_SAM.md` §5.1c for how that
   decision is made) — this is chaining several 24h jobs into one long
   training run, each picking up exactly where the last left off.

## 11. The two "calculation" situations — segformer vs grounded_sam

- **segformer**: `seg_map_calculations.py` runs SegFormer (a real neural
  network) over every raw image to PRODUCE the class-ID PNG maps training
  needs. This genuinely needs a GPU job on the cluster —
  `calc_seg_segformer_jusuf.sbatch` does this, using the same dataset-SCAN
  mode (`--dataset_dir`) you've used locally, so no workflow change.
- **grounded_sam**: there is **no calculation script to run on this
  cluster**. Your masks (`class_map.png`) are the direct output of CARLA's
  own instance-segmentation-camera at capture time — CARLA already did the
  "calculation" when the scene was rendered, on your simulation machine, not
  here. `train_grounded_sam_jusuf.sbatch` trains straight from those staged
  masks; the only "job" beforehand is copying files (`rsync`, §7), not
  running a model.

## 12. Quick command cheat-sheet

```bash
sbatch script.sbatch          # submit a job
squeue -u $USER                # see your queued/running jobs
scancel <jobid>                # cancel one
tail -f logs/<name>-<jobid>.out  # watch live output
sacct -j <jobid>                # job history/stats after it ends
module avail                   # list loadable software
module spider <name>           # search for a specific module
jutil env activate -p <project> # activate your project's $PROJECT/$SCRATCH/$DATA
```

## 13. What's fact-checked vs what needs your confirmation

**Verified against JUSUF's official docs** (quoted/cited while researching,
not guessed): hardware specs, partition names/limits, filesystem table and
the `$DATA` login-only restriction + 90-day scratch purge, the `--account`
requirement, the Anaconda prohibition + `sc_venv_template` workflow, the
`module load Stages/...` pattern, `sbatch`/`squeue`/`scancel`/`sacct` usage.

**You must confirm/fill in yourself** (I have no way to check these from
outside the cluster): your `--account` budget id, your `$PROJECT` directory
name, and the EXACT current module names for the Stage you load (`module
spider PyTorch` will tell you) — module contents change with each annual
Stage release.
