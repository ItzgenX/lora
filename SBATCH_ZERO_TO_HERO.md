# SBATCH / SLURM on JUSUF — Zero to Hero (Grounded-SAM pipeline)

Standalone version of the guide, scoped entirely to the **grounded_sam
branch** — read this on its own, no need to cross-reference the segformer
version. Companion script: `train_grounded_sam_jusuf.sbatch`. Every JUSUF
fact below is sourced from the official docs at
`apps.fz-juelich.de/jsc/hps/jusuf/` — nothing is guessed.

---

## 0. What "SBATCH" actually IS, mechanically — read this even if §1-12 already made sense

§1 below tells you WHAT a cluster is and WHAT to do. This section explains
the one thing that trips up almost everyone the first time: **`#SBATCH`
lines look like comments, and to `bash` they ARE comments — so how do they
configure anything at all?**

**The short answer: two different programs read the same file, for two
different purposes, and only one of them understands `#SBATCH`.**

Take the smallest possible example:
```bash
#!/bin/bash -x
#SBATCH --time=00:10:00
#SBATCH --job-name=my-first-job

echo "hello from the compute node"
```
- To **`bash`** (the shell that eventually RUNS this file), a line starting
  with `#` is a comment, full stop — `bash` skips both `#SBATCH` lines
  exactly like it would skip a line that just says `# some notes to self`.
  If you ran `bash script.sh` directly, this file would just print
  `hello from the compute node` and the `#SBATCH` lines would have zero effect.
- But you never run it with plain `bash`. You run it with **`sbatch
  script.sh`** — a completely different program, SLURM's own submission
  tool. `sbatch` opens the file itself, BEFORE any of it executes, and
  specifically scans for lines matching the pattern `#SBATCH <flag>
  <value>` (they can be mixed among ordinary `#`-comments — see the real
  block at the top of [train_grounded_sam_jusuf.sbatch](slurm/train_grounded_sam_jusuf.sbatch),
  which has a whole paragraph of plain comments interleaved right next to
  its `#SBATCH` lines, and `sbatch` correctly ignores the plain ones while
  reading the `#SBATCH` ones). Every `#SBATCH` line it finds becomes one
  entry in a **resource request** — "give me 1 GPU, 32 CPUs, up to 24
  hours" — which `sbatch` sends to the SLURM controller.

**What happens next, in order, is the whole mental model you need:**
1. You type `sbatch train_grounded_sam_jusuf.sbatch` on a login node.
2. `sbatch` reads the file, extracts every `#SBATCH` line, builds the
   resource request, and hands it to SLURM's scheduler daemon.
3. `sbatch` **returns immediately** — you get your terminal prompt back in
   under a second, with a line like `Submitted batch job 123456`. It did
   **not** wait for a GPU, and it did **not** run your script. It only
   queued the request.
4. **Sometime later** — seconds, minutes, or hours, depending on how busy
   the cluster is — SLURM finds a node matching your request and actually
   executes your script there, **top to bottom, as plain bash**, starting
   from `#!/bin/bash -x`. At this point, every `#SBATCH` line is now just
   an inert comment again (its job — configuring the request — already
   happened back in step 2) — the script's REAL bash commands (module
   loads, the `srun` line, everything in §5-§7 below) are what actually run.
5. Output goes to the `--output=`/`--error=` files (§5) because, by the
   time step 4 happens, you are not watching — you're not even necessarily
   logged in anymore. That's the entire reason `sbatch` exists instead of
   just running `python train.py` directly: it lets you queue work that
   runs unattended, later, on a machine you're not currently using.

This is why §1's step 3 ("You submit it... It queues") and step 4 ("SLURM...
runs your script there, unattended") are two SEPARATE moments in time, often
far apart — and why editing a `#SBATCH` line has NO effect on a job that's
already running (its resource request was locked in back at submission time).

*(This mechanism — `#SBATCH` parsing, queue, delayed execution — is general
SLURM behaviour, true on any SLURM cluster, not something specific to
JUSUF. §1 below still starts with a few more general cluster concepts
before the facts turn JUSUF-specific — hardware numbers, partition names,
filesystem paths — which is exactly what §12 lists as verified against
JUSUF's own docs.)*

## 1. What a cluster actually is, and why you can't just run `python train.py`

A cluster is hundreds of shared computers ("nodes"). You never log into a
GPU machine directly:

1. You log into a **login node** (small, shared, no GPU) via SSH.
2. You write a **job script** — a shell script with `#SBATCH` lines
   describing what you need.
3. You **submit** it: `sbatch train_grounded_sam_jusuf.sbatch`. It queues.
4. **SLURM** (the scheduler) finds you a free GPU node and runs your script
   there, unattended.
5. Output goes to log files — you're not connected to the node while it runs.

The script must be fully self-contained: it loads its own software, finds
its own data, saves its own results — nothing from your login session
carries over to the compute node.

## 2. JUSUF's hardware, in plain numbers

**45 GPU nodes**, each: **1× NVIDIA V100 GPU (16GB)**, 2× AMD EPYC 7742 (128
CPU cores total), 256GB RAM, 1TB local disk. One GPU per node — "give me a
GPU node" and "give me a V100" mean the same thing here. OS: Rocky Linux 9.

## 3. Partitions — which queue to submit to

| Partition | Nodes | Max walltime | Internet | Use for |
|---|---|---|---|---|
| `gpus` | 1–39 | 24h (6h on restart) | **No** | Your real training runs |
| `develgpus` | 1–6 | 24h (same) | **Yes** | First-time testing (e.g. downloading the SD1.5 base model needs internet) |

Same hardware either way. Use `develgpus` once to prove the script works
end-to-end (especially if the base model isn't cached on `$PROJECT` yet and
needs a download), then `gpus` for the real run.

## 4. Why there is only ONE job type for this pipeline — no separate "calc" step

This is the point worth being explicit about, since the segformer pipeline
(a different branch of this same repo) DOES need a GPU calculation job
before training, and it's easy to expect grounded_sam needs the mirror image
of that. It doesn't, and here's exactly why:

- SegFormer branch: `seg_map_calculations.py` runs a real neural network
  (SegFormer) over your raw photos to PRODUCE class-ID maps. No such maps
  exist until that job runs — hence a dedicated GPU calc job.
- Grounded-SAM branch: your masks (`class_map.png` — 16-bit PNG, 1280x800,
  CARLA class ids) are the **direct output of CARLA's own
  instance-segmentation-camera** at the moment you captured the scene in
  the simulator. That computation already happened, on your simulation
  machine, before this repo was ever involved. `GroundedSamEncoder` — the
  class this branch uses in place of a live model — deliberately has no
  working `forward()` (`src/encoders/grounded_sam_encoder.py`, "Tier 2" was
  intentionally never built: no GroundingDINO/SAM code exists anywhere in
  this repo). There is nothing to run on this cluster to "calculate"
  anything for this pipeline — training reads your already-captured masks
  directly.

So the whole workflow is: **stage your files (§7) → train (§8-9)**. No
compute step in between. This holds regardless of which `resize_mode` you
train with (§5) — squaring the map to match the image happens live, inside
the training dataloader, every time a sample is read, not as a separate
pre-computation — so switching `letterbox` <-> `CenterCrop` between two
submitted jobs never needs a recalculation step either.

**Optional, and NOT a GPU job:** two lightweight CPU-only sanity scripts
exist to sanity-check your masks before you spend GPU time —
`check_seg_map_format.py` (confirms format/mode/id-range of one map) and
`scan_seg_map_classes.py` (confirms the id range across a whole manifest).
Run these directly on a **login node**, no `sbatch` needed — they're fast,
CPU-only, and a login node can read `$SCRATCH` just as well as a compute
node can:
```bash
python check_seg_map_format.py --seg_map $SCRATCH/<project>/grounded_sam/000000/class_map.png
python scan_seg_map_classes.py --json_file $SCRATCH/<project>/grounded_sam/train.jsonl
```

## 5. Anatomy of `train_grounded_sam_jusuf.sbatch` — every line explained

```bash
#!/bin/bash -x
```
"This is a bash script"; `-x` logs every command run — useful since you
can't watch the job live.

```bash
#SBATCH --job-name=loradapter-gs
```
A name you choose, shown in the queue.

```bash
#SBATCH --account=<YOUR_BUDGET_ACCOUNT>
```
**Required** — which project's compute-hour budget to charge. Get this from
your PI/JSC project page; I cannot know it.

```bash
#SBATCH --partition=gpus
```
Which queue (§3).

```bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
```
One machine, one copy of your program — this training is single-GPU, not
distributed across nodes.

```bash
#SBATCH --gres=gpu:1
```
Explicitly requests 1 GPU (technically optional here since every node has
exactly one, kept for clarity).

```bash
#SBATCH --cpus-per-task=32
```
CPU cores for your task — mainly used by PyTorch's DataLoader workers
(decoding your 1280x800 masks/images in parallel with GPU compute). Node has
128 cores total.

```bash
#SBATCH --time=24:00:00
```
Wall-clock limit, `HH:MM:SS`. Your job is killed the instant this elapses.
24h is the max on `gpus`. See §9 for what happens when a real run needs
longer.

```bash
#SBATCH --signal=B:USR1@300
```
300 seconds before the 24h limit, SLURM sends your script a `SIGUSR1`
warning (`B:` = to the batch script itself). `grounded_sam_training.py` — the SAME
training script both branches use — already listens for this
(`grounded_sam_training.py:90-93`, confirmed byte-identical on this branch) and
responds by saving a checkpoint and exiting cleanly instead of being killed
mid-write. Without this line, SLURM just kills the job with no warning at
the 24h mark.

```bash
#SBATCH --output=logs/train-%j.out
#SBATCH --error=logs/train-%j.err
```
Log file locations; `%j` = SLURM's job ID, so every submission gets its own
file, never overwritten.

```bash
set -euo pipefail
```
Bash safety: stop on first error, error on unset variables, propagate
pipeline failures. Not SLURM-specific, just good practice.

Everything after the `#SBATCH` block (module loads, venv activation, `cd`,
env vars, the final `srun` line) is ordinary bash — SLURM's job ends once
your script starts running; the rest executes top to bottom like it would
on your own machine.

```bash
srun python grounded_sam_training.py experiment=train_grounded_sam ...
```
`srun` launches your program inside the resources SLURM allocated to this
job — what actually puts your Python process on the GPU node. For a
single-GPU job this behaves like plain `python foo.py`, but `srun` also
gets your job's resource usage tracked (visible later via `sacct`).
`experiment=train_grounded_sam` is the ONE line that differs from the
segformer training script — it selects `configs/experiment/train_grounded_sam.yaml`
(29-class CARLA palette, `GroundedSamEncoder` slot-filler) instead of
`train_seg.yaml`.

**`resize_mode` — pick which squaring technique this job trains with**
(user decision 2026-07-20, GROUNDED_SAM.md §5.0b): add `resize_mode=letterbox`
or `resize_mode=CenterCrop` to the `srun` line. Say nothing and you get
`letterbox` (this project's current default — `SquarePad`, keeps 100% of the
scene, adds a flat pad band). `CenterCrop` is the ORIGINAL stock LoRAdapter
recipe (`Resize`+`CenterCrop`) — no pad band, but crops ~37% of frame width
off the left/right. The output folder always names which one ran
(`outputs/train/grounded_sam_letterbox/...` vs `..._CenterCrop/...`), so
submitting two jobs — one per mode — never collides and both are directly
comparable by generated-image quality afterward:
```bash
srun python grounded_sam_training.py experiment=train_grounded_sam resize_mode=letterbox ...
srun python grounded_sam_training.py experiment=train_grounded_sam resize_mode=CenterCrop ...
```

### 5a. The block right before that `srun` line — sizing to THIS GPU

`grounded_sam_training.py` never auto-scales its batch size — it just reads a static
`data.batch_size=4` from `configs/experiment/train_grounded_sam.yaml`
(confirmed: same "4 batch x 4 accum = 16 effective batch" baseline as the
segformer branch's config, hand-tuned for a ~12GB reference GPU). Left as
-is on a 16GB V100, that leaves real VRAM unused every step. The script runs
this BEFORE the `srun` line, on the job's real allocated GPU (no `srun`
needed for this part — the whole batch script already executes ON the
compute node, not the login node):
```bash
read -r REC_BATCH REC_ACCUM <<< "$(python -c "
import torch
from recommend_training_params import recommend_batch_and_accum
total_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
b, a, _, _ = recommend_batch_and_accum(total_gb)
print(b, a)
")"
```
This reuses `recommend_training_params.py`'s OWN validated formula — nothing
new invented. `REC_BATCH`/`REC_ACCUM` are then passed as
`data.batch_size=${REC_BATCH} gradient_accumulation_steps=${REC_ACCUM}` on
the `srun` line. The formula holds `batch_size * accum` (the "effective
batch") fixed at 16 — the value `learning_rate=1e-4` was validated at — so
only how that 16 is SPLIT changes with the GPU, never the learning rate's
validity. Verified by execution: on a real 12GB GPU this returns `(4, 4)`
unchanged (confirms it's a no-op on the size it was already tuned for); fed
JUSUF's documented 16GB V100 spec it returns `(5, 3)` (effective batch 15,
≈16 — the same real function, not a hand-guessed number).

## 6. Modules — how software gets loaded

- `module avail` — list what's loadable now.
- `module spider <name>` — search everywhere for something, e.g.
  `module spider PyTorch`.
- `module load <name>` — load it.
- `module purge` — unload everything first (the script starts with this so
  you always get a clean, known state).

Software is versioned by annual **Stage** (`module load Stages/2024`) — all
modules in one Stage are built to work together; mixing across Stages
causes mysterious linker errors, so always load a Stage first.

**I could not verify today's exact module names** — `GCC`/`OpenMPI`/`CUDA`/
`cuDNN`/`PyTorch` in the script are placeholders following the documented
pattern. Run `module spider PyTorch` yourself and correct the line if it
differs; module contents change every Stage release.

## 7. Python environment — why NOT conda, and file placement

**Anaconda is prohibited on JSC clusters** (licensing) — use JSC's
`sc_venv_template` instead of your local `loradapter` conda env. One-time
setup on a login node:
```bash
cd $PROJECT/<your_project>
git clone https://gitlab.jsc.fz-juelich.de/kesselheim1/sc_venv_template
# edit sc_venv_template/modules.sh to match the module load line in §6
# edit sc_venv_template/requirements.txt -- see below
bash sc_venv_template/setup.sh
```
`requirements.txt` (everything from this repo's `environment.yaml` pip
section EXCEPT torch/torchvision — those come from the `PyTorch` module,
never pip-install them separately or they'll conflict with the module's
CUDA build):
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
Every job script activates it with
`source $PROJECT/<project>/sc_venv_template/activate.sh` — the cluster
equivalent of `conda activate loradapter`.

**Filesystems** — the #1 real gotcha:

| Variable | Access | For | Danger |
|---|---|---|---|
| `$HOME` | login + compute | tiny personal files | small quota |
| `$PROJECT` | login + compute | your code (this repo) | — |
| `$SCRATCH` | login + compute | active dataset/checkpoints DURING training | auto-deleted after 90 days untouched |
| `$DATA` | **login ONLY** | long-term dataset storage | **compute nodes cannot read this** |
| `$ARCHIVE` | login only | cold long-term storage | slow |

Your CARLA captures presumably live on `$DATA` long-term. Before submitting
the training job, stage a working copy onto `$SCRATCH` from a **login node**:
```bash
rsync -a $DATA/<project>/grounded_sam/ $SCRATCH/<project>/grounded_sam/
```
Then point the job at the `$SCRATCH` copy (as `train_grounded_sam_jusuf.sbatch`
already does via `DATA_DIR`). Re-run the `rsync` if you return after a long
gap — `$SCRATCH` purges files untouched for 90 days.

Also run `jutil env activate -p <project>` once per login session —
`$PROJECT`/`$SCRATCH`/`$DATA` only resolve correctly after that.

## 8. Submitting and watching the job

```bash
sbatch train_grounded_sam_jusuf.sbatch    # submit -> "Submitted batch job 123456"
squeue -u $USER                            # PENDING or RUNNING?
tail -f logs/train-123456.out              # live-follow output
scancel 123456                             # cancel by mistake? kill it
sacct -j 123456 --format=JobID,Elapsed,State,MaxRSS   # stats after it ends
```

**Recommended first run** — a tiny smoke test on `develgpus` before trusting
the real 24h job (mirrors how this pipeline was smoke-tested locally before
any real run): point `data.json_file`/`val_json_file` at a 2-image manifest
and add `epochs=1 val_steps=2 ckpt_steps=50` to the `srun python` line, then:
```bash
sbatch --partition=develgpus --time=00:15:00 train_grounded_sam_jusuf.sbatch
```

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
§5a, this is `REC_BATCH * REC_ACCUM`, printed by the script itself
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
   sbatch --partition=develgpus --time=00:20:00 train_grounded_sam_jusuf.sbatch
   ```
2. Open the `.out` log. `grounded_sam_training.py`'s progress bar prints a live
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

## 10. The 24-hour wall — chaining jobs for a multi-day run

Real training on a large CARLA dataset will likely exceed 24h. This is
normal and handled by resuming:

1. Job runs up to 24h.
2. 300s before the limit, SLURM sends `SIGUSR1` (§5's `--signal` line).
3. `grounded_sam_training.py`'s handler (`grounded_sam_training.py:90-93`) finishes the
   current step, force-saves a checkpoint, exits cleanly
   (`grounded_sam_training.py:759-802`) — no corrupted/missing checkpoint.
4. The job's `.out` log prints the checkpoint path.
5. Resubmit, setting `RESUME_CKPT` in the script to that path —
   `lora.struct.ckpt_path=<path>` picks up exactly where it left off.
6. Repeat until `early_stop_patience` decides training is done (see
   `GROUNDED_SAM.md` §5.1c for how that decision is made — same shared
   mechanism as the segformer branch).

## 11. Quick command cheat-sheet

```bash
sbatch script.sbatch            # submit a job
squeue -u $USER                 # your queued/running jobs
scancel <jobid>                 # cancel one
tail -f logs/<name>-<jobid>.out # watch live output
sacct -j <jobid>                # job history/stats
module avail                    # list loadable software
module spider <name>            # search for a specific module
jutil env activate -p <project> # activate $PROJECT/$SCRATCH/$DATA
```

## 12. What's fact-checked vs what needs your confirmation

**General SLURM knowledge, not JUSUF-specific** (§0): the `#SBATCH`-parsing/
queue/delayed-execution mechanism is standard SLURM behaviour, true on any
SLURM cluster — not sourced from JUSUF's docs specifically, and not
something that could differ per-cluster.

**Verified against JUSUF's official docs**: hardware specs, partition
names/limits, the filesystem table + `$DATA` login-only restriction + 90-day
`$SCRATCH` purge, the `--account` requirement, the Anaconda prohibition +
`sc_venv_template` workflow, the `module load Stages/...` pattern,
`sbatch`/`squeue`/`scancel`/`sacct` usage.

**Verified against THIS repo's code** (not the cluster docs): the
`SIGUSR1`-triggers-checkpoint-then-exit behaviour (`grounded_sam_training.py:90-93,
759-802`), confirmed identical on the grounded_sam branch by diffing against
the segformer branch's copy of the same file.

**You must confirm/fill in yourself**: your `--account` budget id, your
`$PROJECT` directory name, and the exact current module names for the Stage
you load (`module spider PyTorch` tells you) — these change every Stage.

## 13. File inventory — every file this guide talks about

Same provenance tagging as [GROUNDED_SAM.md §7.0](GROUNDED_SAM.md):
🟦 **ORIGINAL — unchanged** (stock repo, untouched), 🟨 **MODIFIED**
(pre-existed, usually built for the SegFormer pipeline, adapted here), 🟩
**NEW** (written specifically for this work).

| File | Provenance | Where it's covered |
|---|---|---|
| [slurm/train_grounded_sam_jusuf.sbatch](slurm/train_grounded_sam_jusuf.sbatch) | 🟩 NEW | The whole guide — §0 explains the mechanism, §5-5a its exact contents |
| [SBATCH_ZERO_TO_HERO.md](SBATCH_ZERO_TO_HERO.md) | 🟩 NEW | This file |
| [grounded_sam_training.py](grounded_sam_training.py) | 🟨 MODIFIED *(renamed from `seg_training.py` 2026-07-20; the `srun` line launches it)* | §5, §9, §10 |
| [configs/train_seg.yaml](configs/train_seg.yaml) | 🟨 MODIFIED *(shared Hydra base config both pipelines layer onto)* | §5 |
| [recommend_training_params.py](recommend_training_params.py) | 🟨 MODIFIED | §5a's GPU-sizing block, §9's step-count math |
| [src/encoders/grounded_sam_encoder.py](src/encoders/grounded_sam_encoder.py) | 🟩 NEW | §4 (explains why `forward()` deliberately raises, so no calc job exists) |
| [configs/experiment/train_grounded_sam.yaml](configs/experiment/train_grounded_sam.yaml) | 🟩 NEW | §5 (`experiment=train_grounded_sam`), §5a (its hardcoded `batch_size`) |
| [check_seg_map_format.py](check_seg_map_format.py) | 🟩 NEW | §4's optional login-node sanity check |
| [scan_seg_map_classes.py](scan_seg_map_classes.py) | 🟩 NEW | §4's optional login-node sanity check |
| `train.py` | 🟦 ORIGINAL — unchanged | §1, mentioned once for contrast (`python train.py` vs `sbatch ...`) |

**Not present on this branch at all** (mentioned only for contrast, so you
don't go looking for it here): `seg_map_calculations.py` — the SegFormer
branch's GPU calc job (§4). Grounded-SAM has no equivalent; your masks are
already CARLA-computed before this repo is ever involved.

**What this guide does NOT cover** (by design, not an oversight): what
`grounded_sam_training.py` actually DOES once it's running — the diffusion
model, LoRA, the conditioning mechanism. This guide is only about getting
that script running unattended on a cluster; for the mechanism itself, see
[GROUNDED_SAM.md Part A](GROUNDED_SAM.md) (original-repo fundamentals) and
§5.1c (the training-step code, same mechanism whether it runs locally or on
JUSUF — nothing about that code cares which machine it's on).
