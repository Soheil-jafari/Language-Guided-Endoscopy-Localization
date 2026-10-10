# Running the whole project with one command (HPC / batch-scheduler guide)

This guide is for running the complete experiment on a shared GPU cluster **without anyone
touching the code**. One command downloads the data and pretrained weights, extracts the video
frames, prepares the labels and train/val/test splits, trains the model, evaluates it on the
held-out videos and writes a short human-readable report.

```bash
bash run.sh --preset full --root /path/to/big/scratch/lgel
```

If a job is killed (wall-clock limit, node failure), submit the **same command again**: finished
steps are skipped and training continues from its last finished epoch.

---

## 0. Quick start on the project cluster (Slurm, A100, `/mnt/scratch/users/daiz1/soheil`)

Everything below is already set up for this cluster in `hpc/`:

* `hpc/lgel.sbatch` is the Slurm job: partition `gpu`, 1 node, 1 GPU, 16 CPUs, 64 GB RAM, and the
  `libs/nvidia-cuda/12.4.0/bin` and `apps/miniconda3` modules.
* `hpc/submit.sh` is the only command to run. It submits that job and lets only one job run at a time
  (always submit through it, never `sbatch` directly: that guarantee relies on it).
* The conda environment is created automatically by the first job, at
  `/mnt/scratch/users/daiz1/soheil/conda/envs/lgel`, not in the home directory.

**Once, on the login node:**

```bash
cd /mnt/scratch/users/daiz1/soheil
git clone https://github.com/Soheil-jafari/Language-Guided-Endoscopy-Localization.git code
( umask 077 && cat > links.env )   # paste the two lines from Soheil's email, press Enter, then Ctrl-D
```

The two lines look like `LGEL_DATA_URL='https://...'` and `LGEL_WEIGHTS_URL='https://...'`. The
file can only be read by you and is never copied into the repository. The links are never printed
in logs or reports, and never passed on a command line (where other users could see them).

**Then three jobs, strictly one after the other.** Each command returns at once and the job waits
in the queue. Before starting the next step, wait until the job has finished (`squeue -u $USER`
shows nothing), send back its report, and wait for Soheil's OK.

```bash
bash code/hpc/submit.sh smoke            # ~30 min: creates the environment, tests the GPU, the weights link and Mamba
bash code/hpc/submit.sh pilot            # a few hours: downloads and prepares the real data, short training, time estimate
bash code/hpc/submit.sh full --repeat 2  # the real experiment: baseline model, then advanced model (20 epochs each)
```

`full` trains two models one after the other on the same data and splits: first the **baseline**,
then the **advanced** model (the baseline plus the Mamba temporal head, the bi-level consistency
loss, evidential uncertainty and confidence fusion). Together they are likely to take longer than
one 72-hour job; `--repeat 2` queues a second job that simply continues where the first stopped
(the pilot's report says how long one model takes).

* **The report to send back** after each job is the newest `.tar.gz` in `lgel_smoke/outbox/` (smoke)
  or `lgel/outbox/` (pilot, full). Its name is also in `outbox/LATEST.txt` and at the end of the job
  log.
  * It holds results, metrics, settings, data checks and logs, at most about 80 MB unpacked
    (typically 5-20 MB as `.tar.gz`). Checkpoints and data are never included.
  * If something large had to be left out or shortened, its `MANIFEST.txt` says so.
  * A report is written whether the job finishes, fails, or is stopped. If the pipeline could not
    write its own (e.g. the environment could not be set up), a smaller one holds the end of the job
    log. Only a node crash or a hard kill of the whole job can prevent it; then send the job log
    `lgel_smoke/logs/slurm-<jobid>.out` (smoke) or `lgel/logs/slurm-<jobid>.out` (pilot, full).
* **Time limit:** about 5 minutes before a job's time limit, the job stops itself cleanly and writes
  its report. Submitting the same command again continues from the last finished epoch.
  * `bash code/hpc/submit.sh full --repeat 2` queues a follow-up job up front. It starts when the
    first one ends, whatever the outcome: if the first job failed for a reason that is still there,
    the follow-up fails the same way within minutes.
* **The trained models** are `lgel/results/full_baseline_seed42/model_weights.pth` and
  `lgel/results/full_advanced_seed42/model_weights.pth` (each the best checkpoint without optimizer
  state); `lgel/results/full_seed42_COMPARISON.md` puts their test results side by side. The full
  checkpoints stay in `lgel/runs/`.
* **Out of GPU memory:** pass options after the preset, e.g.
  `bash code/hpc/submit.sh full --batch-size 4 --accum 48` (same effective batch). They apply to the
  model still being trained: a baseline that has already finished is left exactly as it is. The
  folder, the preset and the links cannot be changed this way.
* **If `sbatch` refuses the job** (for example it asks for an account or QoS, or says the requested
  node configuration is not available), give the site's value like this:
  `LGEL_SBATCH_OPTS="--account=NAME" bash code/hpc/submit.sh ...` (or e.g. `--cpus-per-task=8`).
* **Space:** a full run needs about 200 GB at its peak and about 200,000 files, well within the
  1 TB. Files in this scratch space must not be deleted automatically while the project runs.

The rest of this guide explains the pipeline in general and applies to any cluster.

---

## 1. What is needed

| Resource | Requirement |
|---|---|
| GPU | **Exactly one** NVIDIA GPU with **>= 24 GB** memory, Volta to Hopper generation: V100-32GB, A100, A40, L40/L40S, H100, RTX 3090/4090, RTX A6000. The code is single-GPU by design, so request one GPU per job (if a job sees several, only the first is used). Ampere or newer train in bf16 automatically, older GPUs in fp16. **Not supported:** Blackwell GPUs (RTX 50xx, B100/B200), because the pinned PyTorch 2.5.1 + CUDA 12.4 has no kernels for them. Every GPU job first runs a short real GPU test and stops with a clear message if the GPU cannot be used. |
| CPU | 8+ cores recommended (frame extraction runs in parallel, one video per process). |
| RAM | 32 GB or more recommended. |
| Disk | About **200 GB free** under `--root`. The peak is roughly 150-170 GB, while the ~70 GB zip and the ~80 GB of extracted videos coexist. The zip is deleted after extraction, and the videos can also be deleted with `--cleanup-raw`. Later jobs only check for the space their remaining steps need. |
| Files | About **200,000 files** under `--root` (one JPEG per second of video, about 180,000). Clusters often limit the number of files (inode quota) as well as bytes; check both, e.g. with `quota -s` or `df -ih`. The free-space check cannot see quotas. |
| Internet | Needed for the **first** run only (dataset, weights, a small text model, the optical-flow network used by the advanced model's loss, and the packages: conda-forge, PyPI, download.pytorch.org, and github.com for the ready-made Mamba build). If compute nodes have no internet see section 5. |
| Software | `conda` (Miniforge/Anaconda), Python 3.12 (installed by `setup_env.sh`), an NVIDIA driver that supports CUDA 12.4 (driver 525.60 or newer; `nvidia-smi` shows the driver version), Linux with glibc >= 2.28 (RHEL/Rocky/Alma 8+, Ubuntu 20.04+; `ldd --version` shows it). |

## 2. Install (once)

```bash
git clone https://github.com/Soheil-jafari/Language-Guided-Endoscopy-Localization.git lgel && cd lgel
cp site_env.sh.example site_env.sh   # optional: add the cluster's `module load` lines and cache locations
bash setup_env.sh                    # creates the conda env "lgel" (roughly 10 minutes, needs internet)
```

`setup_env.sh` also installs `mamba-ssm` 2.2.4 for the advanced model: a ready-made build matching
PyTorch 2.5 / CUDA 12 / Python 3.12 is downloaded from its GitHub releases, so no CUDA compiler is
needed. Only if that download fails is it compiled from source, which needs `nvcc` (e.g. the cluster's
CUDA module) and 20-60 minutes. This step lives in `install_mamba.sh`; if it fails, the environment still
works for the baseline model, and `run.sh` retries the download at the start of every later job.

* The environment is built from the conda-forge channel only, so no Anaconda Terms-of-Service
  prompt can stall it.
* Conda environments and pip/conda caches go to the user's default locations (often the home
  directory). If home is small or purged, set `CONDA_ENVS_PATH`, `CONDA_PKGS_DIRS` and
  `PIP_CACHE_DIR` in `site_env.sh` to a persistent location that compute nodes can read.
* `setup_env.sh` marks the environment as complete only at the very end. If it is interrupted,
  just run it again; `run.sh` refuses to use a half-installed environment.

## 3. Run

The two download links are **not stored in the repository**. The student supplies them privately
as environment variables. They must be **direct** links that download the file itself without a
login: a share *page* (for example a Baidu Pan page) does not work. A Google Drive link to a file
shared as "Anyone with the link" also works (it is fetched with `gdown`). If the files are already on
the cluster, use `--data-zip` (the zip) or `--data-dir` (the unzipped folder), and
`--weights-path` (the checkpoint) instead.

```bash
export LGEL_DATA_URL='<direct link to the Cholec80 zip>'
export LGEL_WEIGHTS_URL='<direct link to the pretrained M2CRL checkpoint>'
```

Three steps, each a separate job:

**Step A: 5-minute plumbing check on fake data.**

```bash
bash run.sh --preset smoke --root $SCRATCH/lgel_smoke --synthetic 6 --random-init
```

This generates 6 small fake videos and proves the environment, GPU, frame extraction, training,
prediction and evaluation all run on this cluster. The numbers it produces are meaningless; only
"ALL DONE" matters. Run it inside a GPU job.

**Step B: pilot run (all videos, 3 short epochs).**

```bash
bash run.sh --preset pilot --root $SCRATCH/lgel
```

This prepares the real data once, checks that the model learns, and measures the speed.
`results/pilot_seed42/SUMMARY.md` then contains a **time estimate for the full run**. Use it to
choose the job time limit: each job must fit at least one whole epoch including validation,
because training resumes from the last finished epoch.

**Step C: the real experiment** (same `--root`, so the prepared data is reused):

```bash
bash run.sh --preset full --root $SCRATCH/lgel
```

| Preset | What it does | Use it for |
|---|---|---|
| `smoke` | 6 videos, 1 epoch, small batch | checking that everything is wired up |
| `pilot` | all videos, 3 epochs on 10 % of the training windows | a first "is it learning?" result and a time estimate |
| `full` | all videos, 20 epochs, all training windows, effective batch size 192 - the baseline model, then the advanced model | the final numbers |

`--variant baseline|advanced|both` overrides which model(s) a preset trains (`full`: both; `smoke`,
`pilot`: baseline). The smoke test does not train the advanced model, but it runs one training step
of it on the GPU (`check_advanced`), so a problem with Mamba or the other add-ons shows up in the
first 30 minutes, not after the baseline's 20 epochs.

Note: `full` uses **all** training windows (`--subset-ratio 1.0`), whereas the repository's
default configuration samples 20 %. Pass `--subset-ratio 0.2` to reproduce that.

`--root` is required for `pilot` and `full`. Point it at a large scratch filesystem: home
directories are usually too small. Check that the scratch purge policy will not delete files
during the project.

Frame extraction is CPU work (a few hours). To avoid holding a GPU during it, the data can be
prepared in a CPU-only job first:

```bash
bash run.sh --preset pilot --root $SCRATCH/lgel --to-stage audit    # CPU job, no GPU needed
```

### Several runs in one `--root`

`pilot`, `full` and extra seeds (`--seed N`) share one `--root`: the data is prepared once, and
each run has its own `runs/<run>/` and `results/<run>/` (run name `<preset>_seed<seed>`, or
`full_baseline_seed<seed>` / `full_advanced_seed<seed>` for the two models of `full`, unless
`--run-name` is given).

* If two jobs start on a fresh `--root` at the same time, the second waits until the first has
  prepared the data. It is still better to prepare the data once (step B) before starting
  parallel runs.
* The same run submitted twice at the same time is refused; the second job stops immediately.
* This protection uses the file system's locks, which the system releases automatically when a
  job ends or is killed. **The safe default is still one job at a time per `--root`.** Locks only
  protect jobs on different nodes if the file system shares them between nodes (e.g. Lustre mounted
  with `flock`, not `localflock`; ask the admins if you plan to run jobs in parallel).
* If the file system has no lock support at all, a job stops at once with a message. Then make sure
  only one job at a time uses that `--root`, and add `--single-job` (or `export LGEL_SINGLE_JOB=1`).
* Every run records which data it was trained on. If the shared data is later rebuilt differently
  (other split seed, other frame rate, `--force` of a data stage), that run's checkpoints and
  results are **refused** rather than reused, because they no longer match the data. That
  prevents, for example, evaluating a model on test videos it was trained on. Start a new run
  name instead.

## 4. Time limits and resuming

Schedulers kill jobs at their wall-clock limit. Every step is resumable:

* Submit the **same command again** (same `--root`, same flags). Finished steps are skipped;
  training resumes from `runs/<run>/latest_model.pth`, the last finished epoch.
* A step that is killed half-way is redone from its own start. Downloads continue where they
  stopped (Google-Drive links too), and a transfer that stalls is retried automatically.
* A job killed while saving is handled: the checkpoint is written atomically, and the metrics log
  is repaired to match it on the next start.
* Request the **same GPU type and one GPU** for every resubmission of a run.
* Changing settings after training has produced a checkpoint is refused on purpose (it would mix
  two experiments). Use a new `--run-name` for a different experiment.

## 5. If compute nodes have no internet

Do the downloads once on a node with internet, then run offline. The download step also checks
and unpacks the ~70 GB archive and builds the model once on the CPU (a few GB of RAM), so use a
node where that is allowed: a login node only if the site permits it, otherwise a transfer or CPU
node.

```bash
# on a node with internet (no GPU needed); repeat for every --root you will use:
bash run.sh --preset full  --root $SCRATCH/lgel       --to-stage fetch_data
bash run.sh --preset smoke --root $SCRATCH/lgel_smoke --synthetic 6 --random-init --to-stage fetch_data

# inside the GPU job (no links needed, nothing is downloaded or installed):
bash run.sh --preset full --root $SCRATCH/lgel --offline
```

With `--offline` nothing touches the network. If something is missing, for example a download
that did not finish, the job stops at once and says what to run on the internet node.

## 6. Output

Everything lives under `--root`:

```
<root>/
  raw/ ................ downloaded zip / extracted videos (zip deleted after extraction)
  data/ ............... extracted frames, labels, splits.json, triplets, audit report
  runs/<run>/ ......... checkpoints (latest_model.pth, best_model.pth), training_metrics.jsonl
  results/<run>/ ...... SUMMARY.md   <- start here
                        summary.json, training_curves.png,
                        proposed_{val,test}.csv (predictions), *_reference.csv (ground truth),
                        proposed_calibration.json, proposed_test_metrics.json
  logs/ ............... one log file per step (train.log, extract_frames.log, gpu_check.log, ...)
  state/ .............. "step finished" markers and job locks
  outbox/ ............. one small report (.tar.gz) per job, to send back; newest name in LATEST.txt
```

`SUMMARY.md` contains:

* the data counts;
* the per-epoch training table, including minutes per epoch;
* validation metrics, where the decision threshold is chosen;
* test metrics, with the threshold fixed from validation;
* a per-query breakdown.

Test videos are never used for training, model selection or threshold selection.

The one command produces the frame-level results of the baseline and the advanced model (`full`),
and `results/<run>_COMPARISON.md` with both side by side. The temporal-segment evaluation and the
external comparison models described in `README.md` (CLIP, X-CLIP, Moment-DETR) are separate tools
and are not run automatically.

## 7. Example Slurm job for other clusters (TEMPLATE)

For the project cluster use `hpc/submit.sh` (section 0). On another Slurm cluster, adapt `hpc/lgel.sbatch`
or this template; the partition, account, module and time names below are placeholders.

```bash
#!/bin/bash
#SBATCH --job-name=lgel
#SBATCH --partition=<gpu-partition>
#SBATCH --account=<project-account>
#SBATCH --gres=gpu:1                 # exactly one GPU
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00              # must fit one whole epoch (see the pilot SUMMARY.md)
#SBATCH --output=lgel_%j.out

export LGEL_DATA_URL='<direct link>'       # not needed if the data is already downloaded
export LGEL_WEIGHTS_URL='<direct link>'
cd /path/to/lgel
bash run.sh --preset full --root /path/to/scratch/lgel
```

Resubmitting is safe. On clusters with a per-job time limit the job can also be chained
(`sbatch --dependency=afterany:<jobid> ...`).

## 8. Options worth knowing

| Flag | Meaning |
|---|---|
| `--root DIR` | Where everything is stored. Put it on scratch / a large filesystem. |
| `--run-name NAME` | Name of this training run (default `<preset>_seed<seed>`; for `full` `full_baseline_seed<seed>` and `full_advanced_seed<seed>`, or `NAME_baseline` / `NAME_advanced`). |
| `--variant baseline\|advanced\|both` | Which model(s) to train (default: both for `full`, baseline otherwise). |
| `--seed N` | Training seed (default 42). Different seeds can share one `--root`. |
| `--synthetic N` | Use N generated fake videos instead of real data (plumbing test only). |
| `--epochs N`, `--subset-ratio X` | Override the preset. |
| `--batch-size N`, `--accum N` | Micro-batch and gradient accumulation. When lowering the batch size, raise `--accum` so the product stays the preset's value: 192 for `full` (8 x 24), 32 for `pilot` (8 x 4), 4 for `smoke` (2 x 2). |
| `--pred-batch-size N` | Batch size of the final prediction step (default 8); lower it if prediction runs out of GPU memory. |
| `--amp-dtype auto/fp16/bf16` | Mixed precision; `auto` picks bf16 on GPUs with native support (Ampere or newer), fp16 otherwise. |
| `--splits-json FILE` | Use a given `{"train":[...],"val":[...],"test":[...]}` split instead of a new random one (seed 42, 64/8/8 videos). |
| `--cleanup-raw` | Delete the raw videos after frame extraction to save ~80 GB (frames cannot then be re-extracted without downloading again). |
| `--keep-zip` | Keep the downloaded zip. |
| `--offline` | Never touch the network, never install software. |
| `--single-job` | Only for file systems without file locks: confirms that one job at a time uses the `--root`. |
| `--from-stage S`, `--to-stage S`, `--stages a,b` | Run only part of the pipeline. `python main.py --list-stages` prints the order. |
| `--force S` | Redo stage `S` and everything after it. Refused while another job is using the `--root`. |
| `--dry-run` | Print what would run; nothing is created, changed or deleted. |

## 9. Troubleshooting

| Symptom | Fix |
|---|---|
| `CUDA out of memory` or `killed` during training | Lower `--batch-size` (8 -> 4 -> 2) and raise `--accum` so that the product stays the preset's value (section 8), then resubmit. This is accepted as long as no checkpoint has been written yet. |
| Out of memory during `predict` | Add `--pred-batch-size 2` and resubmit. |
| `the GPU cannot run PyTorch kernels` | The GPU is too new for PyTorch 2.5.1 (e.g. RTX 50xx), or the NVIDIA driver is too old. Request another GPU type; details in `logs/gpu_check.log`. |
| `no CUDA GPU is visible` | The job did not get a GPU; check the `--gres` / partition line. |
| `--root is required for the pilot/full presets` | Add `--root /path/on/scratch/lgel` (or set `LGEL_ROOT`). |
| `only N GB free under <root>` | Point `--root` at a larger filesystem, or lower `--min-free-gb` if you are sure. |
| `Disk quota exceeded` / `No space left on device` with free GB shown | The file-count (inode) or user quota is full; see section 1. |
| `Run ... already exists with different settings` | You changed flags after training started. Restore the flags, or use a new `--run-name`. |
| `the prepared data changed since run ... was started` | The shared data was rebuilt differently after that run started; its results would be invalid. Use a new `--run-name`. |
| `stage ... was already finished with different settings` | Same idea for data stages: use a new `--root`, or `--force <stage>`. |
| `run ... is already being processed by another job` | The same run is already running. Wait for that job. |
| `waiting: another job is preparing the shared data` | Normal when two jobs start together; this one continues when the other has prepared the data. |
| `conda env 'lgel' is incomplete or out of date` | Run `bash setup_env.sh` on a node with internet, then resubmit. |
| `--offline: ... has not been (completely) downloaded` | Run the section 5 download command on a node with internet, then resubmit. |
| `the link for ... returned a web page, not the file` | The link is a share/login page or has expired. Supply a direct download link and resubmit. |
| `The downloaded checkpoint does not load` | The weights link points to the wrong file. The bad file is set aside; fix the link and resubmit. |
| `the downloaded dataset archive is corrupt or is not a zip` | The bad file was deleted; resubmit to download it again. If it repeats, the data link does not point to the zip itself. |
| `the dataset archive uses a compression method Python cannot unpack` | The zip was made with e.g. Windows' Deflate64. Re-create it with standard zip compression, or unzip it and use `--data-dir`. |
| `N frame file(s) are missing ... extracting again` | Frames were deleted (e.g. by a scratch purge); they are re-extracted automatically if the raw videos still exist. |
| `does not support file locks ... add --single-job` | The scratch file system cannot lock files. Run one job at a time on this `--root` and add `--single-job`. |
| `the prepared data is incomplete` | Training was started before the data stages finished (e.g. with `--stages train`). Run the full command. |
| Download stops or the zip is incomplete | Resubmit: downloads continue and the size is verified. Google-Drive links need `gdown` (installed by `setup_env.sh`) and may hit Google's daily quota. |
| Everything is slow at the start | Frame extraction of ~80 videos is CPU-bound; it scales with `--workers` (default: up to 16). |
| Training is much slower than the pilot estimate | Training reads many small JPEGs; slow shared file systems can dominate. Compare `train min` in the pilot `SUMMARY.md` with the GPU's speed, and ask whether node-local scratch is available. |

The per-step logs in `<root>/logs/` contain the full output of every underlying script.

## 10. What has and has not been verified

**Verified** with 176 automated tests plus end-to-end runs on synthetic Cholec80-format data with a
deliberately **tiny** model on CPU:

* **The complete chain:** download/extract, frame extraction, annotation repair, splits, audit, training, prediction, evaluation and summary.
* **Interruptions:**
  * a real SIGKILL during training, followed by a resume;
  * a job killed between saving a checkpoint and logging its metrics;
  * a half-written log line;
  * a partial weights download, which is continued over HTTP;
  * downloads that stall or break repeatedly: each new attempt continues from the bytes already on
    disk (real curl against a test server), and only a server that cannot continue a file is
    downloaded from the start again (into a separate file, which replaces the partial one only
    when it is complete and not a web page);
  * a link that expires half-way (an error status, an error page, a short error message, a refused
    size request): the partial download is kept and the job says what is wrong with the link;
  * Google-Drive downloads with gdown 5 and 6 (`setup_env.sh` installs 6), against a local server:
    an interrupted transfer is continued; gdown 4 is refused;
  * an "internet first, offline later" run;
  * `--offline` refusing to download;
  * download links that return a web page or a wrong file, then corrected;
  * corrupt archives, and archives Python cannot unpack;
  * SIGTERM from the scheduler during frame extraction.
* **Several jobs on one `--root`:**
  * `pilot` followed by `full`;
  * two jobs started at the same moment (the data is prepared once);
  * the same run submitted twice (the duplicate is refused);
  * a run refused after its split was changed;
  * a job killed during data preparation and resubmitted at once;
  * a file system without lock support (the job stops unless `--single-job` is given).
* **Re-extraction:** frames are extracted again after the frame rate changes, or when frame files go missing.
* **Disk check:** an offline job after the download is not refused for the space the download already uses.
* **Exact resume:** killing training after epoch 2 and resuming gives bit-identical weights, optimizer/scheduler state and metrics to an uninterrupted run.
* **Pretrained checkpoint format:** checkpoints in the official M2CRL training format (`teacher`/`student`), and plain state dicts, load with strict checks of every encoder tensor.
* **Project-cluster scripts:** `hpc/submit.sh` and `hpc/lgel.sbatch` were run with stand-ins for Slurm
  and the module system:
  * a smoke job using the weights link, and a pilot downloading through `links.env`;
  * the exported weights used by `predict.py`;
  * the early time-limit warning stopping training cleanly (the report is written and `train.py` is
    stopped first), followed by a resubmission that resumes;
  * a hard stop the way Slurm does it at the limit, and a job whose conda module cannot be loaded
    (both still leave a report);
  * a dropped terminal (for someone running `run.sh` by hand): the step is stopped and the report
    written, even though the terminal can no longer be written to;
  * no private link in any log, result or report, and none on a command line (downloads read the
    link from a private temporary file); parts of a link that identify the file on their own (such
    as a Google Drive file id in an error message) are hidden too;
  * `submit.sh` refusing bad options, links on the command line, empty links, a second
    submission, and an unreadable queue;
  * the `full` preset training the baseline, then the advanced model (Mamba head, bi-level loss,
    uncertainty, confidence fusion) on the same data, a stop during the advanced training and a
    resubmission that skips the finished baseline and resumes the advanced model, one report and
    a comparison table with both. (On CPU, with a stand-in for the GPU-only Mamba layer.)
* **Mamba installation:** `setup_env.sh`'s Mamba step, run against PyTorch 2.5.1 + CUDA 12.4,
  downloaded the ready-made `mamba-ssm` 2.2.4 build without any compiler, and the real advanced
  model (4 official Mamba layers, uncertainty output, confidence fusion) was built with it. The
  Python 3.12 build the cluster needs exists on GitHub. If Mamba cannot be installed, the
  environment still works for the baseline and the advanced model is refused with a clear message.
* **Environment scripts:** with a real conda (Miniforge, conda 26.7), `run.sh` activates and checks
  exactly the environment folder the job specifies (never another environment with the same name),
  and `setup_env.sh` creates it there up to the PyTorch download.

**Not verified** (could not be tested where this was written):

* The real-size model, real GPU training, mixed precision and memory use on actual hardware,
  including the Mamba kernels running on a GPU (the smoke test's `check_advanced` step does this).
* Creating the environment from scratch (conda-forge + PyTorch CUDA 12.4 wheels): the build
  machine could not reach those servers.
* The real cluster itself (Slurm, modules, conda, GPU): the smoke job checks it in about 30 minutes.
* The real dataset archive, the real M2CRL checkpoint file, and the final download links.
* Training time and the quality of the final results. Run the `pilot` preset first.

Known data issue handled automatically: in the official Cholec80 release, the phase files of
video15 and video37 contain one extra row one frame past the end of the video. The pipeline
removes such a row (only if it merely repeats the last phase), writes the cleaned copy to
`data/annotations_clean/` and records what it did in `annotations_sanitized.json`. The original
files are not modified.

The dataset is distributed by its owners under their own terms. Check that you may use and move
it on the cluster before uploading it anywhere. `backbone/vision_transformer.py` is adapted from
TimeSformer (CC BY-NC 4.0) and keeps that licence; see the README's "Third-party code and data".
