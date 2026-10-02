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

## 1. What is needed

| Resource | Requirement |
|---|---|
| GPU | **Exactly one** NVIDIA GPU with **>= 24 GB** memory, Volta to Hopper generation: V100-32GB, A100, A40, L40/L40S, H100, RTX 3090/4090, RTX A6000. The code is single-GPU by design, so request one GPU per job (if a job sees several, only the first is used). Ampere or newer train in bf16 automatically, older GPUs in fp16. **Not supported:** Blackwell GPUs (RTX 50xx, B100/B200), because the pinned PyTorch 2.5.1 + CUDA 12.4 has no kernels for them. Every GPU job first runs a short real GPU test and stops with a clear message if the GPU cannot be used. |
| CPU | 8+ cores recommended (frame extraction runs in parallel, one video per process). |
| RAM | 32 GB or more recommended. |
| Disk | About **200 GB free** under `--root`. The peak is roughly 150-170 GB, while the ~70 GB zip and the ~80 GB of extracted videos coexist. The zip is deleted after extraction, and the videos can also be deleted with `--cleanup-raw`. |
| Files | About **200,000 files** under `--root` (one JPEG per second of video, about 180,000). Clusters often limit the number of files (inode quota) as well as bytes; check both, e.g. with `quota -s` or `df -ih`. The free-space check cannot see quotas. |
| Internet | Needed for the **first** run only (dataset, weights, a small text model). If compute nodes have no internet see section 5. |
| Software | `conda` (Miniforge/Anaconda), Python 3.12 (installed by `setup_env.sh`), an NVIDIA driver that supports CUDA 12.4 (driver 525.60 or newer; `nvidia-smi` shows the driver version), Linux with glibc >= 2.28 (RHEL/Rocky/Alma 8+, Ubuntu 20.04+; `ldd --version` shows it). |

## 2. Install (once)

```bash
git clone https://github.com/Soheil-jafari/Language-Guided-Endoscopy-Localization.git lgel && cd lgel
cp site_env.sh.example site_env.sh   # optional: add the cluster's `module load` lines and cache locations
bash setup_env.sh                    # creates the conda env "lgel" (roughly 10 minutes, needs internet)
```

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
login: a share *page* (for example a Baidu Pan page) does not work. If the files are already on
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
| `full` | all videos, 20 epochs, all training windows, effective batch size 192 | the final numbers |

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
each run has its own `runs/<run>/` and `results/<run>/` (run name `<preset>_seed<seed>` unless
`--run-name` is given).

* If two jobs start on a fresh `--root` at the same time, the second waits until the first has
  prepared the data. It is still better to prepare the data once (step B) before starting
  parallel runs.
* The same run submitted twice at the same time is refused; the second job stops immediately.
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
  stopped, except for Google-Drive links, which start again.
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
```

`SUMMARY.md` contains:

* the data counts;
* the per-epoch training table, including minutes per epoch;
* validation metrics, where the decision threshold is chosen;
* test metrics, with the threshold fixed from validation;
* a per-query breakdown.

Test videos are never used for training, model selection or threshold selection.

The one command produces the frame-level results of the proposed model. The temporal-segment
evaluation and the baseline models described in `README.md` are separate tools and are not run
automatically.

## 7. Example Slurm job (TEMPLATE - adapt it)

The scheduler on the target cluster has not been checked. The partition, account, module and time
names below are placeholders that must be replaced with the cluster's real ones.

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
| `--run-name NAME` | Name of this training run (default `<preset>_seed<seed>`). |
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
| `the downloaded file is not a zip archive` | The data link does not point to the zip itself. Fix the link and resubmit. |
| `the prepared data is incomplete` | Training was started before the data stages finished (e.g. with `--stages train`). Run the full command. |
| Download stops or the zip is incomplete | Resubmit: downloads continue and the size is verified. Google-Drive links need `gdown` (installed by `setup_env.sh`) and may hit Google's daily quota. |
| Everything is slow at the start | Frame extraction of ~80 videos is CPU-bound; it scales with `--workers` (default: up to 16). |
| Training is much slower than the pilot estimate | Training reads many small JPEGs; slow shared file systems can dominate. Compare `train min` in the pilot `SUMMARY.md` with the GPU's speed, and ask whether node-local scratch is available. |

The per-step logs in `<root>/logs/` contain the full output of every underlying script.

## 10. What has and has not been verified

**Verified** with 74 automated tests plus end-to-end runs on synthetic Cholec80-format data with a
deliberately **tiny** model on CPU:

* **The complete chain:** download/extract, frame extraction, annotation repair, splits, audit, training, prediction, evaluation and summary.
* **Interruptions:**
  * a real SIGKILL during training, followed by a resume;
  * a job killed between saving a checkpoint and logging its metrics;
  * a half-written log line;
  * a partial weights download, which is continued over HTTP;
  * an "internet first, offline later" run;
  * `--offline` refusing to download;
  * download links that return a web page or a wrong file, then corrected.
* **Several jobs on one `--root`:**
  * `pilot` followed by `full`;
  * two jobs started at the same moment (the data is prepared once);
  * the same run submitted twice (the duplicate is refused);
  * a run refused after its split was changed.
* **Re-extraction:** frames are extracted again after the frame rate changes.
* **Exact resume:** killing training after epoch 2 and resuming gives bit-identical weights, optimizer/scheduler state and metrics to an uninterrupted run.
* **Pretrained checkpoint format:** checkpoints in the official M2CRL training format (`teacher`/`student`), and plain state dicts, load with strict checks of every encoder tensor.
* **Environment scripts:** `run.sh` activates and checks the conda env in a non-interactive shell (Miniforge, conda 26.7), and `setup_env.sh` runs up to the PyTorch download.

**Not verified** (could not be tested where this was written):

* The real-size model, real GPU training, mixed precision and memory use on actual hardware.
* Creating the environment from scratch (conda-forge + PyTorch CUDA 12.4 wheels): the build
  machine could not reach those servers.
* Slurm or any other scheduler.
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
