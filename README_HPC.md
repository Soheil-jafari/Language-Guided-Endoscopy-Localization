# Running the whole project with one command (HPC / batch-scheduler guide)

This guide is for running the complete experiment on a shared GPU cluster **without anyone
touching the code**. One command downloads the data and pretrained weights, extracts the video
frames, prepares the labels and train/val/test splits, trains the model, evaluates it on the
held-out videos and writes a short human-readable report.

```bash
bash run.sh --preset full --root /path/to/big/scratch/lgel
```

Nothing else has to be run by hand. If the job is killed (wall-clock limit, node failure), submit
the **same command again**: finished steps are skipped and training continues from its last
finished epoch.

---

## 1. What is needed

| Resource | Requirement |
|---|---|
| GPU | 1 CUDA GPU with **>= 24 GB** memory (A100 / L40 / RTX 4090 / 5090 class). The code is single-GPU by design; asking for more GPUs will not make it faster (but several runs, e.g. different seeds, can run side by side). Ampere or newer GPUs train in bf16 automatically; older ones (V100, T4) use fp16. |
| CPU | 8+ cores recommended (frame extraction runs in parallel, one video per process). |
| RAM | 32 GB or more recommended. |
| Disk | About **200 GB free** under `--root`. The peak is roughly 150-170 GB, while the ~70 GB zip and the ~80 GB of extracted videos coexist. The zip is deleted after extraction, and the videos can also be deleted with `--cleanup-raw`. |
| Internet | Needed for the **first** run only (dataset, weights, a small text model). If compute nodes have no internet see section 5. |
| Software | `conda` (Miniforge/Anaconda), Python 3.12 (installed by `setup_env.sh`), an NVIDIA driver that supports CUDA 12.4 wheels (`nvidia-smi` shows the driver version), Linux with glibc >= 2.28 (RHEL/Rocky/Alma 8+, Ubuntu 20.04+; `ldd --version` shows it). |

## 2. Install (once)

```bash
git clone https://github.com/Soheil-jafari/Language-Guided-Endoscopy-Localization.git lgel && cd lgel
# If conda needs a `module load`, copy site_env.sh.example to site_env.sh and edit it (optional).
bash setup_env.sh          # creates the conda env "lgel" (roughly 10 minutes, needs internet)
```

The environment is built from the conda-forge channel only, so no Anaconda Terms-of-Service
prompt can stall it.

If the cluster's driver is older or newer, pick the matching PyTorch build, e.g.
`TORCH_CUDA=cu121 bash setup_env.sh` (see https://pytorch.org/get-started/previous-versions/).

## 3. Run

The two download links are **not stored in the repository**. The student supplies them as
environment variables (direct, no-login links to the dataset zip and to the pretrained backbone
checkpoint):

```bash
export LGEL_DATA_URL='<direct link to the Cholec80 zip>'
export LGEL_WEIGHTS_URL='<direct link to the pretrained M2CRL checkpoint>'
```

If the files are already on the cluster's storage, use `--data-zip` (the zip), `--data-dir` (the
unzipped folder) and `--weights-path` (the checkpoint) instead of the links.

**Step A - 5-minute plumbing check on a fake mini-dataset (recommended before the big job):**

```bash
bash run.sh --preset smoke --root $SCRATCH/lgel_smoke --synthetic 6 --random-init
```

This generates 6 small fake videos and proves the environment, GPU, frame extraction, training,
prediction and evaluation all run on this cluster. The numbers it produces are meaningless
(fake data, 1 epoch); only "ALL DONE" matters. Run it inside a GPU job. If `LGEL_WEIGHTS_URL`
is set, dropping `--random-init` also tests the weights link and that the checkpoint loads.

**Step B - the real experiment:**

```bash
bash run.sh --preset full --root $SCRATCH/lgel
```

| Preset | What it does | Use it for |
|---|---|---|
| `smoke` | 6 videos, 1 epoch, small batch | checking that everything is wired up |
| `pilot` | all videos, 3 epochs on 10 % of the training windows | a first "is it learning?" result and a time estimate, before committing GPU hours |
| `full` | all videos, 20 epochs, all training windows, effective batch size 192 | the final numbers |

Note: `full` uses **all** training windows (`--subset-ratio 1.0`), whereas the repository's
default configuration samples 20 %. Pass `--subset-ratio 0.2` to reproduce that.

The total run time has not been measured on this hardware. Run `pilot` first and read the epoch
time from `logs/train.log` to estimate `full`.

`pilot` and `full` (and extra seeds, `--seed N`) can use the **same** `--root`: the data is
downloaded and prepared once, and each run gets its own `runs/<run>/` and `results/<run>/`
(run name `<preset>_seed<seed>` unless `--run-name` is given).

`--root` is required for `pilot` and `full`. Point it at a large scratch filesystem: home
directories are usually too small, and the free-space check cannot see per-user quotas.

## 4. Time limits and resuming

Schedulers kill jobs at their wall-clock limit. Because every step is resumable:

* Submit the **same command again** (same `--root`, same flags). Finished steps are skipped;
  training resumes from `runs/<run>/latest_model.pth`.
* A step that is killed half-way is redone from its own start (downloads continue where they
  stopped).
* Changing settings after training has produced a checkpoint is refused on purpose (it would mix
  two experiments). Use a new `--run-name` (or a new `--root`) for a different experiment.

## 5. If compute nodes have no internet

Do the downloads once on a login node (or any node with internet), then run offline:

```bash
# on a node with internet (CPU only, no GPU needed):
bash run.sh --preset full --root $SCRATCH/lgel --to-stage fetch_data

# inside the GPU job (no links needed, nothing is downloaded):
bash run.sh --preset full --root $SCRATCH/lgel --offline
```

`--to-stage fetch_data` downloads the dataset, the weights and the small text model, then stops.
The second command reuses them. Frame extraction (the slow CPU part) then happens inside the job.

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
  logs/ ............... one log file per step (train.log, extract_frames.log, ...)
  state/ .............. "step finished" markers (delete one to redo that step)
```

`SUMMARY.md` contains the data counts, per-epoch training curve table, validation metrics
(where the decision threshold is chosen), test metrics (threshold fixed from validation) and a
per-query breakdown. Test videos are never used for training or threshold selection.

## 7. Example Slurm job (TEMPLATE - adapt it)

The scheduler on the target cluster has not been checked. The partition, account, module and time
names below are placeholders that must be replaced with the cluster's real ones.

```bash
#!/bin/bash
#SBATCH --job-name=lgel
#SBATCH --partition=<gpu-partition>
#SBATCH --account=<project-account>
#SBATCH --gres=gpu:1                 # one GPU is enough
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00              # resubmit the same script if it hits this limit
#SBATCH --output=lgel_%j.out

export LGEL_DATA_URL='<direct link>'       # not needed if the data is already downloaded
export LGEL_WEIGHTS_URL='<direct link>'
cd /path/to/lgel
bash run.sh --preset full --root /path/to/scratch/lgel
```

Resubmitting is safe and idempotent. On clusters with a per-job time limit the job can also be
chained (`sbatch --dependency=afterany:<jobid> ...`).

## 8. Options worth knowing

| Flag | Meaning |
|---|---|
| `--root DIR` | Where everything is stored. Put it on scratch / a large filesystem. |
| `--run-name NAME` | Name of this training run (default `<preset>_seed<seed>`). |
| `--seed N` | Training seed (default 42). Different seeds can share one `--root`. |
| `--synthetic N` | Use N generated fake videos instead of real data (plumbing test only). |
| `--epochs N`, `--subset-ratio X` | Override the preset. |
| `--batch-size N`, `--accum N` | Micro-batch and gradient accumulation; keep `batch-size x accum = 192` (the project default) so the optimisation stays the same. |
| `--amp-dtype auto/fp16/bf16` | Mixed precision; `auto` picks bf16 when the GPU supports it. |
| `--splits-json FILE` | Use a given `{"train":[...],"val":[...],"test":[...]}` split instead of a new random one (seed 42, 64/8/8 videos). |
| `--cleanup-raw` | Delete the raw videos after frame extraction to save ~80 GB. |
| `--keep-zip` | Keep the downloaded zip. |
| `--offline` | Never touch the network. |
| `--from-stage S`, `--to-stage S`, `--stages a,b` | Run only part of the pipeline. `python main.py --list-stages` prints the order. |
| `--force S` | Redo stage `S` and everything after it. |
| `--dry-run` | Print what would run, do nothing. |

## 9. Troubleshooting

| Symptom | Fix |
|---|---|
| `CUDA out of memory` or `killed` during training | Lower `--batch-size` (8 -> 4 -> 2) and raise `--accum` so that the product stays 192, then resubmit. This is accepted as long as no checkpoint has been written yet. |
| `no CUDA GPU is visible` | The job did not get a GPU; check the `--gres` / partition line. |
| `--root is required for the pilot/full presets` | Add `--root /path/on/scratch/lgel` (or set `LGEL_ROOT`). |
| `only N GB free under <root>` | Point `--root` at a larger filesystem, or lower `--min-free-gb` if you are sure. |
| `Run ... already exists with different settings` | You changed flags after training started. Restore the flags, or use a new `--run-name`. |
| `stage ... was already finished with different settings` | Same idea for data stages: use a new `--root`, or `--force <stage>`. |
| Download stops or the zip is incomplete | Just resubmit: downloads continue and the size is verified. Google-Drive links need `gdown` (installed by `setup_env.sh`) and may hit Google's daily quota. |
| `Python package ... is not importable` | The conda env is missing or not active: run `bash setup_env.sh`. |
| Everything is slow at the start | Frame extraction of ~80 videos is CPU-bound; it scales with `--workers` (default: up to 16). |

The per-step logs in `<root>/logs/` contain the full output of every underlying script.

## 10. What has and has not been verified

Verified (automated tests, 58 passing, plus end-to-end runs):

* The full chain on a synthetic Cholec80-format dataset with a deliberately **tiny** model on CPU:
  download/extract, frame extraction, annotation repair, splits, audit, training, resume,
  prediction, evaluation, summary, including a killed-and-resumed download, an
  "internet first, offline later" two-step run, and `pilot` followed by `full` in one `--root`.
* Killing training after epoch 2 and resuming gives bit-identical weights, optimizer/scheduler
  state and metrics to an uninterrupted run (tiny model).
* `run.sh` activating the conda env in a non-interactive shell (Miniforge, conda 26.7), and
  `setup_env.sh` up to the PyTorch download.

Not verified (could not be tested where this was written):

* The real-size model, real GPU training, mixed precision and memory use on actual hardware.
* Creating the environment from scratch (conda-forge + PyTorch CUDA 12.4 wheels): the build
  machine could not reach those servers.
* Slurm or any other scheduler.
* The final dataset and weights download links, and Google-Drive links.
* Training time and the quality of the final results. Run the `pilot` preset first.

Known data issue handled automatically: in the official Cholec80 release, the phase files of
video15 and video37 contain one extra row one frame past the end of the video. The pipeline
removes such a row (only if it merely repeats the last phase), writes the cleaned copy to
`data/annotations_clean/` and records what it did in `annotations_sanitized.json`. The original
files are not modified.

The dataset is distributed by its owners under their own terms. Check that you may use and move
it on the cluster before uploading it anywhere. `backbone/vision_transformer.py` is adapted from
TimeSformer (CC BY-NC 4.0) and keeps that licence; see the README's "Third-party code and data".
