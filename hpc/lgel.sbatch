#!/bin/bash -l
#SBATCH -p gpu
#SBATCH --gpus=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=3-00:00:00
#SBATCH --job-name=lgel
#
# Slurm job for the whole pipeline on the project cluster. Do not submit this file directly: use
#     bash hpc/submit.sh smoke|pilot|full
# which sets the output file, the time limit and the paths below, and makes sure only one job runs at
# a time. The job creates the conda environment on its first run, then downloads, prepares, trains,
# evaluates, and writes a report to send back (<root>/outbox/*.tar.gz).

set -o pipefail
: "${LGEL_REPO:?submit with: bash hpc/submit.sh smoke|pilot|full}"
: "${LGEL_BASE:?}" "${LGEL_PRESET:?}" "${LGEL_ROOT:?}"
echo "job ${SLURM_JOB_ID:-?} on $(hostname) at $(date) | preset=$LGEL_PRESET root=$LGEL_ROOT"

module load libs/nvidia-cuda/12.4.0/bin || echo "note: CUDA module not loaded (PyTorch brings its own CUDA runtime)"
module load apps/miniconda3 || { echo "ERROR: 'module load apps/miniconda3' failed"; exit 1; }
set -u                                         # (module scripts are not always 'set -u' safe)

# Conda environment, package caches and temporary files go under the project folder, not $HOME.
export CONDA_ENVS_PATH="$LGEL_BASE/conda/envs" CONDA_PKGS_DIRS="$LGEL_BASE/conda/pkgs"
export PIP_CACHE_DIR="$LGEL_BASE/conda/pip-cache" TMPDIR="$LGEL_BASE/tmp"
mkdir -p "$CONDA_ENVS_PATH" "$CONDA_PKGS_DIRS" "$PIP_CACHE_DIR" "$TMPDIR"

# The two private download links (never stored in the repository).
if [ -f "$LGEL_BASE/links.env" ]; then
  set -a
  # shellcheck disable=SC1091
  source "$LGEL_BASE/links.env"
  set +a
fi

EXTRA=()
if [ "$LGEL_PRESET" = smoke ]; then
  EXTRA=(--synthetic 6)                        # fake videos: tests software, GPU and the weights link
  [ -z "${LGEL_WEIGHTS_URL:-}" ] && EXTRA+=(--random-init)
fi

# --single-job: submit.sh guarantees one job at a time, so this also works if the scratch file
# system cannot lock files (where locks do work, they stay on).
# `exec`: the pipeline becomes the job's main process. At the time limit (or scancel) Slurm then
# waits for it to write its report bundle instead of killing it as soon as this script exits.
# The last lines of the job log show where the report is ("REPORT: ...").
exec bash "$LGEL_REPO/run.sh" --preset "$LGEL_PRESET" --root "$LGEL_ROOT" --single-job \
  ${EXTRA[@]+"${EXTRA[@]}"} "$@"
