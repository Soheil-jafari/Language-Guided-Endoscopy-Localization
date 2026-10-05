#!/bin/bash -l
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=3-00:00:00
#SBATCH --signal=B:USR1@300
#SBATCH --job-name=lgel
#
# Slurm job for the whole pipeline on the project cluster. Do not submit this file directly: use
#     bash hpc/submit.sh smoke|pilot|full
# which sets the output file, the time limit and the paths below, and makes sure only one job runs at
# a time. The job creates the conda environment on its first run, then downloads, prepares, trains,
# evaluates, and writes a report to send back (<root>/outbox/*.tar.gz).
#
# Stopping: about 5 minutes before the time limit Slurm sends USR1 to this script (--signal above);
# it then stops the pipeline cleanly (the report is written, training resumes from its last finished
# epoch when the same command is submitted again). A SIGTERM (scancel) is handled the same way.

set -o pipefail
: "${LGEL_REPO:?submit with: bash hpc/submit.sh smoke|pilot|full}"
: "${LGEL_BASE:?}" "${LGEL_PRESET:?}" "${LGEL_ROOT:?}"
JOB="${SLURM_JOB_ID:-local}"
SLURM_LOG="$LGEL_ROOT/logs/slurm-$JOB.out"
mkdir -p "$LGEL_ROOT/logs"
START_MARK="$LGEL_ROOT/logs/.job-$JOB.started"
touch "$START_MARK"
RC=1
CHILD=

# Whatever happens below, end with a report: the pipeline writes its own; if it never got to run
# (module or environment problems), a small one is made from the end of this job's log.
finish() { bash "$LGEL_REPO/hpc/fallback_report.sh" "$LGEL_ROOT" "$JOB" "$RC" "$SLURM_LOG" "$LGEL_PRESET" "$START_MARK"; }
trap finish EXIT
trap 'echo "stop requested before the pipeline started"; RC=143; exit 143' USR1 TERM INT

echo "job $JOB on $(hostname) at $(date) | preset=$LGEL_PRESET root=$LGEL_ROOT"

module load libs/nvidia-cuda/12.4.0/bin || echo "note: CUDA module not loaded (PyTorch brings its own CUDA runtime)"
if ! module load apps/miniconda3; then
  echo "ERROR: 'module load apps/miniconda3' failed"
  RC=3; exit 3
fi
module list 2>&1 || true
set -u                                         # (module scripts are not always 'set -u' safe)

# Conda environment (an exact location), package caches and temporary files go under the project
# folder, not $HOME.
export LGEL_ENV_PREFIX="$LGEL_BASE/conda/envs/lgel"
export CONDA_ENVS_PATH="$LGEL_BASE/conda/envs" CONDA_PKGS_DIRS="$LGEL_BASE/conda/pkgs"
export PIP_CACHE_DIR="$LGEL_BASE/conda/pip-cache" TMPDIR="$LGEL_BASE/tmp"
mkdir -p "$CONDA_ENVS_PATH" "$CONDA_PKGS_DIRS" "$PIP_CACHE_DIR" "$TMPDIR"

# The two private download links (never stored in the repository, never printed).
if [ -f "$LGEL_BASE/links.env" ]; then
  set -a
  # shellcheck disable=SC1091
  source "$LGEL_BASE/links.env"
  set +a
fi

has_arg() {   # has_arg OPTION ARGS... : is --option (or --option=...) among the extra arguments?
  local opt="$1"; shift
  local x
  for x in "$@"; do case "$x" in "$opt"|"$opt"=*) return 0 ;; esac; done
  return 1
}

EXTRA=()
if [ "$LGEL_PRESET" = smoke ]; then
  # Fake videos: tests the software, the GPU and (if given) the real weights.
  if ! has_arg --data-dir "$@" && ! has_arg --data-zip "$@" && ! has_arg --synthetic "$@"; then
    EXTRA+=(--synthetic 6)
  fi
  if [ -z "${LGEL_WEIGHTS_URL:-}" ] && ! has_arg --weights-path "$@" && ! has_arg --random-init "$@"; then
    EXTRA+=(--random-init)
  fi
else
  # Fail now (seconds), not after installing the environment, if a needed link is missing.
  if [ ! -f "$LGEL_ROOT/state/fetch_data.done.json" ] && [ -z "${LGEL_DATA_URL:-}" ] \
     && ! has_arg --data-dir "$@" && ! has_arg --data-zip "$@"; then
    echo "ERROR: no dataset link: LGEL_DATA_URL is empty in $LGEL_BASE/links.env"
    RC=2; exit 2
  fi
  if [ ! -f "$LGEL_ROOT/state/fetch_weights.done.json" ] && [ -z "${LGEL_WEIGHTS_URL:-}" ] \
     && ! has_arg --weights-path "$@" && ! has_arg --random-init "$@"; then
    echo "ERROR: no weights link: LGEL_WEIGHTS_URL is empty in $LGEL_BASE/links.env"
    RC=2; exit 2
  fi
fi

# Run the pipeline in its own process group, so that a stop reaches all of it (including the
# environment installation on the first run). --single-job: submit.sh lets only one job run at a
# time, so this also works if the scratch file system cannot lock files (where locks work they stay on).
set -m
bash "$LGEL_REPO/run.sh" --preset "$LGEL_PRESET" --root "$LGEL_ROOT" --single-job \
  ${EXTRA[@]+"${EXTRA[@]}"} "$@" &
CHILD=$!
stop_pipeline() {
  echo "$(date): $1 - stopping the pipeline cleanly (submit the same command again to continue)"
  kill -TERM -- "-$CHILD" 2>/dev/null || kill -TERM "$CHILD" 2>/dev/null
}
trap 'stop_pipeline "the time limit is near (USR1)"' USR1
trap 'stop_pipeline "SIGTERM (time limit or scancel)"' TERM
trap 'stop_pipeline "SIGINT"' INT
while :; do
  wait "$CHILD"
  kill -0 "$CHILD" 2>/dev/null || break        # wait returns early when a signal arrives
done
wait "$CHILD"
RC=$?
echo "pipeline finished with exit code $RC at $(date)"
exit "$RC"
