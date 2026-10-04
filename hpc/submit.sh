#!/usr/bin/env bash
# Submit the pipeline to Slurm on the project cluster. Run on the login node, from anywhere:
#
#   bash hpc/submit.sh smoke          # ~30 min: creates the conda env, tests GPU + software on fake data
#   bash hpc/submit.sh pilot          # a few hours: real data prepared once, short training, time estimate
#   bash hpc/submit.sh full           # about a day: the real experiment
#   bash hpc/submit.sh full --repeat 2    # same, plus a follow-up job that continues if 72 h were not enough
#
# Anything after the preset is passed on to main.py, e.g.  bash hpc/submit.sh full --batch-size 4 --accum 48
# Everything is written under $LGEL_BASE (default /mnt/scratch/users/daiz1/soheil):
#   lgel_smoke/  (smoke test)   lgel/  (pilot and full)   conda/  (environment)   links.env  (private links)
# After each job the file to send back is <root>/outbox/<newest>.tar.gz (name in <root>/outbox/LATEST.txt).
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE="${LGEL_BASE:-/mnt/scratch/users/daiz1/soheil}"

PRESET="${1:-}"
case "$PRESET" in
  smoke) ROOT="$BASE/lgel_smoke"; TIME="0-04:00:00" ;;
  pilot) ROOT="$BASE/lgel";       TIME="1-00:00:00" ;;
  full)  ROOT="$BASE/lgel";       TIME="3-00:00:00" ;;
  *) echo "usage: bash hpc/submit.sh smoke|pilot|full [--repeat N] [main.py options]" >&2; exit 2 ;;
esac
shift
REPEAT=1
ARGS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --repeat) REPEAT="${2:?--repeat needs a number}"; shift 2 ;;
    *) ARGS+=("$1"); shift ;;
  esac
done
case "$REPEAT" in ''|*[!0-9]*) echo "--repeat needs a number" >&2; exit 2 ;; esac

command -v sbatch >/dev/null 2>&1 || { echo "ERROR: sbatch not found: run this on the cluster's login node" >&2; exit 1; }

# One job at a time: two jobs must never use the same folder simultaneously.
RUNNING="$(squeue -h -u "$USER" -n lgel -o '%i %j %T' 2>/dev/null || true)"
if [ -n "$RUNNING" ]; then
  echo "ERROR: an lgel job is already queued or running:" >&2
  echo "$RUNNING" >&2
  echo "Wait until it has finished (or cancel it with: scancel <jobid>), then submit again." >&2
  exit 1
fi

if [ "$PRESET" != smoke ] && [ ! -f "$BASE/links.env" ] && [ ! -f "$ROOT/state/fetch_data.done.json" ]; then
  echo "ERROR: $BASE/links.env is missing. It holds the two private download links:" >&2
  echo "       cp $REPO/hpc/links.env.example $BASE/links.env   and paste the links into it." >&2
  exit 1
fi

mkdir -p "$ROOT/logs"
DEPENDENCY=()
for _ in $(seq 1 "$REPEAT"); do
  JOB="$(sbatch --parsable --job-name=lgel --time="$TIME" --output="$ROOT/logs/slurm-%j.out" \
         ${DEPENDENCY[@]+"${DEPENDENCY[@]}"} \
         --export="ALL,LGEL_REPO=$REPO,LGEL_BASE=$BASE,LGEL_PRESET=$PRESET,LGEL_ROOT=$ROOT" \
         "$REPO/hpc/lgel.sbatch" ${ARGS[@]+"${ARGS[@]}"})"
  JOB="${JOB%%;*}"
  echo "submitted $PRESET job $JOB   (live log: $ROOT/logs/slurm-$JOB.out)"
  DEPENDENCY=(--dependency="afterany:$JOB")
done
echo "Watch it with: squeue -u $USER      Everything ends up in: $ROOT"
echo "File to send back after the job: $ROOT/outbox/ (newest file; its name is in outbox/LATEST.txt)"
