#!/usr/bin/env bash
# Submit the pipeline to Slurm on the project cluster. Run on the login node, from anywhere:
#
#   bash hpc/submit.sh smoke          # ~30 min: creates the conda env, tests GPU + software on fake data
#   bash hpc/submit.sh pilot          # a few hours: real data prepared once, short training, time estimate
#   bash hpc/submit.sh full           # estimated about a day: the real experiment
#   bash hpc/submit.sh full --repeat 2    # same, plus one follow-up job in case 72 h are not enough
#
# Anything after the preset is passed on to main.py, e.g.  bash hpc/submit.sh full --batch-size 4 --accum 48
# (except --root/--preset, which this script sets, and the download links, which belong in links.env).
# Site-specific sbatch options, if the cluster ever asks for them:  LGEL_SBATCH_OPTS="--account=xyz" bash ...
# Everything is written under $LGEL_BASE (default /mnt/scratch/users/daiz1/soheil):
#   lgel_smoke/ (smoke test)  lgel/ (pilot and full)  conda/ (environment)  links.env (private links)
# After each job the file to send back is <root>/outbox/<newest>.tar.gz (name in <root>/outbox/LATEST.txt).
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE="${LGEL_BASE:-/mnt/scratch/users/daiz1/soheil}"
die() { echo "ERROR: $*" >&2; exit 1; }

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
    --repeat) [ $# -ge 2 ] || die "--repeat needs a number"; REPEAT="$2"; shift 2 ;;
    --repeat=*) REPEAT="${1#--repeat=}"; shift ;;
    --root|--root=*|--preset|--preset=*)
      die "$1: the folder and the preset are set by this script (use smoke, pilot or full)" ;;
    --data-url|--data-url=*|--weights-url|--weights-url=*)
      die "${1%%=*}: put the download links in $BASE/links.env, never on the command line (it is visible to other users)" ;;
    *) ARGS+=("$1"); shift ;;
  esac
done
case "$REPEAT" in ''|*[!0-9]*) die "--repeat needs a whole number between 1 and 100" ;; esac
[ "$REPEAT" -ge 1 ] && [ "$REPEAT" -le 100 ] || die "--repeat needs a whole number between 1 and 100"
case "$REPO$BASE" in *,*) die "the paths must not contain commas (Slurm's --export cannot pass them): $REPO, $BASE" ;; esac

command -v sbatch >/dev/null 2>&1 || die "sbatch not found: run this on the cluster's login node"
mkdir -p "$BASE"

# Only one submission at a time (an atomic lock directory), and only one job at a time (the queue).
LOCK="$BASE/.submit.lock"
BUSY="another submit.sh is running. If none is, remove the folder $LOCK (and $LOCK.reclaim if it exists) and try again."
take_lock() { mkdir "$LOCK" 2>/dev/null && echo "$(hostname) $$" > "$LOCK/owner"; }
if ! take_lock; then
  # A lock left by a submit.sh on this host that no longer runs is taken over - by one submit.sh at a time
  # (the .reclaim folder), and only if the owner is still that dead process, so a live lock is never removed.
  OWNER="$(cat "$LOCK/owner" 2>/dev/null || true)"
  if [ "${OWNER%% *}" = "$(hostname)" ] && [ -n "${OWNER##* }" ] && ! kill -0 "${OWNER##* }" 2>/dev/null \
     && mkdir "$LOCK.reclaim" 2>/dev/null; then
    if [ "$(cat "$LOCK/owner" 2>/dev/null || true)" = "$OWNER" ]; then rm -rf "$LOCK"; fi
    take_lock; GOT=$?
    rmdir "$LOCK.reclaim"
    [ "$GOT" -eq 0 ] || die "$BUSY"
  else
    die "$BUSY"
  fi
fi
trap 'rm -rf "$LOCK"' EXIT
trap 'exit 130' INT TERM

ME="${USER:-$(id -un)}"
if ! RUNNING="$(squeue -h -u "$ME" -n lgel -o '%i %j %T' 2>"$LOCK/squeue.err")"; then
  die "could not ask Slurm which jobs are running ($(cat "$LOCK/squeue.err")). Try again in a few minutes."
fi
if [ -n "$RUNNING" ]; then
  echo "ERROR: an lgel job is already queued or running:" >&2
  echo "$RUNNING" >&2
  echo "Wait until it has finished (or cancel it with: scancel <jobid>), then submit again." >&2
  exit 1
fi

if [ "$PRESET" != smoke ]; then
  need_data=1; need_weights=1
  [ -f "$ROOT/state/fetch_data.done.json" ] && need_data=0
  [ -f "$ROOT/state/fetch_weights.done.json" ] && need_weights=0
  for x in ${ARGS[@]+"${ARGS[@]}"}; do
    case "$x" in --data-dir|--data-dir=*|--data-zip|--data-zip=*) need_data=0 ;;
                 --weights-path|--weights-path=*|--random-init) need_weights=0 ;; esac
  done
  if [ "$need_data$need_weights" != "00" ]; then
    [ -f "$BASE/links.env" ] || die "$BASE/links.env is missing. It holds the two private download links (see README_HPC.md, section 0)."
    # shellcheck disable=SC1091
    if ! ( set -a; . "$BASE/links.env"; { [ "$need_data" = 0 ] || [ -n "${LGEL_DATA_URL:-}" ]; } &&
                                         { [ "$need_weights" = 0 ] || [ -n "${LGEL_WEIGHTS_URL:-}" ]; } ); then
      die "a link is empty in $BASE/links.env (LGEL_DATA_URL and LGEL_WEIGHTS_URL must both be filled in)."
    fi
  fi
  if [ -n "$(find "$BASE/links.env" -perm /077 2>/dev/null)" ]; then
    echo "note: $BASE/links.env can be read by other users; 'chmod 600 $BASE/links.env' keeps the links private."
  fi
fi

mkdir -p "$ROOT/logs"
read -r -a SITE_OPTS <<< "${LGEL_SBATCH_OPTS:-}"
DEPENDENCY=()
for i in $(seq 1 "$REPEAT"); do
  OUT="$(sbatch --parsable --job-name=lgel --time="$TIME" --output="$ROOT/logs/slurm-%j.out" \
         ${SITE_OPTS[@]+"${SITE_OPTS[@]}"} ${DEPENDENCY[@]+"${DEPENDENCY[@]}"} \
         --export="ALL,LGEL_REPO=$REPO,LGEL_BASE=$BASE,LGEL_PRESET=$PRESET,LGEL_ROOT=$ROOT" \
         "$REPO/hpc/lgel.sbatch" ${ARGS[@]+"${ARGS[@]}"})" || die "sbatch refused the job (see the message above)"
  JOB="${OUT%%;*}"
  case "$JOB" in ''|*[!0-9]*) die "unexpected answer from sbatch: '$OUT'" ;; esac
  if [ "$i" = 1 ]; then echo "submitted $PRESET job $JOB   (live log: $ROOT/logs/slurm-$JOB.out)"
  else echo "submitted follow-up job $JOB (starts after the previous one ends, whatever its outcome)"; fi
  DEPENDENCY=(--dependency="afterany:$JOB")
done
echo "Watch it with: squeue -u $ME      Everything ends up in: $ROOT"
echo "When it has finished, send back the newest file in $ROOT/outbox/ (its name is in outbox/LATEST.txt)."
