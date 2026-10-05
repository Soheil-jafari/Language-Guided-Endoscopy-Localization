#!/usr/bin/env bash
# Called by hpc/lgel.sbatch when the job ends. If the Python pipeline wrote its own report for this
# job, it does nothing. Otherwise (e.g. the conda module or the environment installation failed
# before Python could start) it writes a small report with the end of the job log, so there is
# always one file to send back. Uses only bash and coreutils; links are redacted.
#
#   bash hpc/fallback_report.sh ROOT JOB_ID EXIT_CODE SLURM_LOG PRESET START_MARKER
set -u
ROOT="$1"; JOB="$2"; RC="$3"; SLURM_LOG="$4"; PRESET="$5"; START="$6"
OUT="$ROOT/outbox"
mkdir -p "$OUT" 2>/dev/null || exit 0

# The pipeline already reported this job (a report newer than the job's start marker exists).
PATTERN='*.tar.gz'; [ "$JOB" = local ] || PATTERN="*-job${JOB}-*.tar.gz"      # this job's own report
if [ -f "$START" ] && [ -n "$(find "$OUT" -maxdepth 1 -name "$PATTERN" -newer "$START" -print -quit 2>/dev/null)" ]; then
  exit 0
fi

case "$RC" in
  0) STATUS=OK ;;
  129|138|143) STATUS=STOPPED ;;
  *) # did the pipeline itself get going (its log was written during this job)? Then it crashed;
     # otherwise the environment/setup failed before it could start.
     if [ -f "$ROOT/logs/main.log" ] && [ "$ROOT/logs/main.log" -nt "$START" ]; then STATUS=FAILED
     else STATUS=SETUP-FAILED; fi ;;
esac
NAME="${PRESET}-$(date +%Y%m%d-%H%M%S)-job${JOB}-${STATUS}.tar.gz"
WORK="$(mktemp -d "$OUT/.fallback.XXXXXX" 2>/dev/null)" || exit 0
trap 'rm -rf "$WORK"' EXIT

# Every link is cut down to its host. ('%' is the delimiter because it never occurs in the expression,
# so no sed implementation can misread it.)
redact() { sed -E 's%((https?|ftp)://)([^/[:space:]@"<>]*@)?([^/?#@[:space:]"<>]+)[^[:space:]"<>]*%\1\4/...%g'; }
{
  echo "report: $NAME"
  echo "outcome: $STATUS (exit code $RC) - written by hpc/fallback_report.sh because the pipeline did not write its own"
  echo "job: $JOB  host: $(hostname)  written: $(date)"
  echo "contents: the end of the Slurm job log (and of logs/main.log if it exists)"
} > "$WORK/MANIFEST.txt"
if [ -f "$SLURM_LOG" ]; then
  tail -c 2000000 "$SLURM_LOG" 2>/dev/null | redact > "$WORK/$(basename "$SLURM_LOG").tail.txt"
fi
if [ -f "$ROOT/logs/main.log" ]; then
  tail -c 2000000 "$ROOT/logs/main.log" 2>/dev/null | redact > "$WORK/main.log.tail.txt"
fi

tar -czf "$OUT/.$NAME.tmp" -C "$WORK" . 2>/dev/null && mv -f "$OUT/.$NAME.tmp" "$OUT/$NAME" || exit 0
printf '%s\n' "$NAME" > "$OUT/.LATEST.fallback.tmp" && mv -f "$OUT/.LATEST.fallback.tmp" "$OUT/LATEST.txt"
echo "REPORT: $OUT/$NAME - send this file to Soheil."
