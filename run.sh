#!/usr/bin/env bash
# THE ONE COMMAND.  Everything (download, frame extraction, labels, splits, training, evaluation)
# is driven by main.py; this wrapper only makes sure the conda environment exists and is active.
#
#   export LGEL_DATA_URL='<direct link to the Cholec80 zip>'
#   export LGEL_WEIGHTS_URL='<direct link to the M2CRL checkpoint>'
#   bash run.sh --preset full --root /path/to/big/scratch/lgel
#
# Re-running the same command resumes where it stopped (finished stages are skipped, training
# continues from the last finished epoch).  See README_HPC.md for details and options.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# NOTE: no `cd` here - relative paths given on the command line (e.g. --root ./lgel) must stay
# relative to the directory the command is run from.  main.py itself does not depend on the cwd.
if [ -f "$REPO/site_env.sh" ]; then source "$REPO/site_env.sh"; fi   # optional 'module load ...' lines

OFFLINE=0
for arg in "$@"; do [ "$arg" = "--offline" ] && OFFLINE=1; done

needs_setup() {   # $1 = why; never installs anything in an --offline job
  if [ "$OFFLINE" = "1" ]; then
    echo "ERROR: $1, and --offline jobs must not install software." >&2
    echo "       Run 'bash setup_env.sh' once on a machine with internet, then resubmit." >&2
    exit 1
  fi
  echo "$1 - running setup_env.sh now (needs internet; ~10 min)"
  bash "$REPO/setup_env.sh"
}

if [ "${LGEL_SKIP_ENV:-0}" != "1" ]; then
  ENV_NAME="${LGEL_ENV:-lgel}"
  if command -v conda >/dev/null 2>&1; then
    set +u                                     # conda's shell scripts are not all 'set -u' safe
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"
    if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
      needs_setup "conda env '$ENV_NAME' does not exist"
    fi
    conda activate "$ENV_NAME"
    # setup_env.sh writes this marker only after every package installed and imported correctly,
    # so an environment left half-installed by an interrupted setup is detected and completed.
    WANT="$("$CONDA_PREFIX/bin/python" -c 'import hashlib,sys; print(hashlib.sha256(open(sys.argv[1],"rb").read()).hexdigest())' "$REPO/requirements.txt" 2>/dev/null || true)"
    if [ ! -x "$CONDA_PREFIX/bin/python" ] || [ -z "$WANT" ] || \
       [ "$(cat "$CONDA_PREFIX/.lgel_env_ready" 2>/dev/null || true)" != "$WANT" ]; then
      conda deactivate                         # let setup_env.sh start from a clean shell
      needs_setup "conda env '$ENV_NAME' is incomplete or out of date"
      conda activate "$ENV_NAME"
    fi
    set -u
  else
    echo "conda not found; using the python on PATH"
  fi
fi

PY="$(command -v python || command -v python3 || true)"
if [ -z "$PY" ]; then echo "ERROR: no python found (run 'bash setup_env.sh' first)" >&2; exit 1; fi
exec "$PY" "$REPO/main.py" "$@"
