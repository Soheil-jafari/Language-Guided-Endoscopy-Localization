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
  ENV_PREFIX="${LGEL_ENV_PREFIX:-}"            # an exact location (set by the cluster job), else by name
  if command -v conda >/dev/null 2>&1; then
    set +u                                     # conda's shell scripts are not all 'set -u' safe
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"
    if [ -n "$ENV_PREFIX" ]; then
      TARGET="$ENV_PREFIX"
      [ -x "$ENV_PREFIX/bin/python" ] || needs_setup "conda env $ENV_PREFIX does not exist"
    else
      TARGET="$ENV_NAME"
      if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
        needs_setup "conda env '$ENV_NAME' does not exist"
      fi
    fi
    conda activate "$TARGET"
    if [ -n "$ENV_PREFIX" ] && [ "$(cd "$CONDA_PREFIX" 2>/dev/null && pwd -P)" != "$(cd "$ENV_PREFIX" 2>/dev/null && pwd -P)" ]; then
      echo "ERROR: conda activated $CONDA_PREFIX instead of $ENV_PREFIX" >&2
      exit 1
    fi
    # setup_env.sh writes this marker only after every package installed and imported correctly,
    # so an environment left half-installed by an interrupted setup is detected and completed.
    # (the same fingerprint of requirements.txt + setup_env.sh that setup_env.sh writes)
    WANT="$("$CONDA_PREFIX/bin/python" -c 'import hashlib,sys; h=hashlib.sha256(); [h.update(open(f,"rb").read()) for f in sys.argv[1:]]; print(h.hexdigest())' "$REPO/requirements.txt" "$REPO/setup_env.sh" "$REPO/install_mamba.sh" 2>/dev/null || true)"
    if [ ! -x "$CONDA_PREFIX/bin/python" ] || [ -z "$WANT" ] || \
       [ "$(cat "$CONDA_PREFIX/.lgel_env_ready" 2>/dev/null || true)" != "$WANT" ]; then
      conda deactivate                         # let setup_env.sh start from a clean shell
      needs_setup "conda env '$TARGET' is incomplete or out of date"
      conda activate "$TARGET"
    elif [ "$OFFLINE" != "1" ] && [ -e "$CONDA_PREFIX/.lgel_mamba_missing" ]; then
      # An earlier Mamba installation failed (e.g. a network hiccup): retry only that download. Never fatal:
      # the baseline model does not need Mamba, and the advanced model's check reports it if still missing.
      echo "the Mamba package is missing from conda env '$TARGET' - retrying its installation"
      LGEL_MAMBA_NO_COMPILE=1 bash "$REPO/install_mamba.sh" "$CONDA_PREFIX/bin/python" \
        || echo "WARNING: Mamba is still missing; the baseline model can run, the advanced model cannot"
    fi
    set -u
  else
    echo "conda not found; using the python on PATH"
  fi
fi

PY="$(command -v python || command -v python3 || true)"
if [ -z "$PY" ]; then echo "ERROR: no python found (run 'bash setup_env.sh' first)" >&2; exit 1; fi
exec "$PY" "$REPO/main.py" "$@"
