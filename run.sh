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
cd "$(dirname "$0")"
[ -f site_env.sh ] && source site_env.sh          # optional: site-specific 'module load ...' lines

if [ "${LGEL_SKIP_ENV:-0}" != "1" ]; then
  ENV_NAME="${LGEL_ENV:-lgel}"
  if command -v conda >/dev/null 2>&1; then
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"
    if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
      echo "conda env '$ENV_NAME' not found - creating it now (needs internet; ~10 min)"
      bash setup_env.sh
    fi
    conda activate "$ENV_NAME"
  else
    echo "conda not found; using the python on PATH ($(command -v python))"
  fi
fi

exec python main.py "$@"
