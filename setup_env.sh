#!/usr/bin/env bash
# Create the conda environment for this project (run ONCE, on a machine with internet access).
#
#   bash setup_env.sh                 # env name "lgel", CUDA 12.4 PyTorch wheels
#   LGEL_ENV=myenv TORCH_CUDA=cu121 bash setup_env.sh
#
# Needs conda/mamba on PATH (on many clusters: `module load anaconda` or similar first; put
# such lines in site_env.sh, which this script and run.sh both source if it exists).
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "$REPO/site_env.sh" ]; then source "$REPO/site_env.sh"; fi

ENV_NAME="${LGEL_ENV:-lgel}"
TORCH_CUDA="${TORCH_CUDA:-cu124}"        # must be supported by the cluster's NVIDIA driver (check `nvidia-smi`)

if ! command -v conda >/dev/null 2>&1; then
  echo "ERROR: conda not found. Load it (e.g. 'module load anaconda') or install miniforge, then re-run." >&2
  exit 1
fi

# The pinned numpy/scipy/pandas wheels need glibc >= 2.28 (RHEL/Rocky/Alma 8+, Ubuntu 20.04+).
GLIBC="$(ldd --version 2>/dev/null | head -1 | grep -oE '[0-9]+\.[0-9]+$' || true)"
if [ -n "$GLIBC" ] && [ "$(printf '%s\n2.28\n' "$GLIBC" | sort -V | head -1)" != "2.28" ]; then
  echo "WARNING: this system has glibc $GLIBC (< 2.28, e.g. CentOS 7). Some pinned packages in" >&2
  echo "         requirements.txt have no prebuilt wheels for it and pip will try to compile them." >&2
fi

set +u                                   # conda's shell scripts are not all 'set -u' safe
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "conda env '$ENV_NAME' already exists - reusing it"
else
  # conda-forge only: Anaconda's default channels require an interactive Terms-of-Service
  # acceptance in recent conda versions, which would stall a batch job.
  conda create -y -n "$ENV_NAME" -c conda-forge --override-channels python=3.12 pip
fi
conda activate "$ENV_NAME"
if [ ! -x "$CONDA_PREFIX/bin/python" ]; then     # env left half-created by an interrupted run
  conda install -y -n "$ENV_NAME" -c conda-forge --override-channels python=3.12 pip
  conda activate "$ENV_NAME"
fi
set -u
PY="$CONDA_PREFIX/bin/python"                    # never fall back to a system python
rm -f "$CONDA_PREFIX/.lgel_env_ready"

"$PY" -m pip install --upgrade pip
"$PY" -m pip install "torch==2.5.1" "torchvision==0.20.1" --index-url "https://download.pytorch.org/whl/${TORCH_CUDA}"
"$PY" -m pip install -r "$REPO/requirements.txt"  # pinned versions; torch lines are already satisfied
"$PY" -m pip install gdown matplotlib             # optional helpers: Google-Drive links, training-curve plot

"$PY" - <<'EOF'
import torch, torchvision, transformers, cv2, pandas, sklearn, einops, PIL
print("torch", torch.__version__, "| CUDA available on THIS node:", torch.cuda.is_available())
print("(on a login node without a GPU 'False' is normal; it must be True inside the GPU job)")
EOF
# Written last: run.sh treats an environment without this stamp (or with a stale one) as incomplete.
"$PY" -c 'import hashlib,sys; print(hashlib.sha256(open(sys.argv[1],"rb").read()).hexdigest())' \
  "$REPO/requirements.txt" > "$CONDA_PREFIX/.lgel_env_ready"
echo "OK: environment '$ENV_NAME' is ready.  Next (see README_HPC.md): bash run.sh --preset smoke --root <scratch>/lgel_smoke --synthetic 6 --random-init"
