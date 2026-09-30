#!/usr/bin/env bash
# Create the conda environment for this project (run ONCE, on a machine with internet access).
#
#   bash setup_env.sh                 # env name "lgel", CUDA 12.4 PyTorch wheels
#   LGEL_ENV=myenv TORCH_CUDA=cu121 bash setup_env.sh
#
# Needs conda/mamba on PATH (on many clusters: `module load anaconda` or similar first; put
# such lines in site_env.sh, which this script and run.sh both source if it exists).
set -euo pipefail
cd "$(dirname "$0")"
[ -f site_env.sh ] && source site_env.sh

ENV_NAME="${LGEL_ENV:-lgel}"
TORCH_CUDA="${TORCH_CUDA:-cu124}"        # must be supported by the cluster's NVIDIA driver (check `nvidia-smi`)

if ! command -v conda >/dev/null 2>&1; then
  echo "ERROR: conda not found. Load it (e.g. 'module load anaconda') or install miniforge, then re-run." >&2
  exit 1
fi
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"

if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "conda env '$ENV_NAME' already exists - reusing it"
else
  conda create -y -n "$ENV_NAME" python=3.12 pip
fi
conda activate "$ENV_NAME"

python -m pip install --upgrade pip
python -m pip install "torch==2.5.1" "torchvision==0.20.1" --index-url "https://download.pytorch.org/whl/${TORCH_CUDA}"
python -m pip install -r requirements.txt          # pinned versions; torch lines are already satisfied
python -m pip install gdown matplotlib             # optional helpers: Google-Drive links, training-curve plot

python - <<'EOF'
import torch, torchvision, transformers, cv2, pandas, sklearn
print("torch", torch.__version__, "| CUDA available on THIS node:", torch.cuda.is_available())
print("(on a login node without a GPU 'False' is normal; it must be True inside the GPU job)")
EOF
echo "OK: environment '$ENV_NAME' is ready.  Next: bash run.sh --preset smoke ..."
