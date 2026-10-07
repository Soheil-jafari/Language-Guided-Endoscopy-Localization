#!/usr/bin/env bash
# Install the official Mamba package (mamba-ssm) into the project's Python environment. Called by
# setup_env.sh, and by run.sh to retry after a failed attempt (then without compiling).
#
#   bash install_mamba.sh /path/to/env/bin/python
#
# Exit code 0 when Mamba imports afterwards, 1 otherwise (it never stops the caller by itself).
set -uo pipefail
PY="${1:?usage: bash install_mamba.sh /path/to/python}"
CONDA_PREFIX="$("$PY" -c 'import sys; print(sys.prefix)')"

# Mamba (the advanced model's temporal head). The ready-made mamba-ssm build for this torch / CUDA /
# Python is downloaded from its GitHub releases (with time limits and retries) and installed: no
# compiler is needed. Only if that fails is it compiled from source (needs nvcc; 20-60 minutes).
# If Mamba cannot be installed at all, the environment still works for the baseline model; the
# advanced model's check (part of every smoke and full run) then stops with a clear message.
MAMBA_VERSION=2.2.4
mamba_ok() { "$PY" -c 'import mamba_ssm, selective_scan_cuda' >/dev/null 2>&1; }
if ! mamba_ok; then
  # same name rule as mamba-ssm's own installer: cu<major>, torch<major.minor>, the C++ ABI, the Python tag
  read -r MAMBA_ASSET MAMBA_LOCAL < <("$PY" - "$MAMBA_VERSION" <<'EOF'
import platform, sys, torch
v, py = sys.argv[1], f'cp{sys.version_info.major}{sys.version_info.minor}'
tv = '.'.join(torch.__version__.split('+')[0].split('.')[:2])
cuda, abi = (torch.version.cuda or '0').split('.')[0], str(torch._C._GLIBCXX_USE_CXX11_ABI).upper()
tags = f'{py}-{py}-linux_{platform.machine()}'
print(f'mamba_ssm-{v}+cu{cuda}torch{tv}cxx11abi{abi}-{tags}.whl', f'mamba_ssm-{v}-{tags}.whl')
EOF
) || true
  MAMBA_DIR="$(mktemp -d "${TMPDIR:-/tmp}/lgel-mamba.XXXXXX")"
  if curl -fsSL --retry 5 --retry-delay 10 --retry-max-time 2400 --connect-timeout 30 --max-time 1800 \
       --speed-limit 51200 --speed-time 120 -o "$MAMBA_DIR/$MAMBA_LOCAL" \
       "https://github.com/state-spaces/mamba/releases/download/v$MAMBA_VERSION/$MAMBA_ASSET"; then
    "$PY" -m pip install --no-deps --force-reinstall "$MAMBA_DIR/$MAMBA_LOCAL" || true
  else
    echo "note: could not download the ready-made Mamba build $MAMBA_ASSET"
  fi
  rm -rf "$MAMBA_DIR"
fi
if ! mamba_ok && [ "${LGEL_MAMBA_NO_COMPILE:-0}" != "1" ] && command -v nvcc >/dev/null 2>&1; then
  echo "No ready-made Mamba build could be installed: compiling it from source (20-60 minutes) ..."
  "$PY" -m pip install "setuptools>=70,<85" "wheel>=0.43,<0.49" packaging ninja || true
  # each nvcc process needs several GB: at most 8 in parallel, to stay within the job's memory
  MAMBA_FORCE_BUILD=TRUE MAX_JOBS="${MAX_JOBS:-8}" "$PY" -m pip install --no-build-isolation --no-cache-dir \
    --no-deps --force-reinstall "mamba-ssm==$MAMBA_VERSION" || true
fi
if mamba_ok; then
  rm -f "$CONDA_PREFIX/.lgel_mamba_missing"
else
  echo "WARNING: the Mamba package (mamba-ssm $MAMBA_VERSION) could not be installed (no ready-made build" >&2
  echo "         downloadable from github.com, and no CUDA compiler to build it). The baseline model works;" >&2
  echo "         the advanced model is refused by its check until this is fixed. run.sh retries the" >&2
  echo "         download at the start of every (non-offline) job until it succeeds." >&2
  "$PY" -c 'import mamba_ssm, selective_scan_cuda' 2>&1 | tail -2 >&2 || true
  touch "$CONDA_PREFIX/.lgel_mamba_missing"
fi
mamba_ok
