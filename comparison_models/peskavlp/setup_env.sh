#!/usr/bin/env bash
# Create (once) the separate Python environment for the PeskaVLP comparison and install the official
# SurgVLP/PeskaVLP code (CAMMA, https://github.com/CAMMA-public/SurgVLP, pinned commit) into it.
#
#   bash comparison_models/peskavlp/setup_env.sh /path/to/envs/lgel-peskavlp
#
# PeskaVLP pins transformers 4.30.2 and numpy < 2, which would break the main environment, so it lives in
# its own conda environment next to it. Like setup_env.sh, it never reads or changes the user's personal
# packages (~/.local), and it is marked complete only at the very end (a re-run finishes an interrupted
# installation). Called automatically by the paper preset; needs internet.
set -euo pipefail
export PYTHONNOUSERSITE=1
unset PYTHONPATH PYTHONHOME PIP_TARGET PIP_PREFIX PIP_ROOT
export PIP_USER=0
PREFIX="${1:?usage: bash setup_env.sh ENV_PREFIX}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMMIT=85e858998eec47614614b89f6af1363e0ad3f47b
TORCH_INDEX="${LGEL_TORCH_INDEX:-https://download.pytorch.org/whl/cu124}"
STAMP="$PREFIX/.lgel_peskavlp_ready"
WANT="$(cat "$HERE/setup_env.sh" "$HERE/requirements.txt" | sha256sum | cut -c1-64)"

if [ -f "$STAMP" ] && [ "$(cat "$STAMP")" = "$WANT" ] && \
   "$PREFIX/bin/python" -c 'import surgvlp, torch, transformers' >/dev/null 2>&1; then
  echo "PeskaVLP environment ready: $PREFIX"
  exit 0
fi

if [ ! -x "$PREFIX/bin/python" ]; then
  if [ -e "$PREFIX" ]; then
    echo "$PREFIX exists but has no python (interrupted creation?) - moving it aside"
    mv "$PREFIX" "$PREFIX.broken-$(date +%Y%m%d-%H%M%S)"
  fi
  CONDA="${CONDA_EXE:-$(command -v conda || true)}"
  if [ -n "$CONDA" ]; then
    mkdir -p "$(dirname "$PREFIX")"
    "$CONDA" create -y -p "$PREFIX" -c conda-forge --override-channels python=3.11 pip
  elif command -v python3.11 >/dev/null 2>&1; then          # no conda (e.g. a workstation): a virtualenv
    python3.11 -m venv "$PREFIX"
  else
    echo "ERROR: neither conda nor python3.11 is available to create the PeskaVLP environment." >&2
    exit 1
  fi
fi
PY="$PREFIX/bin/python"
rm -f "$STAMP"
if ! "$PY" -m pip --version >/dev/null 2>&1; then
  "$PY" -m ensurepip --upgrade
fi
if ! PIP_LIST="$(PIP_QUIET=0 PIP_GLOBAL=0 PIP_SITE=0 "$PY" -m pip config list 2>&1)"; then
  echo "ERROR: could not read pip's configuration:" >&2; echo "$PIP_LIST" >&2; exit 1
fi
PIP_LOCATIONS="$(printf '%s\n' "$PIP_LIST" | grep -E "^(global|install|:env:)[.](target|prefix|root)=('[^']|\"[^\"])" || true)"
if [ -n "$PIP_LOCATIONS" ]; then
  echo "ERROR: pip's configuration sets an install location (outside this environment):" >&2
  echo "$PIP_LOCATIONS" >&2; exit 1
fi

"$PY" -m pip install "torch==2.5.1" "torchvision==0.20.1" --index-url "$TORCH_INDEX"
"$PY" -m pip install -r "$HERE/requirements.txt"
# mmengine only provides the model registry/config here; without its dependencies, so that it does not
# pull in opencv-python (the GUI build needs system libraries that compute nodes often lack).
"$PY" -m pip install --no-deps "mmengine==0.7.0"
# The official code at a fixed commit, without its own (looser) requirements: GitHub's archive of that
# commit, or (if a proxy refuses archive downloads) the same commit through git. LGEL_SURGVLP_SOURCE can
# name a local copy instead (a checkout of that commit, or its zip). Its setup.py needs pkg_resources,
# which current setuptools no longer has, so it is built with the pinned setuptools of requirements.txt
# (--no-build-isolation) instead of whatever pip would fetch.
SURGVLP=(--no-deps --no-build-isolation)
if [ -n "${LGEL_SURGVLP_SOURCE:-}" ]; then
  "$PY" -m pip install "${SURGVLP[@]}" "$LGEL_SURGVLP_SOURCE"
elif ! "$PY" -m pip install "${SURGVLP[@]}" "https://github.com/CAMMA-public/SurgVLP/archive/$COMMIT.zip"; then
  echo "the archive download failed; fetching the same commit with git instead"
  "$PY" -m pip install "${SURGVLP[@]}" "git+https://github.com/CAMMA-public/SurgVLP.git@$COMMIT"
fi
"$PY" - "$COMMIT" <<'EOF'
import sys, importlib.metadata as md
dist = md.distribution('surgvlp')
direct = dist.read_text('direct_url.json') or ''
# pip records where a package came from; the commit must be the pinned one (an archive URL or a git commit).
if sys.argv[1] not in direct and 'file://' not in direct:
    sys.exit(f'surgvlp was installed from an unexpected source: {direct}')
EOF

"$PY" - <<'EOF'
import torch, transformers, numpy, surgvlp
print('PeskaVLP environment: torch', torch.__version__, '| transformers', transformers.__version__,
      '| numpy', numpy.__version__, '| surgvlp models', surgvlp.available_models())
EOF
echo "$WANT" > "$STAMP"
echo "OK: PeskaVLP environment ready: $PREFIX"
