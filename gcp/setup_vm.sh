#!/usr/bin/env bash
# Run ON the GCP Deep Learning VM (PyTorch image) after you SSH in and have the
# repo checked out. Installs the extra Python deps into the preinstalled CUDA
# PyTorch environment WITHOUT touching torch, and verifies the GPU is visible.
#
#   cd ~/pacbayes-peft && bash gcp/setup_vm.sh
set -euo pipefail
cd "$(dirname "$0")/.."

# Newer Deep Learning images ship only `python3` (no conda `python` symlink);
# older conda images have `python`. Prefer whichever exists.
PY="$(command -v python || command -v python3)"
[ -n "$PY" ] || { echo "no python interpreter found"; exit 1; }
echo "using interpreter: $PY"

# The run scripts (run_task.sh etc.) call `python` directly. On images that only
# ship `python3`, create a symlink so those scripts work unchanged.
if ! command -v python >/dev/null 2>&1; then
  echo "no 'python' on PATH; symlinking python -> $PY"
  sudo ln -sf "$PY" /usr/local/bin/python
fi

echo "== GPU =="
nvidia-smi || { echo "nvidia-smi failed: GPU/driver not ready"; exit 1; }

echo "== torch / CUDA (preinstalled) =="
"$PY" -c "import torch; print('torch', torch.__version__, '| cuda available:', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no gpu')"

echo "== installing extra deps =="
"$PY" -m pip install --upgrade pip
"$PY" -m pip install -r gcp/requirements-extra.txt

echo "== import check =="
"$PY" -c "import transformers, peft, datasets, evaluate, matplotlib; print('deps OK:', transformers.__version__, peft.__version__)"
echo "setup_vm.sh complete."
