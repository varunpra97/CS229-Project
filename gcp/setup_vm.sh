#!/usr/bin/env bash
# Run ON the GCP Deep Learning VM (PyTorch image) after you SSH in and have the
# repo checked out. Installs the extra Python deps into the preinstalled CUDA
# PyTorch environment WITHOUT touching torch, and verifies the GPU is visible.
#
#   cd ~/pacbayes-peft && bash gcp/setup_vm.sh
set -euo pipefail
cd "$(dirname "$0")/.."

echo "== GPU =="
nvidia-smi || { echo "nvidia-smi failed: GPU/driver not ready"; exit 1; }

echo "== torch / CUDA (preinstalled) =="
python -c "import torch; print('torch', torch.__version__, '| cuda available:', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no gpu')"

echo "== installing extra deps =="
pip install --upgrade pip
pip install -r gcp/requirements-extra.txt

echo "== import check =="
python -c "import transformers, peft, datasets, evaluate, matplotlib; print('deps OK:', transformers.__version__, peft.__version__)"
echo "setup_vm.sh complete."
