#!/usr/bin/env bash
# Native Python env (after setup/host-setup.sh): Python 3.14 venv with mlir-aie 1.4.4 + Peano + system pyxrt + torch CPU.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
python3.14 -m venv .venv
.venv/bin/pip install -U pip wheel setuptools numpy
.venv/bin/pip install "mlir_aie==1.4.4" -f https://github.com/Xilinx/mlir-aie/releases/expanded_assets/v1.4.4
.venv/bin/pip install "llvm-aie==22.0.0.2026092101+0006955e" -f https://github.com/Xilinx/llvm-aie/releases/expanded_assets/nightly
.venv/bin/pip install scipy tokenizers safetensors huggingface_hub transformers sentencepiece pyyaml requests
.venv/bin/pip install torch torchvision torchsde --index-url https://download.pytorch.org/whl/cpu
ln -sf /usr/lib/python3/dist-packages/pyxrt*.so .venv/lib/python3.14/site-packages/
.venv/bin/python -c "import pyxrt, aie; print('pyxrt + mlir-aie OK')"
echo "now: source setup/env.sh"
