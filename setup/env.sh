# source this on the host after setup/host-setup.sh and after creating .venv with setup/make-venv.sh
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/.venv/bin/activate"
export MLIR_AIE_INSTALL_DIR="$ROOT/.venv/lib/python3.14/site-packages/mlir_aie"
export PEANO_INSTALL_DIR="$ROOT/.venv/lib/python3.14/site-packages/llvm-aie"
export PATH="$MLIR_AIE_INSTALL_DIR/bin:$PEANO_INSTALL_DIR/bin:$PATH"
export LD_LIBRARY_PATH="$MLIR_AIE_INSTALL_DIR/lib:${LD_LIBRARY_PATH:-}"
