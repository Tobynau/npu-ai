#!/usr/bin/env bash
# Build npu-ai-hawkpoint:latest (Python 3.12 + mlir-aie 1.3.5.dev115+gc69e4c3 + Peano 2026-07-20 + pyxrt for 3.12) on top of
# npu-ai:latest, for reference/hawkpoint-npu-llm. Logs to docker/build-hawkpoint.log.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
docker image inspect npu-ai:latest >/dev/null 2>&1 || { echo "build npu-ai:latest first (docker/build.sh)" >&2; exit 1; }
cp ../reference/hawkpoint-npu-llm/requirements.lock hawkpoint-requirements.lock
# per-commit CI wheel closest to the project's pinned mlir-aie commit 57d7494e99c (see Dockerfile.hawkpoint)
WHL=mlir_aie-1.3.5.dev115+gc69e4c3-cp312-cp312-manylinux_2_35_x86_64.whl
[ -f "wheels/$WHL" ] || curl -fL -o "wheels/$WHL" "https://github.com/Xilinx/mlir-aie/releases/download/latest-wheels-4/${WHL/+/%2B}"
[ -f pyxrt/pyxrt.cpp ] || curl -fL -o pyxrt/pyxrt.cpp "https://raw.githubusercontent.com/Xilinx/XRT/2.21.75/src/python/pybind11/src/pyxrt.cpp"
docker build -f Dockerfile.hawkpoint -t npu-ai-hawkpoint:latest . 2>&1 | tee build-hawkpoint.log
