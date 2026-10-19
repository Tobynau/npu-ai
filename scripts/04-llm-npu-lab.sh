#!/usr/bin/env bash
# LLM test 2: Pedroj-64/npu-lab, where the NPU generates every token (int8 matvec kernels, 4 columns).
# Run inside the container: docker/run.sh scripts/04-llm-npu-lab.sh [prompt]
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NL="$ROOT/reference/npu-lab"
cd "$NL"
mkdir -p third_party models
[ -e third_party/mlir-aie ] || ln -sfn "$ROOT/reference/mlir-aie" third_party/mlir-aie
[ -e models/qwen3s ] || ln -sfn "$ROOT/models/qwen3s" models/qwen3s
export PYTHONUNBUFFERED=1
echo "== kernels needed by Qwen3-0.6B:"; SHAPES=$(python src/qwen_npu.py --model models/qwen3s --shapes); echo "$SHAPES"
echo; echo "== building int8 4-column matvec kernels (first time only, a minute or two)"; time python scripts/build_matvec.py --shared4 $SHAPES
echo; echo "== building bf16 prefill matmul kernels"; time python scripts/build_prefill.py models/qwen3s
echo; echo "== generation on the NPU"; time python src/qwen_npu.py --model models/qwen3s --tokens 48 --profile "${1:-Explain in two sentences what an NPU is and why it is power efficient.}"
