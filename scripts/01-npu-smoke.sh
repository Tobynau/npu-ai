#!/usr/bin/env bash
# Basic NPU functionality test. Run inside the container:  docker/run.sh scripts/01-npu-smoke.sh
# 1) XRT sees the NPU  2) pyxrt opens it  3) IRON compiles+runs a vector add on it  4) a matmul on all 4 columns
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
echo "== ulimit -l (must be unlimited or >= 65536): $(ulimit -l)"
echo; echo "== 1. xrt-smi examine"; xrt-smi examine 2>&1 | sed -n '1,40p'
echo; echo "== 2. pyxrt"
python3 - << 'PY'
import pyxrt
d = pyxrt.device(0)
print("device 0 name:", d.get_info(pyxrt.xrt_info_device.name))
print("bdf:", d.get_info(pyxrt.xrt_info_device.bdf))
PY
echo; echo "== 3. IRON vector_scalar_add (compile with Peano, run on npu1, verify)"
EX=reference/mlir-aie/programming_examples/basic
( cd $EX/vector_scalar_add && time python3 vector_scalar_add.py ) 2>&1 | tail -8
echo; echo "== 4. IRON whole_array matmul 512x512x512 i16 on 4 columns (verify + time)"
( cd $EX/matrix_multiplication/whole_array && time python3 whole_array.py -M 512 -K 512 -N 512 --dtype_in i16 --dtype_out i32 --n-aie-cols 4 ) 2>&1 | tail -8
echo; echo "== smoke test finished"
