`pyxrt.cpp` is `src/python/pybind11/src/pyxrt.cpp` from the XRT tag `2.21.75`
(https://raw.githubusercontent.com/Xilinx/XRT/2.21.75/src/python/pybind11/src/pyxrt.cpp),
the same upstream version as Ubuntu 26.04's `libxrt2 1:2.21.75+dfsg-4`. Ubuntu's `python3-xrt`
only ships the module for CPython 3.14; `Dockerfile.hawkpoint` compiles this file against the
distro headers (`libxrt-dev`) and `libxrt_coreutil.so.2` for the CPython 3.12 venv that
hawkpoint-npu-llm pins. `build-hawkpoint.sh` re-downloads it if it is missing.
