# NPU dev environment for AMD XDNA1 (Phoenix) on Linux.
# Uses Ubuntu 26.04's own XRT packages; the kernel driver (amdxdna) comes from the host.
FROM ubuntu:26.04
ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
      libxrt2 libxrt-npu2 libxrt-utils libxrt-utils-npu libxrt-dev python3-xrt \
      python3 python3-venv python3-dev python3-pip \
      build-essential clang lld cmake ninja-build git git-lfs curl ca-certificates \
      libcurl4-openssl-dev pkg-config uuid-dev kmod pciutils sudo \
    && rm -rf /var/lib/apt/lists/*

# Python 3.14 venv with mlir-aie (IRON) + Peano (llvm-aie) + the system pyxrt
RUN python3 -m venv /opt/venv && /opt/venv/bin/pip install --no-cache-dir -U pip wheel setuptools numpy
COPY wheels/ /tmp/wheels/
RUN /opt/venv/bin/pip install --no-cache-dir /tmp/wheels/mlir_aie-1.4.4-cp314-cp314-manylinux_2_35_x86_64.whl && rm -rf /tmp/wheels
ARG PEANO_PIN=llvm-aie==22.0.0.2026092101+0006955e
RUN /opt/venv/bin/pip install --no-cache-dir "${PEANO_PIN}" \
      -f https://github.com/Xilinx/llvm-aie/releases/expanded_assets/nightly
RUN /opt/venv/bin/pip install --no-cache-dir scipy tokenizers safetensors huggingface_hub transformers sentencepiece pyyaml requests \
    && /opt/venv/bin/pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN ln -sf /usr/lib/python3/dist-packages/pyxrt*.so /opt/venv/lib/python3.14/site-packages/ \
    && /opt/venv/bin/python -c "import pyxrt, aie; print('pyxrt + aie import OK')"

# ComfyUI dependencies (CPU torch already installed above; torchvision/torchsde from the same CPU index)
COPY comfy-requirements.txt /tmp/comfy-requirements.txt
RUN /opt/venv/bin/pip install --no-cache-dir -r /tmp/comfy-requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu \
    && rm /tmp/comfy-requirements.txt

# Non-root user matching the host user so files written to /work keep the right owner
ARG UID=1000
ARG GID=1000
RUN userdel -r ubuntu 2>/dev/null || true; groupadd -g ${GID} npu && useradd -m -u ${UID} -g ${GID} -s /bin/bash npu \
    && echo "npu ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/npu

ENV VIRTUAL_ENV=/opt/venv
ENV MLIR_AIE_INSTALL_DIR=/opt/venv/lib/python3.14/site-packages/mlir_aie
ENV PEANO_INSTALL_DIR=/opt/venv/lib/python3.14/site-packages/llvm-aie
ENV PATH=/opt/venv/bin:${MLIR_AIE_INSTALL_DIR}/bin:${PEANO_INSTALL_DIR}/bin:${PATH}
ENV LD_LIBRARY_PATH=${MLIR_AIE_INSTALL_DIR}/lib
ENV HF_HOME=/hf
USER npu
WORKDIR /work
CMD ["/bin/bash"]
