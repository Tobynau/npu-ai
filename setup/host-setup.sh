#!/usr/bin/env bash
# Native (no Docker) setup for the AMD XDNA1 NPU on Ubuntu 26.04. Needs sudo once.
# After running: log out and back in (memlock), then `source setup/env.sh`.
set -euo pipefail
echo "[1/3] XRT user-space runtime from the Ubuntu archive"
sudo apt-get update
sudo apt-get install -y libxrt2 libxrt-npu2 libxrt-utils libxrt-utils-npu libxrt-dev python3-xrt \
     python3-venv python3-dev build-essential clang lld cmake ninja-build git uuid-dev
echo "[2/3] memlock limit (XRT pins a 64 MB device heap; default hard limit is 8 MB)"
sudo tee /etc/security/limits.d/99-amdxdna.conf >/dev/null << 'LIM'
* soft memlock unlimited
* hard memlock unlimited
LIM
# PipeWire ships 20-pw-defaults.conf with "memlock 8192"; make sure ours sorts later and wins.
echo "[3/3] group membership"
sudo usermod -aG render "$USER"
echo "Done. Log out and back in, then: ulimit -l   (must print 'unlimited') and xrt-smi examine"
