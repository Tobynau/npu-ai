#!/usr/bin/env bash
# Build the npu-ai image. Logs to docker/build.log.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
docker build --build-arg UID="$(id -u)" --build-arg GID="$(id -g)" -t npu-ai:latest . 2>&1 | tee build.log
