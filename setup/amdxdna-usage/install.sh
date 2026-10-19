#!/usr/bin/env bash
# Build, sign (DKMS uses the enrolled MOK) and load the amdxdna driver with NPU usage reporting.
# Needs root; nothing may have /dev/accel/accel0 open while the module is swapped.
set -euo pipefail
[[ $EUID -eq 0 ]] || exec sudo -- "$0" "$@"

NAME=amdxdna-usage
VER=1.0
HERE=$(cd "$(dirname "$0")" && pwd)
KVER=$(uname -r)
PCI=/sys/bus/pci/drivers/amdxdna

npu_holders() {
    local p
    for p in /proc/[0-9]*; do
        ls -l "$p/fd" 2>/dev/null | grep -q '/dev/accel/' && echo "  pid ${p#/proc/} $(cat "$p/comm" 2>/dev/null)"
    done
    return 0
}

[[ $KVER =~ ^7\.2\. ]] || { echo "This source is the 7.2.6 driver; running kernel is $KVER." >&2; exit 1; }

holders=$(npu_holders)
if [[ -n $holders ]]; then
    echo "The NPU is in use; stop these first (docker kill / exit ComfyUI etc.):" >&2
    echo "$holders" >&2
    exit 1
fi

dkms remove "$NAME/$VER" --all >/dev/null 2>&1 || true
rm -rf "/usr/src/$NAME-$VER"
cp -r "$HERE/dkms" "/usr/src/$NAME-$VER"
dkms add "$NAME/$VER"
dkms install "$NAME/$VER" -k "$KVER"

echo "Reloading amdxdna..."
modprobe -r amdxdna
if ! modprobe amdxdna; then
    echo "Patched module failed to load (see dmesg); restoring the stock driver." >&2
    dkms remove "$NAME/$VER" --all || true
    modprobe amdxdna
    exit 1
fi

dev=$(ls -d "$PCI"/0000:* 2>/dev/null | head -1)
for _ in $(seq 50); do [[ -e $dev/npu_busy_percent ]] && break; sleep 0.1; done
if [[ -e $dev/npu_busy_percent ]]; then
    echo "OK: $(modinfo -F filename amdxdna)"
    echo "    npu_busy_percent=$(cat "$dev/npu_busy_percent")  npu_busy_time_us=$(cat "$dev/npu_busy_time_us")  npu_memory_utilization=$(cat "$dev/npu_memory_utilization")"
    echo "Restart Resources to pick up the new files."
else
    echo "Module loaded but $dev/npu_busy_percent is missing; check dmesg." >&2
    exit 1
fi
