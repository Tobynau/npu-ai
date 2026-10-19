# amdxdna-usage: make GNOME Resources show real NPU usage

GNOME Resources (1.10.2) shows "N/A" for the NPU's Total Usage and Memory Usage because the
in-tree `amdxdna` driver does not report either. Resources looks for:

| Resources reads | for | stock 7.2.6 driver |
|---|---|---|
| `/sys/class/accel/accel0/device/npu_busy_percent` | Total Usage | missing |
| `/sys/class/accel/accel0/device/npu_memory_utilization` | Memory Usage | missing |
| `drm-engine-npu-amdxdna: <n> ns` in `/proc/<pid>/fdinfo/<fd>` | per-process NPU column | missing |

There is no hardware counter to read on Phoenix: `DRM_AMDXDNA_QUERY_SENSORS` (per-column
utilization from AMD PMF) returns `EOPNOTSUPP` on the 7640U. So the patch in `dkms/patches/` makes
the driver measure busy time itself, the way GPU drivers do. A command counts as busy from the
moment it is posted to firmware until firmware responds. Each command is weighted by the columns
its hardware context occupies, out of the 4 a context can use on XDNA1, so one 4-column GEMM
kernel running flat out reads 100% and a 1-column IRON test reads 25%. Overlapping contexts on
the same columns are counted once.

New sysfs files on the PCI device (`/sys/bus/pci/devices/0000:c2:00.1/`):

- `npu_busy_time_us`: cumulative busy time, monotonic (same meaning as Intel `ivpu`)
- `npu_busy_percent`: busy % since the previous read at least 100 ms ago
- `npu_memory_utilization`: bytes of buffer objects held by all NPU clients

Not covered: temperature and power. Nothing on this platform exposes them.

## Install

DKMS builds the in-tree v7.2.6 source plus the patch for the running kernel, signs it with the
enrolled MOK (`/var/lib/shim-signed/mok/MOK.der`, already trusted by the kernel), and installs it
to `updates/dkms`, which takes priority over the in-tree module. Stop anything using the NPU first;
the script refuses to run while `/dev/accel/accel0` is open.

```bash
setup/amdxdna-usage/install.sh
```

Then restart Resources. It only builds for 7.2.x kernels (`BUILD_EXCLUSIVE_KERNEL`); on other
kernels the stock driver loads and Resources goes back to N/A.

## Check it

```bash
watch -n1 cat /sys/bus/pci/devices/0000:c2:00.1/npu_busy_percent
```

Run any NPU job (for example `docker/run.sh python3 scripts/05-gemm-test.py`) and the number
should climb, then drop back to 0 when it finishes.

## Remove

```bash
setup/amdxdna-usage/uninstall.sh
```
