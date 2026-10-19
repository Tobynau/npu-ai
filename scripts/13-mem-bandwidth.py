# Memory bandwidth of the Radeon 760M iGPU and of the CPU cores, with torch (1 GiB fp32 buffers).
# Run on the HOST with the ROCm torch of ~/ComfyUI through the shim venv that scripts/12 creates:
#   HSA_OVERRIDE_GFX_VERSION=11.0.0 build/igpu-scratch/venv312/bin/python scripts/13-mem-bandwidth.py cuda
#   OMP_NUM_THREADS=6 build/igpu-scratch/venv312/bin/python scripts/13-mem-bandwidth.py cpu
import sys, time, torch

dev = sys.argv[1] if len(sys.argv) > 1 else "cuda"
n = 256 * 1024 * 1024  # 256M float32 = 1 GiB per buffer
a = torch.ones(n, dtype=torch.float32, device=dev)
b = torch.empty_like(a)
sync = torch.cuda.synchronize if dev == "cuda" else (lambda: None)
if dev == "cuda":
    print("device:", torch.cuda.get_device_name(0))
print(f"threads: {torch.get_num_threads()}" if dev == "cpu" else "")

def bench(name, fn, bytes_moved, reps=10):
    fn(); sync()
    ts = []
    for _ in range(reps):
        t = time.perf_counter(); fn(); sync(); ts.append(time.perf_counter() - t)
    best, med = min(ts), sorted(ts)[len(ts) // 2]
    print(f"{name:28s} best {bytes_moved / best / 1e9:6.1f} GB/s   median {bytes_moved / med / 1e9:6.1f} GB/s")

nb = n * 4
bench("read (sum)", lambda: a.sum(), nb)
bench("copy (read+write)", lambda: b.copy_(a), 2 * nb)
bench("write (fill)", lambda: b.fill_(2.0), nb)
