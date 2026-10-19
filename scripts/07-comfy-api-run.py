#!/usr/bin/env python3
"""Submit a txt2img workflow to a running ComfyUI (API), wait, save the image, report timing and package energy.
usage: 07-comfy-api-run.py --mode cpu|npu|igpu [--attention auto|npu|cpu] [--steps 8] [--ckpt NAME] [--size 512] [--seed 7] [--out outputs/comfy]
"""
import argparse, json, time, urllib.request, urllib.parse, os, sys

ap = argparse.ArgumentParser()
ap.add_argument("--mode", choices=["cpu", "npu", "igpu"], required=True,
                help="igpu = the plain workflow (as cpu) against a ComfyUI server running on the Radeon iGPU")
ap.add_argument("--attention", choices=["auto", "npu", "cpu"], default="auto",
                help="npu mode only: attention QK^T/PV matmuls: auto (gate picks per shape), npu (force), cpu (never)")
ap.add_argument("--server", default="http://127.0.0.1:8188")
ap.add_argument("--ckpt", default="Realistic_Vision_V6.0_NV_B1_fp16.safetensors")
ap.add_argument("--steps", type=int, default=8)
ap.add_argument("--size", type=int, default=512)
ap.add_argument("--seed", type=int, default=7)
ap.add_argument("--cfg", type=float, default=7.0)
ap.add_argument("--prompt", default="photo of a red fox sitting in a snowy forest at sunrise, detailed fur, soft light")
ap.add_argument("--negative", default="blurry, deformed, text, watermark, low quality")
ap.add_argument("--out", default="outputs/comfy")
a = ap.parse_args()

wf = {
    "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": a.ckpt}},
    "2": {"class_type": "CLIPTextEncode", "inputs": {"text": a.prompt, "clip": ["1", 1]}},
    "3": {"class_type": "CLIPTextEncode", "inputs": {"text": a.negative, "clip": ["1", 1]}},
    "4": {"class_type": "EmptyLatentImage", "inputs": {"width": a.size, "height": a.size, "batch_size": 1}},
    "6": {"class_type": "VAEDecode", "inputs": {"samples": ["5", 0], "vae": ["1", 2]}},
    "7": {"class_type": "SaveImage", "inputs": {"images": ["6", 0], "filename_prefix": f"xdna1_{a.mode}_{a.steps}st" + (f"_attn-{a.attention}" if a.mode == "npu" and a.attention != "auto" else "")}},
}
model_ref = ["1", 0]
if a.mode == "npu":
    wf["8"] = {"class_type": "XDNA1NpuLinearOffload",
               "inputs": {"model": ["1", 0], "enabled": True, "min_tokens": 64, "min_features": 320,
                          "skip_names": "time_embed,label_emb,emb_layers", "attention": a.attention,
                          "attn_min_tokens": 256}}
    model_ref = ["8", 0]
    wf["9"] = {"class_type": "XDNA1NpuStatus", "inputs": {"trigger": ["6", 0]}}
wf["5"] = {"class_type": "KSampler", "inputs": {"model": model_ref, "positive": ["2", 0], "negative": ["3", 0],
           "latent_image": ["4", 0], "seed": a.seed, "steps": a.steps, "cfg": a.cfg,
           "sampler_name": "euler", "scheduler": "normal", "denoise": 1.0}}

def api(path, data=None):
    req = urllib.request.Request(a.server + path, data=json.dumps(data).encode() if data is not None else None,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())

# SoC package energy (RAPL; on this APU the counter covers the CPU cores, iGPU and NPU together)
RAPL = "/sys/class/powercap/intel-rapl:0"
def energy_uj():
    try:
        return int(open(RAPL + "/energy_uj").read())
    except (OSError, ValueError):
        return None

e0 = energy_uj()
t0 = time.perf_counter()
pid = api("/prompt", {"prompt": wf, "client_id": "npu-ai"})["prompt_id"]
print(f"[{a.mode}] queued prompt {pid}", flush=True)
hist = None
while True:
    time.sleep(0.5)
    h = api(f"/history/{pid}")
    if pid in h:
        hist = h[pid]; break
    if time.perf_counter() - t0 > 3 * 3600:
        sys.exit("timeout")
elapsed = time.perf_counter() - t0
e1 = energy_uj()
energy = ""
if e0 is not None and e1 is not None:
    if e1 < e0:
        e1 += int(open(RAPL + "/max_energy_range_uj").read())
    j = (e1 - e0) / 1e6
    energy = f", package energy {j:.0f} J (avg {j / elapsed:.1f} W)"
st = hist.get("status", {})
if st.get("status_str") == "error":
    print(json.dumps(st, indent=1)[:3000]); sys.exit(f"[{a.mode}] prompt failed")
os.makedirs(a.out, exist_ok=True)
saved = []
for node_id, out in hist.get("outputs", {}).items():
    for im in out.get("images", []):
        q = urllib.parse.urlencode({"filename": im["filename"], "subfolder": im.get("subfolder", ""), "type": im.get("type", "output")})
        data = urllib.request.urlopen(f"{a.server}/view?{q}", timeout=60).read()
        p = os.path.join(a.out, im["filename"]); open(p, "wb").write(data); saved.append(p)
    for t in out.get("text", []):
        print(f"[{a.mode}] status node: {t}")
print(f"[{a.mode}] done in {elapsed:.1f}s total ({a.steps} steps, {a.size}x{a.size}, cfg {a.cfg}){energy}; images: {saved}", flush=True)
