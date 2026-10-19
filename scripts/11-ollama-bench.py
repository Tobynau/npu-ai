#!/usr/bin/env python3
"""Stream /api/generate requests against an Ollama server and report TTFT, decode and prefill rates.

Used by scripts/11-llm-hawkpoint-ollama-colibri.sh for the CPU-only vs NPU placement comparison of the
hawkpoint-npu-llm Ollama XDNA1 patch. Stdlib only (the hawkpoint image's venv has no requests).

  11-ollama-bench.py --url http://127.0.0.1:11435 --model qwen2.5:0.5b --requests 6 --tokens 32 \
      --label npu --report logs/11-ollama-bench-npu.json

TTFT is wall time from the request to the first streamed token (prompt processing + any model load);
decode tok/s = eval_count / eval_duration and prefill tok/s = prompt_eval_count / prompt_eval_duration
come from Ollama's own counters in the final chunk. Greedy (temperature 0, fixed seed) so the two
placements can be compared token by token.
"""
import argparse
import json
import statistics
import sys
import time
import urllib.request

PROMPTS = [
    "What is an NPU and why is it power efficient?",
    "Explain in two sentences why the sky appears blue.",
    "Write a haiku about a laptop that stays cool.",
    "List three uses of a neural processing unit.",
    "Describe the difference between prefill and decode in a language model.",
    "Why do Ryzen AI chips have a separate AI engine next to the CPU?",
    "Give one reason a small language model can run on a laptop.",
    "What does a matrix-vector product have to do with text generation?",
]


def generate(url, model, prompt, tokens, threads):
    body = {
        "model": model,
        "prompt": prompt,
        "stream": True,
        "options": {"num_predict": tokens, "temperature": 0, "seed": 1, "num_thread": threads},
    }
    req = urllib.request.Request(
        url + "/api/generate", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
    )
    t0 = time.perf_counter()
    first = None
    text = []
    final = None
    with urllib.request.urlopen(req, timeout=900) as r:
        for line in r:
            d = json.loads(line)
            if d.get("response") and first is None:
                first = time.perf_counter()
            text.append(d.get("response", ""))
            if d.get("done"):
                final = d
                break
    t1 = time.perf_counter()
    if final is None:
        raise RuntimeError("stream ended without a done chunk (server stopped?)")
    ev = final.get("eval_count", 0)
    evd = final.get("eval_duration", 0)
    pc = final.get("prompt_eval_count", 0)
    pcd = final.get("prompt_eval_duration", 0)
    return {
        "prompt": prompt,
        "text": "".join(text),
        "ttft_s": (first - t0) if first else None,
        "wall_s": t1 - t0,
        "eval_count": ev,
        "decode_tps": ev / (evd / 1e9) if evd else None,
        "prompt_eval_count": pc,
        "prefill_tps": pc / (pcd / 1e9) if pcd else None,
        "load_s": final.get("load_duration", 0) / 1e9,
        "total_s": final.get("total_duration", 0) / 1e9,
    }


def ps(url):
    with urllib.request.urlopen(url + "/api/ps", timeout=30) as r:
        return json.load(r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:11435")
    ap.add_argument("--model", default="qwen2.5:0.5b")
    ap.add_argument("--requests", type=int, default=6)
    ap.add_argument("--tokens", type=int, default=32)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--label", default="run")
    ap.add_argument("--report", default=None)
    a = ap.parse_args()

    print(f"warm-up ({a.label}): loading {a.model} ...", flush=True)
    warm = generate(a.url, a.model, PROMPTS[0], 8, a.threads)
    print(f"  load {warm['load_s']:.1f} s, first request wall {warm['wall_s']:.1f} s", flush=True)
    status = ps(a.url)
    for m in status.get("models", []):
        print(
            f"  /api/ps: {m.get('name')} size={m.get('size')} size_vram={m.get('size_vram')} "
            f"npu={m.get('npu')} npu_percent={m.get('npu_percent')}",
            flush=True,
        )

    runs = []
    for i in range(a.requests):
        r = generate(a.url, a.model, PROMPTS[i % len(PROMPTS)], a.tokens, a.threads)
        runs.append(r)
        print(
            f"  [{i + 1}/{a.requests}] ttft {r['ttft_s']:.3f} s | decode {r['decode_tps']:.2f} tok/s "
            f"({r['eval_count']} tok) | prefill {r['prefill_tps']:.1f} tok/s ({r['prompt_eval_count']} tok) "
            f"| wall {r['wall_s']:.2f} s",
            flush=True,
        )
    med = lambda k: statistics.median([r[k] for r in runs if r[k] is not None])
    summary = {
        "label": a.label,
        "model": a.model,
        "requests": a.requests,
        "tokens": a.tokens,
        "threads": a.threads,
        "median_decode_tps": med("decode_tps"),
        "min_decode_tps": min(r["decode_tps"] for r in runs),
        "max_decode_tps": max(r["decode_tps"] for r in runs),
        "median_ttft_s": med("ttft_s"),
        "median_prefill_tps": med("prefill_tps"),
        "median_wall_s": med("wall_s"),
        "load_s": warm["load_s"],
        "ps": status,
        "runs": runs,
    }
    print(
        f"SUMMARY {a.label}: decode median {summary['median_decode_tps']:.2f} tok/s "
        f"({summary['min_decode_tps']:.2f}-{summary['max_decode_tps']:.2f}) | TTFT median {summary['median_ttft_s']:.3f} s "
        f"| prefill median {summary['median_prefill_tps']:.1f} tok/s | load {warm['load_s']:.1f} s",
        flush=True,
    )
    print(f"sample ({a.label}): {json.dumps(runs[0]['text'])}", flush=True)
    if a.report:
        with open(a.report, "w") as f:
            json.dump(summary, f, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
