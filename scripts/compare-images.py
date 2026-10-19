#!/usr/bin/env python3
"""Pixel difference between two images (same seed, CPU vs NPU). usage: compare-images.py a.png b.png"""
import sys, numpy as np
from PIL import Image
a = np.asarray(Image.open(sys.argv[1]).convert("RGB"), dtype=np.float32)
b = np.asarray(Image.open(sys.argv[2]).convert("RGB"), dtype=np.float32)
d = np.abs(a - b)
print(f"mean |diff| {d.mean():.2f}/255, median {np.median(d):.1f}, 95th pct {np.percentile(d,95):.1f}, "
      f"pixels within 8/255 on all channels: {(d.max(-1) <= 8).mean()*100:.1f}%")
