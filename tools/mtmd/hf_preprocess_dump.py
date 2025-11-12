#!/usr/bin/env python3
"""
hf_preprocess_dump.py
Create a planar C,H,W float32 dump matching the C++ E2VL preprocessing (resize->center crop->normalize SigLIP mean=0.5 std=0.5).
Usage: python3 hf_preprocess_dump.py input.jpg out.bin
"""
import sys
from PIL import Image
import numpy as np

if len(sys.argv) < 3:
    print("usage: hf_preprocess_dump.py INPUT_IMAGE OUT_BIN")
    sys.exit(2)

inp = sys.argv[1]
out = sys.argv[2]

image_size = 448
mean = np.array([0.5, 0.5, 0.5], dtype=np.float32)
std = np.array([0.5, 0.5, 0.5], dtype=np.float32)

img = Image.open(inp).convert('RGB')
# Resize short side to image_size, keep aspect, then center crop
w, h = img.size
if w < h:
    new_w = image_size
    new_h = int(round(h * (image_size / w)))
else:
    new_h = image_size
    new_w = int(round(w * (image_size / h)))
img = img.resize((new_w, new_h), resample=Image.BICUBIC)
# center crop
left = (new_w - image_size) // 2
top = (new_h - image_size) // 2
img = img.crop((left, top, left + image_size, top + image_size))
arr = np.array(img).astype(np.float32) / 255.0
# Normalize (SigLIP style): (x - mean)/std
arr = (arr - mean) / std
# Convert to planar C,H,W
planar = np.transpose(arr, (2,0,1)).astype(np.float32)
planar.tofile(out)
print(f"wrote {out}: shape={planar.shape}, bytes={planar.nbytes}")
