#!/usr/bin/env python3
# Standalone reference for the MiniMax-M3 vision tower + projector.
# No transformers imports: ports modeling_minimax_m3_vl.py (vision classes) and
# image_processing_minimax_m3_vl.py (smart_resize + patch flatten) directly.
# Loads vision weights from the two safetensors shards (26 & 59) and dumps a
# parity .npz that the clip.cpp graph must reproduce.
#
# usage:  python m3_vision_ref.py [image.png] [model_dir_with_shards]
#   - no model_dir  -> downloads shards 26 & 59 from HF (~10 GB, cached)
#   - no image      -> generates a gradient test.png so it runs out of the box

import os, sys, math, glob
import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open

IMG       = sys.argv[1] if len(sys.argv) > 1 else "test.png"
MODEL_DIR = sys.argv[2] if len(sys.argv) > 2 else None
REPO      = "MiniMaxAI/MiniMax-M3"
SHARDS    = ["model-00026-of-00059.safetensors", "model-00059-of-00059.safetensors"]

# ---- config (from config.json: vision_config) ----
PATCH, TPS, MERGE = 14, 2, 2
HID, HEADS, LAYERS, FFN = 1280, 16, 32, 5120
HEAD_DIM = HID // HEADS                      # 80
EPS, ROPE_THETA = 1e-5, 10000.0
MAX_PIXELS, MIN_PIXELS = 451584, 4 * 28 * 28
MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1)
STD  = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1)
VIS_PREFIXES = ("vision_tower.", "multi_modal_projector.", "patch_merge_mlp.")

torch.manual_seed(0)

# ---- locate / load vision weights (fp32) ----
if MODEL_DIR:
    shard_paths = glob.glob(os.path.join(MODEL_DIR, "*.safetensors"))
else:
    from huggingface_hub import hf_hub_download
    shard_paths = [hf_hub_download(REPO, s) for s in SHARDS]

W = {}
for sh in shard_paths:
    with safe_open(sh, framework="pt") as f:
        for k in f.keys():
            if k.startswith(VIS_PREFIXES):
                W[k] = f.get_tensor(k).float()
assert W, "no vision tensors found in shards"
g = lambda n: W[n]
lin = lambda x, p: F.linear(x, g(p + ".weight"), g(p + ".bias"))
ln  = lambda x, p: F.layer_norm(x, (x.shape[-1],), g(p + ".weight"), g(p + ".bias"), EPS)

# ---- preprocessing (port of image_processing_minimax_m3_vl) ----
def smart_resize(h, w, factor=PATCH * MERGE, max_pixels=MAX_PIXELS, min_pixels=MIN_PIXELS):
    hb = max(factor, round(h / factor) * factor)
    wb = max(factor, round(w / factor) * factor)
    if hb * wb > max_pixels:
        beta = math.sqrt(h * w / max_pixels)
        hb = math.floor(h / beta / factor) * factor
        wb = math.floor(w / beta / factor) * factor
    elif hb * wb < min_pixels:
        beta = math.sqrt(min_pixels / (h * w))
        hb = math.ceil(h * beta / factor) * factor
        wb = math.ceil(w * beta / factor) * factor
    return hb, wb

from PIL import Image
if not os.path.exists(IMG):
    a = np.zeros((448, 448, 3), np.uint8)
    a[..., 0] = np.linspace(0, 255, 448)[None, :]
    a[..., 1] = np.linspace(0, 255, 448)[:, None]
    a[..., 2] = 128
    Image.fromarray(a).save("test.png"); IMG = "test.png"

img = Image.open(IMG).convert("RGB")
x = torch.from_numpy(np.array(img)).permute(2, 0, 1).float()      # [3,H,W]
Hr, Wr = smart_resize(x.shape[1], x.shape[2])
x = F.interpolate(x.unsqueeze(0), size=(Hr, Wr), mode="bicubic",
                  align_corners=False, antialias=True).squeeze(0)
x = (x / 255.0 - MEAN) / STD                                      # [3,Hr,Wr]
x = x.unsqueeze(0)                                                # [T=1,3,Hr,Wr]
if x.shape[0] % TPS:                                              # pad T to TPS
    x = torch.cat([x, x[-1:].repeat(TPS - x.shape[0] % TPS, 1, 1, 1)], 0)
x = x.unsqueeze(0)                                                # [B=1,T=2,3,Hr,Wr]
gt, gh, gw = x.shape[1] // TPS, Hr // PATCH, Wr // PATCH
x = x.view(1, gt, TPS, 3, gh // MERGE, MERGE, PATCH, gw // MERGE, MERGE, PATCH)
x = x.permute(0, 1, 4, 7, 5, 8, 3, 2, 6, 9)
pixel_values = x.reshape(1, gt * gh * gw, 3 * TPS * PATCH * PATCH).squeeze(0)  # [N,1176]
grid_thw = torch.tensor([[gt, gh, gw]])
N = gt * gh * gw

# ---- 3D RoPE (port of MiniMaxM3VL3DRotaryEmbedding) ----
def rope_cos_sin(grid_thw):
    axis_dim = 2 * ((2 * (HEAD_DIM // 2) // 3) // 2)              # 26
    m, coords = MERGE, []
    for t, h, w in grid_thw.tolist():
        hi = torch.arange(h).unsqueeze(1).expand(-1, w).reshape(h // m, m, w // m, m).permute(0, 2, 1, 3).flatten()
        wi = torch.arange(w).unsqueeze(0).expand(h, -1).reshape(h // m, m, w // m, m).permute(0, 2, 1, 3).flatten()
        ti = torch.arange(t).repeat_interleave(h * w)
        coords.append(torch.stack([ti, hi.repeat(t), wi.repeat(t)], -1))
    coords = torch.cat(coords).float()
    inv = 1.0 / (ROPE_THETA ** (torch.arange(0, axis_dim, 2).float() / axis_dim))  # [13]
    freqs = torch.cat([coords[:, i:i + 1] * inv for i in range(3)], -1)            # [N,39]
    emb = torch.cat([freqs, freqs], -1)                                           # [N,78]
    return emb.cos(), emb.sin()

def rotate_half(t):
    a, b = t[..., :t.shape[-1] // 2], t[..., t.shape[-1] // 2:]
    return torch.cat([-b, a], -1)

def apply_rope(q, k, cos, sin):          # q,k: [N,heads,80]; cos,sin: [N,78]
    rd = cos.shape[-1]
    c, s = cos[:, None, :], sin[:, None, :]
    def f(t):
        tr, tp = t[..., :rd], t[..., rd:]
        return torch.cat([tr * c + rotate_half(tr) * s, tp], -1)
    return f(q), f(k)

# ---- vision tower forward ----
pe = g("vision_tower.vision_model.embeddings.patch_embedding.weight")   # [1280,3,2,14,14]
h = pixel_values.view(-1, 3, TPS, PATCH, PATCH)
h = F.conv3d(h, pe, stride=(TPS, PATCH, PATCH)).view(-1, HID)           # [N,1280]
h = ln(h, "vision_tower.vision_model.pre_layrnorm")
cos, sin = rope_cos_sin(grid_thw)
for i in range(LAYERS):
    p = f"vision_tower.vision_model.encoder.layers.{i}."
    r = h
    c = ln(h, p + "layer_norm1")
    q = lin(c, p + "self_attn.q_proj").view(-1, HEADS, HEAD_DIM)
    k = lin(c, p + "self_attn.k_proj").view(-1, HEADS, HEAD_DIM)
    v = lin(c, p + "self_attn.v_proj").view(-1, HEADS, HEAD_DIM)
    q, k = apply_rope(q, k, cos, sin)
    q, k, v = (t.permute(1, 0, 2).unsqueeze(0) for t in (q, k, v))      # [1,heads,N,80]
    o = F.scaled_dot_product_attention(q, k, v).squeeze(0)             # [heads,N,80]
    o = lin(o.permute(1, 0, 2).reshape(-1, HID), p + "self_attn.out_proj")
    h = r + o
    r = h
    c = ln(h, p + "layer_norm2")
    c = lin(F.gelu(lin(c, p + "mlp.fc1")), p + "mlp.fc2")
    h = r + c
last_hidden = h                                                        # [N,1280]

# ---- projector (per-patch MLP -> group-4 -> merge MLP) ----
pp = lin(F.gelu(lin(last_hidden, "multi_modal_projector.linear_1")),
         "multi_modal_projector.linear_2")                            # [N,6144]
pp = pp.reshape(pp.shape[0] // (MERGE * MERGE), -1)                    # [N/4,24576]
proj_out = lin(F.gelu(lin(pp, "patch_merge_mlp.linear_1")),
               "patch_merge_mlp.linear_2")                            # [N/4,6144]

np.savez("m3_vis_parity.npz",
         pixel_values=pixel_values.numpy(), grid_thw=grid_thw.numpy(),
         cos=cos.numpy(), sin=sin.numpy(),
         last_hidden=last_hidden.numpy(), proj_out=proj_out.numpy())

print(f"grid_thw = {grid_thw.tolist()}  N = {N}")
print(f"pixel_values {tuple(pixel_values.shape)}  last_hidden {tuple(last_hidden.shape)}  proj_out {tuple(proj_out.shape)}")
print(f"proj_out  mean={proj_out.mean():+.5f}  std={proj_out.std():.5f}  [0,:5]={proj_out[0,:5].tolist()}")
print("saved m3_vis_parity.npz")
