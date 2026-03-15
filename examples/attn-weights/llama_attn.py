"""
Minimal ctypes wrapper for llama.cpp attention weight extraction API.
"""

import ctypes
import os
import sys

# Find the shared library
_LIB_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "build", "bin")
_LIB_NAMES = ["libllama.dylib", "libllama.so", "llama.dll"]

_lib = None
for name in _LIB_NAMES:
    path = os.path.join(_LIB_DIR, name)
    if os.path.exists(path):
        _lib = ctypes.CDLL(path)
        break

if _lib is None:
    raise RuntimeError(f"Cannot find libllama in {_LIB_DIR}. Build first with: cmake --build build")


# --- Types ---

class llama_model(ctypes.Structure):
    pass

class llama_context(ctypes.Structure):
    pass

class llama_vocab(ctypes.Structure):
    pass

llama_token = ctypes.c_int32
llama_pos = ctypes.c_int32
llama_seq_id = ctypes.c_int32


class llama_batch(ctypes.Structure):
    _fields_ = [
        ("n_tokens",  ctypes.c_int32),
        ("token",     ctypes.POINTER(llama_token)),
        ("embd",      ctypes.POINTER(ctypes.c_float)),
        ("pos",       ctypes.POINTER(llama_pos)),
        ("n_seq_id",  ctypes.POINTER(ctypes.c_int32)),
        ("seq_id",    ctypes.POINTER(ctypes.POINTER(llama_seq_id))),
        ("logits",    ctypes.POINTER(ctypes.c_int8)),
    ]


# Model params - we only need to get the default and possibly modify a few fields
# Since the struct is large and complex, we'll treat it as opaque bytes
# and use the C function to get defaults
class llama_model_params(ctypes.Structure):
    _fields_ = [
        ("_opaque", ctypes.c_uint8 * 256),  # oversized, safe
    ]


# We need the exact layout of llama_context_params to set attn_weights and flash_attn_type
# Let's read it from the header. The key fields we need are at known positions.
# Instead of matching the exact struct, we'll use the C API defaults and patch bytes.

# Enums
LLAMA_FLASH_ATTN_TYPE_AUTO     = -1
LLAMA_FLASH_ATTN_TYPE_DISABLED = 0
LLAMA_FLASH_ATTN_TYPE_ENABLED  = 1


# --- Function signatures ---

# void llama_backend_init(void)
_lib.llama_backend_init.argtypes = []
_lib.llama_backend_init.restype = None

# void llama_backend_free(void)
_lib.llama_backend_free.argtypes = []
_lib.llama_backend_free.restype = None

# llama_model_params llama_model_default_params(void)
_lib.llama_model_default_params.argtypes = []
_lib.llama_model_default_params.restype = llama_model_params

# llama_model * llama_model_load_from_file(const char * path, llama_model_params params)
_lib.llama_model_load_from_file.argtypes = [ctypes.c_char_p, llama_model_params]
_lib.llama_model_load_from_file.restype = ctypes.POINTER(llama_model)

# void llama_model_free(llama_model * model)
_lib.llama_model_free.argtypes = [ctypes.POINTER(llama_model)]
_lib.llama_model_free.restype = None

# const llama_vocab * llama_model_get_vocab(const llama_model * model)
_lib.llama_model_get_vocab.argtypes = [ctypes.POINTER(llama_model)]
_lib.llama_model_get_vocab.restype = ctypes.POINTER(llama_vocab)

# int32_t llama_model_n_layer(const llama_model * model)
_lib.llama_model_n_layer.argtypes = [ctypes.POINTER(llama_model)]
_lib.llama_model_n_layer.restype = ctypes.c_int32

# int32_t llama_model_n_head(const llama_model * model)
_lib.llama_model_n_head.argtypes = [ctypes.POINTER(llama_model)]
_lib.llama_model_n_head.restype = ctypes.c_int32

# int32_t llama_tokenize(const llama_vocab *, const char *, int32_t, llama_token *, int32_t, bool, bool)
_lib.llama_tokenize.argtypes = [
    ctypes.POINTER(llama_vocab), ctypes.c_char_p, ctypes.c_int32,
    ctypes.POINTER(llama_token), ctypes.c_int32, ctypes.c_bool, ctypes.c_bool
]
_lib.llama_tokenize.restype = ctypes.c_int32

# llama_batch llama_batch_init(int32_t n_tokens, int32_t embd, int32_t n_seq_max)
_lib.llama_batch_init.argtypes = [ctypes.c_int32, ctypes.c_int32, ctypes.c_int32]
_lib.llama_batch_init.restype = llama_batch

# void llama_batch_free(llama_batch batch)
_lib.llama_batch_free.argtypes = [llama_batch]
_lib.llama_batch_free.restype = None

# int32_t llama_decode(llama_context * ctx, llama_batch batch)
_lib.llama_decode.argtypes = [ctypes.POINTER(llama_context), llama_batch]
_lib.llama_decode.restype = ctypes.c_int32

# void llama_free(llama_context * ctx)
_lib.llama_free.argtypes = [ctypes.POINTER(llama_context)]
_lib.llama_free.restype = None

# void llama_set_attn_heads(llama_context *, const int32_t * layers, const int32_t * heads, size_t n_pairs)
_lib.llama_set_attn_heads.argtypes = [
    ctypes.POINTER(llama_context),
    ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32), ctypes.c_size_t
]
_lib.llama_set_attn_heads.restype = None

# float * llama_get_attn_ith(llama_context * ctx, int32_t i)
_lib.llama_get_attn_ith.argtypes = [ctypes.POINTER(llama_context), ctypes.c_int32]
_lib.llama_get_attn_ith.restype = ctypes.POINTER(ctypes.c_float)

# int32_t llama_get_attn_n_kv(llama_context * ctx)
_lib.llama_get_attn_n_kv.argtypes = [ctypes.POINTER(llama_context)]
_lib.llama_get_attn_n_kv.restype = ctypes.c_int32

# float * llama_get_logits_ith(llama_context * ctx, int32_t i)
_lib.llama_get_logits_ith.argtypes = [ctypes.POINTER(llama_context), ctypes.c_int32]
_lib.llama_get_logits_ith.restype = ctypes.POINTER(ctypes.c_float)

# uint32_t llama_n_ctx(const llama_context * ctx)
_lib.llama_n_ctx.argtypes = [ctypes.POINTER(llama_context)]
_lib.llama_n_ctx.restype = ctypes.c_uint32

# void llama_synchronize(llama_context * ctx)
_lib.llama_synchronize.argtypes = [ctypes.POINTER(llama_context)]
_lib.llama_synchronize.restype = None

# int32_t llama_vocab_n_tokens(const llama_vocab * vocab)
_lib.llama_vocab_n_tokens.argtypes = [ctypes.POINTER(llama_vocab)]
_lib.llama_vocab_n_tokens.restype = ctypes.c_int32

# llama_token llama_vocab_bos(const llama_vocab * vocab)
_lib.llama_vocab_bos.argtypes = [ctypes.POINTER(llama_vocab)]
_lib.llama_vocab_bos.restype = llama_token

# llama_token llama_vocab_eos(const llama_vocab * vocab)
_lib.llama_vocab_eos.argtypes = [ctypes.POINTER(llama_vocab)]
_lib.llama_vocab_eos.restype = llama_token

# int32_t llama_token_to_piece(const llama_vocab *, llama_token, char *, int32_t, int32_t, bool)
_lib.llama_token_to_piece.argtypes = [
    ctypes.POINTER(llama_vocab), llama_token,
    ctypes.POINTER(ctypes.c_char), ctypes.c_int32, ctypes.c_int32, ctypes.c_bool
]
_lib.llama_token_to_piece.restype = ctypes.c_int32


# --- Context creation ---
# Since llama_context_params is complex and may change layout between versions,
# we use a helper approach: call llama_context_default_params from C, then patch
# the specific fields we need.

# We need to know the struct size and field offsets. Let's use a small C helper.
# Actually, let's just build a properly aligned struct by reading the header.
# The key insight: we can create context via a C helper function.

# For now, let's use ctypes.c_uint8 array as an opaque blob and set fields at
# known byte offsets. This is fragile but works for our specific build.

def _create_context(model_ptr, n_ctx=512, n_batch=512, attn_weights=True, n_gpu_layers=0):
    """Create a llama_context with attention weights enabled.

    Uses a small C shim compiled on-the-fly to avoid struct layout issues.
    """
    import tempfile, subprocess

    shim_src = r"""
#include "llama.h"
#include <stdlib.h>

// Export a function that creates a context with the right params
__attribute__((visibility("default")))
struct llama_context * create_ctx_with_attn(
        struct llama_model * model,
        int n_ctx, int n_batch, int attn_weights, int n_gpu_layers) {
    struct llama_context_params params = llama_context_default_params();
    params.n_ctx = n_ctx;
    params.n_batch = n_batch;
    params.n_ubatch = n_batch;
    params.attn_weights = attn_weights ? true : false;
    params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    params.offload_kqv = n_gpu_layers > 0;
    return llama_init_from_model(model, params);
}
"""
    llama_dir = os.path.join(os.path.dirname(__file__), "..", "..")
    include_dir = os.path.join(llama_dir, "include")
    lib_dir = os.path.join(llama_dir, "build", "bin")
    ggml_include = os.path.join(llama_dir, "ggml", "include")

    with tempfile.NamedTemporaryFile(suffix=".c", mode="w", delete=False) as f:
        f.write(shim_src)
        src_path = f.name

    shim_lib = os.path.join(lib_dir, "libllama_attn_shim.dylib")
    if sys.platform == "linux":
        shim_lib = os.path.join(lib_dir, "libllama_attn_shim.so")

    cmd = [
        "cc", "-shared", "-fPIC", "-o", shim_lib, src_path,
        f"-I{include_dir}", f"-I{ggml_include}",
        f"-L{lib_dir}", "-lllama",
        f"-Wl,-rpath,{lib_dir}",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    os.unlink(src_path)
    if result.returncode != 0:
        raise RuntimeError(f"Failed to compile shim: {result.stderr}")

    shim = ctypes.CDLL(shim_lib)
    shim.create_ctx_with_attn.argtypes = [
        ctypes.POINTER(llama_model), ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int
    ]
    shim.create_ctx_with_attn.restype = ctypes.POINTER(llama_context)

    ctx = shim.create_ctx_with_attn(model_ptr, n_ctx, n_batch, 1 if attn_weights else 0, n_gpu_layers)
    if not ctx:
        raise RuntimeError("Failed to create llama_context")
    return ctx


# --- High-level helpers ---

def tokenize(vocab_ptr, text, add_bos=True, special=True):
    """Tokenize text, returning a list of token ids."""
    text_bytes = text.encode("utf-8")
    buf = (llama_token * (len(text_bytes) + 32))()
    n = _lib.llama_tokenize(vocab_ptr, text_bytes, len(text_bytes), buf, len(buf), add_bos, special)
    if n < 0:
        buf = (llama_token * (-n))()
        n = _lib.llama_tokenize(vocab_ptr, text_bytes, len(text_bytes), buf, len(buf), add_bos, special)
    return list(buf[:n])


def decode_batch(ctx_ptr, tokens, output_last_only=True):
    """Decode a batch of tokens. Returns llama_decode return code."""
    n = len(tokens)
    batch = _lib.llama_batch_init(n, 0, 1)

    for i in range(n):
        batch.token[i] = tokens[i]
        batch.pos[i] = i
        batch.n_seq_id[i] = 1
        # Write seq_id value into the pre-allocated buffer (don't replace the pointer)
        batch.seq_id[i][0] = 0
        batch.logits[i] = 1 if (not output_last_only or i == n - 1) else 0

    batch.n_tokens = n
    ret = _lib.llama_decode(ctx_ptr, batch)
    _lib.llama_batch_free(batch)
    return ret


def decode_single(ctx_ptr, token, pos, output=True):
    """Decode a single token at a given position."""
    batch = _lib.llama_batch_init(1, 0, 1)
    batch.token[0] = token
    batch.pos[0] = pos
    batch.n_seq_id[0] = 1
    batch.seq_id[0][0] = 0  # Write value into pre-allocated buffer
    batch.logits[0] = 1 if output else 0
    batch.n_tokens = 1
    ret = _lib.llama_decode(ctx_ptr, batch)
    _lib.llama_batch_free(batch)
    return ret


def get_attn_weights(ctx_ptr, token_idx, n_pairs, n_ctx):
    """Get attention weights for a given output token index.

    Returns numpy array of shape (n_pairs, n_kv) or None.
    """
    import numpy as np

    ptr = _lib.llama_get_attn_ith(ctx_ptr, token_idx)
    if not ptr:
        return None

    n_kv = _lib.llama_get_attn_n_kv(ctx_ptr)
    if n_kv <= 0:
        return None

    # Layout: [n_pairs * n_ctx] floats, each pair has n_ctx floats, first n_kv valid
    result = np.zeros((n_pairs, n_kv), dtype=np.float32)
    for p in range(n_pairs):
        offset = p * n_ctx
        arr = (ctypes.c_float * n_kv).from_address(ctypes.addressof(ptr.contents) + offset * 4)
        result[p] = np.frombuffer(arr, dtype=np.float32)

    return result


def argmax_logits(ctx_ptr, token_idx, n_vocab):
    """Get the argmax of logits for a given output token."""
    ptr = _lib.llama_get_logits_ith(ctx_ptr, token_idx)
    if not ptr:
        return -1
    logits = (ctypes.c_float * n_vocab).from_address(ctypes.addressof(ptr.contents))
    import numpy as np
    return int(np.argmax(np.frombuffer(logits, dtype=np.float32)))


# --- Public API ---

def init():
    _lib.llama_backend_init()

def cleanup():
    _lib.llama_backend_free()

def load_model(path, n_gpu_layers=0):
    params = _lib.llama_model_default_params()
    # n_gpu_layers is at offset 0 in llama_model_params (first field)
    # Actually let's just use default params for simplicity
    model = _lib.llama_model_load_from_file(path.encode(), params)
    if not model:
        raise RuntimeError(f"Failed to load model from {path}")
    return model

def create_context(model, n_ctx=512, n_batch=512, attn_weights=True):
    return _create_context(model, n_ctx, n_batch, attn_weights)

def set_attn_heads(ctx, layers, heads):
    n = len(layers)
    assert len(heads) == n
    l_arr = (ctypes.c_int32 * n)(*layers)
    h_arr = (ctypes.c_int32 * n)(*heads)
    _lib.llama_set_attn_heads(ctx, l_arr, h_arr, n)

def get_vocab(model):
    return _lib.llama_model_get_vocab(model)

def n_layer(model):
    return _lib.llama_model_n_layer(model)

def n_head(model):
    return _lib.llama_model_n_head(model)

def n_vocab(vocab):
    return _lib.llama_vocab_n_tokens(vocab)

def n_ctx(ctx):
    return _lib.llama_n_ctx(ctx)

def vocab_eos(vocab):
    return _lib.llama_vocab_eos(vocab)

def free_context(ctx):
    _lib.llama_free(ctx)

def token_to_piece(vocab, token_id, special=True):
    """Convert a single token ID to its string piece."""
    buf = (ctypes.c_char * 256)()
    n = _lib.llama_token_to_piece(vocab, token_id, buf, 256, 0, special)
    if n > 0:
        return buf[:n].decode("utf-8", errors="replace")
    return ""

def free_model(model):
    _lib.llama_model_free(model)
