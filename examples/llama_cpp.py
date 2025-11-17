import sys
import os
import ctypes
from ctypes import (
    c_int,
    c_float,
    c_char_p,
    c_void_p,
    c_bool,
    POINTER,
    _Pointer,  # type: ignore
    Structure,
    Array,
    c_uint8,
    c_size_t,
)
import pathlib


# Load the library
def _load_shared_library(lib_base_name: str):
    # Determine the file extension based on the platform
    if sys.platform.startswith("linux"):
        lib_ext = ".so"
    elif sys.platform == "darwin":
        lib_ext = ".dylib"
    elif sys.platform == "win32":
        lib_ext = ".dll"
    else:
        raise RuntimeError("Unsupported platform")

    # Construct the paths to the possible shared library names
    _base_path = pathlib.Path(__file__).parent.resolve()
    _base_path_parent = pathlib.Path(__file__).parent.parent.resolve()
    # Searching for the library in the current directory under the name "libllama" (default name
    # for llamacpp) and "llama" (default name for this repo)
    _lib_paths = [
        _base_path / f"lib{lib_base_name}{lib_ext}",
        _base_path_parent / f"lib{lib_base_name}{lib_ext}",
        _base_path / f"{lib_base_name}{lib_ext}",
    ]

    if "LLAMA_CPP_LIB" in os.environ:
        lib_base_name = os.environ["LLAMA_CPP_LIB"]
        _lib = pathlib.Path(lib_base_name)
        _base_path = _lib.parent.resolve()
        _lib_paths = [_lib.resolve()]

    cdll_args = dict()  # type: ignore
    # Add the library directory to the DLL search path on Windows (if needed)
    if sys.platform == "win32" and sys.version_info >= (3, 8):
        os.add_dll_directory(str(_base_path))
        if "CUDA_PATH" in os.environ:
            os.add_dll_directory(os.path.join(os.environ["CUDA_PATH"], "bin"))
            os.add_dll_directory(os.path.join(os.environ["CUDA_PATH"], "lib"))
        cdll_args["winmode"] = 0

    # Try to load the shared library, handling potential errors
    for _lib_path in _lib_paths:
        if _lib_path.exists():
            try:
                return ctypes.CDLL(str(_lib_path), **cdll_args)
            except Exception as e:
                raise RuntimeError(f"Failed to load shared library '{_lib_path}': {e}")

    raise FileNotFoundError(
        f"Shared library with base name '{lib_base_name}' not found"
    )


# Specify the base name of the shared library to load
_lib_base_name = "llama"

# Load the library
_lib = _load_shared_library(_lib_base_name)

# Misc
c_float_p = POINTER(c_float)
c_uint8_p = POINTER(c_uint8)
c_size_t_p = POINTER(c_size_t)

# llama.h bindings

# #define LLAMA_FILE_MAGIC_GGJT        0x67676a74u // 'ggjt'
LLAMA_FILE_MAGIC_GGJT = ctypes.c_uint(0x67676A74)
# #define LLAMA_FILE_MAGIC_GGLA        0x67676c61u // 'ggla'
LLAMA_FILE_MAGIC_GGLA = ctypes.c_uint(0x67676C61)
# #define LLAMA_FILE_MAGIC_GGMF        0x67676d66u // 'ggmf'
LLAMA_FILE_MAGIC_GGMF = ctypes.c_uint(0x67676D66)
# #define LLAMA_FILE_MAGIC_GGML        0x67676d6cu // 'ggml'
LLAMA_FILE_MAGIC_GGML = ctypes.c_uint(0x67676D6C)
# #define LLAMA_FILE_MAGIC_GGSN        0x6767736eu // 'ggsn'
LLAMA_FILE_MAGIC_GGSN = ctypes.c_uint(0x6767736E)

# #define LLAMA_FILE_VERSION           3
LLAMA_FILE_VERSION = c_int(3)
LLAMA_FILE_MAGIC = LLAMA_FILE_MAGIC_GGJT
LLAMA_FILE_MAGIC_UNVERSIONED = LLAMA_FILE_MAGIC_GGML
LLAMA_SESSION_MAGIC = LLAMA_FILE_MAGIC_GGSN
LLAMA_SESSION_VERSION = c_int(1)

# struct llama_context;
llama_context_p = c_void_p


# typedef int llama_token;
llama_token = c_int
llama_token_p = POINTER(llama_token)


# typedef struct llama_token_data {
#     llama_token id; // token id
#     float logit;    // log-odds of the token
#     float p;        // probability of the token
# } llama_token_data;
class llama_token_data(Structure):
    _fields_ = [
        ("id", llama_token),
        ("logit", c_float),
        ("p", c_float),
    ]


llama_token_data_p = POINTER(llama_token_data)


# typedef struct llama_token_data_array {
#     llama_token_data * data;
#     size_t size;
#     bool sorted;
# } llama_token_data_array;
class llama_token_data_array(Structure):
    _fields_ = [
        ("data", llama_token_data_p),
        ("size", c_size_t),
        ("sorted", c_bool),
    ]


llama_token_data_array_p = POINTER(llama_token_data_array)

# typedef void (*llama_progress_callback)(float progress, void *ctx);
llama_progress_callback = ctypes.CFUNCTYPE(None, c_float, c_void_p)


# struct llama_context_params {
#     int n_ctx;        // text context
#     int n_gpu_layers; // number of layers to store in VRAM
#     int seed;         // RNG seed, -1 for random

#     bool f16_kv;     // use fp16 for KV cache
#     bool logits_all; // the llama_eval() call computes all logits, not just the last one
#     bool vocab_only; // only load the vocabulary, no weights
#     bool use_mmap;   // use mmap if possible
#     bool use_mlock;  // force system to keep model in RAM
#     bool embedding;  // embedding mode only


#     // called with a progress value between 0 and 1, pass NULL to disable
#     llama_progress_callback progress_callback;
#     // context pointer passed to the progress callback
#     void * progress_callback_user_data;
# };
class llama_context_params(Structure):
    _fields_ = [
        ("n_ctx", c_int),
        ("n_gpu_layers", c_int),
        ("seed", c_int),
        ("f16_kv", c_bool),
        (
            "logits_all",
            c_bool,
        ),
        ("vocab_only", c_bool),
        ("use_mmap", c_bool),
        ("use_mlock", c_bool),
        ("embedding", c_bool),
        ("progress_callback", llama_progress_callback),
        ("progress_callback_user_data", c_void_p),
    ]


llama_context_params_p = POINTER(llama_context_params)

# enum llama_ftype {
#     LLAMA_FTYPE_ALL_F32              = 0,
#     LLAMA_FTYPE_MOSTLY_F16           = 1, // except 1d tensors
#     LLAMA_FTYPE_MOSTLY_Q4_0          = 2, // except 1d tensors
#     LLAMA_FTYPE_MOSTLY_Q4_1          = 3, // except 1d tensors
#     LLAMA_FTYPE_MOSTLY_Q4_1_SOME_F16 = 4, // tok_embeddings.weight and output.weight are F16
#     // LLAMA_FTYPE_MOSTLY_Q4_2       = 5, // support has been removed
#     // LLAMA_FTYPE_MOSTLY_Q4_3       = 6, // support has been removed
#     LLAMA_FTYPE_MOSTLY_Q8_0          = 7, // except 1d tensors
#     LLAMA_FTYPE_MOSTLY_Q5_0          = 8, // except 1d tensors
#     LLAMA_FTYPE_MOSTLY_Q5_1          = 9, // except 1d tensors
# };
LLAMA_FTYPE_ALL_F32 = c_int(0)
LLAMA_FTYPE_MOSTLY_F16 = c_int(1)
LLAMA_FTYPE_MOSTLY_Q4_0 = c_int(2)
LLAMA_FTYPE_MOSTLY_Q4_1 = c_int(3)
LLAMA_FTYPE_MOSTLY_Q4_1_SOME_F16 = c_int(4)
LLAMA_FTYPE_MOSTLY_Q8_0 = c_int(7)
LLAMA_FTYPE_MOSTLY_Q5_0 = c_int(8)
LLAMA_FTYPE_MOSTLY_Q5_1 = c_int(9)


# LLAMA_API struct llama_context_params llama_context_default_params();
def llama_context_default_params() -> llama_context_params:
    return _lib.llama_context_default_params()


_lib.llama_context_default_params.argtypes = []
_lib.llama_context_default_params.restype = llama_context_params


# LLAMA_API bool llama_mmap_supported();
def llama_mmap_supported() -> bool:
    return _lib.llama_mmap_supported()


_lib.llama_mmap_supported.argtypes = []
_lib.llama_mmap_supported.restype = c_bool


# LLAMA_API bool llama_mlock_supported();
def llama_mlock_supported() -> bool:
    return _lib.llama_mlock_supported()


_lib.llama_mlock_supported.argtypes = []
_lib.llama_mlock_supported.restype = c_bool


# // TODO: not great API - very likely to change
# // Initialize the llama + ggml backend
# // Call once at the start of the program
# LLAMA_API void llama_init_backend();
def llama_init_backend():
    return _lib.llama_init_backend()


_lib.llama_init_backend.argtypes = []
_lib.llama_init_backend.restype = None


# LLAMA_API int64_t llama_time_us();
def llama_time_us() -> int:
    return _lib.llama_time_us()


_lib.llama_time_us.argtypes = []
_lib.llama_time_us.restype = ctypes.c_int64


# // Various functions for loading a ggml llama model.
# // Allocate (almost) all memory needed for the model.
# // Return NULL on failure
# LLAMA_API struct llama_context * llama_init_from_file(
#                             const char * path_model,
#         struct llama_context_params   params);
def llama_init_from_file(
    path_model: bytes, params: llama_context_params
) -> llama_context_p:
    return _lib.llama_init_from_file(path_model, params)


_lib.llama_init_from_file.argtypes = [c_char_p, llama_context_params]
_lib.llama_init_from_file.restype = llama_context_p


# Frees all allocated memory
# LLAMA_API void llama_free(struct llama_context * ctx);
def llama_free(ctx: llama_context_p):
    return _lib.llama_free(ctx)


_lib.llama_free.argtypes = [llama_context_p]
_lib.llama_free.restype = None


# TODO: not great API - very likely to change
# Returns 0 on success
# nthread - how many threads to use. If <=0, will use std::thread::hardware_concurrency(), else the number given
# LLAMA_API int llama_model_quantize(
#         const char * fname_inp,
#         const char * fname_out,
#     enum llama_ftype   ftype,
#         int          nthread);
def llama_model_quantize(
    fname_inp: bytes, fname_out: bytes, ftype: c_int, nthread: c_int
) -> int:
    return _lib.llama_model_quantize(fname_inp, fname_out, ftype, nthread)


_lib.llama_model_quantize.argtypes = [c_char_p, c_char_p, c_int, c_int]
_lib.llama_model_quantize.restype = c_int


# Apply a LoRA adapter to a loaded model
# path_base_model is the path to a higher quality model to use as a base for
# the layers modified by the adapter. Can be NULL to use the current loaded model.
# The model needs to be reloaded before applying a new adapter, otherwise the adapter
# will be applied on top of the previous one
# Returns 0 on success
# LLAMA_API int llama_apply_lora_from_file(
#         struct llama_context * ctx,
#                   const char * path_lora,
#                   const char * path_base_model,
#                          int   n_threads);
def llama_apply_lora_from_file(
    ctx: llama_context_p,
    path_lora: c_char_p,
    path_base_model: c_char_p,
    n_threads: c_int,
) -> int:
    return _lib.llama_apply_lora_from_file(ctx, path_lora, path_base_model, n_threads)


_lib.llama_apply_lora_from_file.argtypes = [llama_context_p, c_char_p, c_char_p, c_int]
_lib.llama_apply_lora_from_file.restype = c_int


# Returns the number of tokens in the KV cache
# LLAMA_API int llama_get_kv_cache_token_count(const struct llama_context * ctx);
def llama_get_kv_cache_token_count(ctx: llama_context_p) -> int:
    return _lib.llama_get_kv_cache_token_count(ctx)


_lib.llama_get_kv_cache_token_count.argtypes = [llama_context_p]
_lib.llama_get_kv_cache_token_count.restype = c_int


# Sets the current rng seed.
# LLAMA_API void llama_set_rng_seed(struct llama_context * ctx, int seed);
def llama_set_rng_seed(ctx: llama_context_p, seed: c_int):
    return _lib.llama_set_rng_seed(ctx, seed)


_lib.llama_set_rng_seed.argtypes = [llama_context_p, c_int]
_lib.llama_set_rng_seed.restype = None


# Returns the maximum size in bytes of the state (rng, logits, embedding
# and kv_cache) - will often be smaller after compacting tokens
# LLAMA_API size_t llama_get_state_size(const struct llama_context * ctx);
def llama_get_state_size(ctx: llama_context_p) -> int:
    return _lib.llama_get_state_size(ctx)


_lib.llama_get_state_size.argtypes = [llama_context_p]
_lib.llama_get_state_size.restype = c_size_t


# Copies the state to the specified destination address.
# Destination needs to have allocated enough memory.
# Returns the number of bytes copied
# LLAMA_API size_t llama_copy_state_data(struct llama_context * ctx, uint8_t * dst);
def llama_copy_state_data(
    ctx: llama_context_p, dst  # type: Array[c_uint8]
) -> int:
    return _lib.llama_copy_state_data(ctx, dst)


_lib.llama_copy_state_data.argtypes = [llama_context_p, c_uint8_p]
_lib.llama_copy_state_data.restype = c_size_t


# Set the state reading from the specified address
# Returns the number of bytes read
# LLAMA_API size_t llama_set_state_data(struct llama_context * ctx, uint8_t * src);
def llama_set_state_data(
    ctx: llama_context_p, src  # type: Array[c_uint8]
) -> int:
    return _lib.llama_set_state_data(ctx, src)


_lib.llama_set_state_data.argtypes = [llama_context_p, c_uint8_p]
_lib.llama_set_state_data.restype = c_size_t


# Save/load session file
# LLAMA_API bool llama_load_session_file(struct llama_context * ctx, const char * path_session, llama_token * tokens_out, size_t n_token_capacity, size_t * n_token_count_out);
def llama_load_session_file(
    ctx: llama_context_p,
    path_session: bytes,
    tokens_out,  # type: Array[llama_token]
    n_token_capacity: c_size_t,
    n_token_count_out,  # type: _Pointer[c_size_t]
) -> int:
    return _lib.llama_load_session_file(
        ctx, path_session, tokens_out, n_token_capacity, n_token_count_out
    )


_lib.llama_load_session_file.argtypes = [
    llama_context_p,
    c_char_p,
    llama_token_p,
    c_size_t,
    c_size_t_p,
]
_lib.llama_load_session_file.restype = c_size_t


# LLAMA_API bool llama_save_session_file(struct llama_context * ctx, const char * path_session, const llama_token * tokens, size_t n_token_count);
def llama_save_session_file(
    ctx: llama_context_p,
    path_session: bytes,
    tokens,  # type: Array[llama_token]
    n_token_count: c_size_t,
) -> int:
    return _lib.llama_save_session_file(ctx, path_session, tokens, n_token_count)


_lib.llama_save_session_file.argtypes = [
    llama_context_p,
    c_char_p,
    llama_token_p,
    c_size_t,
]
_lib.llama_save_session_file.restype = c_size_t


# Run the llama inference to obtain the logits and probabilities for the next token.
# tokens + n_tokens is the provided batch of new tokens to process
# n_past is the number of tokens to use from previous eval calls
# Returns 0 on success
# LLAMA_API int llama_eval(
#         struct llama_context * ctx,
#            const llama_token * tokens,
#                          int   n_tokens,
#                          int   n_past,
#                          int   n_threads);
def llama_eval(
    ctx: llama_context_p,
    tokens,  # type: Array[llama_token]
    n_tokens: c_int,
    n_past: c_int,
    n_threads: c_int,
) -> int:
    return _lib.llama_eval(ctx, tokens, n_tokens, n_past, n_threads)


_lib.llama_eval.argtypes = [llama_context_p, llama_token_p, c_int, c_int, c_int]
_lib.llama_eval.restype = c_int


# Convert the provided text into tokens.
# The tokens pointer must be large enough to hold the resulting tokens.
# Returns the number of tokens on success, no more than n_max_tokens
# Returns a negative number on failure - the number of tokens that would have been returned
# TODO: not sure if correct
# LLAMA_API int llama_tokenize(
#         struct llama_context * ctx,
#                   const char * text,
#                  llama_token * tokens,
#                          int   n_max_tokens,
#                         bool   add_bos);
def llama_tokenize(
    ctx: llama_context_p,
    text: bytes,
    tokens,  # type: Array[llama_token]
    n_max_tokens: c_int,
    add_bos: c_bool,
) -> int:
    return _lib.llama_tokenize(ctx, text, tokens, n_max_tokens, add_bos)


_lib.llama_tokenize.argtypes = [llama_context_p, c_char_p, llama_token_p, c_int, c_bool]
_lib.llama_tokenize.restype = c_int


# LLAMA_API int llama_n_vocab(const struct llama_context * ctx);
def llama_n_vocab(ctx: llama_context_p) -> int:
    return _lib.llama_n_vocab(ctx)


_lib.llama_n_vocab.argtypes = [llama_context_p]
_lib.llama_n_vocab.restype = c_int


# LLAMA_API int llama_n_ctx  (const struct llama_context * ctx);
def llama_n_ctx(ctx: llama_context_p) -> int:
    return _lib.llama_n_ctx(ctx)


_lib.llama_n_ctx.argtypes = [llama_context_p]
_lib.llama_n_ctx.restype = c_int


# LLAMA_API int llama_n_embd (const struct llama_context * ctx);
def llama_n_embd(ctx: llama_context_p) -> int:
    return _lib.llama_n_embd(ctx)


_lib.llama_n_embd.argtypes = [llama_context_p]
_lib.llama_n_embd.restype = c_int


# Token logits obtained from the last call to llama_eval()
# The logits for the last token are stored in the last row
# Can be mutated in order to change the probabilities of the next token
# Rows: n_tokens
# Cols: n_vocab
# LLAMA_API float * llama_get_logits(struct llama_context * ctx);
def llama_get_logits(
    ctx: llama_context_p,
):  # type: (...) -> Array[float] # type: ignore
    return _lib.llama_get_logits(ctx)


_lib.llama_get_logits.argtypes = [llama_context_p]
_lib.llama_get_logits.restype = c_float_p


# Get the embeddings for the input
# shape: [n_embd] (1-dimensional)
# LLAMA_API float * llama_get_embeddings(struct llama_context * ctx);
def llama_get_embeddings(
    ctx: llama_context_p,
):  # type: (...) -> Array[float] # type: ignore
    return _lib.llama_get_embeddings(ctx)


_lib.llama_get_embeddings.argtypes = [llama_context_p]
_lib.llama_get_embeddings.restype = c_float_p


# Token Id -> String. Uses the vocabulary in the provided context
# LLAMA_API const char * llama_token_to_str(const struct llama_context * ctx, llama_token token);
def llama_token_to_str(ctx: llama_context_p, token: llama_token) -> bytes:
    return _lib.llama_token_to_str(ctx, token)


_lib.llama_token_to_str.argtypes = [llama_context_p, llama_token]
_lib.llama_token_to_str.restype = c_char_p

# Special tokens


# LLAMA_API llama_token llama_token_bos();
def llama_token_bos() -> int:
    return _lib.llama_token_bos()


_lib.llama_token_bos.argtypes = []
_lib.llama_token_bos.restype = llama_token


# LLAMA_API llama_token llama_token_eos();
def llama_token_eos() -> int:
    return _lib.llama_token_eos()


_lib.llama_token_eos.argtypes = []
_lib.llama_token_eos.restype = llama_token


# LLAMA_API llama_token llama_token_nl();
def llama_token_nl() -> int:
    return _lib.llama_token_nl()


_lib.llama_token_nl.argtypes = []
_lib.llama_token_nl.restype = llama_token


# Sampling functions


# @details Repetition penalty described in CTRL academic paper https://arxiv.org/abs/1909.05858, with negative logit fix.
# LLAMA_API void llama_sample_repetition_penalty(struct llama_context * ctx, llama_token_data_array * candidates, const llama_token * last_tokens, size_t last_tokens_size, float penalty);
def llama_sample_repetition_penalty(
    ctx: llama_context_p,
    candidates,  # type: _Pointer[llama_token_data_array]
    last_tokens_data,  # type: Array[llama_token]
    last_tokens_size: c_int,
    penalty: c_float,
):
    return _lib.llama_sample_repetition_penalty(
        ctx, candidates, last_tokens_data, last_tokens_size, penalty
    )


_lib.llama_sample_repetition_penalty.argtypes = [
    llama_context_p,
    llama_token_data_array_p,
    llama_token_p,
    c_int,
    c_float,
]
_lib.llama_sample_repetition_penalty.restype = None


# @details Frequency and presence penalties described in OpenAI API https://platform.openai.com/docs/api-reference/parameter-details.
# LLAMA_API void llama_sample_frequency_and_presence_penalties(struct llama_context * ctx, llama_token_data_array * candidates, const llama_token * last_tokens, size_t last_tokens_size, float alpha_frequency, float alpha_presence);
def llama_sample_frequency_and_presence_penalties(
    ctx: llama_context_p,
    candidates,  # type: _Pointer[llama_token_data_array]
    last_tokens_data,  # type: Array[llama_token]
    last_tokens_size: c_int,
    alpha_frequency: c_float,
    alpha_presence: c_float,
):
    return _lib.llama_sample_frequency_and_presence_penalties(
        ctx,
        candidates,
        last_tokens_data,
        last_tokens_size,
        alpha_frequency,
        alpha_presence,
    )


_lib.llama_sample_frequency_and_presence_penalties.argtypes = [
    llama_context_p,
    llama_token_data_array_p,
    llama_token_p,
    c_int,
    c_float,
    c_float,
]
_lib.llama_sample_frequency_and_presence_penalties.restype = None


# @details Sorts candidate tokens by their logits in descending order and calculate probabilities based on logits.
# LLAMA_API void llama_sample_softmax(struct llama_context * ctx, llama_token_data_array * candidates);
def llama_sample_softmax(
    ctx: llama_context_p, candidates  # type: _Pointer[llama_token_data]
):
    return _lib.llama_sample_softmax(ctx, candidates)


_lib.llama_sample_softmax.argtypes = [
    llama_context_p,
    llama_token_data_array_p,
]
_lib.llama_sample_softmax.restype = None


# @details Top-K sampling described in academic paper "The Curious Case of Neural Text Degeneration" https://arxiv.org/abs/1904.09751
# LLAMA_API void llama_sample_top_k(struct llama_context * ctx, llama_token_data_array * candidates, int k, size_t min_keep);
def llama_sample_top_k(
    ctx: llama_context_p,
    candidates,  # type: _Pointer[llama_token_data_array]
    k: c_int,
    min_keep: c_size_t,
):
    return _lib.llama_sample_top_k(ctx, candidates, k, min_keep)


_lib.llama_sample_top_k.argtypes = [
    llama_context_p,
    llama_token_data_array_p,
    c_int,
    c_size_t,
]
_lib.llama_sample_top_k.restype = None


# @details Nucleus sampling described in academic paper "The Curious Case of Neural Text Degeneration" https://arxiv.org/abs/1904.09751
# LLAMA_API void llama_sample_top_p(struct llama_context * ctx, llama_token_data_array * candidates, float p, size_t min_keep);
def llama_sample_top_p(
    ctx: llama_context_p,
    candidates,  # type: _Pointer[llama_token_data_array]
    p: c_float,
    min_keep: c_size_t,
):
    return _lib.llama_sample_top_p(ctx, candidates, p, min_keep)


_lib.llama_sample_top_p.argtypes = [
    llama_context_p,
    llama_token_data_array_p,
    c_float,
    c_size_t,
]
_lib.llama_sample_top_p.restype = None


# @details Tail Free Sampling described in https://www.trentonbricken.com/Tail-Free-Sampling/.
# LLAMA_API void llama_sample_tail_free(struct llama_context * ctx, llama_token_data_array * candidates, float z, size_t min_keep);
def llama_sample_tail_free(
    ctx: llama_context_p,
    candidates,  # type: _Pointer[llama_token_data_array]
    z: c_float,
    min_keep: c_size_t,
):
    return _lib.llama_sample_tail_free(ctx, candidates, z, min_keep)


_lib.llama_sample_tail_free.argtypes = [
    llama_context_p,
    llama_token_data_array_p,
    c_float,
    c_size_t,
]
_lib.llama_sample_tail_free.restype = None


# @details Locally Typical Sampling implementation described in the paper https://arxiv.org/abs/2202.00666.
# LLAMA_API void llama_sample_typical(struct llama_context * ctx, llama_token_data_array * candidates, float p, size_t min_keep);
def llama_sample_typical(
    ctx: llama_context_p,
    candidates,  # type: _Pointer[llama_token_data_array]
    p: c_float,
    min_keep: c_size_t,
):
    return _lib.llama_sample_typical(ctx, candidates, p, min_keep)


_lib.llama_sample_typical.argtypes = [
    llama_context_p,
    llama_token_data_array_p,
    c_float,
    c_size_t,
]
_lib.llama_sample_typical.restype = None


# LLAMA_API void llama_sample_temperature(struct llama_context * ctx, llama_token_data_array * candidates, float temp);
def llama_sample_temperature(
    ctx: llama_context_p,
    candidates,  # type: _Pointer[llama_token_data_array]
    temp: c_float,
):
    return _lib.llama_sample_temperature(ctx, candidates, temp)


_lib.llama_sample_temperature.argtypes = [
    llama_context_p,
    llama_token_data_array_p,
    c_float,
]
_lib.llama_sample_temperature.restype = None


# @details Mirostat 1.0 algorithm described in the paper https://arxiv.org/abs/2007.14966. Uses tokens instead of words.
# @param candidates A vector of `llama_token_data` containing the candidate tokens, their probabilities (p), and log-odds (logit) for the current position in the generated text.
# @param tau  The target cross-entropy (or surprise) value you want to achieve for the generated text. A higher value corresponds to more surprising or less predictable text, while a lower value corresponds to less surprising or more predictable text.
# @param eta The learning rate used to update `mu` based on the error between the target and observed surprisal of the sampled word. A larger learning rate will cause `mu` to be updated more quickly, while a smaller learning rate will result in slower updates.
# @param m The number of tokens considered in the estimation of `s_hat`. This is an arbitrary value that is used to calculate `s_hat`, which in turn helps to calculate the value of `k`. In the paper, they use `m = 100`, but you can experiment with different values to see how it affects the performance of the algorithm.
# @param mu Maximum cross-entropy. This value is initialized to be twice the target cross-entropy (`2 * tau`) and is updated in the algorithm based on the error between the target and observed surprisal.
# LLAMA_API llama_token llama_sample_token_mirostat(struct llama_context * ctx, llama_token_data_array * candidates, float tau, float eta, int m, float * mu);
def llama_sample_token_mirostat(
    ctx: llama_context_p,
    candidates,  # type: _Pointer[llama_token_data_array]
    tau: c_float,
    eta: c_float,
    m: c_int,
    mu,  # type: _Pointer[c_float]
) -> int:
    return _lib.llama_sample_token_mirostat(ctx, candidates, tau, eta, m, mu)


_lib.llama_sample_token_mirostat.argtypes = [
    llama_context_p,
    llama_token_data_array_p,
    c_float,
    c_float,
    c_int,
    c_float_p,
]
_lib.llama_sample_token_mirostat.restype = llama_token


# @details Mirostat 2.0 algorithm described in the paper https://arxiv.org/abs/2007.14966. Uses tokens instead of words.
# @param candidates A vector of `llama_token_data` containing the candidate tokens, their probabilities (p), and log-odds (logit) for the current position in the generated text.
# @param tau  The target cross-entropy (or surprise) value you want to achieve for the generated text. A higher value corresponds to more surprising or less predictable text, while a lower value corresponds to less surprising or more predictable text.
# @param eta The learning rate used to update `mu` based on the error between the target and observed surprisal of the sampled word. A larger learning rate will cause `mu` to be updated more quickly, while a smaller learning rate will result in slower updates.
# @param mu Maximum cross-entropy. This value is initialized to be twice the target cross-entropy (`2 * tau`) and is updated in the algorithm based on the error between the target and observed surprisal.
# LLAMA_API llama_token llama_sample_token_mirostat_v2(struct llama_context * ctx, llama_token_data_array * candidates, float tau, float eta, float * mu);
def llama_sample_token_mirostat_v2(
    ctx: llama_context_p,
    candidates,  # type: _Pointer[llama_token_data_array]
    tau: c_float,
    eta: c_float,
    mu,  # type: _Pointer[c_float]
) -> int:
    return _lib.llama_sample_token_mirostat_v2(ctx, candidates, tau, eta, mu)


_lib.llama_sample_token_mirostat_v2.argtypes = [
    llama_context_p,
    llama_token_data_array_p,
    c_float,
    c_float,
    c_float_p,
]
_lib.llama_sample_token_mirostat_v2.restype = llama_token


# @details Selects the token with the highest probability.
# LLAMA_API llama_token llama_sample_token_greedy(struct llama_context * ctx, llama_token_data_array * candidates);
def llama_sample_token_greedy(
    ctx: llama_context_p,
    candidates,  # type: _Pointer[llama_token_data_array]
) -> int:
    return _lib.llama_sample_token_greedy(ctx, candidates)


_lib.llama_sample_token_greedy.argtypes = [
    llama_context_p,
    llama_token_data_array_p,
]
_lib.llama_sample_token_greedy.restype = llama_token


# @details Randomly selects a token from the candidates based on their probabilities.
# LLAMA_API llama_token llama_sample_token(struct llama_context * ctx, llama_token_data_array * candidates);
def llama_sample_token(
    ctx: llama_context_p,
    candidates,  # type: _Pointer[llama_token_data_array]
) -> int:
    return _lib.llama_sample_token(ctx, candidates)


_lib.llama_sample_token.argtypes = [
    llama_context_p,
    llama_token_data_array_p,
]
_lib.llama_sample_token.restype = llama_token


# Performance information


# LLAMA_API void llama_print_timings(struct llama_context * ctx);
def llama_print_timings(ctx: llama_context_p):
    _lib.llama_print_timings(ctx)


_lib.llama_print_timings.argtypes = [llama_context_p]
_lib.llama_print_timings.restype = None


# LLAMA_API void llama_reset_timings(struct llama_context * ctx);
def llama_reset_timings(ctx: llama_context_p):
    _lib.llama_reset_timings(ctx)


_lib.llama_reset_timings.argtypes = [llama_context_p]
_lib.llama_reset_timings.restype = None


# Print system information
# LLAMA_API const char * llama_print_system_info(void);
def llama_print_system_info() -> bytes:
    return _lib.llama_print_system_info()


_lib.llama_print_system_info.argtypes = []
_lib.llama_print_system_info.restype = c_char_p

###################################################################################################


_llama_initialized = False

if not _llama_initialized:
    llama_init_backend()
    _llama_initialized = True
