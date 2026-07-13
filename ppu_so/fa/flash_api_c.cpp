// libppu_fa.so — torch-free C ABI shim over the flash-attention submodule's forward kernels.
//
// Upstream's entry points (mha_fwd / mha_varlen_fwd in csrc/flash_attn/flash_api.cpp) take at::Tensor and drag in
// libtorch, and so does upstream's `run_mha_fwd` (it lives in that same torch TU). We use NEITHER. Instead we fill
// Flash_fwd_params from raw device pointers exactly the way set_params_fprop does, and dispatch the per-headdim
// template `run_mha_fwd_<elem_type, kHeadDim, Is_causal>` ourselves (the same FP16_SWITCH/HEADDIM_SWITCH/BOOL_SWITCH
// nest upstream's run_mha_fwd uses), forcing the non-split path. Those templates are explicitly instantiated by the
// flash_fwd_hdim*_sm80.cu files we compile into this .so, so nothing from flash_api.cpp is needed.
//
// This .so links ONLY the CUDA runtime + the FA kernel .cu — never libtorch. (flash.h does #include an ATen header
// for the `at::PhiloxCudaState philox_args` field, so torch HEADERS are a compile-time dependency; the type is a
// header-only POD and no torch symbol is referenced, so nothing links. Verify: `ldd libppu_fa.so | grep -i torch`
// must be empty.)
//
// Exports exactly the symbol ggml-cuda/ppu-fa-so.h declares. cutlass/FA templates stay inside this .so.
//
// NOTE on mask: FA2's sm80 forward has no additive-mask pointer -- attention shape is expressed via is_causal /
// window only. So the ggml additive mask is NOT consumed here; the caller must only engage this path for
// full or pure-causal attention (that gating lives in the ggml hook). alibi/varlen/paged-KV are out of scope.

#include <cuda_runtime.h>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <mutex>
#include <stdexcept>
#include <cstdio>

// From the submodule (build with -I .../csrc/flash_attn/src):
#include <cutlass/numeric_types.h>   // cutlass::half_t / bfloat16_t -- static_switch.h names them but doesn't include
#include "namespace_config.h"
#include "flash.h"           // Flash_fwd_params, run_mha_fwd_<T, Headdim, Is_causal>
#include "static_switch.h"   // FP16_SWITCH, HEADDIM_SWITCH, BOOL_SWITCH

#ifndef M_LOG2E
#define M_LOG2E 1.4426950408889634074
#endif

// The FA launch templates call C10_CUDA_CHECK / C10_CUDA_KERNEL_LAUNCH_CHECK (flash_fwd_launch_template.h), which
// bottom out in this ONE libtorch symbol. Rather than patch the submodule, we define it ourselves: the header
// declaration is satisfied, nothing links against libtorch, and a CUDA error still surfaces as an exception (caught
// below -> non-zero rc -> ggml falls back to its inline path). This is the only torch symbol the FA kernels need;
// `nm -D -u libppu_fa.so | grep c10` must come back empty once this is in.
namespace c10::cuda {
void c10_cuda_check_implementation(const int32_t err, const char * filename, const char * function_name,
                                   const int line_number, const bool /*include_device_assertions*/) {
    if (err != cudaSuccess) {
        char buf[512];
        snprintf(buf, sizeof(buf), "[ppu-fa] CUDA error %d (%s) at %s:%d in %s",
                 err, cudaGetErrorString((cudaError_t) err), filename ? filename : "?", line_number,
                 function_name ? function_name : "?");
        throw std::runtime_error(buf);
    }
}
} // namespace c10::cuda

using FLASH_NAMESPACE::Flash_fwd_params;

namespace {

// Per-call device scratch the kernel writes but the ggml caller doesn't need: softmax_lse [b*h*sq] float, and
// rng_state [2] uint64. Cached, grow-on-demand. Guarded by a mutex; the buffers are only written by the kernel we
// launch on the caller's stream, so concurrent callers on different streams would race -- documented single-stream
// assumption, same as upstream's per-call torch::empty. TODO: move the scratch into the ABI if that ever changes.
std::mutex g_mtx;
void * g_lse = nullptr;  size_t g_lse_bytes = 0;
void * g_rng = nullptr;  size_t g_rng_bytes = 0;

void * ensure(void ** p, size_t * cap, size_t bytes) {
    if (*cap < bytes) {
        if (*p) { cudaFree(*p); *p = nullptr; }
        if (cudaMalloc(p, bytes) != cudaSuccess) { *p = nullptr; *cap = 0; return nullptr; }
        *cap = bytes;
    }
    return *p;
}

int round_multiple(int x, int m) { return (x + m - 1) / m * m; }

// Upstream's run_mha_fwd (flash_api.cpp), minus the split-kv branch: we always set num_splits=1.
void run_mha_fwd_nosplit(Flash_fwd_params & params, cudaStream_t stream) {
    FP16_SWITCH(!params.is_bf16, [&] {
        HEADDIM_SWITCH(params.d, [&] {
            BOOL_SWITCH(params.is_causal, Is_causal, [&] {
                FLASH_NAMESPACE::run_mha_fwd_<elem_type, kHeadDim, Is_causal>(params, stream);
            });
        });
    });
}

} // namespace

extern "C" int ppu_flash_attn_fwd(
        const void * q, const void * k, const void * v, void * o,
        int batch, int seqlen_q, int seqlen_k, int n_heads_q, int n_heads_kv, int head_dim,
        long long q_batch_stride, long long q_head_stride, long long q_row_stride,
        long long k_batch_stride, long long k_head_stride, long long k_row_stride,
        long long v_batch_stride, long long v_head_stride, long long v_row_stride,
        long long o_batch_stride, long long o_head_stride, long long o_row_stride,
        float scale, float logit_softcap, int is_causal, int dtype, void * stream) {

    // Coverage gate (mirror mha_fwd's TORCH_CHECKs).
    if (head_dim > 256 || head_dim <= 0 || (head_dim % 8) != 0) return 1;
    if (n_heads_kv <= 0 || (n_heads_q % n_heads_kv) != 0)       return 1;
    if (seqlen_k <= 0 || seqlen_q <= 0 || batch <= 0)           return 1;
    if (dtype != 0 && dtype != 1)                               return 1;

    Flash_fwd_params p;
    std::memset(&p, 0, sizeof(p));   // == set_params_fprop's `params = {}`

    // pointers
    p.q_ptr = const_cast<void *>(q);
    p.k_ptr = const_cast<void *>(k);
    p.v_ptr = const_cast<void *>(v);
    p.o_ptr = o;

    // strides (elements) -- caller-supplied so the .so consumes ggml's native layout without repacking.
    // head_dim is contiguous (stride 1); FA's "row" stride is the seqlen step.
    p.q_row_stride = q_row_stride;   p.q_head_stride = q_head_stride;   p.q_batch_stride = q_batch_stride;
    p.k_row_stride = k_row_stride;   p.k_head_stride = k_head_stride;   p.k_batch_stride = k_batch_stride;
    p.v_row_stride = v_row_stride;   p.v_head_stride = v_head_stride;   p.v_batch_stride = v_batch_stride;
    p.o_row_stride = o_row_stride;   p.o_head_stride = o_head_stride;   p.o_batch_stride = o_batch_stride;

    // dims + rounding (mha_fwd: head_size_rounded / seqlen_*_rounded)
    p.b = batch; p.h = n_heads_q; p.h_k = n_heads_kv; p.h_h_k_ratio = n_heads_q / n_heads_kv;
    p.seqlen_q = seqlen_q;  p.seqlen_k = seqlen_k;  p.d = head_dim;
    p.d_rounded        = round_multiple(head_dim, head_dim <= 128 ? 32 : 64);
    p.seqlen_q_rounded = round_multiple(seqlen_q, 128);
    p.seqlen_k_rounded = round_multiple(seqlen_k, 128);

    // scales + softcap (set_params_fprop)
    if (logit_softcap > 0.0f) {
        p.softcap            = scale / logit_softcap;
        p.scale_softmax      = logit_softcap;
        p.scale_softmax_log2 = logit_softcap * (float) M_LOG2E;
    } else {
        p.softcap            = 0.0f;
        p.scale_softmax      = scale;
        p.scale_softmax_log2 = scale * (float) M_LOG2E;
    }

    // dropout disabled -> keep-prob 1 (set_params_fprop)
    p.p_dropout = 1.0f;  p.p_dropout_in_uint8_t = 255;  p.rp_dropout = 1.0f;
    p.scale_softmax_rp_dropout = p.rp_dropout * p.scale_softmax;

    // causal / window. Upstream: is_causal == (window_left < 0 && window_right == 0); a full (non-causal, non-local)
    // pass keeps BOTH at -1 so the kernel's Is_local stays false. A single query row can't be causally masked
    // (bottom-right alignment), so mha_fwd downgrades seqlen_q==1 to non-causal -- do the same.
    if (seqlen_q == 1) { is_causal = 0; }
    if (is_causal) { p.window_size_left = -1; p.window_size_right =  0; p.is_causal = true;  }
    else           { p.window_size_left = -1; p.window_size_right = -1; p.is_causal = false; }

    p.is_bf16 = (dtype == 1);
    p.is_seqlens_k_cumulative = true;
    p.unpadded_lse = false;
    p.seqlenq_ngroups_swapped = false;
    p.num_splits = 1;   // force the non-split kernel -> no accum buffers needed

    // scratch the kernel writes
    {
        std::lock_guard<std::mutex> lk(g_mtx);
        p.softmax_lse_ptr = ensure(&g_lse, &g_lse_bytes, (size_t) batch * n_heads_q * seqlen_q * sizeof(float));
        p.rng_state       = (uint64_t *) ensure(&g_rng, &g_rng_bytes, 2 * sizeof(uint64_t));
    }
    if (!p.softmax_lse_ptr || !p.rng_state) return 3;   // OOM -> fall back to inline

    try {
        run_mha_fwd_nosplit(p, reinterpret_cast<cudaStream_t>(stream));
    } catch (const std::exception & e) {
        fprintf(stderr, "%s -> inline fallback\n", e.what());
        return 4;
    }
    return cudaGetLastError() == cudaSuccess ? 0 : 4;
}
