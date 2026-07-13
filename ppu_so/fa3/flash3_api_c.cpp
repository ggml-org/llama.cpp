// libppu_fa3.so -- torch-free C ABI shim over FlashAttention-3 (hopper/) sm90 forward kernels. Same ABI as the FA2
// shim (ppu-fa-so.h's ppu_flash_attn_fwd), so llama.cpp's dlopen hook is unchanged -- just point GGML_PPU_FA_SO at
// this .so to use the Hopper wgmma/TMA/warp-specialized kernels instead of the sm80 FA2 fallback.
//
// FA3's mha_fwd (flash_api.cpp) is torch-heavy AND does a lot of host setup (dynamic split, paged KV, varlen
// scheduler). We only need the SIMPLE path: sm90, non-split, non-paged, non-varlen, fp16/bf16, dv==d, no softcap,
// no dropout, PackGQA=false (GQA handled via head strides). For that, the instance files instantiate exactly
// run_mha_fwd_<90, T, hdim, hdim, /*Split*/false, /*PagedKVNonTMA*/false, /*Has_softcap*/false, /*PackGQA*/false>,
// which we call directly -- bypassing the torch entry entirely. We replicate set_params_fprop (verified vs
// hopper/flash_api.cpp) for the fields these kernels read, and allocate the softmax_lse + the persistent-scheduler
// tile_count_semaphore (sm90 always needs it). Links only the CUDA runtime, NOT libtorch.

#include <cuda_runtime.h>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <mutex>
#include <stdexcept>
#include <cstdio>

#include <cutlass/numeric_types.h>   // cutlass::half_t / bfloat16_t
#include "flash.h"                    // Flash_fwd_params + run_mha_fwd_<...> template decl

// Flash_fwd_params + run_mha_fwd_<...> are global (flash.h has no namespace) and declared there.

// The FA3 sm90 launch templates call C10_CUDA_CHECK etc.; define the one libtorch symbol they bottom out in so we
// link no libtorch (same trick as the FA2 shim). Throws on CUDA error -> caught below -> non-zero rc -> ggml inline.
namespace c10::cuda {
void c10_cuda_check_implementation(const int32_t err, const char *, const char *, const int, const bool) {
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString((cudaError_t) err));
}
}

namespace {
std::mutex g_mtx;
void * g_lse = nullptr;  size_t g_lse_bytes = 0;
int  * g_sem = nullptr;  size_t g_sem_ints  = 0;
void * ensure(void ** p, size_t * cap, size_t bytes) {
    if (*cap < bytes) { if (*p) cudaFree(*p); if (cudaMalloc(p, bytes) != cudaSuccess) { *p=nullptr; *cap=0; return nullptr; } *cap = bytes; }
    return *p;
}
int round_multiple(int x, int m) { return (x + m - 1) / m * m; }
int round_up_headdim(int d) { return d <= 64 ? 64 : d <= 96 ? 96 : d <= 128 ? 128 : d <= 192 ? 192 : 256; }

// dispatch run_mha_fwd_<90, T, hd, hd, false,false,false,false> by (dtype, head_dim)
template <typename T>
bool dispatch_hd(int hd, Flash_fwd_params & p, cudaStream_t s) {
    switch (hd) {
        case 64:  run_mha_fwd_<90, T, 64,  64,  false,false,false,false>(p, s); return true;
        case 96:  run_mha_fwd_<90, T, 96,  96,  false,false,false,false>(p, s); return true;
        case 128: run_mha_fwd_<90, T, 128, 128, false,false,false,false>(p, s); return true;
        case 192: run_mha_fwd_<90, T, 192, 192, false,false,false,false>(p, s); return true;
        case 256: run_mha_fwd_<90, T, 256, 256, false,false,false,false>(p, s); return true;
        default:  return false;
    }
}
} // namespace

// Only referenced by the launch template's varlen branch (Varlen && ...), which is never taken here (non-varlen).
// Define an empty stub so the .so links without compiling the torch-heavy flash_api.cpp that owns the real one.
void prepare_varlen_num_blocks(Flash_fwd_params &, cudaStream_t, bool, int, int, bool) {}

extern "C" int ppu_flash_attn_fwd(
        const void * q, const void * k, const void * v, void * o,
        int batch, int seqlen_q, int seqlen_k, int n_heads_q, int n_heads_kv, int head_dim,
        long long q_batch_stride, long long q_head_stride, long long q_row_stride,
        long long k_batch_stride, long long k_head_stride, long long k_row_stride,
        long long v_batch_stride, long long v_head_stride, long long v_row_stride,
        long long o_batch_stride, long long o_head_stride, long long o_row_stride,
        float scale, float logit_softcap, int is_causal, int dtype, void * stream) {

    if (head_dim > 256 || head_dim <= 0 || (head_dim % 8) != 0) return 1;
    if (n_heads_kv <= 0 || (n_heads_q % n_heads_kv) != 0)       return 1;
    if (seqlen_k <= 0 || seqlen_q <= 0 || batch <= 0)           return 1;
    if (dtype != 0 && dtype != 1)                               return 1;
    if (logit_softcap > 0.0f)                                   return 1;   // no softcap instances built

    // check the device is actually sm90 (these kernels are Hopper-only)
    int dev = 0; cudaGetDevice(&dev);
    cudaDeviceProp prop; cudaGetDeviceProperties(&prop, dev);
    if (prop.major != 9) return 1;

    Flash_fwd_params p;
    std::memset(&p, 0, sizeof(p));

    p.q_ptr = const_cast<void*>(q); p.k_ptr = const_cast<void*>(k); p.v_ptr = const_cast<void*>(v); p.o_ptr = o;
    p.q_row_stride = q_row_stride; p.q_head_stride = q_head_stride; p.q_batch_stride = q_batch_stride;
    p.k_row_stride = k_row_stride; p.k_head_stride = k_head_stride; p.k_batch_stride = k_batch_stride;
    p.v_row_stride = v_row_stride; p.v_head_stride = v_head_stride; p.v_batch_stride = v_batch_stride;
    p.v_dim_stride = 1;   // contiguous head_dim
    p.o_row_stride = o_row_stride; p.o_head_stride = o_head_stride; p.o_batch_stride = o_batch_stride;

    p.b = batch; p.b_k = batch; p.h = n_heads_q; p.h_k = n_heads_kv;
    p.seqlen_q = seqlen_q; p.seqlen_k = seqlen_k;
    p.seqlen_q_rounded = round_multiple(seqlen_q, 128);
    p.seqlen_k_rounded = round_multiple(seqlen_k, 128);
    p.d = head_dim; p.d_rounded = round_up_headdim(head_dim);
    p.dv = head_dim; p.dv_rounded = p.d_rounded;

    p.scale_softmax = scale; p.softcap = 0.0f;
    p.p_dropout = 1.0f; p.p_dropout_in_uint8_t = 255; p.rp_dropout = 1.0f;

    // causal / full (FA3 set_params_fprop convention). Full: wl<0 && wr<0 -> not causal, not local.
    int wl = -1, wr = is_causal ? 0 : -1;
    p.is_causal = wl < 0 && wr == 0;
    p.is_local  = false;
    if (wl < 0) wl = seqlen_k - 1;
    if (wr < 0) wr = seqlen_q - 1;
    p.window_size_left = wl; p.window_size_right = wr; p.attention_chunk = 0;

    p.is_bf16 = (dtype == 1); p.is_e4m3 = false; p.is_fp32 = false;
    p.arch = 90; p.num_sm = prop.multiProcessorCount;
    p.num_splits = 1; p.num_splits_dynamic_ptr = nullptr; p.num_m_blocks_ptr = nullptr; p.prepare_varlen_pdl = false;
    p.pack_gqa = false; p.page_table = nullptr; p.pagedkv_tma = false;

    {
        std::lock_guard<std::mutex> lk(g_mtx);
        p.softmax_lse_ptr = ensure(&g_lse, &g_lse_bytes, (size_t) batch * n_heads_q * seqlen_q * sizeof(float));
        // persistent-scheduler counter (sm90 needs it); zero it each call. Generous size covers any offset.
        size_t need = (size_t) 256 + (size_t) batch * n_heads_kv;
        int * sem = (int *) ensure((void**)&g_sem, &g_sem_ints, need * sizeof(int));
        if (sem) cudaMemsetAsync(sem, 0, need * sizeof(int), (cudaStream_t) stream);
        p.tile_count_semaphore = sem;
    }
    if (!p.softmax_lse_ptr || !p.tile_count_semaphore) return 3;

    try {
        bool ok = (dtype == 1) ? dispatch_hd<cutlass::bfloat16_t>(head_dim, p, (cudaStream_t) stream)
                               : dispatch_hd<cutlass::half_t>    (head_dim, p, (cudaStream_t) stream);
        if (!ok) return 1;
    } catch (const std::exception & e) {
        fprintf(stderr, "[ppu-fa3] %s -> inline fallback\n", e.what());
        return 4;
    }
    return cudaGetLastError() == cudaSuccess ? 0 : 4;
}
