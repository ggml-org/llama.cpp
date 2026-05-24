#include "common.cuh"

#include "tma-transfer.h"

// TMA-based transfer layer for moving KV cache data between pinned
// system RAM and GPU VRAM.  On SM 100+ the kernel can use the TMA
// engine; otherwise it falls back to cudaMemcpyAsync.
//
// We define our own copy of the 16-byte TMA descriptor type here to
// avoid pulling in tma.cuh host-side helpers (which are gated behind
// #ifndef __CUDA_ARCH__ and cause visibility issues in .cu translation
// units).  The layout matches ggml_cuda_tma_desc from tma.cuh.

#pragma pack(push, 1)
struct tma_transfer_desc {
    uint64_t d[2];
};
#pragma pack(pop)
static_assert(sizeof(tma_transfer_desc) == 16);

struct ggml_tma_transfer {
    tma_transfer_desc * desc;      // device-side TMA descriptor (16 bytes)
    cudaStream_t stream;
    size_t num_elements;
    size_t elem_size;
    void * src_pinned;
    void * dst_vram;
    bool use_tma;                  // true if SM >= 100 and kernel is verified
};

// TODO (TMA kernel): The device-side kernel for SM 100+ is not yet
// implemented. When ready, add a __global__ function guarded by
// #if __CUDA_ARCH__ >= 1000 that uses cp.async.bulk PTX to transfer
// data from the pinned source (TMA descriptor) to global VRAM.
// The commit/wait_group PTX wrappers are in tma.cuh (note: their PTX
// syntax requires verification against ptxas).

bool ggml_tma_init_transfer(ggml_tma_transfer_t * out,
    void * src_pinned,
    void * dst_vram,
    size_t num_elements,
    size_t elem_size,
    void * stream) {

    if (!out) return false;
    if (!src_pinned || !dst_vram || num_elements == 0) {
        *out = nullptr;
        return false;
    }

    ggml_tma_transfer * transfer = new ggml_tma_transfer();
    transfer->src_pinned  = src_pinned;
    transfer->dst_vram    = dst_vram;
    transfer->num_elements = num_elements;
    transfer->elem_size   = elem_size > 0 ? elem_size : 2;  // default float16/bf16
    transfer->stream      = (cudaStream_t)stream;
    transfer->desc        = nullptr;
    transfer->use_tma     = false; // conservative: memcpy until TMA kernel is verified

    // Build a 1D TMA descriptor pointing at the pinned system RAM source.
    // Encoding matches ggml_cuda_tma_make_load_desc_1d from tma.cuh:
    //   d[0] = lower 48 bits of address
    //   d[1] = lower 48 bits of byte count
    {
        size_t num_bytes = num_elements * transfer->elem_size;
        tma_transfer_desc host_desc;
        uint64_t addr = (uint64_t)src_pinned & 0xFFFFFFFFFFFFUL;
        host_desc.d[0] = addr;
        host_desc.d[1] = num_bytes & 0xFFFFFFFFFFFFUL;

        CUDA_CHECK(cudaMalloc(&transfer->desc, sizeof(tma_transfer_desc)));
        cudaMemcpyAsync(transfer->desc, &host_desc, sizeof(tma_transfer_desc),
                        cudaMemcpyHostToDevice, transfer->stream);
    }

    *out = transfer;
    return true;
}

void ggml_tma_launch_transfer(ggml_tma_transfer_t transfer) {
    if (!transfer) return;

    size_t bytes = transfer->num_elements * transfer->elem_size;

    if (transfer->use_tma) {
        // TODO: launch ggml_tma_kv_transfer_kernel for SM 100+.
        // The kernel is guarded by #if __CUDA_ARCH__ >= 1000 and will
        // only be compiled for Blackwell targets. The host-side launch
        // is gated by the use_tma runtime flag, which is currently
        // always false (conservative default until TMA is verified).
        GGML_UNUSED(transfer);
    }

    // Fallback: async memcpy from pinned RAM to VRAM.
    // This is the active path for all current configurations.
    cudaMemcpyAsync(transfer->dst_vram, transfer->src_pinned, bytes,
                    cudaMemcpyHostToDevice, transfer->stream);
}

void ggml_tma_free_transfer(ggml_tma_transfer_t transfer) {
    if (!transfer) return;
    if (transfer->desc) {
        cudaFree(transfer->desc);
    }
    delete transfer;
}
