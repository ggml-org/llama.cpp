#include "cuda-dump-dequant.cuh"

#include "common.cuh"
#include "convert.cuh"
#include "dequantize.cuh"

#include "tessera-debug.h"

#include <cstdint>
#include <mutex>
#include <string>
#include <unordered_set>
#include <vector>

//
// Tessera Layer 1: CUDA-side dequant sidecar dump helper.
//
// Materializes the full dequantized F32 weight of a quantized `src0` to a
// device scratch buffer, copies it to the host, and writes the rows to a
// `.dequant.f32` sidecar via common/tessera-debug.h. The hook is a no-op
// when the dequant sidecar is disabled (default). See
// docs/runtime-aware-pipeline.md for the format and acceptance criteria.
//
// Implementation notes (decisions the user / orchestrator should review):
//
//   - Per-process dedup: a single static set tracks tensor names that have
//     been dumped. The sidecar writer's `open_dequant_writer` truncates the
//     file on first open and reuses the open file for subsequent calls with
//     the same (name, rows, cols) -- so re-calling would APPEND data to an
//     already-written file and corrupt it. The dedup makes the dump truly
//     one-shot per (process, tensor).
//
//   - Batched matmul (ne2 > 1, e.g. MoE experts): the sidecar gets the first
//     expert only. The dequantized data is structurally identical across
//     experts (the kernel produces the same per-row bytes for any slice
//     along ne2), so this is acceptable for the "kernel dequant fidelity"
//     measurement. A future layer (L2 or L5) can re-dump per-expert if it
//     needs the per-expert breakdown.
//
//   - Q8_K: not covered by ggml_get_to_fp32_cuda (which dispatches to the
//     convert.cu F32 contig path). We have a tiny dedicated kernel below.
//     All other matmul types go through the existing dispatcher.
//
//   - Open question from the L1 doc (kernel-internal dedup by (tensor, row)
//     for batched dequant in the MMQ path) is left for a future layer; here
//     we just dump the materialized weight once per matmul call.
//
//   - Cost: a full F32 materialization + a synchronous D2H copy per dumped
//     tensor. This is intentional; L1 measures what the kernel actually
//     computes, not what the offline reference thinks. Memory pressure on
//     large models is the documented trade-off.
//

// Q8_K: float d, int8 qs[QK_K]. Simple per-block dequant, one block per
// super-block, 32 threads cooperating.
static __global__ void dump_dequant_q8_K_kernel(const void * __restrict__ vx, float * __restrict__ yy) {
    const int64_t ib = blockIdx.x;
    const block_q8_K * x = (const block_q8_K *) vx + ib;
    float * y = yy + ib * QK_K;
    const float d = x->d;
    for (int j = threadIdx.x; j < QK_K; j += blockDim.x) {
        y[j] = static_cast<float>(x->qs[j]) * d;
    }
}

void ggml_cuda_dump_dequant(ggml_backend_cuda_context & ctx, const ggml_tensor * src0) {
    // Gate: no env var / no flag set -> no-op. Cheap branch, no I/O.
    if (!tessera_debug::dequant_debug_enabled()) {
        return;
    }
    if (src0 == nullptr || src0->name == nullptr || !ggml_is_quantized(src0->type)) {
        return;
    }

    // Dedup + sidecar serialization. The dedup set protects against
    // re-dumping the same tensor (which would corrupt the file: the
    // foundation's open_dequant_writer truncates only on the first call per
    // (name, shape); subsequent calls would append). The mutex also
    // serializes the sidecar writes themselves (the foundation's API is
    // not thread-safe; we may be called from multiple CUDA streams).
    static std::mutex                      g_mutex;
    static std::unordered_set<std::string> g_dumped;
    const std::string key(src0->name);
    std::lock_guard<std::mutex> lock(g_mutex);
    if (g_dumped.count(key) != 0) {
        return;
    }
    g_dumped.insert(key);

    const int64_t ne0 = src0->ne[0];
    const int64_t ne1 = src0->ne[1];
    // Dump the first slice along ne2/ne3. For batched matmul this is the
    // first expert; for the common 2-D weight case it is the whole tensor.
    const int64_t n_dump = ne0 * ne1;
    if (n_dump <= 0) {
        return;
    }

    cudaStream_t stream = ctx.stream();

    // Scratch F32 buffer on device, drawn from the per-stream pool.
    ggml_cuda_pool_alloc<float> dst_d(ctx.pool(), n_dump);

    // Dispatch: Q8_K is the only matmul-eligible type missing from the
    // convert.cu F32 contig dispatcher.
    if (src0->type == GGML_TYPE_Q8_K) {
        const int64_t nb = n_dump / QK_K;
        dump_dequant_q8_K_kernel<<<nb, 32, 0, stream>>>(src0->data, dst_d.get());
    } else {
        to_fp32_cuda_t fn = ggml_get_to_fp32_cuda(src0->type);
        if (fn == nullptr) {
            // Unsupported type: drop the dedup key so a future, valid call
            // (e.g. after a relink) can still dump.
            g_dumped.erase(key);
            return;
        }
        fn(src0->data, dst_d.get(), n_dump, stream);
    }

    // The dequant kernel must complete before the host copy. This is the
    // explicit L1 cost the spec calls out: a full F32 materialization plus
    // a synchronous D2H copy per dumped tensor.
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Pull the dequantized weight to the host in one shot. The pool alloc
    // stays alive (RAII) until the function returns, so the device pointer
    // remains valid for the copy.
    std::vector<float> dst_h(static_cast<size_t>(n_dump));
    CUDA_CHECK(cudaMemcpy(dst_h.data(), dst_d.get(),
                          static_cast<size_t>(n_dump) * sizeof(float),
                          cudaMemcpyDeviceToHost));

    // Sidecar write: header + one row per ne0 index, each row is ne1 floats
    // wide. Matches the convention in docs/runtime-aware-pipeline.md 1.2:
    // `open(name, ne0, ne1); for r in 0..ne0 write(r, scratch + r*ne1, ne1)`.
    tessera_debug::open_dequant_writer(src0->name, ne0, ne1);
    for (int64_t r = 0; r < ne0; r++) {
        tessera_debug::write_dequant_row(r, dst_h.data() + r * ne1, ne1);
    }
    tessera_debug::close_dequant_writer();
}
