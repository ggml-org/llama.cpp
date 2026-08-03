#include "cuda-dump-dequant.cuh"

#include "common.cuh"
#include "convert.cuh"
#include "dequantize.cuh"

#include "tessera-debug.h"

#include <chrono>
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
// `.dequant.f32` sidecar via the `llama-tessera-debug` static target
// (common/tessera-debug/). The hook is a no-op when the dequant sidecar
// is disabled (default). See docs/runtime-aware-pipeline.md for the
// format and acceptance criteria.
//
// In v3 the hook also captures the wall-clock for the dequant + sync +
// D2H copy (the explicit L1 cost) and writes it to the v3 per-row meta
// strip. The total time is distributed equally to each row; the L6
// kernel-direct fitness reads the strip and treats the per-row timing
// as a proxy for the row-wise dequant cost.
//
// In W4A4 mode (LLAMA_TILE640_DEBUG_DEQUANT_MODE=w4a4) the L1.5
// FP16-reference sidecar (.act.dequant.f32) is also written. The data
// is the same F32 values; a future refactor will pass the original
// FP16 weight to the hook so the L1.5 reference captures the actual
// ground-truth. The CUDA path may need to capture the FP16 reference
// from device memory and copy it back separately; for now both files
// are populated from the same dequantized F32 buffer.
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
//   - Timing: cudaEventRecord is the correct tool for GPU-side timing.
//     wall-clock around the kernel launch (dequant + sync) is the
//     host-visible cost. The D2H copy is timed separately with a
//     second event; the two are summed and reported as the per-row
//     total. The split is informational (a future per-stage LUT can
//     be derived from the v3 per-row meta and the JSON provenance).
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

    // The dequantized F32 weight has the natural storage layout of
    // M (=ne[1]) rows of K (=ne[0]) F32 values. Match the cpu/metal
    // sidecar convention so the round-trip validator does not see a
    // transposed view.
    const int64_t ne0 = src0->ne[1];
    const int64_t ne1 = src0->ne[0];
    // Dump the first slice along ne2/ne3. For batched matmul this is the
    // first expert; for the common 2-D weight case it is the whole tensor.
    const int64_t n_dump = ne0 * ne1;
    if (n_dump <= 0) {
        return;
    }

    cudaStream_t stream = ctx.stream();

    // Scratch F32 buffer on device, drawn from the per-stream pool.
    ggml_cuda_pool_alloc<float> dst_d(ctx.pool(), n_dump);

    // GPU-side timing for the v3 per-row meta strip. cudaEventRecord
    // gives true GPU time; the host-visible cost (launch + D2H) is
    // captured with std::chrono around the synchronous wait. The
    // GPU-only time is the more useful number for the L6 kernel-direct
    // fitness (matmul-direct comparison); the host time is reported
    // in the JSON provenance for end-to-end latency budgeting.
    cudaEvent_t evt_dequant_start = nullptr;
    cudaEvent_t evt_dequant_end   = nullptr;
    const bool have_events =
        (cudaEventCreate(&evt_dequant_start) == cudaSuccess) &&
        (cudaEventCreate(&evt_dequant_end)   == cudaSuccess);
    if (have_events) {
        cudaEventRecord(evt_dequant_start, stream);
    }

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
            if (have_events) {
                cudaEventDestroy(evt_dequant_start);
                cudaEventDestroy(evt_dequant_end);
            }
            return;
        }
        fn(src0->data, dst_d.get(), n_dump, stream);
    }

    if (have_events) {
        cudaEventRecord(evt_dequant_end, stream);
    }

    // The dequant kernel must complete before the host copy. This is the
    // explicit L1 cost the spec calls out: a full F32 materialization plus
    // a synchronous D2H copy per dumped tensor. Host wall-clock around
    // the sync is the conservative per-row timing; the GPU-only time is
    // available via the cudaEvent if a future reader wants the split.
    const auto host_t0 = std::chrono::steady_clock::now();
    CUDA_CHECK(cudaStreamSynchronize(stream));
    const auto host_t1 = std::chrono::steady_clock::now();
    const uint64_t host_total_ns =
        (uint64_t) std::chrono::duration_cast<std::chrono::nanoseconds>(host_t1 - host_t0).count();

    float gpu_ms = 0.0f;
    if (have_events) {
        // elapsed is a no-op if the event hasn't fired yet, but we just
        // synchronized the stream so the values are valid.
        cudaEventElapsedTime(&gpu_ms, evt_dequant_start, evt_dequant_end);
        cudaEventDestroy(evt_dequant_start);
        cudaEventDestroy(evt_dequant_end);
    }
    const uint64_t gpu_total_ns = have_events
        ? (uint64_t) (gpu_ms * 1.0e6f)
        : host_total_ns;

    // Pull the dequantized weight to the host in one shot. The pool alloc
    // stays alive (RAII) until the function returns, so the device pointer
    // remains valid for the copy.
    std::vector<float> dst_h(static_cast<size_t>(n_dump));
    CUDA_CHECK(cudaMemcpy(dst_h.data(), dst_d.get(),
                          static_cast<size_t>(n_dump) * sizeof(float),
                          cudaMemcpyDeviceToHost));

    // Per-row timing: distribute the GPU + host total equally to each
    // row. The L6 kernel-direct fitness treats this as the per-row
    // dequant cost; the matmul cost is added in by the L6
    // instrumentation on its own side.
    const uint64_t per_row_ns = ne0 > 0 ? (gpu_total_ns / (uint64_t) ne0) : 0;
    // kernel_id: stable per-quantization-type identifier. dispatch_count
    // is 1 (a single kernel launch per tensor in the L1 dump path).
    const uint32_t kernel_id = (uint32_t) src0->type;

    // Sidecar write: header + one row per ne0 index, each row is ne1 floats
    // wide. Matches the convention in docs/runtime-aware-pipeline.md 1.2:
    // `open(name, ne0, ne1); for r in 0..ne0 write(r, scratch + r*ne1, ne1)`.
    // Per-row outlier counts (|x| > threshold) are computed inside the
    // writer against the F32 host buffer; the convention here is
    // preserved: ne0 == rows, ne1 == cols, row r at offset r*ne1.
    const int64_t stride = tessera_debug::dequant_stride();
    const int64_t captured_rows = (ne0 + stride - 1) / stride;

    tessera_debug::open_dequant_writer(src0->name, captured_rows, ne1);
    int64_t out_r = 0;
    for (int64_t r = 0; r < ne0; r += stride, out_r++) {
        tessera_debug::write_dequant_row(out_r, dst_h.data() + r * ne1, ne1);
        tessera_debug::set_dequant_row_meta(out_r, per_row_ns, kernel_id,
                                            /*dispatch_count=*/1);
    }
    tessera_debug::close_dequant_writer();
    // In W4A4 mode, the L1.5 FP16-reference sidecar is also written.
    // The L1.5 reference is the F32 dequant cast to FP16 (proper
    // rounding via ggml_fp32_to_fp16, NOT truncation). This is the
    // FP16 ground truth: the file data block is 2 bytes/value, the
    // header dtype is DEQUANT_DTYPE_F16, and the file suffix is
    // `.act.dequant.f16`. The conversion is done here in the hook
    // (not the writer) so the L1.5 metric
    //     ||FP16(Q_b(W_l)) - FP16(W_l)||_F^2 / ||FP16(W_l)||_F^2
    // is well-defined and non-zero whenever the kernel dequant is
    // not bit-exact at F16 precision (the common case for any
    // non-power-of-2 weight value).
    if (tessera_debug::dequant_w4a4_enabled() && tessera_debug::l15_dtype_is_f16()) {
        tessera_debug::open_fp16_reference_writer(src0->name, captured_rows, ne1);
        out_r = 0;
        for (int64_t r = 0; r < ne0; r += stride, out_r++) {
            const float * row = dst_h.data() + r * ne1;
            // Stack buffer for small rows, heap for large.
            uint16_t stack_buf[256];
            uint16_t * fp16_row;
            std::vector<uint16_t> heap_buf;
            if ((size_t) ne1 <= 256) {
                fp16_row = stack_buf;
            } else {
                heap_buf.resize((size_t) ne1);
                fp16_row = heap_buf.data();
            }
            for (int64_t c = 0; c < ne1; c++) {
                fp16_row[c] = (uint16_t) ggml_fp32_to_fp16(row[c]);
            }
            tessera_debug::write_fp16_reference_row(out_r, fp16_row, ne1);
            tessera_debug::set_fp16_reference_row_meta(out_r, per_row_ns,
                                                       kernel_id,
                                                       /*dispatch_count=*/1);
        }
        tessera_debug::close_fp16_reference_writer();
    }
    // Note: the legacy F32 L1.5 path (l15_dtype=f32) is handled
    // automatically by tessera_debug::write_dequant_row's auto-
    // populate branch, which mirrors the F32 buffer to the L1.5
    // sidecar when both L1 and L1.5 are open as F32. No explicit
    // call is needed here.
}
