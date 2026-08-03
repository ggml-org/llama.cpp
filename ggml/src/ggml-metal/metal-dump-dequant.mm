//
// metal-dump-dequant.mm
//
// Tessera Layer 1 dequant sidecar helper for the Metal backend
// (see docs/runtime-aware-pipeline.md).
//
// Two producer paths:
//
//   1. Non-Tile640 quantized matmuls (q4_0/q8_0/...): `metal_dump_dequant`
//      fills a host buffer via the CPU reference `traits->to_float` and
//      streams it to the sidecar. The Metal `kernel_cpy_q_*_f32` cpy
//      pipeline dispatches ne00 threads per row, each producing a T4x4
//      (16 floats), which overflows a flat dump buffer for small rows;
//      the CPU reference is the correct dense F32 dump for these types
//      and matches the Metal per-element dequant under the ULP spec.
//
//   2. Tile640 matmuls (GGML_OP_TILE640_MATMUL / _MATMUL_ID):
//      `metal_dump_dequant_tile640` dispatches the row-aware
//      `kernel_TILE640_DEQUANT` Metal kernel (one thread per element,
//      no row overflow) into a shared device buffer, reads it back, and
//      streams it to the sidecar. This is the runtime-faithful path: it
//      captures the GPU's actual dequant output, including the sparse
//      outlier addback that the CPU `dequantize_row_tessera_t640` trait
//      omits. The CPU trait also re-dequantizes from the flat packed
//      layout, while the GPU reads the six separate component tensors
//      the matmul consumes; using the GPU kernel removes both gaps so
//      the L1 fitness signal reflects what the matmul really does.
//
// In v3 the hooks also capture per-row wall-clock timing (around the
// dequant call) and write it to the v3 per-row meta strip. The L6
// kernel-direct fitness reads the strip and treats the per-row timing
// as a proxy for the row-wise dequant cost. The L1 spec describes the
// timing as "dequant + matmul"; the matmul portion is not measured at
// this hook (it fires before the matmul).
//
// No-op when the dequant debug hook is not enabled. The hook is off by
// default; activate via `--tessera-dequant-dir PATH` or
// `LLAMA_TILE640_DEBUG_DEQUANT_DIR`.
//
// Linking note: `tessera_debug::*` lives in the `llama-tessera-debug`
// static target (common/tessera-debug/). The link dependency is
// declared in ggml/src/ggml-metal/CMakeLists.txt; symbols resolve at
// link time. No weak redeclarations, no `-undefined dynamic_lookup`.
//

#import "metal-dump-dequant.h"

#import "ggml-metal-device.h"
#import "ggml-metal-impl.h"   // ggml_metal_kargs_tile640_dequant
#import "ggml.h"
#import "ggml-backend-impl.h" // ggml_backend_buffer_t

#include "tessera-debug.h"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <vector>

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

void metal_dump_dequant(
        ggml_metal_device_t dev,
        const struct ggml_tensor * src0,
        int64_t rows,
        int64_t cols,
        const char * name) {
    (void) dev;
    if (src0 == nullptr || name == nullptr) {
        return;
    }
    if (!tessera_debug::dequant_debug_enabled()) {
        return;
    }
    if (!ggml_is_quantized(src0->type)) {
        return;
    }
    if (rows <= 0 || cols <= 0) {
        return;
    }

    const int64_t expected_els = (int64_t) src0->ne[0] * src0->ne[1] * src0->ne[2] * src0->ne[3];
    if (rows * cols != expected_els) {
        fprintf(stderr, "metal-dump-dequant: '%s' size mismatch (rows*cols=%lld, ne0*ne1*ne2*ne3=%lld); skipping\n",
                name, (long long) (rows * cols), (long long) expected_els);
        return;
    }

    const ggml_type_traits * traits = ggml_get_type_traits(src0->type);
    if (traits == nullptr || traits->to_float == nullptr) {
        fprintf(stderr, "metal-dump-dequant: no to_float trait for type %d ('%s')\n",
                (int) src0->type, name);
        return;
    }

    // Allocate a host-side F32 buffer and run the reference dequant.
    // The tensor must be contiguous; non-contiguous matmul weights
    // are exceedingly rare in practice.
    if (src0->data == nullptr) {
        return;
    }

    std::vector<float> host_buf((size_t) expected_els);

    // Wall-clock around the single host dequant call. We don't have a
    // command-buffer fence here (this is host-side work, not a GPU
    // dispatch), so std::chrono::steady_clock is the right tool. The
    // total time is distributed equally to each row in the per-row
    // meta strip; the L6 fitness treats this as the per-row dequant
    // cost (a per-row GPU dequant kernel is the future work).
    const auto t0 = std::chrono::steady_clock::now();
    traits->to_float(src0->data, host_buf.data(), expected_els);
    const auto t1 = std::chrono::steady_clock::now();
    const uint64_t total_ns =
        (uint64_t) std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
    const uint64_t per_row_ns = rows > 0 ? (total_ns / (uint64_t) rows) : 0;

    // kernel_id for the v3 per-row meta: a stable per-quantization-type
    // identifier. dispatch_count is 1 for the single host dequant call.
    const uint32_t kernel_id = (uint32_t) src0->type;

    // The dump is laid out as `rows` rows of `cols` F32 values. The
    // reference dequant is row-major, so row r starts at host_buf +
    // r*cols. (The caller chooses the row/col split; for a 2D weight
    // [K, N] that means rows=N, cols=K.)
    const int64_t stride = tessera_debug::dequant_stride();
    const int64_t captured_rows = (rows + stride - 1) / stride;

    tessera_debug::open_dequant_writer(name, captured_rows, cols);
    int64_t out_r = 0;
    for (int64_t r = 0; r < rows; r += stride, out_r++) {
        tessera_debug::write_dequant_row(out_r, host_buf.data() + r * cols, cols);
        tessera_debug::set_dequant_row_meta(out_r, per_row_ns, kernel_id,
                                            /*dispatch_count=*/1);
    }
    tessera_debug::close_dequant_writer();
    // Per-row outlier counts (|x| > threshold) are sealed in the sidecar
    // file's per-row strip. The CPU-side host_buf ref-dequant matches
    // the Metal kernel's per-element dequant under the ULP tolerance,
    // so counting on the host gives the L3 metric the same signal it
    // would get from a GPU-side count.
    //
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
        tessera_debug::open_fp16_reference_writer(name, captured_rows, cols);
        out_r = 0;
        for (int64_t r = 0; r < rows; r += stride, out_r++) {
            const float * row = host_buf.data() + r * cols;
            // Stack buffer for small rows, heap for large.
            uint16_t stack_buf[256];
            uint16_t * fp16_row;
            std::vector<uint16_t> heap_buf;
            if ((size_t) cols <= 256) {
                fp16_row = stack_buf;
            } else {
                heap_buf.resize((size_t) cols);
                fp16_row = heap_buf.data();
            }
            for (int64_t c = 0; c < cols; c++) {
                fp16_row[c] = (uint16_t) ggml_fp32_to_fp16(row[c]);
            }
            tessera_debug::write_fp16_reference_row(out_r, fp16_row, cols);
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

// Resolve a tensor's host Metal buffer + offset. Mirrors the static
// helper in ggml-metal-ops.cpp (not exported). Returns metal=nil if the
// tensor has no Metal backing yet (caller treats as a skip).
static ggml_metal_buffer_id metal_dump_get_buffer_id(const struct ggml_tensor * t) {
    if (t == nullptr) {
        return { nullptr, 0 };
    }
    ggml_backend_buffer_t buffer = t->view_src ? t->view_src->buffer : t->buffer;
    if (buffer == nullptr) {
        return { nullptr, 0 };
    }
    ggml_metal_buffer_t ctx = (ggml_metal_buffer_t) buffer->context;
    if (ctx == nullptr) {
        return { nullptr, 0 };
    }
    return ggml_metal_buffer_get_id(ctx, t);
}

void metal_dump_dequant_tile640(
        ggml_metal_device_t dev,
        ggml_metal_encoder_t enc,
        ggml_metal_cmd_buf_t cmd_buf,
        const struct ggml_tensor * op,
        int64_t row_width,
        int64_t n_rows,
        const char * name) {
    if (op == nullptr || name == nullptr || enc == nullptr || cmd_buf == nullptr) {
        return;
    }
    if (!tessera_debug::dequant_debug_enabled()) {
        return;
    }
    if (row_width <= 0 || n_rows <= 0) {
        return;
    }

    // Six Tile640 weight components. src[0..5] must all be resident on
    // the GPU (the matmul is about to read them); if any is missing we
    // are too early in the graph build and silently skip.
    ggml_metal_buffer_id comp[6];
    for (int i = 0; i < 6; ++i) {
        comp[i] = metal_dump_get_buffer_id(op->src[i]);
        if (comp[i].metal == nullptr) {
            fprintf(stderr,
                    "metal-dump-dequant-t640: '%s' src[%d] has no Metal buffer; skipping\n",
                    name, i);
            return;
        }
    }

    const int64_t n_elements = row_width * n_rows;
    const int64_t out_bytes  = n_elements * (int64_t) sizeof(float);

    id<MTLDevice> device = (__bridge id<MTLDevice>) ggml_metal_device_get_obj(dev);
    if (device == nil) {
        fprintf(stderr, "metal-dump-dequant-t640: '%s' no Metal device; skipping\n", name);
        return;
    }

    ggml_metal_library_t lib = ggml_metal_device_get_library(dev);
    if (lib == nullptr) {
        fprintf(stderr, "metal-dump-dequant-t640: '%s' no Metal library; skipping\n", name);
        return;
    }

    // Shared (host-visible) output buffer. On Apple Silicon unified
    // memory, reading the contents pointer after the command buffer
    // completes is safe without a blit. The buffer is retained by the
    // completed handler block and released once the sidecar write is
    // done. We deliberately do NOT call waitUntilCompleted here: the
    // backend is mid-encode on the same shared queue, and a synchronous
    // wait would deadlock. Instead the readback + sidecar write run in
    // a completed handler after the command buffer finishes.
    id<MTLBuffer> out_buf = [device newBufferWithLength:(NSUInteger) out_bytes
                                               options:MTLResourceStorageModeShared];
    if (out_buf == nil) {
        fprintf(stderr,
                "metal-dump-dequant-t640: '%s' failed to allocate %lld-byte output buffer; skipping\n",
                name, (long long) out_bytes);
        return;
    }

    // Compile (cached) the row-aware dequant pipeline. Its kernel writes
    // exactly n_elements floats, one thread per element, so there is no
    // row-overflow into adjacent data (unlike kernel_cpy_q_*_f32).
    struct ggml_metal_pipeline_with_params pipeline =
        ggml_metal_library_get_pipeline_tile640_dequant(lib);
    if (pipeline.pipeline == nullptr) {
        fprintf(stderr,
                "metal-dump-dequant-t640: '%s' kernel_TILE640_DEQUANT pipeline missing; skipping\n",
                name);
        return;
    }

    ggml_metal_kargs_tile640_dequant args;
    args.row_width  = (int32_t) row_width;
    args.n_rows     = (int32_t) n_rows;
    args.n_elements = (int32_t) n_elements;

    // Encode into the backend's live encoder for this command buffer.
    // The dispatch runs on the same queue as the matmul; ordering is
    // preserved because both are in the same command buffer.
    const auto t0 = std::chrono::steady_clock::now();
    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer(enc, comp[0], 1); // packed
    ggml_metal_encoder_set_buffer(enc, comp[1], 2); // page_scales
    ggml_metal_encoder_set_buffer(enc, comp[2], 3); // lane_scales
    ggml_metal_encoder_set_buffer(enc, comp[3], 4); // outlier_row_offsets
    ggml_metal_encoder_set_buffer(enc, comp[4], 5); // outlier_cols
    ggml_metal_encoder_set_buffer(enc, comp[5], 6); // outlier_vals
    ggml_metal_buffer_id dst_bid = { (__bridge void *) out_buf, 0 };
    ggml_metal_encoder_set_buffer(enc, dst_bid, 7); // dst

    // One thread per element. The kernel clamps to n_elements, so the
    // rounded-up threadgroup count is safe.
    const int nthreads = 256;
    const int ngroups  = (int) ((n_elements + nthreads - 1) / nthreads);
    ggml_metal_encoder_dispatch_threadgroups(enc, ngroups, 1, 1, nthreads, 1, 1);

    // Deferred readback + sidecar write. out_buf is retained by the
    // block; the tensor name is copied into an NSString so the block does
    // not capture the (short-lived) `const char * name` pointer. The L1
    // fitness reads the sidecar after inference, so the asynchronous
    // timing is safe; the file appears shortly after the matmul returns.
    //
    // NOTE: the tessera_debug sidecar writer is documented as not
    // thread-safe (one matmul kernel at a time per process). If two
    // Tile640 matmuls finish on different command buffers in the same
    // window, their completed handlers could interleave and corrupt the
    // writer state. B5 ships the single-tensor verification; serializing
    // the writer (a mutex in common/tessera-debug) is the follow-up for
    // the end-to-end model run.
    id<MTLCommandBuffer> cmd = (__bridge id<MTLCommandBuffer>) cmd_buf;
    const int64_t stride = tessera_debug::dequant_stride();
    const int64_t captured_rows = (n_rows + stride - 1) / stride;
    // kernel_id 't640' marks the row as a GPU Tile640 dequant so the
    // reader can distinguish it from the CPU to_float path.
    const uint32_t kernel_id = 0x74643734;
    NSString * name_ns = [NSString stringWithUTF8String:name];

    [cmd addCompletedHandler:^(id<MTLCommandBuffer> cb) {
        (void) cb;
        @autoreleasepool {
            const auto t1 = std::chrono::steady_clock::now();
            const uint64_t total_ns = (uint64_t)
                std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
            const uint64_t per_row_ns = n_rows > 0 ? (total_ns / (uint64_t) n_rows) : 0;

            const float * gpu_data = (const float *) [out_buf contents];

            tessera_debug::open_dequant_writer([name_ns UTF8String],
                                               captured_rows, row_width);
            int64_t out_r = 0;
            for (int64_t r = 0; r < n_rows; r += stride, out_r++) {
                tessera_debug::write_dequant_row(out_r, gpu_data + r * row_width, row_width);
                tessera_debug::set_dequant_row_meta(out_r, per_row_ns, kernel_id,
                                                    /*dispatch_count=*/1);
            }
            tessera_debug::close_dequant_writer();
            // L1.5 FP16-reference sidecar: in W4A4 mode with the new
            // F16 L1.5 dtype (the default), the L1.5 ground truth is
            // the GPU's dequantized F32 weight cast to FP16 (proper
            // rounding). The conversion is done here in the hook
            // (not the writer) so the L1.5 metric
            //     ||FP16(Q_b(W_l)) - FP16(W_l)||_F^2 / ||FP16(W_l)||_F^2
            // is well-defined and non-zero whenever the GPU dequant
            // is not bit-exact at F16 precision. The legacy F32 L1.5
            // path is auto-populated by write_dequant_row.
            if (tessera_debug::dequant_w4a4_enabled() &&
                tessera_debug::l15_dtype_is_f16()) {
                const char * cname = [name_ns UTF8String];
                tessera_debug::open_fp16_reference_writer(cname, captured_rows, row_width);
                out_r = 0;
                for (int64_t r = 0; r < n_rows; r += stride, out_r++) {
                    const float * row = gpu_data + r * row_width;
                    uint16_t stack_buf[256];
                    uint16_t * fp16_row;
                    std::vector<uint16_t> heap_buf;
                    if ((size_t) row_width <= 256) {
                        fp16_row = stack_buf;
                    } else {
                        heap_buf.resize((size_t) row_width);
                        fp16_row = heap_buf.data();
                    }
                    for (int64_t c = 0; c < row_width; c++) {
                        fp16_row[c] = (uint16_t) ggml_fp32_to_fp16(row[c]);
                    }
                    tessera_debug::write_fp16_reference_row(out_r, fp16_row, row_width);
                    tessera_debug::set_fp16_reference_row_meta(out_r, per_row_ns,
                                                               kernel_id,
                                                               /*dispatch_count=*/1);
                }
                tessera_debug::close_fp16_reference_writer();
            }
        }
    }];
}
