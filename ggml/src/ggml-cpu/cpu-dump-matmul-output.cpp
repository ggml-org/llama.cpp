//
// cpu-dump-matmul-output.cpp
//
// CPU-side helper for the Tessera runtime-aware calibration pipeline
// (Layer 2, see docs/runtime-aware-pipeline.md). See cpu-dump-matmul-output.h
// for the public API and a description of the call pattern.
//

#include "cpu-dump-matmul-output.h"

#include "ggml.h"
#include "ggml-cpu-impl.h"

#include "tessera-matmul-output.h"

#include <chrono>
#include <cstdint>
#include <cstring>
#include <vector>

void cpu_dump_matmul_output(
        const struct ggml_compute_params * params,
        const struct ggml_tensor * dst,
        const char              * tensor_name) {
    if (!tessera_matmul_output::matmul_output_capture_enabled()) {
        return;
    }
    if (params == nullptr || dst == nullptr || tensor_name == nullptr || tensor_name[0] == '\0') {
        return;
    }
    // Only thread 0 of the matmul op writes the sidecar (the same
    // pattern as the L1 dequant sidecar). Subsequent threads return
    // immediately; the writer is single-stream per open/close sequence.
    if (params->ith != 0) {
        return;
    }
    // Audit fix 3: filter by tensor name. The matmul op is called for
    // every matmul in the graph, including RoPE rotation matrix
    // lookups, token-embedding gathers, and small bias-projection
    // matmuls that are not in the L2 scope. The allowlist keeps
    // only the linear layer matmuls of the per-block transformer
    // (attn_q/k/v/output, ffn_gate/up/down) and the model head
    // (output.weight). The rest are silently skipped - the sidecar
    // dir stays focused on the tensors that have a meaningful
    // forward-pass differential to score.
    if (!tessera_matmul_output::matmul_output_tensor_allowed(tensor_name)) {
        return;
    }
    // The matmul dst layout: (ne0, ne1, ne2, ne3) row-major F32, where
    // ne1 is the number of tokens, ne0 is the output dimension. The
    // sidecar stores one row per token (rows = ne1, cols = ne0).
    const int64_t ne0 = dst->ne[0];
    const int64_t ne1 = dst->ne[1];
    if (ne0 <= 0 || ne1 <= 0) {
        return;
    }
    if (dst->data == nullptr) {
        return;
    }
    // The dst must be F32 (matmul ops produce F32 in ggml-cpu). The
    // Metal matmul produces F16/F32; the CPU path here is F32 only.
    // The L1 sidecar's v3 schema supports F32 (dtype = 0).
    if (dst->type != GGML_TYPE_F32) {
        return;
    }

    const int64_t stride = tessera_matmul_output::matmul_output_stride();
    const int64_t captured_rows = (ne1 + stride - 1) / stride;

    // Audit fix 2: the open_matmul_output_writer enforces the hard
    // cap (MATMUL_OUTPUT_MAX_ROWS = 4096) and silently rejects
    // over-cap tensors with a one-time warning. The pre-check here
    // is a fast-fail that avoids the more expensive dst-row-pointer
    // computation in the over-cap case. Both layers are guarded so
    // the cap holds even if open_matmul_output_writer's check is
    // bypassed by a future change.
    if (captured_rows > tessera_matmul_output::MATMUL_OUTPUT_MAX_ROWS) {
        return;
    }

    // Remember the current tensor name so we can detect an
    // audit-fix-4 drop. If the open was rejected because of a
    // shape mismatch (decode-time reopen of a prefill-time
    // tensor), we want to skip the subsequent write + close
    // calls so the rejected invocation does not corrupt the
    // prefill data via the still-open prefill stream.
    const std::string tensor_name_str(tensor_name);
    tessera_matmul_output::open_matmul_output_writer(tensor_name, captured_rows, ne0);

    // If the open was silently dropped (shape mismatch - typical
    // for decode reopening a prefill tensor), bail out. The
    // close call at the end of THIS invocation would otherwise
    // re-seal the prefill v2/v3 strips using the decode-side
    // g_stream state (which was NOT touched by the rejected
    // open, so this is safe), but the write calls in the loop
    // below would land in the prefill stream and overwrite
    // prefill data. We skip the whole loop + close to be safe.
    //
    // We detect the drop by checking whether the g_stream's
    // current tensor_name matches ours. After a successful
    // open, g_stream.tensor_name == our name. After a dropped
    // open, g_stream is unchanged (it still holds the prefill
    // stream), so the names match too. We distinguish the two by
    // checking the rows/cols: after a drop, g_stream.rows is the
    // prefill rows (large), not our decode rows (small). If the
    // rows don't match, the open was dropped.
    if (tessera_matmul_output::matmul_output_stream_rows() != captured_rows ||
        tessera_matmul_output::matmul_output_stream_cols() != ne0) {
        // The open was dropped. Skip the writes and close.
        return;
    }

    const int64_t nb1 = dst->nb[1];
    const int64_t nb2 = dst->nb[2];
    const int64_t nb3 = dst->nb[3];
    const char * base = (const char *) dst->data;

    for (int64_t r = 0; r < captured_rows; r++) {
        const int64_t src_r = r * stride;
        // Resolve the row pointer. For ne1 == captured_rows the row
        // stride is nb1 (the natural layout); for higher-dim dst
        // tensors (ne2 > 1, ne3 > 1) we collapse the higher dims
        // into a contiguous (src_r)-th row by stepping through ne2 * ne3
        // tiles. Most attention / FFN matmul outputs are 2D so this
        // branch is uncommon in practice.
        const char * row_ptr;
        if (dst->ne[2] == 1 && dst->ne[3] == 1) {
            row_ptr = base + src_r * nb1;
        } else {
            const int64_t r2 = src_r / dst->ne[1];
            const int64_t r1 = src_r % dst->ne[1];
            row_ptr = base + r2 * nb2 + r1 * nb1;
            (void) nb3;  // kept for future 4D use
        }

        const auto t0 = std::chrono::steady_clock::now();
        // The matmul itself has just run; this hook only copies + writes
        // the F32 row, so the timing is dominated by the I/O cost.
        // Kernel_id 0 is reserved for "generic CPU" in this L2 hook.
        tessera_matmul_output::write_matmul_output_row(r, (const float *) row_ptr, ne0);
        const auto t1 = std::chrono::steady_clock::now();
        const uint64_t ns = (uint64_t)
            std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
        tessera_matmul_output::set_matmul_output_row_meta(r, ns, /*kernel_id=*/0, /*dispatch_count=*/1);
    }

    tessera_matmul_output::close_matmul_output_writer();
    // The per-row sample counter is now sealed in the sidecar. The L2
    // Python reader (tools/tessera/runtime_probe.py) consumes this
    // file alongside the BF16 sidecar to compute the per-tensor
    // divergence metrics (max_abs, mean_abs, relative_frobenius,
    // top1_mismatch, top5_mismatch). The sidecar is the kernel's
    // actual matmul output, not a Python recomputation, so the
    // forward-pass differential captures kernel-level arithmetic
    // differences (F16 precision, blocked compute, etc.) that the
    // offline weight-level L2 cannot see.
}
