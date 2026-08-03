#pragma once

//
// tessera-matmul-output.h
//
// Tessera Layer-2 matmul-output sidecar writer (Layer 2 of the
// runtime-aware calibration pipeline, see docs/runtime-aware-pipeline.md).
//
// The L1 sidecar (`tessera-debug.h`) captures the kernel's effective
// dequantized WEIGHT per matmul invocation. This header captures the
// kernel's matmul OUTPUT (the F32 dst tensor of the matmul op) for the
// same set of tensors, so the L2 forward-pass differential can compare
// the two forwards' post-matmul distributions at every position.
//
// The on-disk layout is the v3 TDQT header (magic = "TPMO" so the file
// is recognizable as a matmul-output sidecar distinct from a
// dequant-output sidecar; same v3 header shape: 28-byte v1 header +
// 12-byte v2 header + per-row outlier strip + per-row v3 meta strip).
// The data block is F32, row-major, with one row per matmul invocation
// (rows = n_tokens for a (n_tokens x output_dim) dst). One file per
// tensor name; first call opens, subsequent calls append.
//
// Concurrency: thread-safe. A recursive mutex serializes open/write/close
// for the per-tensor SidecarStream state, mirroring the L1 sidecar's
// thread-safety contract (Metal addCompletedHandler blocks can fire in
// parallel across multiple Tile640 matmuls).
//
// Hook sites: `ggml_compute_forward_mul_mat` in ggml-cpu/ggml-cpu.c
// calls `cpu_dump_matmul_output(dst, params, src0, tensor_name)` once
// per matmul when the sidecar is enabled. The hook is off by default;
// it activates only when a non-empty matmul-output directory is
// configured via `set_matmul_output_dir()` (typically from the
// `--tessera-matmul-output-dir` CLI flag or the
// `LLAMA_TILE640_DEBUG_MATMUL_OUTPUT_DIR` env var).
//
//

#include <cstdint>
#include <string>

namespace tessera_matmul_output {

    // Suffix for the matmul-output sidecar file kind. Distinct from
    // the L1 sidecar suffix so the L2 Python reader and the L1
    // reader can route the file without inspecting the magic.
    static constexpr const char * MATMUL_OUTPUT_FILE_SUFFIX = ".matmul-output.f32";

    // The header magic is the four ASCII bytes "TPMO" (Tessera PMatmul
    // Output). It is distinct from the L1/L1.5 "TDQT" magic so the two
    // file kinds cannot be confused if both are written to the same
    // directory (e.g. during a calibration pass that enables both L1
    // dequant capture and L2 matmul-output capture).
    static constexpr char MATMUL_OUTPUT_FILE_MAGIC[4] = { 'T', 'P', 'M', 'O' };

    // Hard cap on the number of rows (captured / post-stride) per
    // tensor. The matmul hook in ggml-cpu/ggml-cpu.c refuses to capture
    // any tensor whose captured row count exceeds this value (a warning
    // is logged once per offending tensor and the capture is silently
    // dropped). This is the audit-mandated disk-usage safety bound:
    // 4096 rows * 4 bytes per F32 * max_output_dim (e.g. 151936 for
    // a 12B output) is bounded to ~2.5 GB per file even in the
    // pathological case, which is small enough to fit on a free
    // 1 TB disk with many other tensors.
    static constexpr int64_t MATMUL_OUTPUT_MAX_ROWS = 4096;

    // Default capture stride. Stride N captures rows 0, N, 2N, ... so
    // the per-file data block is rows/stride * cols * 4 bytes. The
    // default 32 is the audit-mandated safe default: a 32k-token
    // forward pass yields 1024 captured rows per tensor, well below
    // the 4096-row cap, and a 12B model at 32k ctx writes at most
    // ~1 GB of sidecar data per forward pass (vs. 32 GB with
    // stride=1). Can be overridden via set_matmul_output_stride() or
    // the LLAMA_TILE640_DEBUG_MATMUL_OUTPUT_STRIDE env var.
    static constexpr int64_t MATMUL_OUTPUT_DEFAULT_STRIDE = 32;

    // Returns true when a matmul-output sidecar directory is configured
    // (either via the env var `LLAMA_TILE640_DEBUG_MATMUL_OUTPUT_DIR`
    // at process start or via an earlier call to `set_matmul_output_dir`).
    // Cheap; no I/O.
    bool matmul_output_capture_enabled();

    // Returns true iff the tensor name matches the audit-mandated
    // L2 allowlist: only the linear layer matmuls of the per-layer
    // transformer block (blk.N.attn_[qkvo].weight and
    // blk.N.ffn_(gate|up|down).weight) and the model head
    // (output.weight). Token embedding, RoPE rotation matrices,
    // normalization gamma vectors, attention bias vectors, and any
    // other small ops that go through ggml_compute_forward_mul_mat
    // are silently skipped. This cuts the file count from "every
    // matmul in the graph" (~hundreds for a 0.5B model, ~thousands
    // for a 12B model) to "the linear layers that the L2 differential
    // is actually designed to score".
    //
    // The allowlist is intentionally conservative: a tensor that the
    // L2 schema does not have a clear forward-pass metric for
    // (e.g. embeddings) is excluded. The full list is:
    //   blk\.[0-9]+\.attn_(q|k|v|output)\.weight
    //   blk\.[0-9]+\.ffn_(gate|up|down)\.weight
    //   output\.weight
    bool matmul_output_tensor_allowed(const char * tensor_name);

    // Configure (or reconfigure) the matmul-output sidecar directory.
    // Last write wins. Passing an empty path disables the sidecar output
    // and closes any currently open sidecar file.
    void set_matmul_output_dir(const std::string & path);

    // Returns the currently-configured directory (empty when disabled).
    const std::string & matmul_output_dir();

    // Configure the matmul-output capture stride. When stride > 1, only
    // every Nth row of the dst tensor is written (rows 0, N, 2N, ...).
    // Default is MATMUL_OUTPUT_DEFAULT_STRIDE (32). Can also be set via
    // the env var `LLAMA_TILE640_DEBUG_MATMUL_OUTPUT_STRIDE` at process
    // start. Values < 1 are clamped to 1.
    void set_matmul_output_stride(int64_t stride);

    // Returns the currently-configured capture stride (default
    // MATMUL_OUTPUT_DEFAULT_STRIDE).
    int64_t matmul_output_stride();

    // Returns true if the most recent open call to the sidecar writer
    // APPENDED to an existing file (the audit-mandated "second-and-
    // onward" mode) rather than truncating it. The first call to
    // open_matmul_output_writer for a given tensor name always
    // truncates (creates a fresh file); subsequent calls for the
    // same tensor name (e.g. prefill then decode, or
    // multi-prompt imatrix chunks) append. The Python L2 reader can
    // use this flag to detect that the file holds multi-invocation
    // data; the per-row v3 strip records the dispatch count.
    bool matmul_output_last_open_appended();

    // Returns the row count of the currently-open sidecar stream, or
    // -1 if no stream is open. The CPU hook uses this to detect
    // audit-fix-4 drops: if open_matmul_output_writer() was called
    // with shape (rows, cols) and the post-call stream still has
    // a different (rows, cols), the open was dropped (different
    // shape; prefill data was preserved; decode write must skip).
    int64_t matmul_output_stream_rows();

    // Returns the column count of the currently-open sidecar stream,
    // or -1 if no stream is open. See matmul_output_stream_rows().
    int64_t matmul_output_stream_cols();

    // Open (or reuse) the matmul-output sidecar file for `tensor_name`
    // and write its v3 header. The first call opens the file at
    // `<dir>/<tensor_name>.matmul-output.f32` in truncating write mode;
    // subsequent calls with the same name return the cached writer. If
    // the writer is already open with a different `rows`/`cols`, the
    // current implementation closes the file and reopens it (the
    // mismatch is logged as a warning).
    void open_matmul_output_writer(const char * tensor_name,
                                   int64_t rows, int64_t cols);

    // Append a single row of `n` F32 values (the F32 dst of the matmul
    // op) to the currently-open sidecar file. `n` should match the
    // `cols` passed to `open_matmul_output_writer`; otherwise a warning
    // is logged and the shorter count is written. No-op if the writer
    // is not open.
    void write_matmul_output_row(int64_t row_idx, const float * data, int64_t n);

    // Record the per-row v3 metadata (wall-clock timing + dispatch
    // metadata) for row `row_idx` in the currently-open sidecar. The
    // strip is finalized at `close_matmul_output_writer` time.
    // `timing_ns` captures the wall-clock for the matmul work as
    // observed by the hook; `kernel_id` is the backend's identifier for
    // the matmul kernel (CPU BLAS / CPU generic / Metal); `dispatch_count`
    // is the number of kernel dispatches for this row (typically 1).
    void set_matmul_output_row_meta(int64_t row_idx,
                                    uint64_t timing_ns,
                                    uint32_t kernel_id,
                                    uint32_t dispatch_count);

    // Flush and close the currently-open sidecar file. On v3 files,
    // also seeks back to the per-row v3 strip and writes the collected
    // values, then updates the file-header total. Idempotent. Safe to
    // call when no file is open.
    void close_matmul_output_writer();

} // namespace tessera_matmul_output
