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

    // Returns true when a matmul-output sidecar directory is configured
    // (either via the env var `LLAMA_TILE640_DEBUG_MATMUL_OUTPUT_DIR`
    // at process start or via an earlier call to `set_matmul_output_dir`).
    // Cheap; no I/O.
    bool matmul_output_capture_enabled();

    // Configure (or reconfigure) the matmul-output sidecar directory.
    // Last write wins. Passing an empty path disables the sidecar output
    // and closes any currently open sidecar file.
    void set_matmul_output_dir(const std::string & path);

    // Returns the currently-configured directory (empty when disabled).
    const std::string & matmul_output_dir();

    // Configure the matmul-output capture stride. When stride > 1, only
    // every Nth row of the dst tensor is written (rows 0, N, 2N, ...).
    // Default is 1 (capture all rows). Can also be set via the env var
    // `LLAMA_TILE640_DEBUG_MATMUL_OUTPUT_STRIDE` at process start.
    // Values < 1 are clamped to 1.
    void set_matmul_output_stride(int64_t stride);

    // Returns the currently-configured capture stride (default 1).
    int64_t matmul_output_stride();

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
