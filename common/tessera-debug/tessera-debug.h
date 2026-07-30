#pragma once

//
// tessera-debug.h
//
// Tessera debug sidecar writers (Layer 1 of the runtime-aware calibration
// pipeline, see docs/runtime-aware-pipeline.md).
//
// The CPU/CUDA/Metal dequant instrumentation calls these helpers to dump
// dequantized weights to disk at runtime. The hook is off by default; it
// activates only when a non-empty dequant directory is configured via
// `set_dequant_dir()` (typically from the `--tessera-dequant-dir` CLI flag
// or the `LLAMA_TILE640_DEBUG_DEQUANT_DIR` environment variable).
//
// Concurrency: not thread-safe. One matmul kernel at a time per process.
//
// Typical call pattern from a backend instrumentation site:
//
//   if (tessera_debug::dequant_debug_enabled()) {
//       // materialize the dequantized weight into a F32 scratch buffer
//       // (one row at a time, or all rows up front)
//       tessera_debug::open_dequant_writer(t->name, ne0, ne1);
//       for (int64_t r = 0; r < ne0; r++) {
//           tessera_debug::write_dequant_row(r, scratch + r * ne1, ne1);
//       }
//       tessera_debug::close_dequant_writer();
//   }
//
// Each call site is responsible for its own scratch buffer; the sidecar
// writer is purely a stream sink and does no quantization work itself.
// The writer counts per-row outliers (|x| > threshold) as a Tier 0
// sensitivity signal consumed by the L3 metric and the L5 IterQuant
// orchestrator. See tessera/l3_outlier_report.py.
//

#include <cstdint>
#include <string>

namespace tessera_debug {

    // On-disk file format. All multi-byte fields are little-endian.
    //
    //   offset    size   field
    //   ------    ----   -----------------------------------------------
    //         0      4   magic              = "TDQT" (ASCII, no NUL)
    //         4      4   version            = DEQUANT_FILE_VERSION, uint32
    //         8      8   rows               = int64
    //        16      8   cols               = int64
    //        24      4   dtype              = DEQUANT_DTYPE_F32, uint32
    //   -- end of v1 header (28 bytes) ---------------------------------
    //        28      4   outlier_threshold  = float32, |x| cutoff (default 6.0)
    //        32      8   outlier_count_total= int64, sum of per-row counts
    //   -- end of v2 header (40 bytes) ---------------------------------
    //        40  R*4   row_outlier_count   = int32 per row, R = rows
    //   -- end of per-row header strip (40 + R*4 bytes) ----------------
    //   40+R*4 R*C*4 data                 = F32, row-major, streamed via
    //                                       `write_dequant_row`
    //
    // The v1 fields (magic, version, rows, cols, dtype) are unchanged from
    // format version 1. v2 adds the two new file-header fields and the
    // per-row outlier-count strip. A v1 reader (one that ignores the
    // version field and assumes F32 data starts at offset 28) will see
    // garbage on a v2 file; readers that check the version field and
    // dispatch on it are forward- and backward-compatible: a v2 reader
    // reads v1 files (data at offset 28) and v2 files (per-row strip then
    // data at offset 40 + R*4). The shipped Python reader
    // (tools/tessera/l3_outlier_report.py) does exactly this.
    //
    // The header is written by `open_dequant_writer`; the per-row strip
    // is filled by `close_dequant_writer` (after the F32 stream is
    // complete) by seeking back; the data is streamed by
    // `write_dequant_row`.

    static constexpr char     DEQUANT_FILE_MAGIC[4]   = { 'T', 'D', 'Q', 'T' };
    static constexpr uint32_t DEQUANT_FILE_VERSION    = 2;
    static constexpr uint32_t DEQUANT_DTYPE_F32       = 0;
    static constexpr uint32_t DEQUANT_DTYPE_F16       = 1;
    static constexpr uint32_t DEQUANT_DTYPE_BF16      = 2;

    // Default outlier threshold for the per-row |x| > t count. Chosen to
    // match the LLM.int8() precedent (Dettmers et al., 2022): 0.1% of
    // channels in transformer attention/FFN weights exceed ~6.0 in
    // absolute value, and those channels dominate the quantization loss
    // landscape. The sidecar stores the actual threshold used so the
    // reader does not need an out-of-band config.
    static constexpr float    DEQUANT_DEFAULT_OUTLIER_THRESHOLD = 6.0f;

    // Returns true when a dequant sidecar directory is configured (either
    // via the env var `LLAMA_TILE640_DEBUG_DEQUANT_DIR` at process start
    // or via an earlier call to `set_dequant_dir`). Cheap; no I/O.
    bool dequant_debug_enabled();

    // Configure (or reconfigure) the dequant sidecar directory. Last
    // write wins. Passing an empty path disables the sidecar output and
    // closes any currently open sidecar file.
    void set_dequant_dir(const std::string & path);

    // Configure the |x| > threshold cutoff used by the per-row outlier
    // counter. The threshold is recorded in the sidecar file header so
    // the reader can reproduce the count. Last write wins; defaults to
    // DEQUANT_DEFAULT_OUTLIER_THRESHOLD. No effect on an already-open
    // sidecar file (the header is written once at open time).
    void set_outlier_threshold(float threshold);

    // Returns the currently-configured outlier threshold (post any
    // set_outlier_threshold call, or the default if none).
    float outlier_threshold();

    // Open (or reuse) the sidecar file for `tensor_name` and write its
    // 40-byte v2 header. The first call opens the file at
    // `<dequant_dir>/<tensor_name>.dequant.f32` in truncating write mode;
    // subsequent calls with the same name return the cached writer. If
    // the writer is already open with a different `rows`/`cols`, the
    // current implementation closes the file and reopens it (the
    // mismatch is logged as a warning). See implementation comment.
    void open_dequant_writer(const char * tensor_name, int64_t rows, int64_t cols);

    // Append a single row of `n` F32 values to the currently-open
    // sidecar file. `n` should match the `cols` passed to
    // `open_dequant_writer`; otherwise a warning is logged and the
    // shorter count is written. No-op if the writer is not open.
    // `row_idx` is currently informational only (the file is purely
    // sequential, the per-row outlier-count strip is finalized at close).
    void write_dequant_row(int64_t row_idx, const float * data, int64_t n);

    // Flush and close the currently-open sidecar file. On v2 files,
    // also seeks back to the per-row outlier-count strip and writes the
    // collected counts, then updates the file-header total. Idempotent.
    // Safe to call when no file is open.
    void close_dequant_writer();

} // namespace tessera_debug
