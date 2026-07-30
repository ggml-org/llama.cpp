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
//

#include <cstdint>
#include <string>

namespace tessera_debug {

    // On-disk file format. All multi-byte fields are little-endian.
    //
    //   offset  size   field
    //   ------  ----   -----------------------------------------------
    //        0     4   magic        = "TDQT" (ASCII, no NUL)
    //        4     4   version      = DEQUANT_FILE_VERSION, uint32
    //        8     8   rows         = int64
    //       16     8   cols         = int64
    //       24     4   dtype        = DEQUANT_DTYPE_F32, uint32
    //       28  R*C*4 data         = F32, row-major, streamed via
    //                               `write_dequant_row`
    //
    // The header is written by `open_dequant_writer`; the data is
    // streamed by `write_dequant_row` and the file is closed by
    // `close_dequant_writer`.

    static constexpr char     DEQUANT_FILE_MAGIC[4]   = { 'T', 'D', 'Q', 'T' };
    static constexpr uint32_t DEQUANT_FILE_VERSION    = 1;
    static constexpr uint32_t DEQUANT_DTYPE_F32       = 0;
    static constexpr uint32_t DEQUANT_DTYPE_F16       = 1;
    static constexpr uint32_t DEQUANT_DTYPE_BF16      = 2;

    // Returns true when a dequant sidecar directory is configured (either
    // via the env var `LLAMA_TILE640_DEBUG_DEQUANT_DIR` at process start
    // or via an earlier call to `set_dequant_dir`). Cheap; no I/O.
    bool dequant_debug_enabled();

    // Configure (or reconfigure) the dequant sidecar directory. Last
    // write wins. Passing an empty path disables the sidecar output and
    // closes any currently open sidecar file.
    void set_dequant_dir(const std::string & path);

    // Open (or reuse) the sidecar file for `tensor_name` and write its
    // 28-byte header. The first call opens the file at
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
    // sequential).
    void write_dequant_row(int64_t row_idx, const float * data, int64_t n);

    // Flush and close the currently-open sidecar file. Idempotent.
    // Safe to call when no file is open.
    void close_dequant_writer();

} // namespace tessera_debug
