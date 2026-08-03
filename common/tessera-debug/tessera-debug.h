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
// Concurrency: thread-safe. A recursive mutex serializes the per-tensor
// open/write_rows/close sequence, so concurrent backend callbacks (e.g.
// Metal addCompletedHandler blocks firing in parallel across multiple
// Tile640 matmuls) cannot interleave their writes on the file-static
// SidecarStream state. Each public entry point acquires the lock.
//
// The on-disk format is described in `DEQUANT_FILE_LAYOUT_V3` below.
//
// Typical call pattern from a backend instrumentation site:
//
//   if (tessera_debug::dequant_debug_enabled()) {
//       // materialize the dequantized weight into a F32 scratch buffer
//       // (one row at a time, or all rows up front)
//       tessera_debug::open_dequant_writer(t->name, ne0, ne1);
//       for (int64_t r = 0; r < ne0; r++) {
//           auto t0 = std::chrono::steady_clock::now();
//           tessera_debug::write_dequant_row(r, scratch + r * ne1, ne1);
//           auto t1 = std::chrono::steady_clock::now();
//           uint64_t ns = (uint64_t)
//               std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
//           tessera_debug::set_dequant_row_meta(r, ns, kernel_id, 1);
//       }
//       tessera_debug::close_dequant_writer();
//   }
//
// Each call site is responsible for its own scratch buffer; the sidecar
// writer is purely a stream sink and does no quantization work itself.
// The writer counts per-row outliers (|x| > threshold) as a Tier 0
// sensitivity signal consumed by the L3 metric and the L5 IterQuant
// orchestrator.
//

#include <cstdint>
#include <string>

namespace tessera_debug {

    // On-disk file format. All multi-byte fields are little-endian.
    //
    //   offset              size   field
    //   ------              ----   --------------------------------------
    //         0                4   magic              = "TDQT" (ASCII, no NUL)
    //         4                4   version            = DEQUANT_FILE_VERSION, uint32 (= 3)
    //         8                8   rows               = int64
    //        16                8   cols               = int64
    //        24                4   dtype              = DEQUANT_DTYPE_F32, uint32
    //   -- end of v1 header (28 bytes) ----------------------------------
    //        28                4   outlier_threshold  = float32, |x| cutoff (default 6.0)
    //        32                8   outlier_count_total= int64, sum of per-row counts
    //   -- end of v2 header (40 bytes) ----------------------------------
    //        40           R * 4   row_outlier_count   = int32 per row, R = rows
    //   -- end of per-row v2 strip (40 + R*4 bytes) ---------------------
    //   40+R*4           R * 24   row_v3_meta         = per-row, 24 bytes:
    //                        8       timing_ns            uint64
    //                        4       kernel_id            uint32
    //                        4       dispatch_count       uint32
    //                        8       reserved             uint64 (zero for now)
    //   -- end of per-row v3 strip (40 + R*4 + R*24 bytes) -------------
    //   40+R*4+R*24     R * C*4   data               = F32, row-major, streamed
    //                                                  via `write_dequant_row`
    //
    // Reader compatibility:
    //   - v1 reader: reads data at offset 28, sees header corruption on v2/v3
    //     files. Backward compat path: a v1 reader that does NOT check the
    //     version field will read 4 bytes of magic + 4 bytes of version + ...
    //     + the v1 header, then continue at offset 28 and read the F32 data.
    //     On a v3 file that "data" is actually the outlier_threshold +
    //     outlier_count_total + the per-row outlier_count strip + the v3
    //     per-row strip, which is garbage as F32. The shipped Python
    //     reader (tools/tessera/l3_sidecar_v3_reader.py) dispatches on
    //     the version field and skips ahead; a naive v1 reader will produce
    //     nonsense (NOT supported as a compatibility mode in this layout).
    //   - v2 reader: reads the v2 fields, treats the v3 per-row strip as
    //     opaque padding, then reads data at offset 40 + R*4. The
    //     per-row v3 strip is skipped (it is not needed for the v2
    //     contract).
    //   - v3 reader: reads all fields including the per-row v3 strip.
    //     A v3 reader reading a v1 or v2 file produces zeros for the
    //     missing fields (timing, kernel_id, dispatch_count) and
    //     recognizes the data offset accordingly.
    //
    // The L1.5 reference sidecar (`.act.dequant.f32`) uses the same v3
    // schema; the F32 data is the F32-cast of the FP16 reference (in the
    // quantized-source case this is identical to the L1 dequant output
    // until a future refactor passes the FP16 reference to the hook).
    //
    // The header is written by `open_dequant_writer` (and
    // `open_fp16_reference_writer`); the per-row strips are filled by
    // `close_dequant_writer` (and `close_fp16_reference_writer`) by
    // seeking back; the data is streamed by `write_dequant_row` (and
    // `write_fp16_reference_row`).

    static constexpr char     DEQUANT_FILE_MAGIC[4]   = { 'T', 'D', 'Q', 'T' };
    static constexpr uint32_t DEQUANT_FILE_VERSION    = 3;
    static constexpr uint32_t DEQUANT_DTYPE_F32       = 0;
    static constexpr uint32_t DEQUANT_DTYPE_F16       = 1;
    static constexpr uint32_t DEQUANT_DTYPE_BF16      = 2;

    // Suffixes for the three sidecar file kinds:
    //   - `.dequant.f32`         L1 dequant sidecar (always written when
    //                             the hook is enabled).
    //   - `.act.dequant.f32`     L1.5 reference sidecar, F32 data block
    //                             (legacy W4A4 mode; preserved for back-compat
    //                             with existing reader/test code).
    //   - `.act.dequant.f16`     L1.5 reference sidecar, FP16 data block
    //                             (default when W4A4 is enabled; the whole
    //                             point of L1.5 is the FP16 ground truth).
    // The W4A4 mode controls whether the L1.5 sidecar is written at all
    // (see `set_dequant_mode` / `dequant_w4a4_enabled`); the L1.5 dtype
    // is independent (see `set_l15_dtype` / `l15_dtype_is_f16`).
    static constexpr const char * DEQUANT_FILE_SUFFIX_L1      = ".dequant.f32";
    static constexpr const char * DEQUANT_FILE_SUFFIX_L15_F32 = ".act.dequant.f32";
    static constexpr const char * DEQUANT_FILE_SUFFIX_L15_F16 = ".act.dequant.f16";

    // Dtype strings for `set_l15_dtype`. The L1.5 reference is stored as
    // FP16 by default (the whole point of L1.5 is the FP16 ground truth,
    // distinct from the F32 L1 dequant). "f32" is the legacy W4A4
    // behavior; kept available for back-compat with existing reader code
    // and for users who explicitly want a higher-precision reference.
    static constexpr const char * L15_DTYPE_F16 = "f16";
    static constexpr const char * L15_DTYPE_F32 = "f32";

    // Default outlier threshold for the per-row |x| > t count. Chosen to
    // match the LLM.int8() precedent (Dettmers et al., 2022): 0.1% of
    // channels in transformer attention/FFN weights exceed ~6.0 in
    // absolute value, and those channels dominate the quantization loss
    // landscape. The sidecar stores the actual threshold used so the
    // reader does not need an out-of-band config.
    static constexpr float    DEQUANT_DEFAULT_OUTLIER_THRESHOLD = 6.0f;

    // Mode string for the W4A4 calibration mode. When set, the kernel
    // writes BOTH the L1 dequant sidecar (existing) and the L1.5 FP16
    // reference sidecar (new). Without this mode, only the L1 sidecar
    // is written (back-compat).
    static constexpr const char * DEQUANT_MODE_W4A4 = "w4a4";

    // Returns true when a dequant sidecar directory is configured (either
    // via the env var `LLAMA_TILE640_DEBUG_DEQUANT_DIR` at process start
    // or via an earlier call to `set_dequant_dir`). Cheap; no I/O.
    bool dequant_debug_enabled();

    // Configure (or reconfigure) the dequant sidecar directory. Last
    // write wins. Passing an empty path disables the sidecar output and
    // closes any currently open sidecar file.
    void set_dequant_dir(const std::string & path);

    // Configure the dequant mode. Empty string (default) writes only the
    // L1 sidecar; "w4a4" writes both the L1 and L1.5 sidecars. The mode
    // can also be set via the `LLAMA_TILE640_DEBUG_DEQUANT_MODE` env var
    // at process start. Last write wins; no effect on an already-open
    // sidecar file.
    void set_dequant_mode(const std::string & mode);

    // Returns the currently-configured mode (empty by default, or "w4a4"
    // when the W4A4 calibration mode is active).
    const std::string & dequant_mode();

    // Returns true iff the current mode is "w4a4" (L1.5 sidecar enabled).
    bool dequant_w4a4_enabled();

    // Configure the L1.5 reference dtype. "f16" (default) writes the L1.5
    // sidecar as FP16 ground truth (2 bytes/value, file suffix
    // `.act.dequant.f16`); "f32" writes the legacy F32 reference (4
    // bytes/value, suffix `.act.dequant.f32`). The choice is independent
    // of the W4A4 mode toggle above; both are required for the L1.5 file
    // to be written at all. Last write wins; can also be set via the
    // `LLAMA_TESSERA_L15_DTYPE` env var at process start. No effect on an
    // already-open sidecar file (the header is written once at open time,
    // and the dtype is recorded in it).
    void set_l15_dtype(const std::string & dtype);

    // Returns the currently-configured L1.5 dtype (default "f16").
    const std::string & l15_dtype();

    // Convenience: true when the L1.5 dtype is FP16 (the default).
    bool l15_dtype_is_f16();

    // Configure the |x| > threshold cutoff used by the per-row outlier
    // counter. The threshold is recorded in the sidecar file header so
    // the reader can reproduce the count. Last write wins; defaults to
    // DEQUANT_DEFAULT_OUTLIER_THRESHOLD. No effect on an already-open
    // sidecar file (the header is written once at open time).
    void set_outlier_threshold(float threshold);

    // Returns the currently-configured outlier threshold (post any
    // set_outlier_threshold call, or the default if none).
    float outlier_threshold();

    // Configure the row capture stride. When stride > 1, only every
    // Nth row is written to the sidecar (rows 0, N, 2N, ...). Default
    // is 1 (capture all rows). Can also be set via the env var
    // `LLAMA_TILE640_DEBUG_DEQUANT_STRIDE` at process start. Values
    // < 1 are clamped to 1.
    void set_dequant_stride(int64_t stride);

    // Returns the currently-configured row capture stride (default 1).
    int64_t dequant_stride();

    // Open (or reuse) the L1 dequant sidecar file for `tensor_name` and
    // write its v3 header. The first call opens the file at
    // `<dequant_dir>/<tensor_name>.dequant.f32` in truncating write mode;
    // subsequent calls with the same name return the cached writer. If
    // the writer is already open with a different `rows`/`cols`, the
    // current implementation closes the file and reopens it (the
    // mismatch is logged as a warning). See implementation comment.
    void open_dequant_writer(const char * tensor_name, int64_t rows, int64_t cols);

    // Append a single row of `n` F32 values to the currently-open L1
    // sidecar file. `n` should match the `cols` passed to
    // `open_dequant_writer`; otherwise a warning is logged and the
    // shorter count is written. No-op if the writer is not open.
    // `row_idx` is currently informational only (the file is purely
    // sequential, the per-row strips are finalized at close).
    void write_dequant_row(int64_t row_idx, const float * data, int64_t n);

    // Record the per-row v3 metadata (wall-clock timing + kernel-launch
    // metadata) for row `row_idx` in the currently-open L1 sidecar. The
    // strip is finalized at `close_dequant_writer` time. The `timing_ns`
    // field captures the wall-clock for the dequant + matmul work as
    // observed by the hook; `kernel_id` is the backend's identifier for
    // the dequant kernel (CPU/MM/Qx); `dispatch_count` is the number of
    // kernel dispatches for this row (typically 1). `reserved` is zero
    // for now and reserved for future fields. No-op if the writer is
    // not open or `row_idx` is out of range (a warning is logged).
    void set_dequant_row_meta(int64_t row_idx,
                              uint64_t    timing_ns,
                              uint32_t    kernel_id,
                              uint32_t    dispatch_count);

    // Flush and close the currently-open L1 sidecar file. On v2/v3
    // files, also seeks back to the per-row outlier-count strip and the
    // per-row v3 strip and writes the collected values, then updates the
    // file-header total. Idempotent. Safe to call when no file is open.
    void close_dequant_writer();

    // Open (or reuse) the L1.5 FP16-reference sidecar file for
    // `tensor_name` and write its v3 header. The file path and the
    // on-disk dtype depend on the L1.5 dtype config (see `set_l15_dtype`):
    //   - dtype "f16" -> `<dequant_dir>/<tensor_name>.act.dequant.f16`,
    //     header dtype = DEQUANT_DTYPE_F16, data block is 2 bytes/value.
    //   - dtype "f32" -> `<dequant_dir>/<tensor_name>.act.dequant.f32`,
    //     header dtype = DEQUANT_DTYPE_F32, data block is 4 bytes/value
    //     (legacy W4A4 behavior; preserved for back-compat).
    // The data is the FP16 ground truth (or the F32 reference, when
    // dtype = "f32") captured at the runtime hook. No-op unless the mode
    // is "w4a4" (see `set_dequant_mode` and `dequant_w4a4_enabled`).
    void open_fp16_reference_writer(const char * tensor_name, int64_t rows, int64_t cols);

    // Append a single row of `n` FP16 values to the currently-open L1.5
    // sidecar file. The dtype must be "f16" (the default) - the call is
    // a no-op otherwise with a warning to stderr. The backend hook is
    // expected to convert from F32 to FP16 (via `ggml_fp32_to_fp16`,
    // proper rounding) before calling. `n` should match the `cols`
    // passed to `open_fp16_reference_writer`; a mismatch logs a warning
    // and the shorter count is written.
    void write_fp16_reference_row(int64_t row_idx, const uint16_t * data, int64_t n);

    // Back-compat helper: convert `n` F32 values to FP16 (proper
    // rounding via ggml_fp32_to_fp16) and write them as a single row of
    // the L1.5 sidecar. Used by call sites that already hold the F32
    // buffer and want the writer to do the conversion. The dtype must
    // be "f16" (the default) - the call is a no-op otherwise. This
    // convenience is the same path the legacy F32 `write_fp16_reference_row`
    // callers used to take; the writer now converts to FP16 internally
    // instead of writing the F32 buffer as the reference data.
    void write_fp16_reference_row_from_f32(int64_t row_idx, const float * data, int64_t n);

    // Record the per-row v3 metadata for the L1.5 sidecar. Same fields
    // and semantics as `set_dequant_row_meta`.
    void set_fp16_reference_row_meta(int64_t row_idx,
                                     uint64_t    timing_ns,
                                     uint32_t    kernel_id,
                                     uint32_t    dispatch_count);

    // Flush and close the currently-open L1.5 sidecar file. Same
    // semantics as `close_dequant_writer`.
    void close_fp16_reference_writer();

    // Provenance sidecar API. Each L1 / L1.5 sidecar file is paired
    // with a JSON provenance file at
    // `<dequant_dir>/<tensor_name>.dequant.f32.provenance.json` (or
    // `*.act.dequant.f32.provenance.json` for the L1.5 sidecar). The
    // JSON schema is:
    //   {
    //     "model": "<env TESSERA_TELEMETRY_MODEL>",
    //     "calibration_corpus": "<env TESSERA_TELEMETRY_CALIBRATION_CORPUS>",
    //     "calibration_corpus_hash": "<env TESSERA_TELEMETRY_CALIBRATION_CORPUS_HASH>",
    //     "kernel_version": "<git describe of HEAD>",
    //     "l1_sidecar_version": <int, DEQUANT_FILE_VERSION at write>,
    //     "imatrix_version": <int, currently 2>,
    //     "created_at": "<ISO 8601 UTC>",
    //     "tessera_main_tip": "<short SHA of local main branch>"
    //   }
    //
    // The provenance file is written by the close_*_writer functions
    // when a sidecar is successfully written. The `kernel_version` and
    // `tessera_main_tip` fields are auto-populated from the build
    // (see tessera-build-info.h, generated by CMake at configure time).
    void set_telemetry_model(const std::string & model);
    void set_telemetry_calibration_corpus(const std::string & corpus);
    void set_telemetry_calibration_corpus_hash(const std::string & hash);

    // Returns the auto-populated kernel version string (compile-time
    // value, "<branch> <short-sha>" of HEAD at configure time).
    const char * tessera_kernel_version();

    // Returns the auto-populated tessera_main_tip string (compile-time
    // value, short SHA of the local `main` branch at configure time).
    const char * tessera_main_tip();

    // imatrix version (for provenance). Currently 2.
    int tessera_imatrix_version();

} // namespace tessera_debug
