#include "tessera-debug.h"

// Intentionally NOT including common/log.h: this translation unit is
// compiled into the llama-tessera-debug static library, which sits
// below common/ in the layering hierarchy. Pulling in log.h would
// drag common/log.cpp (and the rest of llama-common) into the link
// for every backend. The sidecar writer is a low-level stream sink;
// its diagnostic messages go straight to stderr instead of through
// the common log infrastructure.

#include "tessera-build-info.h"

// FP16 -> F32 conversion for the FP16 row outlier counter. We do not
// pull in ggml.h from this translation unit (the sidecar writer is a
// pure stream sink, below the ggml layering); the local copy of
// ggml_compute_fp16_to_fp32 mirrors ggml/src/ggml-impl.h:384-405 and
// is bit-identical to the canonical implementation. The F32 -> FP16
// direction (used by the public `write_fp16_reference_row_from_f32`
// convenience helper) is the symmetric routine below.
namespace {
union fp32_bits {
    uint32_t as_bits;
    float    as_value;
};
inline float fp32_from_bits(uint32_t w) {
    fp32_bits u; u.as_bits = w; return u.as_value;
}
inline uint32_t fp32_to_bits(float f) {
    fp32_bits u; u.as_value = f; return u.as_bits;
}
inline float local_fp16_to_fp32(uint16_t h) {
    const uint32_t w = (uint32_t) h << 16;
    const uint32_t sign = w & UINT32_C(0x80000000);
    const uint32_t two_w = w + w;
    const uint32_t exp_offset = UINT32_C(0xE0) << 23;
    const float exp_scale = 0x1.0p-112f;
    const float normalized_value = fp32_from_bits((two_w >> 4) + exp_offset) * exp_scale;
    const uint32_t magic_mask = UINT32_C(126) << 23;
    const float magic_bias = 0.5f;
    const float denormalized_value = fp32_from_bits((two_w >> 17) | magic_mask) - magic_bias;
    const uint32_t denormalized_cutoff = UINT32_C(1) << 27;
    const uint32_t result = sign |
        (two_w < denormalized_cutoff ? fp32_to_bits(denormalized_value) : fp32_to_bits(normalized_value));
    return fp32_from_bits(result);
}
inline uint16_t local_fp32_to_fp16(float f) {
    const float scale_to_inf = 0x1.0p+112f;
    const float scale_to_zero = 0x1.0p-110f;
    float base = (fabsf(f) * scale_to_inf) * scale_to_zero;

    const uint32_t w = fp32_to_bits(f);
    const uint32_t shl1_w = w + w;
    const uint32_t sign = w & UINT32_C(0x80000000);
    uint32_t bias = shl1_w & UINT32_C(0xFF000000);
    if (bias < UINT32_C(0x71000000)) {
        bias = UINT32_C(0x71000000);
    }

    base = fp32_from_bits((bias >> 1) + UINT32_C(0x07800000)) + base;
    const uint32_t bits = fp32_to_bits(base);
    const uint32_t exp_bits = (bits >> 13) & UINT32_C(0x00007C00);
    const uint32_t mantissa_bits = bits & UINT32_C(0x00000FFF);
    const uint32_t nonsign = exp_bits + mantissa_bits;
    return (uint16_t) ((sign >> 16) | (shl1_w > UINT32_C(0xFF000000) ? UINT16_C(0x7E00) : nonsign));
}
} // anonymous namespace

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <string>
#include <system_error>
#include <vector>

namespace tessera_debug {

// File-static state. Guarded by g_writer_mutex; a per-tensor
// open/write_rows/close sequence is atomic across concurrent backend
// callbacks (e.g. Metal addCompletedHandler blocks firing in parallel).
static float g_outlier_threshold = DEQUANT_DEFAULT_OUTLIER_THRESHOLD;

// Recursive so the public open/write/close entry points can each acquire
// the lock without deadlocking when one sequence nests another (it does
// not today, but the open_dequant_writer + open_fp16_reference_writer
// pattern means two opens can be in flight from one caller).
static std::recursive_mutex g_writer_mutex;

namespace {

// One-time env-var snapshot. The dequant dir, mode, and telemetry fields
// can also be set programmatically via set_*; the env vars are only
// read on the first call. We use a once-flag pattern via a local static
// init function to avoid pulling in <atomic> for a single bool.
struct EnvState {
    std::string dequant_dir;
    std::string mode;
    std::string l15_dtype;     // L1.5 reference dtype: "f16" (default) or "f32" (legacy)
    std::string model;
    std::string calibration_corpus;
    std::string calibration_corpus_hash;
    int64_t     stride = 1;
    bool        initialized = false;
};

EnvState & env_state() {
    static EnvState s;
    return s;
}

void ensure_env_loaded() {
    EnvState & s = env_state();
    if (s.initialized) {
        return;
    }
    s.initialized = true;
    if (const char * v = std::getenv("LLAMA_TILE640_DEBUG_DEQUANT_DIR"); v != nullptr) {
        s.dequant_dir = v;
    }
    if (const char * v = std::getenv("LLAMA_TILE640_DEBUG_DEQUANT_MODE"); v != nullptr) {
        s.mode = v;
    }
    if (const char * v = std::getenv("LLAMA_TESSERA_L15_DTYPE"); v != nullptr) {
        s.l15_dtype = v;
    }
    if (const char * v = std::getenv("TESSERA_TELEMETRY_MODEL"); v != nullptr) {
        s.model = v;
    }
    if (const char * v = std::getenv("TESSERA_TELEMETRY_CALIBRATION_CORPUS"); v != nullptr) {
        s.calibration_corpus = v;
    }
    if (const char * v = std::getenv("TESSERA_TELEMETRY_CALIBRATION_CORPUS_HASH"); v != nullptr) {
        s.calibration_corpus_hash = v;
    }
    if (const char * v = std::getenv("LLAMA_TILE640_DEBUG_DEQUANT_STRIDE"); v != nullptr) {
        int64_t st = (int64_t) atoll(v);
        s.stride = (st < 1) ? 1 : st;
    }
}

// Per-row v3 metadata. Stored row-major in the file; the per-row strip
// is 24 bytes (8 timing + 4 kernel_id + 4 dispatch_count + 8 reserved).
struct RowV3Meta {
    uint64_t timing_ns      = 0;
    uint32_t kernel_id      = 0;
    uint32_t dispatch_count = 0;
    uint64_t reserved       = 0;
};

// State for a single open sidecar file. Both the L1 and L1.5 sidecars
// use the same v3 layout; they differ in the file suffix and in the
// on-disk dtype of the data block. The L1 sidecar is always F32; the
// L1.5 sidecar's dtype is configurable (FP16 by default, F32 for
// legacy W4A4 back-compat). The header field `dtype` is taken from
// `file_dtype` so a reader that dispatches on the header can locate
// the data correctly.
struct SidecarStream {
    std::ofstream            ofs;
    std::string              tensor_name;
    int64_t                  rows = 0;
    int64_t                  cols = 0;
    std::vector<int32_t>     row_outlier_counts;
    int64_t                  outlier_count_total = 0;
    std::vector<RowV3Meta>   row_meta;        // size = rows
    std::string              suffix;          // ".dequant.f32", ".act.dequant.f32", or ".act.dequant.f16"
    uint32_t                 file_dtype;      // DEQUANT_DTYPE_F32 (=0) or DEQUANT_DTYPE_F16 (=1)
    bool                     open = false;
};

SidecarStream g_l1;
SidecarStream g_l15;

bool sidecar_open_impl(SidecarStream & s,
                       const std::string & dir,
                       const char * tensor_name,
                       int64_t rows, int64_t cols,
                       const char * suffix,
                       uint32_t file_dtype) {
    if (dir.empty() || tensor_name == nullptr) {
        return false;
    }
    if (file_dtype != DEQUANT_DTYPE_F32 && file_dtype != DEQUANT_DTYPE_F16) {
        fprintf(stderr, "tessera_debug: bad file_dtype %u for '%s'\n",
                file_dtype, tensor_name);
        return false;
    }

    // Fast path: same tensor, same shape, same suffix, same dtype ->
    // keep the open file. The dtype check is essential for the L1.5
    // case: a process that switches the L1.5 dtype between open calls
    // (e.g. via set_l15_dtype) would otherwise silently mix F32 and
    // FP16 rows in the same file. Reopen the file with the new dtype
    // instead.
    if (s.open && s.tensor_name == tensor_name &&
        s.rows == rows && s.cols == cols &&
        s.suffix == suffix && s.file_dtype == file_dtype) {
        return true;
    }

    // Shape mismatch on an already-open writer: close + reopen. We
    // intentionally avoid rewriting the header in-place; the simpler
    // behavior is to truncate the file under the same name. The caller
    // is expected to provide a consistent (name, rows, cols, dtype) for
    // any given tensor.
    if (s.open) {
        if (s.tensor_name == tensor_name &&
            (s.rows != rows || s.cols != cols || s.suffix != suffix ||
             s.file_dtype != file_dtype)) {
            fprintf(stderr, "tessera_debug: shape/dtype mismatch for '%s' "
                    "(%lldx%lld dtype=%u suffix=%s -> %lldx%lld dtype=%u suffix=%s); "
                    "closing and reopening\n",
                    tensor_name,
                    (long long) s.rows, (long long) s.cols,
                    (unsigned) s.file_dtype, s.suffix.c_str(),
                    (long long) rows, (long long) cols,
                    (unsigned) file_dtype, suffix);
        }
        s.ofs.close();
        s.tensor_name.clear();
        s.rows = 0;
        s.cols = 0;
        s.row_outlier_counts.clear();
        s.outlier_count_total = 0;
        s.row_meta.clear();
        s.suffix.clear();
        s.file_dtype = 0;
        s.open = false;
    }

    std::filesystem::path out_path =
        std::filesystem::path(dir) /
        (std::string(tensor_name) + suffix);

    std::error_code ec;
    std::filesystem::create_directories(dir, ec);
    if (ec) {
        fprintf(stderr, "tessera_debug: failed to create dir '%s': %s\n",
                dir.c_str(), ec.message().c_str());
        return false;
    }

    s.ofs.open(out_path, std::ios::binary | std::ios::out | std::ios::trunc);
    if (!s.ofs.is_open()) {
        fprintf(stderr, "tessera_debug: failed to open '%s' for writing\n",
                out_path.string().c_str());
        return false;
    }

    // v3 header (40 bytes): v1 fields (28) + outlier_threshold(4) +
    // outlier_count_total(8). The total is written as zero here; the
    // real value is patched in at close_*. The v1 fields are at the
    // same offsets as in v1/v2, so a v1 or v2 reader that does NOT
    // dispatch on version can still locate them (a v1 reader that
    // assumes F32 data at offset 28 will see garbage on a v3 file;
    // readers that check the version field work correctly). The
    // `dtype` field is the L1.5 reference dtype (F32 or F16); the data
    // block size is rows*cols*bytes_per_elem where bytes_per_elem is
    // 4 for F32 and 2 for F16.
    s.ofs.write(DEQUANT_FILE_MAGIC, 4);
    uint32_t version = DEQUANT_FILE_VERSION;
    s.ofs.write(reinterpret_cast<const char *>(&version), sizeof(version));
    s.ofs.write(reinterpret_cast<const char *>(&rows),    sizeof(rows));
    s.ofs.write(reinterpret_cast<const char *>(&cols),    sizeof(cols));
    s.ofs.write(reinterpret_cast<const char *>(&file_dtype), sizeof(file_dtype));

    const float threshold = g_outlier_threshold;
    int64_t     total_zero = 0;
    s.ofs.write(reinterpret_cast<const char *>(&threshold),  sizeof(threshold));
    s.ofs.write(reinterpret_cast<const char *>(&total_zero), sizeof(total_zero));

    // Reserve the per-row outlier-count strip (v2). Filled at close.
    s.row_outlier_counts.assign((size_t) rows, 0);
    s.outlier_count_total = 0;
    const int64_t v2_strip_bytes = (int64_t) rows * (int64_t) sizeof(int32_t);
    std::vector<char> zeros((size_t) v2_strip_bytes, 0);
    s.ofs.write(zeros.data(), (std::streamsize) v2_strip_bytes);

    // Reserve the per-row v3 strip (24 bytes per row). Filled at close.
    s.row_meta.assign((size_t) rows, RowV3Meta{});
    const int64_t v3_strip_bytes = (int64_t) rows * (int64_t) sizeof(RowV3Meta);
    std::vector<char> zeros3((size_t) v3_strip_bytes, 0);
    s.ofs.write(zeros3.data(), (std::streamsize) v3_strip_bytes);

    s.tensor_name = tensor_name;
    s.rows        = rows;
    s.cols        = cols;
    s.suffix      = suffix;
    s.file_dtype  = file_dtype;
    s.open        = true;
    return true;
}

void sidecar_write_row_impl(SidecarStream & s,
                            int64_t row_idx, const float * data, int64_t n) {
    (void) row_idx;
    if (!s.open || data == nullptr || n <= 0) {
        return;
    }
    if (s.file_dtype != DEQUANT_DTYPE_F32) {
        fprintf(stderr, "tessera_debug: sidecar '%s' opened as dtype %u; "
                "refusing F32 row write\n",
                s.tensor_name.c_str(), (unsigned) s.file_dtype);
        return;
    }
    if (n != s.cols) {
        fprintf(stderr, "tessera_debug: row width mismatch for '%s' (cols=%lld, got=%lld); "
                "writing %lld values\n",
                s.tensor_name.c_str(),
                (long long) s.cols, (long long) n, (long long) n);
    }

    int32_t row_count = 0;
    const float t = g_outlier_threshold;
    for (int64_t i = 0; i < n; i++) {
        if (fabsf(data[i]) > t) {
            row_count++;
        }
    }
    if (row_idx >= 0 && row_idx < (int64_t) s.row_outlier_counts.size()) {
        s.row_outlier_counts[(size_t) row_idx] = row_count;
    } else {
        fprintf(stderr, "tessera_debug: row_idx %lld out of range for '%s' (rows=%lld); "
                "counting but not recording per-row\n",
                (long long) row_idx, s.tensor_name.c_str(),
                (long long) s.rows);
    }
    s.outlier_count_total += row_count;

    s.ofs.write(reinterpret_cast<const char *>(data),
                static_cast<std::streamsize>(n * sizeof(float)));
}

void sidecar_write_row_fp16_impl(SidecarStream & s,
                                 int64_t row_idx,
                                 const uint16_t * data,
                                 int64_t n) {
    (void) row_idx;
    if (!s.open || data == nullptr || n <= 0) {
        return;
    }
    if (s.file_dtype != DEQUANT_DTYPE_F16) {
        fprintf(stderr, "tessera_debug: sidecar '%s' opened as dtype %u; "
                "refusing FP16 row write\n",
                s.tensor_name.c_str(), (unsigned) s.file_dtype);
        return;
    }
    if (n != s.cols) {
        fprintf(stderr, "tessera_debug: row width mismatch for '%s' (cols=%lld, got=%lld); "
                "writing %lld values\n",
                s.tensor_name.c_str(),
                (long long) s.cols, (long long) n, (long long) n);
    }

    // Outlier count is computed by upcasting the FP16 to F32 for the
    // threshold comparison. The threshold (default 6.0) is a magnitude
    // cutoff, so the half-precision representation is good enough; the
    // upcast recovers the same |x| to within 1 ULP. This matches the
    // L1 F32 outlier accounting bit-for-bit on inputs that are exactly
    // representable in FP16, and is off by at most one element on
    // inputs that round at FP16 precision.
    int32_t row_count = 0;
    const float t = g_outlier_threshold;
    for (int64_t i = 0; i < n; i++) {
        const float f = local_fp16_to_fp32(data[i]);
        if (fabsf(f) > t) {
            row_count++;
        }
    }
    if (row_idx >= 0 && row_idx < (int64_t) s.row_outlier_counts.size()) {
        s.row_outlier_counts[(size_t) row_idx] = row_count;
    } else {
        fprintf(stderr, "tessera_debug: row_idx %lld out of range for '%s' (rows=%lld); "
                "counting but not recording per-row\n",
                (long long) row_idx, s.tensor_name.c_str(),
                (long long) s.rows);
    }
    s.outlier_count_total += row_count;

    s.ofs.write(reinterpret_cast<const char *>(data),
                static_cast<std::streamsize>(n * sizeof(uint16_t)));
}

void sidecar_set_row_meta_impl(SidecarStream & s,
                               int64_t row_idx,
                               uint64_t timing_ns,
                               uint32_t kernel_id,
                               uint32_t dispatch_count) {
    if (!s.open) {
        return;
    }
    if (row_idx < 0 || row_idx >= s.rows) {
        fprintf(stderr, "tessera_debug: row_idx %lld out of range for '%s' (rows=%lld); "
                "v3 meta not recorded\n",
                (long long) row_idx, s.tensor_name.c_str(),
                (long long) s.rows);
        return;
    }
    s.row_meta[(size_t) row_idx].timing_ns      = timing_ns;
    s.row_meta[(size_t) row_idx].kernel_id      = kernel_id;
    s.row_meta[(size_t) row_idx].dispatch_count = dispatch_count;
    // reserved stays zero
}

void sidecar_close_impl(SidecarStream & s) {
    if (!s.open) {
        s.tensor_name.clear();
        s.rows = 0;
        s.cols = 0;
        s.row_outlier_counts.clear();
        s.outlier_count_total = 0;
        s.row_meta.clear();
        s.suffix.clear();
        s.file_dtype = 0;
        return;
    }

    // Patch the v2 header total at offset 32 (right after the 28-byte
    // v1 header and the 4-byte threshold).
    const std::streamoff total_off = (std::streamoff) (4 + 4 + 8 + 8 + 4 + 4);
    s.ofs.seekp(total_off, std::ios::beg);
    if (!s.ofs) {
        fprintf(stderr, "tessera_debug: seekp to total-offset failed for '%s'; "
                "per-row strip will not be written\n",
                s.tensor_name.c_str());
    } else {
        s.ofs.write(reinterpret_cast<const char *>(&s.outlier_count_total),
                    sizeof(s.outlier_count_total));
    }

    // Patch the per-row outlier-count strip at offset 40, just before
    // the v3 per-row strip.
    const std::streamoff v2_strip_off = (std::streamoff) (4 + 4 + 8 + 8 + 4 + 4 + 8);
    s.ofs.seekp(v2_strip_off, std::ios::beg);
    if (!s.ofs) {
        fprintf(stderr, "tessera_debug: seekp to v2-strip-offset failed for '%s'\n",
                s.tensor_name.c_str());
    } else if (!s.row_outlier_counts.empty()) {
        s.ofs.write(reinterpret_cast<const char *>(s.row_outlier_counts.data()),
                    (std::streamsize) (s.row_outlier_counts.size() * sizeof(int32_t)));
    }

    // Patch the per-row v3 strip (24 bytes per row) right after the v2
    // strip and before the F32 data.
    const std::streamoff v3_strip_off = v2_strip_off +
        (std::streamoff) s.row_meta.size() * (std::streamoff) sizeof(int32_t);
    s.ofs.seekp(v3_strip_off, std::ios::beg);
    if (!s.ofs) {
        fprintf(stderr, "tessera_debug: seekp to v3-strip-offset failed for '%s'\n",
                s.tensor_name.c_str());
    } else if (!s.row_meta.empty()) {
        s.ofs.write(reinterpret_cast<const char *>(s.row_meta.data()),
                    (std::streamsize) (s.row_meta.size() * sizeof(RowV3Meta)));
    }

    s.ofs.flush();
    s.ofs.close();

    // Write the provenance sidecar next to the data sidecar.
    {
        EnvState & es = env_state();
        es.initialized = true;
        // The provenance file lives next to the data file.
        const std::string sidecar_path =
            (std::filesystem::path(env_state().dequant_dir) /
             (s.tensor_name + s.suffix)).string();
        const std::string prov_path = sidecar_path + ".provenance.json";

        // ISO 8601 UTC timestamp. std::time returns UTC seconds since
        // epoch when gmtime is used.
        std::time_t now = std::time(nullptr);
        std::tm tm_utc{};
#if defined(_WIN32)
        gmtime_s(&tm_utc, &now);
#else
        gmtime_r(&now, &tm_utc);
#endif
        char ts[32];
        std::strftime(ts, sizeof(ts), "%Y-%m-%dT%H:%M:%SZ", &tm_utc);

        std::ofstream prov(prov_path, std::ios::out | std::ios::trunc);
        if (prov.is_open()) {
            // The "sidecar_kind" field discriminates the four cases:
            //   - ".dequant.f32"             -> "dequant"          (L1)
            //   - ".act.dequant.f32"         -> "fp16_reference_f32" (L1.5 legacy)
            //   - ".act.dequant.f16"         -> "fp16_reference_f16" (L1.5 default)
            const char * kind = "dequant";
            if (s.suffix == DEQUANT_FILE_SUFFIX_L15_F32) {
                kind = "fp16_reference_f32";
            } else if (s.suffix == DEQUANT_FILE_SUFFIX_L15_F16) {
                kind = "fp16_reference_f16";
            }
            const char * dtype_str = (s.file_dtype == DEQUANT_DTYPE_F16) ? "f16" : "f32";
            prov << "{\n"
                 << "  \"model\": \"" << es.model << "\",\n"
                 << "  \"calibration_corpus\": \"" << es.calibration_corpus << "\",\n"
                 << "  \"calibration_corpus_hash\": \"" << es.calibration_corpus_hash << "\",\n"
                 << "  \"kernel_version\": \"" << TESSERA_KERNEL_VERSION << "\",\n"
                 << "  \"l1_sidecar_version\": " << (int) DEQUANT_FILE_VERSION << ",\n"
                 << "  \"imatrix_version\": " << tessera_imatrix_version() << ",\n"
                 << "  \"created_at\": \"" << ts << "\",\n"
                 << "  \"tessera_main_tip\": \"" << TESSERA_MAIN_TIP << "\",\n"
                 << "  \"sidecar_path\": \"" << sidecar_path << "\",\n"
                 << "  \"sidecar_kind\": \"" << kind << "\",\n"
                 << "  \"sidecar_dtype\": \"" << dtype_str << "\"\n"
                 << "}\n";
            prov.flush();
            prov.close();
        } else {
            fprintf(stderr, "tessera_debug: failed to open provenance '%s' for writing\n",
                    prov_path.c_str());
        }
    }

    s.tensor_name.clear();
    s.rows = 0;
    s.cols = 0;
    s.row_outlier_counts.clear();
    s.outlier_count_total = 0;
    s.row_meta.clear();
    s.suffix.clear();
    s.file_dtype = 0;
    s.open = false;
}

} // namespace

// ---- public API ----

bool dequant_debug_enabled() {
    ensure_env_loaded();
    return !env_state().dequant_dir.empty();
}

void set_dequant_dir(const std::string & path) {
    ensure_env_loaded();
    if (path == env_state().dequant_dir) {
        return;
    }
    // If a file is open under the old dir, close it; the new dir means
    // any subsequent open_*_writer is a fresh start.
    if (g_l1.open) {
        fprintf(stderr, "tessera_debug: closing L1 sidecar for '%s' before reconfiguring dir\n",
                g_l1.tensor_name.c_str());
        g_l1.ofs.close();
        g_l1.tensor_name.clear();
        g_l1.rows = 0;
        g_l1.cols = 0;
        g_l1.row_outlier_counts.clear();
        g_l1.outlier_count_total = 0;
        g_l1.row_meta.clear();
        g_l1.suffix.clear();
        g_l1.file_dtype = 0;
        g_l1.open = false;
    }
    if (g_l15.open) {
        fprintf(stderr, "tessera_debug: closing L1.5 sidecar for '%s' before reconfiguring dir\n",
                g_l15.tensor_name.c_str());
        g_l15.ofs.close();
        g_l15.tensor_name.clear();
        g_l15.rows = 0;
        g_l15.cols = 0;
        g_l15.row_outlier_counts.clear();
        g_l15.outlier_count_total = 0;
        g_l15.row_meta.clear();
        g_l15.suffix.clear();
        g_l15.file_dtype = 0;
        g_l15.open = false;
    }
    env_state().dequant_dir = path;
}

void set_dequant_mode(const std::string & mode) {
    ensure_env_loaded();
    if (mode == env_state().mode) {
        return;
    }
    env_state().mode = mode;
    // Closing open streams when the mode flips is intentional: an L1
    // stream opened under w4a4 mode would be left dangling on a flip
    // back to empty (we'd skip the L1.5 close path, and the next
    // close_dequant_writer would only flush the L1 file). Closing
    // both streams here keeps the invariant simple: the L1 and L1.5
    // streams are always both closed or both open.
    if (g_l1.open || g_l15.open) {
        if (g_l1.open) {
            g_l1.ofs.close();
            g_l1.tensor_name.clear();
            g_l1.rows = 0;
            g_l1.cols = 0;
            g_l1.row_outlier_counts.clear();
            g_l1.outlier_count_total = 0;
            g_l1.row_meta.clear();
            g_l1.suffix.clear();
            g_l1.file_dtype = 0;
            g_l1.open = false;
        }
        if (g_l15.open) {
            g_l15.ofs.close();
            g_l15.tensor_name.clear();
            g_l15.rows = 0;
            g_l15.cols = 0;
            g_l15.row_outlier_counts.clear();
            g_l15.outlier_count_total = 0;
            g_l15.row_meta.clear();
            g_l15.suffix.clear();
            g_l15.file_dtype = 0;
            g_l15.open = false;
        }
        fprintf(stderr, "tessera_debug: closing L1/L1.5 sidecars due to mode flip\n");
    }
}

const std::string & dequant_mode() {
    ensure_env_loaded();
    return env_state().mode;
}

bool dequant_w4a4_enabled() {
    ensure_env_loaded();
    return env_state().mode == DEQUANT_MODE_W4A4;
}

void set_l15_dtype(const std::string & dtype) {
    ensure_env_loaded();
    if (dtype == env_state().l15_dtype) {
        return;
    }
    // A dtype change while a sidecar is open would mix F32 and FP16
    // rows in the same file. Close both streams; the next open_*_writer
    // call will reopen under the new dtype.
    if (g_l1.open || g_l15.open) {
        if (g_l1.open) {
            g_l1.ofs.close();
            g_l1.tensor_name.clear();
            g_l1.rows = 0;
            g_l1.cols = 0;
            g_l1.row_outlier_counts.clear();
            g_l1.outlier_count_total = 0;
            g_l1.row_meta.clear();
            g_l1.suffix.clear();
            g_l1.file_dtype = 0;
            g_l1.open = false;
        }
        if (g_l15.open) {
            g_l15.ofs.close();
            g_l15.tensor_name.clear();
            g_l15.rows = 0;
            g_l15.cols = 0;
            g_l15.row_outlier_counts.clear();
            g_l15.outlier_count_total = 0;
            g_l15.row_meta.clear();
            g_l15.suffix.clear();
            g_l15.file_dtype = 0;
            g_l15.open = false;
        }
        fprintf(stderr, "tessera_debug: closing L1/L1.5 sidecars due to L1.5 dtype flip\n");
    }
    env_state().l15_dtype = dtype;
}

const std::string & l15_dtype() {
    ensure_env_loaded();
    return env_state().l15_dtype;
}

bool l15_dtype_is_f16() {
    ensure_env_loaded();
    // "f16" is the default; the only other accepted value is "f32"
    // (legacy). An empty string is treated as "f16" so unset / unconfig
    // callers get the new default. Other values log a warning and fall
    // back to f16.
    const std::string & d = env_state().l15_dtype;
    if (d.empty()) {
        return true;
    }
    if (d == L15_DTYPE_F16) {
        return true;
    }
    if (d == L15_DTYPE_F32) {
        return false;
    }
    fprintf(stderr, "tessera_debug: unknown LLAMA_TESSERA_L15_DTYPE='%s'; "
            "expected 'f16' or 'f32'; falling back to 'f16'\n", d.c_str());
    return true;
}

void set_outlier_threshold(float threshold) {
    g_outlier_threshold = threshold;
}

float outlier_threshold() {
    return g_outlier_threshold;
}

void set_dequant_stride(int64_t stride) {
    ensure_env_loaded();
    env_state().stride = (stride < 1) ? 1 : stride;
}

int64_t dequant_stride() {
    ensure_env_loaded();
    return env_state().stride;
}

void open_dequant_writer(const char * tensor_name, int64_t rows, int64_t cols) {
    std::lock_guard<std::recursive_mutex> lk(g_writer_mutex);
    ensure_env_loaded();
    if (!sidecar_open_impl(g_l1, env_state().dequant_dir, tensor_name, rows, cols,
                            DEQUANT_FILE_SUFFIX_L1, DEQUANT_DTYPE_F32)) {
        return;
    }
    // In w4a4 mode, also open the L1.5 sidecar. The suffix and on-disk
    // dtype come from the L1.5 dtype config (see set_l15_dtype): the
    // default "f16" opens `.act.dequant.f16` with DEQUANT_DTYPE_F16
    // (the FP16 ground truth - the whole point of L1.5); the legacy
    // "f32" opens `.act.dequant.f32` with DEQUANT_DTYPE_F32. Backends
    // are expected to call open_fp16_reference_writer after
    // open_dequant_writer and use the matching
    // write_fp16_reference_row{,_from_f32} entry point to feed the
    // reference row data.
    if (dequant_w4a4_enabled()) {
        const char * l15_suffix = DEQUANT_FILE_SUFFIX_L15_F16;
        uint32_t     l15_dtype  = DEQUANT_DTYPE_F16;
        if (!l15_dtype_is_f16()) {
            l15_suffix = DEQUANT_FILE_SUFFIX_L15_F32;
            l15_dtype  = DEQUANT_DTYPE_F32;
        }
        sidecar_open_impl(g_l15, env_state().dequant_dir, tensor_name, rows, cols,
                          l15_suffix, l15_dtype);
    }
}

void write_dequant_row(int64_t row_idx, const float * data, int64_t n) {
    std::lock_guard<std::recursive_mutex> lk(g_writer_mutex);
    sidecar_write_row_impl(g_l1, row_idx, data, n);
    // Back-compat: when w4a4 is enabled AND the L1.5 dtype is the
    // legacy "f32", auto-populate the L1.5 file with the same F32
    // data. The FP16 L1.5 path (the default) is the hook's
    // responsibility - the hook does the F32->FP16 conversion
    // (ggml_fp32_to_fp16, proper rounding) and calls
    // write_fp16_reference_row with the FP16 buffer. The two paths
    // are independent; a given hook call always writes to L1, and
    // additionally either auto-populates the F32 L1.5 OR explicitly
    // populates the FP16 L1.5, depending on the L1.5 dtype config.
    if (dequant_w4a4_enabled() && g_l15.open &&
        g_l15.file_dtype == DEQUANT_DTYPE_F32 &&
        data != nullptr && n > 0) {
        sidecar_write_row_impl(g_l15, row_idx, data, n);
    }
}

void set_dequant_row_meta(int64_t row_idx,
                          uint64_t timing_ns,
                          uint32_t kernel_id,
                          uint32_t dispatch_count) {
    std::lock_guard<std::recursive_mutex> lk(g_writer_mutex);
    sidecar_set_row_meta_impl(g_l1, row_idx, timing_ns, kernel_id, dispatch_count);
    if (dequant_w4a4_enabled() && g_l15.open) {
        sidecar_set_row_meta_impl(g_l15, row_idx, timing_ns, kernel_id, dispatch_count);
    }
}

void close_dequant_writer() {
    std::lock_guard<std::recursive_mutex> lk(g_writer_mutex);
    if (dequant_w4a4_enabled() && g_l15.open) {
        sidecar_close_impl(g_l15);
    }
    if (g_l1.open) {
        sidecar_close_impl(g_l1);
    }
}

void open_fp16_reference_writer(const char * tensor_name, int64_t rows, int64_t cols) {
    std::lock_guard<std::recursive_mutex> lk(g_writer_mutex);
    ensure_env_loaded();
    if (!dequant_w4a4_enabled()) {
        // Back-compat: in non-w4a4 mode the L1.5 sidecar is not
        // written. This function becomes a no-op so the existing
        // call pattern (open_dequant_writer + open_fp16_reference_writer)
        // works regardless of mode.
        return;
    }
    const char * l15_suffix = DEQUANT_FILE_SUFFIX_L15_F16;
    uint32_t     l15_dtype  = DEQUANT_DTYPE_F16;
    if (!l15_dtype_is_f16()) {
        l15_suffix = DEQUANT_FILE_SUFFIX_L15_F32;
        l15_dtype  = DEQUANT_DTYPE_F32;
    }
    sidecar_open_impl(g_l15, env_state().dequant_dir, tensor_name, rows, cols,
                      l15_suffix, l15_dtype);
}

void write_fp16_reference_row(int64_t row_idx, const uint16_t * data, int64_t n) {
    std::lock_guard<std::recursive_mutex> lk(g_writer_mutex);
    sidecar_write_row_fp16_impl(g_l15, row_idx, data, n);
}

void write_fp16_reference_row_from_f32(int64_t row_idx, const float * data, int64_t n) {
    std::lock_guard<std::recursive_mutex> lk(g_writer_mutex);
    if (!g_l15.open || data == nullptr || n <= 0) {
        return;
    }
    if (g_l15.file_dtype != DEQUANT_DTYPE_F16) {
        // L1.5 is open as F32; the F32 path is handled by
        // write_dequant_row's auto-populate branch. Calling
        // write_fp16_reference_row_from_f32 on an F32 sidecar would
        // silently up-cast FP16->F32 then write F32, which is not
        // what the caller wants; refuse the call with a warning so
        // the bug surfaces.
        fprintf(stderr, "tessera_debug: write_fp16_reference_row_from_f32 on "
                "an F32 L1.5 sidecar ('%s'); call write_fp16_reference_row on the "
                "FP16 buffer or switch the L1.5 dtype to f16\n",
                g_l15.tensor_name.c_str());
        return;
    }
    // Convert F32 -> FP16 with proper rounding, then write the FP16
    // buffer as a single row. Local stack buffer for the common
    // small-row case; heap fallback for big rows. The conversion is
    // bit-identical to ggml_compute_fp32_to_fp16 in
    // ggml/src/ggml-impl.h:407-431; we keep a local copy here to
    // avoid pulling ggml.h into the writer (which sits below the
    // ggml layer).
    std::vector<uint16_t> fp16_buf;
    uint16_t stack_buf[256];
    if ((size_t) n <= 256) {
        for (int64_t i = 0; i < n; i++) {
            stack_buf[i] = local_fp32_to_fp16(data[i]);
        }
        sidecar_write_row_fp16_impl(g_l15, row_idx, stack_buf, n);
    } else {
        fp16_buf.resize((size_t) n);
        for (int64_t i = 0; i < n; i++) {
            fp16_buf[(size_t) i] = local_fp32_to_fp16(data[i]);
        }
        sidecar_write_row_fp16_impl(g_l15, row_idx, fp16_buf.data(), n);
    }
}

void set_fp16_reference_row_meta(int64_t row_idx,
                                 uint64_t timing_ns,
                                 uint32_t kernel_id,
                                 uint32_t dispatch_count) {
    std::lock_guard<std::recursive_mutex> lk(g_writer_mutex);
    sidecar_set_row_meta_impl(g_l15, row_idx, timing_ns, kernel_id, dispatch_count);
}

void close_fp16_reference_writer() {
    std::lock_guard<std::recursive_mutex> lk(g_writer_mutex);
    sidecar_close_impl(g_l15);
}

void set_telemetry_model(const std::string & model) {
    ensure_env_loaded();
    env_state().model = model;
}

void set_telemetry_calibration_corpus(const std::string & corpus) {
    ensure_env_loaded();
    env_state().calibration_corpus = corpus;
}

void set_telemetry_calibration_corpus_hash(const std::string & hash) {
    ensure_env_loaded();
    env_state().calibration_corpus_hash = hash;
}

const char * tessera_kernel_version() {
    return TESSERA_KERNEL_VERSION;
}

const char * tessera_main_tip() {
    return TESSERA_MAIN_TIP;
}

int tessera_imatrix_version() {
    return 2;
}

} // namespace tessera_debug
