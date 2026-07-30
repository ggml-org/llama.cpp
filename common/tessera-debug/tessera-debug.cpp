#include "tessera-debug.h"

// Intentionally NOT including common/log.h: this translation unit is
// compiled into the llama-tessera-debug static library, which sits
// below common/ in the layering hierarchy. Pulling in log.h would
// drag common/log.cpp (and the rest of llama-common) into the link
// for every backend. The sidecar writer is a low-level stream sink;
// its diagnostic messages go straight to stderr instead of through
// the common log infrastructure.

#include "tessera-build-info.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <string>
#include <system_error>
#include <vector>

namespace tessera_debug {

// File-static state. Not thread-safe; one matmul at a time per process.
static float g_outlier_threshold = DEQUANT_DEFAULT_OUTLIER_THRESHOLD;

namespace {

// One-time env-var snapshot. The dequant dir, mode, and telemetry fields
// can also be set programmatically via set_*; the env vars are only
// read on the first call. We use a once-flag pattern via a local static
// init function to avoid pulling in <atomic> for a single bool.
struct EnvState {
    std::string dequant_dir;
    std::string mode;
    std::string model;
    std::string calibration_corpus;
    std::string calibration_corpus_hash;
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
    if (const char * v = std::getenv("TESSERA_TELEMETRY_MODEL"); v != nullptr) {
        s.model = v;
    }
    if (const char * v = std::getenv("TESSERA_TELEMETRY_CALIBRATION_CORPUS"); v != nullptr) {
        s.calibration_corpus = v;
    }
    if (const char * v = std::getenv("TESSERA_TELEMETRY_CALIBRATION_CORPUS_HASH"); v != nullptr) {
        s.calibration_corpus_hash = v;
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
// use the same layout (v3); they differ only in the file suffix and
// (potentially) in the per-row data semantics.
struct SidecarStream {
    std::ofstream            ofs;
    std::string              tensor_name;
    int64_t                  rows = 0;
    int64_t                  cols = 0;
    std::vector<int32_t>     row_outlier_counts;
    int64_t                  outlier_count_total = 0;
    std::vector<RowV3Meta>   row_meta;        // size = rows
    std::string              suffix;          // ".dequant.f32" or ".act.dequant.f32"
    bool                     open = false;
};

SidecarStream g_l1;
SidecarStream g_l15;

bool sidecar_open_impl(SidecarStream & s,
                       const std::string & dir,
                       const char * tensor_name,
                       int64_t rows, int64_t cols,
                       const char * suffix) {
    if (dir.empty() || tensor_name == nullptr) {
        return false;
    }

    // Fast path: same tensor, same shape -> keep the open file.
    if (s.open && s.tensor_name == tensor_name &&
        s.rows == rows && s.cols == cols) {
        return true;
    }

    // Shape mismatch on an already-open writer: close + reopen. We
    // intentionally avoid rewriting the header in-place; the simpler
    // behavior is to truncate the file under the same name. The caller
    // is expected to provide a consistent (name, rows, cols) for any
    // given tensor.
    if (s.open) {
        if (s.tensor_name == tensor_name &&
            (s.rows != rows || s.cols != cols)) {
            fprintf(stderr, "tessera_debug: shape mismatch for '%s' (%lldx%lld -> %lldx%lld); "
                    "closing and reopening\n",
                    tensor_name,
                    (long long) s.rows, (long long) s.cols,
                    (long long) rows, (long long) cols);
        }
        s.ofs.close();
        s.tensor_name.clear();
        s.rows = 0;
        s.cols = 0;
        s.row_outlier_counts.clear();
        s.outlier_count_total = 0;
        s.row_meta.clear();
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
    // readers that check the version field work correctly).
    s.ofs.write(DEQUANT_FILE_MAGIC, 4);
    uint32_t version = DEQUANT_FILE_VERSION;
    uint32_t dtype   = DEQUANT_DTYPE_F32;
    s.ofs.write(reinterpret_cast<const char *>(&version), sizeof(version));
    s.ofs.write(reinterpret_cast<const char *>(&rows),    sizeof(rows));
    s.ofs.write(reinterpret_cast<const char *>(&cols),    sizeof(cols));
    s.ofs.write(reinterpret_cast<const char *>(&dtype),   sizeof(dtype));

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
    s.open        = true;
    return true;
}

void sidecar_write_row_impl(SidecarStream & s,
                            int64_t row_idx, const float * data, int64_t n) {
    (void) row_idx;
    if (!s.open || data == nullptr || n <= 0) {
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
                 << "  \"sidecar_kind\": \""
                 << (s.suffix == DEQUANT_FILE_SUFFIX_L15 ? "fp16_reference" : "dequant")
                 << "\"\n"
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

void set_outlier_threshold(float threshold) {
    g_outlier_threshold = threshold;
}

float outlier_threshold() {
    return g_outlier_threshold;
}

void open_dequant_writer(const char * tensor_name, int64_t rows, int64_t cols) {
    ensure_env_loaded();
    if (!sidecar_open_impl(g_l1, env_state().dequant_dir, tensor_name, rows, cols,
                            DEQUANT_FILE_SUFFIX_L1)) {
        return;
    }
    // In w4a4 mode, also open the L1.5 sidecar. The data will be the
    // same F32 values written via write_fp16_reference_row, but the
    // file suffix and path differ. Backends are expected to call
    // open_fp16_reference_writer after open_dequant_writer and use
    // write_fp16_reference_row for the reference row data.
    if (dequant_w4a4_enabled()) {
        sidecar_open_impl(g_l15, env_state().dequant_dir, tensor_name, rows, cols,
                          DEQUANT_FILE_SUFFIX_L15);
    }
}

void write_dequant_row(int64_t row_idx, const float * data, int64_t n) {
    sidecar_write_row_impl(g_l1, row_idx, data, n);
    if (dequant_w4a4_enabled() && g_l15.open && data != nullptr && n > 0) {
        sidecar_write_row_impl(g_l15, row_idx, data, n);
    }
}

void set_dequant_row_meta(int64_t row_idx,
                          uint64_t timing_ns,
                          uint32_t kernel_id,
                          uint32_t dispatch_count) {
    sidecar_set_row_meta_impl(g_l1, row_idx, timing_ns, kernel_id, dispatch_count);
    if (dequant_w4a4_enabled() && g_l15.open) {
        sidecar_set_row_meta_impl(g_l15, row_idx, timing_ns, kernel_id, dispatch_count);
    }
}

void close_dequant_writer() {
    if (dequant_w4a4_enabled() && g_l15.open) {
        sidecar_close_impl(g_l15);
    }
    if (g_l1.open) {
        sidecar_close_impl(g_l1);
    }
}

void open_fp16_reference_writer(const char * tensor_name, int64_t rows, int64_t cols) {
    ensure_env_loaded();
    if (!dequant_w4a4_enabled()) {
        // Back-compat: in non-w4a4 mode the L1.5 sidecar is not
        // written. This function becomes a no-op so the existing
        // call pattern (open_dequant_writer + open_fp16_reference_writer)
        // works regardless of mode.
        return;
    }
    sidecar_open_impl(g_l15, env_state().dequant_dir, tensor_name, rows, cols,
                      DEQUANT_FILE_SUFFIX_L15);
}

void write_fp16_reference_row(int64_t row_idx, const float * data, int64_t n) {
    sidecar_write_row_impl(g_l15, row_idx, data, n);
}

void set_fp16_reference_row_meta(int64_t row_idx,
                                 uint64_t timing_ns,
                                 uint32_t kernel_id,
                                 uint32_t dispatch_count) {
    sidecar_set_row_meta_impl(g_l15, row_idx, timing_ns, kernel_id, dispatch_count);
}

void close_fp16_reference_writer() {
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
