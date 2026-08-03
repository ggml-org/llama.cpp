//
// tessera-matmul-output.cpp
//
// Tessera Layer-2 matmul-output sidecar writer. See tessera-matmul-output.h.
//
// This is a thin mirror of the L1 dequant sidecar (tessera-debug.cpp)
// but captures the F32 matmul OUTPUT instead of the dequantized weight.
// The on-disk layout uses a distinct magic ("TPMO") so the two sidecar
// kinds can coexist in one directory and be routed by their magic.
//

#include "tessera-matmul-output.h"

// Pull the L1 sidecar v3 header constants so the v3 shape stays in
// lockstep between the two file kinds. The L1 sidecar owns the schema
// versioning (DEQUANT_FILE_VERSION) and the per-row v3 strip shape.
#include "tessera-debug.h"

// Intentionally NOT including common/log.h: this TU is compiled into
// the llama-tessera-debug static library, which sits below common/ in
// the layering hierarchy. Diagnostic messages go straight to stderr.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <string>
#include <system_error>
#include <vector>

namespace tessera_matmul_output {

// Recursive so the public open/write/close entry points can each acquire
// the lock without deadlocking when one sequence nests another (it does
// not today, but the L1 sidecar uses the same pattern).
static std::recursive_mutex g_writer_mutex;

namespace {

// One-time env-var snapshot. The output dir and stride can also be set
// programmatically via set_*; the env vars are only read on the first
// call.
struct EnvState {
    std::string dir;
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
    if (const char * v = std::getenv("LLAMA_TILE640_DEBUG_MATMUL_OUTPUT_DIR"); v != nullptr) {
        s.dir = v;
    }
    if (const char * v = std::getenv("LLAMA_TILE640_DEBUG_MATMUL_OUTPUT_STRIDE"); v != nullptr) {
        int64_t st = (int64_t) atoll(v);
        s.stride = (st < 1) ? 1 : st;
    }
}

// Per-row v3 metadata. Same shape as the L1 sidecar (8 + 4 + 4 + 8 = 24
// bytes per row) so the v3 reader can parse both kinds uniformly. The
// outlier-count strip (int32 per row) is reused as a "samples seen"
// count so a downstream L2 reader can tell whether the file was
// truncated mid-row.
struct RowV3Meta {
    uint64_t timing_ns      = 0;
    uint32_t kernel_id      = 0;
    uint32_t dispatch_count = 0;
    uint64_t reserved       = 0;
};

// State for a single open sidecar file.
struct SidecarStream {
    std::ofstream ofs;
    std::string   tensor_name;
    int64_t       rows       = 0;     // captured rows (= logical rows / stride)
    int64_t       cols       = 0;     // elements per row (= n_embd for attn, etc.)
    int64_t       next_row   = 0;     // next row to write
    std::vector<RowV3Meta>    row_meta;
    std::vector<int32_t>      row_samples;   // per-row samples seen (= 1 normally)
    int64_t                   total_samples  = 0;
};

static SidecarStream g_stream;

void write_v3_header(SidecarStream & s) {
    // v1 header (28 bytes): magic(4) version(4) rows(8) cols(8) dtype(4)
    s.ofs.write(MATMUL_OUTPUT_FILE_MAGIC, 4);
    uint32_t version = tessera_debug::DEQUANT_FILE_VERSION;   // 3, shared with L1
    s.ofs.write((const char *) &version, sizeof(version));
    s.ofs.write((const char *) &s.rows, sizeof(s.rows));
    s.ofs.write((const char *) &s.cols, sizeof(s.cols));
    uint32_t dtype = tessera_debug::DEQUANT_DTYPE_F32;
    s.ofs.write((const char *) &dtype, sizeof(dtype));

    // v2 header (12 bytes): outlier_threshold(4) outlier_count_total(8)
    // The L2 matmul-output sidecar reuses the v2 outlier-count strip as
    // a per-row sample counter (see SidecarStream::row_samples). The
    // threshold field is set to 0.0 (no outlier count semantics here)
    // and the total is repurposed as the total-sample count so a v3
    // reader can still validate the file.
    float outlier_threshold = 0.0f;
    s.ofs.write((const char *) &outlier_threshold, sizeof(outlier_threshold));
    s.ofs.write((const char *) &s.total_samples, sizeof(s.total_samples));

    // v2 per-row strip (R * 4 bytes) - the per-row sample counter
    std::vector<int32_t> zeros((size_t) s.rows, 0);
    s.ofs.write((const char *) zeros.data(), sizeof(int32_t) * (size_t) s.rows);

    // v3 per-row strip (R * 24 bytes) - reserved for now, finalized at close
    uint8_t row_meta_zero[24];
    std::memset(row_meta_zero, 0, sizeof(row_meta_zero));
    for (int64_t r = 0; r < s.rows; r++) {
        s.ofs.write((const char *) row_meta_zero, sizeof(row_meta_zero));
    }
    s.ofs.flush();
}

void seek_to_row_meta_strip(SidecarStream & s) {
    if (!s.ofs.is_open()) {
        return;
    }
    // v1 header (28) + v2 header (12) + v2 per-row strip (R*4) = 40 + R*4
    const std::streamoff off = (std::streamoff)(40 + s.rows * 4);
    s.ofs.seekp(off, std::ios::beg);
}

void seek_to_data_start(SidecarStream & s) {
    if (!s.ofs.is_open()) {
        return;
    }
    // data starts after v1 header (28) + v2 header (12) + v2 per-row strip
    // (R*4) + v3 per-row strip (R*24) = 40 + R*4 + R*24 = 40 + R*28
    const std::streamoff off = (std::streamoff)(40 + s.rows * 28);
    s.ofs.seekp(off, std::ios::beg);
}

} // namespace

bool matmul_output_capture_enabled() {
    std::lock_guard<std::recursive_mutex> lock(g_writer_mutex);
    ensure_env_loaded();
    return !env_state().dir.empty() || !g_stream.tensor_name.empty();
}

void set_matmul_output_dir(const std::string & path) {
    std::lock_guard<std::recursive_mutex> lock(g_writer_mutex);
    ensure_env_loaded();
    if (path.empty()) {
        // closing existing writer
        if (g_stream.ofs.is_open()) {
            try { g_stream.ofs.flush(); } catch (...) {}
            g_stream.ofs.close();
        }
        g_stream = SidecarStream();
        env_state().dir.clear();
        return;
    }
    env_state().dir = path;
    // We do NOT pre-create the directory here; it is created on the
    // first open_matmul_output_writer call (mirroring the L1 sidecar
    // contract). This avoids surprising the user with an empty dir.
}

const std::string & matmul_output_dir() {
    std::lock_guard<std::recursive_mutex> lock(g_writer_mutex);
    ensure_env_loaded();
    return env_state().dir;
}

void set_matmul_output_stride(int64_t stride) {
    std::lock_guard<std::recursive_mutex> lock(g_writer_mutex);
    ensure_env_loaded();
    env_state().stride = (stride < 1) ? 1 : stride;
}

int64_t matmul_output_stride() {
    std::lock_guard<std::recursive_mutex> lock(g_writer_mutex);
    ensure_env_loaded();
    return env_state().stride;
}

void open_matmul_output_writer(const char * tensor_name,
                               int64_t rows, int64_t cols) {
    if (tensor_name == nullptr || tensor_name[0] == '\0' || rows <= 0 || cols <= 0) {
        return;
    }
    std::lock_guard<std::recursive_mutex> lock(g_writer_mutex);
    ensure_env_loaded();
    if (env_state().dir.empty()) {
        return;
    }
    if (g_stream.tensor_name == tensor_name && g_stream.ofs.is_open()) {
        // Same tensor re-opened (e.g. another matmul invocation). If
        // the rows/cols match, the existing stream is reused. If they
        // differ, close and reopen with the new shape.
        if (g_stream.rows != rows || g_stream.cols != cols) {
            std::fprintf(stderr,
                         "tessera-matmul-output: reopen '%s' with shape (%lld, %lld) -> (%lld, %lld); truncating\n",
                         tensor_name,
                         (long long) g_stream.rows, (long long) g_stream.cols,
                         (long long) rows, (long long) cols);
            g_stream.ofs.flush();
            g_stream.ofs.close();
            g_stream = SidecarStream();
        } else {
            return;   // already open with the same shape
        }
    } else if (g_stream.ofs.is_open()) {
        // Different tensor, close the previous one.
        g_stream.ofs.flush();
        g_stream.ofs.close();
        g_stream = SidecarStream();
    }

    std::filesystem::path dir_path(env_state().dir);
    std::error_code ec;
    std::filesystem::create_directories(dir_path, ec);
    if (ec) {
        std::fprintf(stderr, "tessera-matmul-output: failed to create dir '%s': %s\n",
                     env_state().dir.c_str(), ec.message().c_str());
        return;
    }

    g_stream.tensor_name = tensor_name;
    g_stream.rows        = rows;
    g_stream.cols        = cols;
    g_stream.row_meta.assign((size_t) rows, RowV3Meta());
    g_stream.row_samples.assign((size_t) rows, 0);
    g_stream.total_samples = 0;
    g_stream.next_row      = 0;

    const std::string file_path = (dir_path / (std::string(tensor_name) + MATMUL_OUTPUT_FILE_SUFFIX)).string();
    g_stream.ofs.open(file_path, std::ios::binary | std::ios::out | std::ios::trunc);
    if (!g_stream.ofs.is_open()) {
        std::fprintf(stderr, "tessera-matmul-output: failed to open '%s'\n", file_path.c_str());
        g_stream = SidecarStream();
        return;
    }
    write_v3_header(g_stream);
}

void write_matmul_output_row(int64_t row_idx, const float * data, int64_t n) {
    if (data == nullptr || n <= 0) {
        return;
    }
    std::lock_guard<std::recursive_mutex> lock(g_writer_mutex);
    if (!g_stream.ofs.is_open() || row_idx < 0 || row_idx >= g_stream.rows) {
        if (g_stream.tensor_name.empty()) {
            return;   // sidecar disabled; no warning
        }
        std::fprintf(stderr, "tessera-matmul-output: row %lld out of range for '%s' (rows=%lld)\n",
                     (long long) row_idx, g_stream.tensor_name.c_str(),
                     (long long) g_stream.rows);
        return;
    }
    const int64_t effective_n = (n < g_stream.cols) ? n : g_stream.cols;
    if (n != g_stream.cols) {
        std::fprintf(stderr, "tessera-matmul-output: row %lld n=%lld != cols=%lld for '%s'; truncating\n",
                     (long long) row_idx, (long long) n, (long long) g_stream.cols,
                     g_stream.tensor_name.c_str());
    }
    seek_to_data_start(g_stream);
    // Each row is g_stream.cols F32 values; row stride is g_stream.cols * 4 bytes.
    const std::streamoff row_off = (std::streamoff) row_idx * (std::streamoff) g_stream.cols * 4;
    g_stream.ofs.seekp(row_off, std::ios::cur);
    g_stream.ofs.write((const char *) data, sizeof(float) * (size_t) effective_n);
    g_stream.row_samples[(size_t) row_idx] += 1;
    g_stream.total_samples += 1;
    g_stream.next_row = (row_idx + 1 > g_stream.next_row) ? (row_idx + 1) : g_stream.next_row;
}

void set_matmul_output_row_meta(int64_t row_idx,
                                uint64_t timing_ns,
                                uint32_t kernel_id,
                                uint32_t dispatch_count) {
    std::lock_guard<std::recursive_mutex> lock(g_writer_mutex);
    if (!g_stream.ofs.is_open() || row_idx < 0 || row_idx >= g_stream.rows) {
        return;
    }
    g_stream.row_meta[(size_t) row_idx].timing_ns      = timing_ns;
    g_stream.row_meta[(size_t) row_idx].kernel_id      = kernel_id;
    g_stream.row_meta[(size_t) row_idx].dispatch_count = dispatch_count;
}

void close_matmul_output_writer() {
    std::lock_guard<std::recursive_mutex> lock(g_writer_mutex);
    if (!g_stream.ofs.is_open()) {
        return;
    }

    // Finalize the v2 per-row strip (sample count) and the v3 per-row
    // strip (timing / kernel_id / dispatch_count).
    seek_to_row_meta_strip(g_stream);
    g_stream.ofs.write((const char *) g_stream.row_samples.data(),
                       sizeof(int32_t) * (size_t) g_stream.rows);
    g_stream.ofs.write((const char *) g_stream.row_meta.data(),
                       sizeof(RowV3Meta) * (size_t) g_stream.rows);

    // Update the v2 header total (repurposed as total_samples).
    const std::streamoff v2_off = 28 + 4;   // 32
    g_stream.ofs.seekp(v2_off, std::ios::beg);
    g_stream.ofs.write((const char *) &g_stream.total_samples, sizeof(g_stream.total_samples));

    g_stream.ofs.flush();
    g_stream.ofs.close();
    g_stream = SidecarStream();
}

} // namespace tessera_matmul_output
