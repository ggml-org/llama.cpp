#include "tessera-debug.h"

#include "log.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <system_error>
#include <vector>

namespace tessera_debug {

// File-static state. Not thread-safe; one matmul at a time per process.
static std::string        g_dequant_dir;
static float              g_outlier_threshold = DEQUANT_DEFAULT_OUTLIER_THRESHOLD;
static std::string        g_open_tensor_name; // empty when no file is open
static int64_t            g_open_rows = 0;
static int64_t            g_open_cols = 0;
static std::ofstream      g_ofs;
// Per-row outlier counts collected during write_dequant_row and flushed
// to disk in close_dequant_writer. Sized to g_open_rows at open time.
static std::vector<int32_t> g_row_outlier_counts;
// Cumulative sum kept in sync with g_row_outlier_counts; mirrored to the
// file header at close time.
static int64_t            g_outlier_count_total = 0;

bool dequant_debug_enabled() {
    return !g_dequant_dir.empty();
}

void set_dequant_dir(const std::string & path) {
    if (path == g_dequant_dir) {
        return;
    }
    // If a file is open under the old dir, close it; the new dir means
    // any subsequent open_dequant_writer is a fresh start.
    if (g_ofs.is_open()) {
        LOG_WRN("tessera_debug: closing sidecar for '%s' before reconfiguring dir\n",
                g_open_tensor_name.c_str());
        g_ofs.close();
        g_open_tensor_name.clear();
        g_open_rows = 0;
        g_open_cols = 0;
        g_row_outlier_counts.clear();
        g_outlier_count_total = 0;
    }
    g_dequant_dir = path;
}

void set_outlier_threshold(float threshold) {
    g_outlier_threshold = threshold;
}

float outlier_threshold() {
    return g_outlier_threshold;
}

void open_dequant_writer(const char * tensor_name, int64_t rows, int64_t cols) {
    if (g_dequant_dir.empty() || tensor_name == nullptr) {
        return;
    }

    // Fast path: same tensor, same shape -> keep the open file.
    if (g_ofs.is_open() && g_open_tensor_name == tensor_name &&
        g_open_rows == rows && g_open_cols == cols) {
        return;
    }

    // Shape mismatch on an already-open writer: close + reopen. We
    // intentionally avoid rewriting the header in-place; the simpler
    // behavior is to truncate the file under the same name. The caller
    // is expected to provide a consistent (name, rows, cols) for any
    // given tensor.
    if (g_ofs.is_open()) {
        if (g_open_tensor_name == tensor_name &&
            (g_open_rows != rows || g_open_cols != cols)) {
            LOG_WRN("tessera_debug: shape mismatch for '%s' (%lldx%lld -> %lldx%lld); "
                    "closing and reopening\n",
                    tensor_name,
                    (long long) g_open_rows, (long long) g_open_cols,
                    (long long) rows, (long long) cols);
        }
        g_ofs.close();
        g_open_tensor_name.clear();
        g_open_rows = 0;
        g_open_cols = 0;
        g_row_outlier_counts.clear();
        g_outlier_count_total = 0;
    }

    std::filesystem::path out_path =
        std::filesystem::path(g_dequant_dir) /
        (std::string(tensor_name) + ".dequant.f32");

    std::error_code ec;
    std::filesystem::create_directories(g_dequant_dir, ec);
    if (ec) {
        LOG_ERR("tessera_debug: failed to create dir '%s': %s\n",
                g_dequant_dir.c_str(), ec.message().c_str());
        return;
    }

    g_ofs.open(out_path, std::ios::binary | std::ios::out | std::ios::trunc);
    if (!g_ofs.is_open()) {
        LOG_ERR("tessera_debug: failed to open '%s' for writing\n",
                out_path.string().c_str());
        return;
    }

    // v2 header (40 bytes): v1 fields (28) + outlier_threshold(4) +
    // outlier_count_total(8). The total is written as zero here; the
    // real value is patched in at close_dequant_writer.
    g_ofs.write(DEQUANT_FILE_MAGIC, 4);
    uint32_t version = DEQUANT_FILE_VERSION;
    uint32_t dtype   = DEQUANT_DTYPE_F32;
    g_ofs.write(reinterpret_cast<const char *>(&version), sizeof(version));
    g_ofs.write(reinterpret_cast<const char *>(&rows),    sizeof(rows));
    g_ofs.write(reinterpret_cast<const char *>(&cols),    sizeof(cols));
    g_ofs.write(reinterpret_cast<const char *>(&dtype),   sizeof(dtype));

    const float  threshold = g_outlier_threshold;
    int64_t      total_zero = 0;
    g_ofs.write(reinterpret_cast<const char *>(&threshold), sizeof(threshold));
    g_ofs.write(reinterpret_cast<const char *>(&total_zero), sizeof(total_zero));

    // Reserve the per-row outlier-count strip. We can't fill it yet
    // (the F32 rows haven't been written), so seek past it. close_
    // dequant_writer comes back to write the strip and the final
    // header total.
    g_row_outlier_counts.assign((size_t) rows, 0);
    g_outlier_count_total = 0;
    const int64_t strip_bytes = (int64_t) rows * (int64_t) sizeof(int32_t);
    std::vector<char> zeros((size_t) strip_bytes, 0);
    g_ofs.write(zeros.data(), (std::streamsize) strip_bytes);

    g_open_tensor_name = tensor_name;
    g_open_rows        = rows;
    g_open_cols        = cols;
}

void write_dequant_row(int64_t row_idx, const float * data, int64_t n) {
    (void) row_idx;
    if (!g_ofs.is_open() || data == nullptr || n <= 0) {
        return;
    }
    if (n != g_open_cols) {
        LOG_WRN("tessera_debug: row width mismatch for '%s' (cols=%lld, got=%lld); "
                "writing %lld values\n",
                g_open_tensor_name.c_str(),
                (long long) g_open_cols, (long long) n, (long long) n);
    }

    // Count outliers in this row against the configured threshold.
    // |x| > t using fabsf to keep the comparison branchless-friendly;
    // the sidecar writer is off the hot path so the per-element fabs
    // is not a performance concern.
    int32_t row_count = 0;
    const float t = g_outlier_threshold;
    for (int64_t i = 0; i < n; i++) {
        if (fabsf(data[i]) > t) {
            row_count++;
        }
    }

    if (row_idx >= 0 && row_idx < (int64_t) g_row_outlier_counts.size()) {
        g_row_outlier_counts[(size_t) row_idx] = row_count;
    } else {
        LOG_WRN("tessera_debug: row_idx %lld out of range for '%s' (rows=%lld); "
                "counting but not recording per-row\n",
                (long long) row_idx, g_open_tensor_name.c_str(),
                (long long) g_open_rows);
    }
    g_outlier_count_total += row_count;

    g_ofs.write(reinterpret_cast<const char *>(data),
                static_cast<std::streamsize>(n * sizeof(float)));
}

void close_dequant_writer() {
    if (!g_ofs.is_open()) {
        g_open_tensor_name.clear();
        g_open_rows = 0;
        g_open_cols = 0;
        g_row_outlier_counts.clear();
        g_outlier_count_total = 0;
        return;
    }

    // Patch the v2 header total at offset 32 (right after the 28-byte
    // v1 header and the 4-byte threshold).
    const std::streamoff total_off = (std::streamoff) (4 + 4 + 8 + 8 + 4 + 4);
    g_ofs.seekp(total_off, std::ios::beg);
    if (!g_ofs) {
        LOG_ERR("tessera_debug: seekp to total-offset failed for '%s'; "
                "per-row strip will not be written\n",
                g_open_tensor_name.c_str());
    } else {
        g_ofs.write(reinterpret_cast<const char *>(&g_outlier_count_total),
                    sizeof(g_outlier_count_total));
    }

    // Patch the per-row outlier-count strip at offset 40, just before
    // the F32 data block.
    const std::streamoff strip_off = (std::streamoff) (4 + 4 + 8 + 8 + 4 + 4 + 8);
    g_ofs.seekp(strip_off, std::ios::beg);
    if (!g_ofs) {
        LOG_ERR("tessera_debug: seekp to strip-offset failed for '%s'\n",
                g_open_tensor_name.c_str());
    } else if (!g_row_outlier_counts.empty()) {
        g_ofs.write(reinterpret_cast<const char *>(g_row_outlier_counts.data()),
                    (std::streamsize) (g_row_outlier_counts.size() * sizeof(int32_t)));
    }

    g_ofs.flush();
    g_ofs.close();
    g_open_tensor_name.clear();
    g_open_rows = 0;
    g_open_cols = 0;
    g_row_outlier_counts.clear();
    g_outlier_count_total = 0;
}

} // namespace tessera_debug
