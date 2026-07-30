#include "tessera-debug.h"

#include "log.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <system_error>

namespace tessera_debug {

// File-static state. Not thread-safe; one matmul at a time per process.
static std::string        g_dequant_dir;
static std::string        g_open_tensor_name; // empty when no file is open
static int64_t            g_open_rows = 0;
static int64_t            g_open_cols = 0;
static std::ofstream      g_ofs;

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
    }
    g_dequant_dir = path;
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

    // 28-byte header: magic(4) | version(4) | rows(8) | cols(8) | dtype(4)
    g_ofs.write(DEQUANT_FILE_MAGIC, 4);
    uint32_t version = DEQUANT_FILE_VERSION;
    uint32_t dtype   = DEQUANT_DTYPE_F32;
    g_ofs.write(reinterpret_cast<const char *>(&version), sizeof(version));
    g_ofs.write(reinterpret_cast<const char *>(&rows),    sizeof(rows));
    g_ofs.write(reinterpret_cast<const char *>(&cols),    sizeof(cols));
    g_ofs.write(reinterpret_cast<const char *>(&dtype),   sizeof(dtype));

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
    g_ofs.write(reinterpret_cast<const char *>(data),
                static_cast<std::streamsize>(n * sizeof(float)));
}

void close_dequant_writer() {
    if (g_ofs.is_open()) {
        g_ofs.flush();
        g_ofs.close();
    }
    g_open_tensor_name.clear();
    g_open_rows = 0;
    g_open_cols = 0;
}

} // namespace tessera_debug
