//
// test_l1_sidecar.cpp
//
// End-to-end test for the L1 dequant sidecar writer: open, write rows,
// close, read back via the v3 reader, and verify header + data. Also
// tests the row-capture stride.
//

#include "tessera-debug.h"
#include "tessera-sidecar-v3.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

static const char * TEST_DIR = "/tmp/test_l1_sidecar";

static int test_basic_write_read() {
    const int64_t rows = 4;
    const int64_t cols = 8;

    tessera_debug::set_dequant_stride(1);
    tessera_debug::set_dequant_dir(TEST_DIR);

    if (!tessera_debug::dequant_debug_enabled()) {
        fprintf(stderr, "FAIL: dequant_debug_enabled() returned false after set_dequant_dir\n");
        return -1;
    }

    // write known data
    std::vector<float> data((size_t)(rows * cols));
    for (int64_t i = 0; i < rows * cols; i++) {
        data[(size_t) i] = (float) i * 0.25f;
    }

    tessera_debug::open_dequant_writer("test_basic", rows, cols);
    for (int64_t r = 0; r < rows; r++) {
        tessera_debug::write_dequant_row(r, data.data() + r * cols, cols);
    }
    tessera_debug::close_dequant_writer();

    // read back
    const std::string path = std::string(TEST_DIR) + "/test_basic.dequant.f32";
    ts_sidecar_v3 sc;
    std::string err;
    if (ts_sidecar_v3_read(path.c_str(), &sc, &err) != 0) {
        fprintf(stderr, "FAIL: read: %s\n", err.c_str());
        return -1;
    }

    // verify header
    if (sc.header.version != 3) {
        fprintf(stderr, "FAIL: version = %u, want 3\n", sc.header.version);
        return -1;
    }
    if (sc.header.rows != rows || sc.header.cols != cols) {
        fprintf(stderr, "FAIL: dims = %lldx%lld, want %lldx%lld\n",
                (long long) sc.header.rows, (long long) sc.header.cols,
                (long long) rows, (long long) cols);
        return -1;
    }
    if (sc.header.dtype != 0) {
        fprintf(stderr, "FAIL: dtype = %u, want 0 (F32)\n", sc.header.dtype);
        return -1;
    }

    // verify data
    if ((int64_t) sc.data.size() != rows * cols) {
        fprintf(stderr, "FAIL: data size = %zu, want %lld\n",
                sc.data.size(), (long long)(rows * cols));
        return -1;
    }
    for (int64_t i = 0; i < rows * cols; i++) {
        float expect = (float) i * 0.25f;
        if (sc.data[(size_t) i] != expect) {
            fprintf(stderr, "FAIL: data[%lld] = %f, want %f\n",
                    (long long) i, sc.data[(size_t) i], expect);
            return -1;
        }
    }

    // verify provenance file exists
    const std::string prov_path = path + ".provenance.json";
    if (!std::filesystem::exists(prov_path)) {
        fprintf(stderr, "FAIL: provenance file missing: %s\n", prov_path.c_str());
        return -1;
    }

    return 0;
}

static int test_stride() {
    const int64_t rows = 8;
    const int64_t cols = 4;
    const int64_t stride = 3;
    // rows 0, 3, 6 -> 3 captured rows
    const int64_t captured = (rows + stride - 1) / stride;

    tessera_debug::set_dequant_stride(stride);
    tessera_debug::set_dequant_dir(TEST_DIR);

    if (tessera_debug::dequant_stride() != stride) {
        fprintf(stderr, "FAIL: dequant_stride() = %lld, want %lld\n",
                (long long) tessera_debug::dequant_stride(), (long long) stride);
        return -1;
    }

    std::vector<float> data((size_t)(rows * cols));
    for (int64_t i = 0; i < rows * cols; i++) {
        data[(size_t) i] = (float) i;
    }

    // simulate what cpu_dump_dequant does with stride
    tessera_debug::open_dequant_writer("test_stride", captured, cols);
    int64_t out_r = 0;
    for (int64_t r = 0; r < rows; r += stride, out_r++) {
        tessera_debug::write_dequant_row(out_r, data.data() + r * cols, cols);
    }
    tessera_debug::close_dequant_writer();

    const std::string path = std::string(TEST_DIR) + "/test_stride.dequant.f32";
    ts_sidecar_v3 sc;
    std::string err;
    if (ts_sidecar_v3_read(path.c_str(), &sc, &err) != 0) {
        fprintf(stderr, "FAIL: stride read: %s\n", err.c_str());
        return -1;
    }

    if (sc.header.rows != captured) {
        fprintf(stderr, "FAIL: stride rows = %lld, want %lld\n",
                (long long) sc.header.rows, (long long) captured);
        return -1;
    }
    if (sc.header.cols != cols) {
        fprintf(stderr, "FAIL: stride cols = %lld, want %lld\n",
                (long long) sc.header.cols, (long long) cols);
        return -1;
    }

    // verify captured rows match source rows 0, 3, 6
    for (int64_t cr = 0; cr < captured; cr++) {
        int64_t src_r = cr * stride;
        for (int64_t c = 0; c < cols; c++) {
            float expect = (float)(src_r * cols + c);
            float got = sc.data[(size_t)(cr * cols + c)];
            if (got != expect) {
                fprintf(stderr, "FAIL: stride data[%lld,%lld] = %f, want %f\n",
                        (long long) cr, (long long) c, got, expect);
                return -1;
            }
        }
    }

    // reset stride for other tests
    tessera_debug::set_dequant_stride(1);
    return 0;
}

static int test_disabled() {
    tessera_debug::set_dequant_dir("");
    if (tessera_debug::dequant_debug_enabled()) {
        fprintf(stderr, "FAIL: dequant_debug_enabled() returned true after clearing dir\n");
        return -1;
    }
    return 0;
}

int main() {
    std::filesystem::create_directories(TEST_DIR);

    if (test_basic_write_read() != 0) {
        return 1;
    }
    printf("  basic write/read: OK\n");

    if (test_stride() != 0) {
        return 1;
    }
    printf("  stride capture:   OK\n");

    if (test_disabled() != 0) {
        return 1;
    }
    printf("  disabled gate:    OK\n");

    // cleanup
    std::filesystem::remove_all(TEST_DIR);

    printf("PASS\n");
    return 0;
}
