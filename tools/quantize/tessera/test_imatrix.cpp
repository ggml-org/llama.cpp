//
// test_imatrix.cpp
//
// Smoke test for tessera-imatrix.h. Creates a synthetic .npz in /tmp,
// loads it, verifies values and regime stats.
//

#include "tessera-imatrix.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

static int g_fail = 0;

static void check_close(const char * name, float got, float want) {
    const float tol = 1e-4f * (std::fabs(want) + 1.0f);
    if (std::fabs(got - want) > tol) {
        std::printf("FAIL %-20s got %.7g want %.7g\n", name, (double)got, (double)want);
        g_fail++;
    } else {
        std::printf("ok   %-20s %.7g\n", name, (double)got);
    }
}

static void check_true(const char * name, bool cond) {
    if (!cond) {
        std::printf("FAIL %-20s\n", name);
        g_fail++;
    } else {
        std::printf("ok   %-20s\n", name);
    }
}

// write a uint16 LE
static void put_u16(std::vector<uint8_t> & v, uint16_t x) {
    v.push_back((uint8_t)(x & 0xFF));
    v.push_back((uint8_t)(x >> 8));
}

// write a uint32 LE
static void put_u32(std::vector<uint8_t> & v, uint32_t x) {
    v.push_back((uint8_t)(x & 0xFF));
    v.push_back((uint8_t)((x >> 8) & 0xFF));
    v.push_back((uint8_t)((x >> 16) & 0xFF));
    v.push_back((uint8_t)((x >> 24) & 0xFF));
}

// Build a minimal .npy buffer (v1, <f4, 1-D shape)
static std::vector<uint8_t> make_npy(const float * data, int n) {
    // header string
    char hdr_buf[128];
    std::snprintf(hdr_buf, sizeof(hdr_buf),
                  "{'descr': '<f4', 'fortran_order': False, 'shape': (%d,), }", n);
    size_t hdr_str_len = std::strlen(hdr_buf);

    // pad header so total prefix (10 + hdr_len) is 64-aligned
    size_t total = 10 + hdr_str_len;
    size_t pad = (64 - (total % 64)) % 64;
    hdr_str_len += pad;

    std::vector<uint8_t> out;
    // magic
    out.push_back(0x93);
    out.push_back('N'); out.push_back('U'); out.push_back('M');
    out.push_back('P'); out.push_back('Y');
    // version 1.0
    out.push_back(1);
    out.push_back(0);
    // header length
    put_u16(out, (uint16_t)hdr_str_len);
    // header string + padding
    for (size_t i = 0; i < hdr_str_len; ++i) {
        if (i < std::strlen(hdr_buf)) {
            out.push_back((uint8_t)hdr_buf[i]);
        } else if (i == hdr_str_len - 1) {
            out.push_back('\n');
        } else {
            out.push_back(' ');
        }
    }
    // data
    const uint8_t * p = (const uint8_t *)data;
    out.insert(out.end(), p, p + n * 4);
    return out;
}

// Build a minimal .npz (ZIP with one stored entry)
static void write_test_npz(const char * path, const char * entry_name,
                           const float * data, int n) {
    auto npy = make_npy(data, n);
    uint16_t name_len = (uint16_t)std::strlen(entry_name);
    uint32_t data_sz  = (uint32_t)npy.size();

    std::vector<uint8_t> zip;
    // local file header
    put_u32(zip, 0x04034b50);   // signature
    put_u16(zip, 20);           // version needed
    put_u16(zip, 0);            // flags
    put_u16(zip, 0);            // method: stored
    put_u16(zip, 0);            // mod time
    put_u16(zip, 0);            // mod date
    put_u32(zip, 0);            // crc32 (not checked by our reader)
    put_u32(zip, data_sz);      // compressed size
    put_u32(zip, data_sz);      // uncompressed size
    put_u16(zip, name_len);     // filename length
    put_u16(zip, 0);            // extra field length
    // filename
    for (int i = 0; i < name_len; ++i) {
        zip.push_back((uint8_t)entry_name[i]);
    }
    // data
    zip.insert(zip.end(), npy.begin(), npy.end());

    FILE * f = std::fopen(path, "wb");
    std::fwrite(zip.data(), 1, zip.size(), f);
    std::fclose(f);
}

int main() {
    const char * npz_path = "/tmp/test_imatrix.npz";

    // 8 float32 values with known distribution
    const float vals[8] = { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f };

    write_test_npz(npz_path, "blk.0.attn_q.npy", vals, 8);

    // load
    ts_imatrix im;
    std::string err;
    int rc = ts_imatrix_load_npz(npz_path, &im, &err);
    check_true("load_npz rc==0", rc == 0);
    if (rc != 0) {
        std::printf("  error: %s\n", err.c_str());
        return 1;
    }

    check_true("one entry", im.data.size() == 1);
    check_true("has blk.0.attn_q", im.data.count("blk.0.attn_q") == 1);

    // lookup with .weight suffix normalization
    int64_t dim = 0;
    const float * p = ts_imatrix_lookup(&im, "blk.0.attn_q.weight", &dim);
    check_true("lookup found", p != nullptr);
    check_true("dim == 8", dim == 8);

    if (p) {
        for (int i = 0; i < 8; ++i) {
            char buf[64];
            std::snprintf(buf, sizeof(buf), "val[%d]", i);
            check_close(buf, p[i], vals[i]);
        }
    }

    // lookup miss
    check_true("lookup miss", ts_imatrix_lookup(&im, "blk.1.attn_k", nullptr) == nullptr);

    // regime stats
    ts_imatrix_regime_stats st = ts_imatrix_regime(vals, 8);
    check_true("mean_mag > 0", st.mean_magnitude > 0.0f);
    check_close("mean_mag", st.mean_magnitude, 4.5f);  // mean(1..8) = 4.5

    // kurtosis of uniform-ish 1..8: E[x^4]/E[x^2]^2 - 3
    // E[x^2] = (1+4+9+16+25+36+49+64)/8 = 204/8 = 25.5
    // E[x^4] = (1+16+81+256+625+1296+2401+4096)/8 = 8772/8 = 1096.5
    // kurt = 1096.5 / (25.5^2) - 3 = 1096.5/650.25 - 3 = 1.6863... - 3 = -1.3137
    check_close("kurtosis", st.kurtosis, -1.3137f);

    check_true("eff_rank in [0,1]", st.eff_rank >= 0.0f && st.eff_rank <= 1.0f);
    check_true("p99 > 0", st.p99 > 0.0f);

    std::printf("\np99 = %.4f, eff_rank = %.4f, kurtosis = %.4f\n",
                (double)st.p99, (double)st.eff_rank, (double)st.kurtosis);

    if (g_fail == 0) {
        std::printf("\nall tests passed\n");
        return 0;
    }
    std::printf("\n%d check(s) failed\n", g_fail);
    return 1;
}
