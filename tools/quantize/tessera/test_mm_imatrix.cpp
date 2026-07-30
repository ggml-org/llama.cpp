//
// test_mm_imatrix.cpp
//
// Build:
//   clang++ -std=c++17 -O2 -I tools/quantize/tessera \
//       tools/quantize/tessera/tessera-mm-imatrix.cpp \
//       tools/quantize/tessera/test_mm_imatrix.cpp -o /tmp/test_mm_imatrix
//

#include "tessera-mm-imatrix.h"

#include <cstdio>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>

#define CHECK(cond, msg) do { if (!(cond)) { std::fprintf(stderr, "FAIL: %s\n", msg); return 1; } } while (0)

// ---------------------------------------------------------------------------
// minimal npz writer (stored ZIP entries + NPY v1.0 <f4)
// ---------------------------------------------------------------------------

static void put_u16(std::vector<uint8_t> & b, uint16_t v) {
    b.push_back((uint8_t)(v & 0xff));
    b.push_back((uint8_t)((v >> 8) & 0xff));
}

static void put_u32(std::vector<uint8_t> & b, uint32_t v) {
    for (int i = 0; i < 4; ++i) {
        b.push_back((uint8_t)((v >> (8 * i)) & 0xff));
    }
}

static std::vector<uint8_t> make_npy(const std::vector<float> & vals) {
    std::string hdr = "{'descr': '<f4', 'fortran_order': False, 'shape': (" +
                      std::to_string(vals.size()) + ",), }";
    size_t base = 10 + hdr.size() + 1; // + trailing newline
    size_t pad = (64 - (base % 64)) % 64;
    hdr.append(pad, ' ');
    hdr.push_back('\n');

    std::vector<uint8_t> b;
    b.push_back(0x93);
    const char * magic = "NUMPY";
    for (int i = 0; i < 5; ++i) b.push_back((uint8_t)magic[i]);
    b.push_back(1); // major
    b.push_back(0); // minor
    put_u16(b, (uint16_t)hdr.size());
    for (char c : hdr) b.push_back((uint8_t)c);
    for (float f : vals) {
        uint32_t u;
        std::memcpy(&u, &f, 4);
        put_u32(b, u);
    }
    return b;
}

static void zip_add(std::vector<uint8_t> & z, const std::string & name,
                    const std::vector<uint8_t> & data) {
    put_u32(z, 0x04034b50);
    put_u16(z, 20);                          // version needed
    put_u16(z, 0);                           // flags
    put_u16(z, 0);                           // method = stored
    put_u16(z, 0);                           // mod time
    put_u16(z, 0);                           // mod date
    put_u32(z, 0);                           // crc32 (ignored by reader)
    put_u32(z, (uint32_t)data.size());       // compressed size
    put_u32(z, (uint32_t)data.size());       // uncompressed size
    put_u16(z, (uint16_t)name.size());
    put_u16(z, 0);                           // extra len
    for (char c : name) z.push_back((uint8_t)c);
    z.insert(z.end(), data.begin(), data.end());
}

// ---------------------------------------------------------------------------
// independent reference computations
// ---------------------------------------------------------------------------

static float kurtosis_of(const std::vector<float> & v) {
    double ex2 = 0.0, ex4 = 0.0;
    for (float f : v) {
        double x = f;
        double x2 = x * x;
        ex2 += x2;
        ex4 += x2 * x2;
    }
    double n = (double)v.size();
    ex2 /= n;
    ex4 /= n;
    if (ex2 <= 1e-30) return 0.0f;
    return (float)(ex4 / (ex2 * ex2) - 3.0);
}

static double sumsq(const std::vector<float> & v) {
    double s = 0.0;
    for (float f : v) s += (double)f * (double)f;
    return s;
}

int main() {
    // tensor blk.0.attn_q: all 3 modalities, distinct lengths so the
    // joint-kurtosis weighting is non-trivial.
    std::vector<float> a_text(100), a_image(200), a_audio(50);
    for (size_t i = 0; i < a_text.size();  ++i) a_text[i]  = (i % 2) ? 1.0f : -1.0f; // kurtosis = -2
    for (size_t i = 0; i < a_image.size(); ++i) a_image[i] = (float)((int)(i % 7) - 3) * 0.5f + ((i % 13 == 0) ? 4.0f : 0.0f);
    for (size_t i = 0; i < a_audio.size(); ++i) a_audio[i] = (float)((int)(i % 5) - 2) * 0.3f;

    // tensor blk.2.ffn: text + image only, audio absent.
    std::vector<float> c_text(80), c_image(60);
    for (size_t i = 0; i < c_text.size();  ++i) c_text[i]  = (float)((int)(i % 9) - 4) * 0.25f;
    for (size_t i = 0; i < c_image.size(); ++i) c_image[i] = (float)((int)(i % 3) - 1) * 1.5f;

    std::vector<uint8_t> z;
    zip_add(z, "blk.0.attn_q.text.npy",  make_npy(a_text));
    zip_add(z, "blk.0.attn_q.image.npy", make_npy(a_image));
    zip_add(z, "blk.0.attn_q.audio.npy", make_npy(a_audio));
    zip_add(z, "blk.2.ffn.text.npy",     make_npy(c_text));
    zip_add(z, "blk.2.ffn.image.npy",    make_npy(c_image));

    const char * path = "/tmp/test_mm_imatrix_data.npz";
    FILE * f = std::fopen(path, "wb");
    CHECK(f != nullptr, "open temp npz for write");
    CHECK(std::fwrite(z.data(), 1, z.size(), f) == z.size(), "write temp npz");
    std::fclose(f);

    ts_mm_imatrix mm;
    std::string err;
    int rc = ts_mm_imatrix_load(path, &mm, &err);
    CHECK(rc == 0, err.empty() ? "load failed" : err.c_str());
    CHECK(mm.version == 3, "expected version 3");
    CHECK(mm.data.size() == 2, "expected 2 tensors");

    // --- blk.0.attn_q: per-modality stats populated ---
    auto itA = mm.data.find("blk.0.attn_q");
    CHECK(itA != mm.data.end(), "blk.0.attn_q present");
    const ts_mm_imatrix_entry & A = itA->second;

    for (int m = 0; m < TS_MODALITY_COUNT; ++m) {
        CHECK(A.has_modality[m], "A: all modalities present");
    }
    CHECK(A.counts[TS_MODALITY_TEXT]  == 100, "A text count");
    CHECK(A.counts[TS_MODALITY_IMAGE] == 200, "A image count");
    CHECK(A.counts[TS_MODALITY_AUDIO] == 50,  "A audio count");

    CHECK(std::fabs((double)A.in_sum2[TS_MODALITY_TEXT]  - sumsq(a_text))  < 1e-2, "A text in_sum2");
    CHECK(std::fabs((double)A.in_sum2[TS_MODALITY_IMAGE] - sumsq(a_image)) < 1e-1, "A image in_sum2");
    CHECK(std::fabs((double)A.in_sum2[TS_MODALITY_AUDIO] - sumsq(a_audio)) < 1e-2, "A audio in_sum2");

    // per-modality kurtosis matches an independent computation
    CHECK(std::fabs(A.stats[TS_MODALITY_TEXT].kurtosis  - kurtosis_of(a_text))  < 1e-4, "A text kurtosis");
    CHECK(std::fabs(A.stats[TS_MODALITY_IMAGE].kurtosis - kurtosis_of(a_image)) < 1e-4, "A image kurtosis");
    CHECK(std::fabs(A.stats[TS_MODALITY_AUDIO].kurtosis - kurtosis_of(a_audio)) < 1e-4, "A audio kurtosis");

    // hand-derived value: +/-1 distribution has excess kurtosis -2
    CHECK(std::fabs(A.stats[TS_MODALITY_TEXT].kurtosis - (-2.0f)) < 1e-4, "A text kurtosis == -2");
    CHECK(A.stats[TS_MODALITY_TEXT].mean_magnitude > 0.0f, "A text mean_magnitude populated");

    // accessor
    const ts_imatrix_regime_stats * st = ts_mm_imatrix_modality_stats(&mm, "blk.0.attn_q", TS_MODALITY_IMAGE);
    CHECK(st != nullptr, "A image stats accessor");
    CHECK(std::fabs(st->kurtosis - kurtosis_of(a_image)) < 1e-4, "A image stats kurtosis");

    // missing mask = 0 when all present
    CHECK(ts_mm_imatrix_missing_mask(&A) == 0, "A missing mask == 0");

    // joint kurtosis is the count-weighted average
    double kt = kurtosis_of(a_text);
    double ki = kurtosis_of(a_image);
    double ka = kurtosis_of(a_audio);
    double expected_joint = (100.0 * kt + 200.0 * ki + 50.0 * ka) / (100.0 + 200.0 + 50.0);
    CHECK(std::fabs((double)ts_mm_imatrix_joint_kurtosis(&A) - expected_joint) < 1e-4, "A joint kurtosis weighted");

    // a simple (unweighted) average must differ, else the test is vacuous
    double simple_avg = (kt + ki + ka) / 3.0;
    CHECK(std::fabs(simple_avg - expected_joint) > 1e-3, "weighting is exercised");

    // --- blk.2.ffn: audio absent ---
    auto itC = mm.data.find("blk.2.ffn");
    CHECK(itC != mm.data.end(), "blk.2.ffn present");
    const ts_mm_imatrix_entry & C = itC->second;

    CHECK(C.has_modality[TS_MODALITY_TEXT],  "C text present");
    CHECK(C.has_modality[TS_MODALITY_IMAGE], "C image present");
    CHECK(!C.has_modality[TS_MODALITY_AUDIO], "C audio absent");

    CHECK(ts_mm_imatrix_missing_mask(&C) == (1 << TS_MODALITY_AUDIO), "C missing mask == audio bit");
    CHECK(ts_mm_imatrix_modality_stats(&mm, "blk.2.ffn", TS_MODALITY_AUDIO) == nullptr, "C audio stats null");
    CHECK(ts_mm_imatrix_modality_stats(&mm, "blk.2.ffn", TS_MODALITY_TEXT)  != nullptr, "C text stats non-null");

    // joint kurtosis over present modalities only
    double expected_c = (80.0 * kurtosis_of(c_text) + 60.0 * kurtosis_of(c_image)) / (80.0 + 60.0);
    CHECK(std::fabs((double)ts_mm_imatrix_joint_kurtosis(&C) - expected_c) < 1e-4, "C joint kurtosis weighted");

    std::printf("ok\n");
    return 0;
}
