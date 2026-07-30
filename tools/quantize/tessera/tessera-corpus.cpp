//
// tessera-corpus.cpp
//
// Built-in synthetic calibration corpus. Generated at runtime from a
// fixed-seed xorshift32 PRNG (Box-Muller for Gaussians), so the binary
// carries no static data. Deterministic: same params -> same output.
//

#include "tessera-corpus.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <algorithm>

#include <dirent.h>
#include <sys/stat.h>

static const float TS_PI = 3.141592653589793f;

// ---------------------------------------------------------------------------
// PRNG + small helpers
// ---------------------------------------------------------------------------

static uint32_t ts_xorshift32(uint32_t * state) {
    uint32_t x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    return x;
}

// uniform in [0, 1)
static float ts_corpus_uniform01(uint32_t * state) {
    return ts_xorshift32(state) * (1.0f / 4294967296.0f);
}

// standard normal via Box-Muller
static float ts_corpus_gaussian(uint32_t * state) {
    float u1;
    do {
        u1 = (ts_xorshift32(state) + 1u) * (1.0f / 4294967296.0f);
    } while (u1 <= 0.0f);
    float u2 = ts_corpus_uniform01(state);
    return sqrtf(-2.0f * logf(u1)) * cosf(2.0f * TS_PI * u2);
}

// ---------------------------------------------------------------------------
// generation
// ---------------------------------------------------------------------------

ts_corpus_params ts_corpus_default_params() {
    ts_corpus_params p;
    p.n_tokens      = 256;
    p.in_dim        = 4096;
    p.seed          = 12345;
    p.outlier_rate  = 0.01f;
    p.outlier_scale = 10.0f;
    return p;
}

std::vector<float> ts_corpus_generate(const ts_corpus_params * params) {
    ts_corpus_params p = params ? *params : ts_corpus_default_params();
    if (p.n_tokens < 0) p.n_tokens = 0;
    if (p.in_dim   < 0) p.in_dim   = 0;

    uint32_t st = p.seed ? p.seed : 1u;  // xorshift needs a non-zero state

    std::vector<float> X((size_t)p.n_tokens * (size_t)p.in_dim);
    for (int64_t t = 0; t < p.n_tokens; t++) {
        float * row = X.data() + (size_t)t * (size_t)p.in_dim;
        for (int64_t c = 0; c < p.in_dim; c++) {
            float v = ts_corpus_gaussian(&st);
            if (ts_corpus_uniform01(&st) < p.outlier_rate) {
                v *= p.outlier_scale;
            }
            row[c] = v;
        }
    }
    return X;
}

// ---------------------------------------------------------------------------
// .npy writer (numpy format v1.0, little-endian f32, C order)
// ---------------------------------------------------------------------------

static int ts_npy_write(const char * path, const float * data,
                        int64_t n_tokens, int64_t in_dim, std::string * err) {
    FILE * f = std::fopen(path, "wb");
    if (!f) {
        if (err) *err = std::string("cannot open for write: ") + path;
        return 1;
    }

    char shape[64];
    std::snprintf(shape, sizeof(shape), "(%lld, %lld)",
                  (long long)n_tokens, (long long)in_dim);
    std::string dict =
        std::string("{'descr': '<f4', 'fortran_order': False, 'shape': ") + shape + ", }";

    // pad so the 10-byte preamble + header is a multiple of 64, ending in '\n'
    size_t unpadded = dict.size() + 1;
    size_t pad = (64 - ((10 + unpadded) % 64)) % 64;
    dict.append(pad, ' ');
    dict.push_back('\n');

    const unsigned char magic[6] = { 0x93, 'N', 'U', 'M', 'P', 'Y' };
    const unsigned char ver[2]   = { 1, 0 };
    uint16_t hlen = (uint16_t)dict.size();
    const unsigned char hlen_le[2] = {
        (unsigned char)(hlen & 0xff),
        (unsigned char)((hlen >> 8) & 0xff),
    };

    size_t n = (size_t)n_tokens * (size_t)in_dim;
    bool ok = true;
    ok &= std::fwrite(magic,    1, 6,          f) == 6;
    ok &= std::fwrite(ver,      1, 2,          f) == 2;
    ok &= std::fwrite(hlen_le,  1, 2,          f) == 2;
    ok &= std::fwrite(dict.data(), 1, dict.size(), f) == dict.size();
    ok &= std::fwrite(data, sizeof(float), n,  f) == n;
    std::fclose(f);

    if (!ok) {
        if (err) *err = std::string("write failed: ") + path;
        return 1;
    }
    return 0;
}

int ts_corpus_generate_to_file(const ts_corpus_params * params,
                               const char * output_path,
                               std::string * err_msg) {
    if (!output_path) {
        if (err_msg) *err_msg = "null output_path";
        return 1;
    }
    ts_corpus_params p = params ? *params : ts_corpus_default_params();
    if (p.n_tokens < 0) p.n_tokens = 0;
    if (p.in_dim   < 0) p.in_dim   = 0;

    std::vector<float> X = ts_corpus_generate(params);
    return ts_npy_write(output_path, X.data(), p.n_tokens, p.in_dim, err_msg);
}

// ---------------------------------------------------------------------------
// .npy reader (minimal: v1/v2 header, <f4, C order)
// ---------------------------------------------------------------------------

static bool ts_npy_read_int(const std::string & s, size_t & pos, int64_t & out) {
    while (pos < s.size() && (s[pos] == ' ' || s[pos] == ',')) pos++;
    int64_t v = 0;
    bool any = false;
    while (pos < s.size() && s[pos] >= '0' && s[pos] <= '9') {
        v = v * 10 + (s[pos] - '0');
        pos++;
        any = true;
    }
    if (any) out = v;
    return any;
}

static std::vector<float> ts_npy_read(const char * path,
                                      int64_t * n_tokens, int64_t * in_dim,
                                      std::string * err) {
    FILE * f = std::fopen(path, "rb");
    if (!f) {
        if (err) *err = std::string("cannot open: ") + path;
        return {};
    }

    unsigned char magic[6];
    if (std::fread(magic, 1, 6, f) != 6 || magic[0] != 0x93 ||
        std::memcmp(magic + 1, "NUMPY", 5) != 0) {
        std::fclose(f);
        if (err) *err = std::string("bad npy magic: ") + path;
        return {};
    }

    unsigned char ver[2];
    if (std::fread(ver, 1, 2, f) != 2) {
        std::fclose(f);
        if (err) *err = std::string("short header: ") + path;
        return {};
    }

    size_t hlen;
    if (ver[0] == 1) {
        unsigned char b[2];
        if (std::fread(b, 1, 2, f) != 2) {
            std::fclose(f);
            if (err) *err = std::string("short header: ") + path;
            return {};
        }
        hlen = (size_t)b[0] | ((size_t)b[1] << 8);
    } else {
        unsigned char b[4];
        if (std::fread(b, 1, 4, f) != 4) {
            std::fclose(f);
            if (err) *err = std::string("short header: ") + path;
            return {};
        }
        hlen = (size_t)b[0] | ((size_t)b[1] << 8) |
               ((size_t)b[2] << 16) | ((size_t)b[3] << 24);
    }

    std::string hdr(hlen, '\0');
    if (std::fread(&hdr[0], 1, hlen, f) != hlen) {
        std::fclose(f);
        if (err) *err = std::string("short header: ") + path;
        return {};
    }

    if (hdr.find("<f4") == std::string::npos) {
        std::fclose(f);
        if (err) *err = std::string("unsupported descr (need <f4): ") + path;
        return {};
    }
    if (hdr.find("'fortran_order': True") != std::string::npos) {
        std::fclose(f);
        if (err) *err = std::string("fortran_order not supported: ") + path;
        return {};
    }

    size_t pos = hdr.find("'shape'");
    if (pos == std::string::npos) {
        std::fclose(f);
        if (err) *err = std::string("no shape in header: ") + path;
        return {};
    }
    pos = hdr.find('(', pos);
    if (pos == std::string::npos) {
        std::fclose(f);
        if (err) *err = std::string("bad shape: ") + path;
        return {};
    }
    pos++;
    int64_t a = 0, b = 1;
    ts_npy_read_int(hdr, pos, a);
    ts_npy_read_int(hdr, pos, b);

    size_t n = (size_t)a * (size_t)b;
    std::vector<float> data(n);
    if (n && std::fread(data.data(), sizeof(float), n, f) != n) {
        std::fclose(f);
        if (err) *err = std::string("short data: ") + path;
        return {};
    }
    std::fclose(f);

    if (n_tokens) *n_tokens = a;
    if (in_dim)   *in_dim   = b;
    return data;
}

std::vector<float> ts_corpus_load_directory(const char * dir_path,
                                            int64_t * out_n_tokens,
                                            int64_t * out_in_dim,
                                            std::string * err_msg) {
    if (!dir_path) {
        if (err_msg) *err_msg = "null dir_path";
        return {};
    }

    struct stat st;
    if (stat(dir_path, &st) != 0) {
        if (err_msg) *err_msg = std::string("no such path: ") + dir_path;
        return {};
    }

    // a bare .npy file is accepted directly
    if (S_ISREG(st.st_mode)) {
        return ts_npy_read(dir_path, out_n_tokens, out_in_dim, err_msg);
    }
    if (!S_ISDIR(st.st_mode)) {
        if (err_msg) *err_msg = std::string("not a file or directory: ") + dir_path;
        return {};
    }

    DIR * d = opendir(dir_path);
    if (!d) {
        if (err_msg) *err_msg = std::string("cannot open dir: ") + dir_path;
        return {};
    }
    std::vector<std::string> files;
    struct dirent * e;
    while ((e = readdir(d)) != nullptr) {
        std::string name = e->d_name;
        size_t dot = name.find_last_of('.');
        if (dot != std::string::npos && name.substr(dot) == ".npy") {
            files.push_back(std::string(dir_path) + "/" + name);
        }
    }
    closedir(d);

    if (files.empty()) {
        if (err_msg) *err_msg = std::string("no .npy files in: ") + dir_path;
        return {};
    }
    std::sort(files.begin(), files.end());
    return ts_npy_read(files[0].c_str(), out_n_tokens, out_in_dim, err_msg);
}
