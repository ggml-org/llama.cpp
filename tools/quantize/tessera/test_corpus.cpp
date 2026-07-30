//
// test_corpus.cpp
//
// Smoke test for tessera-corpus.h. Checks determinism, the .npy
// round-trip, and that outlier injection actually produces outliers.
// Returns non-zero on any failure.
//

#include "tessera-corpus.h"

#include <cmath>
#include <cstdio>
#include <cstring>

static int g_fail = 0;

static void check(const char * name, bool ok) {
    if (ok) {
        std::printf("ok   %s\n", name);
    } else {
        std::printf("FAIL %s\n", name);
        g_fail++;
    }
}

int main() {
    ts_corpus_params p = ts_corpus_default_params();
    p.n_tokens = 16;
    p.in_dim   = 64;
    p.seed     = 42;

    // 1-2. generate and check size
    std::vector<float> a = ts_corpus_generate(&p);
    check("size == 16*64", a.size() == (size_t)16 * 64);

    // 3. determinism: same params -> bit-identical output
    std::vector<float> b = ts_corpus_generate(&p);
    check("deterministic", a.size() == b.size() &&
          std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0);

    // 4. write to .npy, read back, verify matches
    const char * path = "/tmp/test_corpus.npy";
    std::string err;
    int rc = ts_corpus_generate_to_file(&p, path, &err);
    check("write npy", rc == 0);
    if (rc != 0) std::printf("     err: %s\n", err.c_str());

    int64_t n_tokens = 0, in_dim = 0;
    std::vector<float> loaded = ts_corpus_load_directory(path, &n_tokens, &in_dim, &err);
    check("read npy dims", n_tokens == 16 && in_dim == 64);
    check("read npy data", loaded.size() == a.size() &&
          std::memcmp(loaded.data(), a.data(), a.size() * sizeof(float)) == 0);
    if (loaded.size() != a.size()) std::printf("     err: %s\n", err.c_str());

    // 5. outliers exist (some value scaled up past 5.0)
    bool has_outlier = false;
    for (float v : a) {
        if (v > 5.0f) { has_outlier = true; break; }
    }
    check("outliers > 5.0", has_outlier);

    if (g_fail == 0) {
        std::printf("\nall tests passed\n");
        return 0;
    }
    std::printf("\n%d check(s) failed\n", g_fail);
    return 1;
}
