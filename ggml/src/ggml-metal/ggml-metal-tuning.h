#pragma once

#include "ggml.h"

#include <cstdint>
#include <vector>

namespace ggml_metal_tuning {

// FA vec selection buckets. ne01 (query rows) splits decode (==1) from batch (>=2), the
// batch side refined into {2,3,4,5}: Q>1 reuses one K/V load across rows, so it only pays
// off once ne01 aligns with Q. ne11 (KV length) is bucketed too, as the Q>1 crossover is
// head-size dependent (small dk crosses late, large dk wins even at short KV).
constexpr int FA_VEC_NE11_BUCKETS[] = { 1024, 4096, 16384 };
constexpr int FA_VEC_NE01_BUCKETS[] = { 2, 3, 4, 5 };

int fa_vec_ne11_bucket(int64_t ne11);
int fa_vec_ne01_bucket(int64_t ne01);

// NE baked into each (dk,dv) baseline instantiation in kernels/fa_vec_*.metal.
// Hand-maintained mirror; keep in sync with those instantiations.
// The Metal test slice covers every legal config for dk=128 and dk=576.
int fa_vec_baseline_ne(int dk, int dv);

// Tuned table has two row kinds. Exact rows key a (ne11_b, ne01_b) bucket. Default rows
// collapse ne11 over one ne01 domain: ne11_b == FA_VEC_NE11_DEFAULT and ne01_b holds the
// domain. fa_vec_pick tries exact bucket -> domain default -> baseline; short KV
// (ne11 < FA_VEC_NE11_BUCKETS[0]) always uses baseline.
constexpr int8_t FA_VEC_NE11_DEFAULT  = -1;
constexpr int8_t FA_VEC_DOMAIN_DECODE = 0;  // ne01 == 1
constexpr int8_t FA_VEC_DOMAIN_BATCH  = 1;  // ne01 >= 2

struct fa_vec_key_t {
    int8_t  family;
    int8_t  dtype;
    int16_t dk;
    int16_t dv;
    int8_t  ne11_b;
    int8_t  ne01_b;
};

static_assert(sizeof(fa_vec_key_t) == 8, "fa_vec_key_t must be tightly packed for memcmp");

struct fa_vec_cfg_t {
    int8_t Q;
    int8_t NE;
};

struct fa_vec_entry_t {
    fa_vec_key_t key;
    fa_vec_cfg_t cfg;
};

// legal NE values for a (dk,dv): NL = 32/NE, require (dk/4)%NL==0 && (dv/4)%NL==0.
// single source shared by the offline tuner and test-backend-ops.
inline std::vector<int> fa_vec_legal_ne(int dk, int dv) {
    std::vector<int> r;
    for (int ne : { 1, 2, 4 }) {
        const int nl = 32 / ne;
        if ((dk / 4) % nl == 0 && (dv / 4) % nl == 0) {
            r.push_back(ne);
        }
    }
    return r;
}

// test/tune-only override; when set, fa_vec_pick returns it directly.
void         fa_vec_set_override(fa_vec_cfg_t cfg);
void         fa_vec_clear_override();
fa_vec_cfg_t fa_vec_baseline_cfg(int dk, int dv);

// Keyed by Apple GPU family; an untuned family matches no row and gets the baseline.
fa_vec_cfg_t fa_vec_pick(int gpu_family, int dtype, int dk, int dv, int64_t ne11, int64_t ne01);

// FA (non-vec) queries per threadgroup (Q) and simdgroups per threadgroup (NSG). Rows key an exact ne11 bucket,
// or all of them with ne11_b == FA_NE11_DEFAULT. fa_pick tries exact bucket -> default -> baseline.
// Keyed by device SKU, as tiles_min is an occupancy threshold that differs within a GPU family; an untuned
// device matches no row and gets the baseline.
// A row applies from tiles_min wide tiles (ceil(ne01/16)*ne02*ne03). ne01 < FA_NE01_MIN, or a last wide tile
// of 1..8 rows in a short batch, keeps the baseline.
constexpr int FA_NE11_BUCKETS[]         = { 4096, 8192, 16384, 32768, 65536 };
constexpr int FA_NE01_MIN               = 64;
constexpr int FA_NE01_MIN_PARTIAL_TILES = 50;  // such a tile pads 1/this of the row work; 50 keeps that under
                                               // the 2% aggregate gain the tuner requires of a row (TUNE_THETA)

constexpr int8_t FA_NE11_DEFAULT = -1;

constexpr int8_t FA_Q_BASELINE = 8;
constexpr int8_t FA_Q_WIDE     = 16;

int fa_ne11_bucket(int64_t ne11);

struct fa_key_t {
    int8_t  device_id;
    int8_t  ne11_b;
    int16_t dk;
    int16_t dv;
};

static_assert(sizeof(fa_key_t) == 6, "fa_key_t must be tightly packed for memcmp");

// NSG is used with the wide tile only - the baseline tile keeps its own choice
struct fa_cfg_t {
    int8_t Q;
    int8_t NSG;
};

struct fa_entry_t {
    fa_key_t key;
    fa_cfg_t cfg;
    int16_t  tiles_min;
};

// test/tune-only override; when set, fa_pick returns it directly.
void fa_set_override(fa_cfg_t cfg);
void fa_clear_override();

fa_cfg_t fa_pick(int device_id, int dk, int dv, int64_t ne11, int64_t ne01, int64_t ne02, int64_t ne03);

}  // namespace ggml_metal_tuning
