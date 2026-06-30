#pragma once

#include "ggml-metal-device.h"  // enum ggml_metal_device_id
#include "ggml.h"

#include <cstdint>

namespace ggml_metal_tuning {

// FA vec selection key buckets: ne11 = KV length, ne01 = query rows.
// ne01 only needs the 1 vs >=2 split: Q>1 reuses one K/V tile load across query
// rows, so the optimum flips to Q>1 exactly when there are >=2 rows to share it.
// ne11 keeps a finer split: the baseline->Q>1 crossover by KV length is head-size
// dependent (small dk crosses late, large dk wins even at short KV).
constexpr int FA_VEC_NE11_BUCKETS[] = { 1024, 4096, 16384 };
constexpr int FA_VEC_NE01_BUCKETS[] = { 2 };

int fa_vec_ne11_bucket(int64_t ne11);
int fa_vec_ne01_bucket(int64_t ne01);

// NE baked into each (dk,dv) baseline instantiation in kernels/fa.metal.
// Hand-maintained mirror; keep in sync with those instantiations (run_fa_vec_tune_check
// exercises every (Q,NE), so a missing instantiation surfaces there).
int fa_vec_baseline_ne(int dk, int dv);

struct fa_vec_key_t {
    int8_t  device_id;
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

// test/tune-only override; when set, fa_vec_pick returns it directly.
void         fa_vec_set_override(fa_vec_cfg_t cfg);
void         fa_vec_clear_override();
bool         fa_vec_override_active();
fa_vec_cfg_t fa_vec_baseline_cfg(int dk, int dv);

// device_id selects a per-SKU row; on a miss, gpu_family (0 if unknown) maps to a representative
// SKU and the table is retried. No match -> baseline.
fa_vec_cfg_t fa_vec_pick(enum ggml_metal_device_id device_id, int gpu_family, int dtype, int dk, int dv, int64_t ne11, int64_t ne01);

}  // namespace ggml_metal_tuning
