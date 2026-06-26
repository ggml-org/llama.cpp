#pragma once

#include "ggml-metal-device.h"  // enum ggml_metal_device_id
#include "ggml.h"

#include <cstdint>

namespace ggml_metal_tuning {

// FA vec selection key buckets: ne11 = KV length, ne01 = query rows.
constexpr int FA_VEC_NE11_BUCKETS[] = { 1024, 4096, 16384 };
constexpr int FA_VEC_NE01_BUCKETS[] = { 2, 3, 5 };

int fa_vec_ne11_bucket(int64_t ne11);
int fa_vec_ne01_bucket(int64_t ne01);

// NE baked into each (dk,dv) baseline instantiation in kernels/fa.metal.
// Must stay in sync with those instantiations (drift guard checks this).
int fa_vec_baseline_ne(int dk, int dv);

// device_id -> Apple GPU family (7..10); 0 for GENERIC/unknown.
int fa_vec_device_family(enum ggml_metal_device_id device_id);

struct fa_vec_key_t {
    int8_t  device_id;  // ggml_metal_device_id; reused as the family value in the family table
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

fa_vec_cfg_t fa_vec_pick(enum ggml_metal_device_id device_id, int dtype, int dk, int dv, int64_t ne11, int64_t ne01);

}  // namespace ggml_metal_tuning
