#include "ggml-metal-tuning.h"

#include <cstddef>
#include <cstring>
#include <iterator>  // std::size

namespace ggml_metal_tuning {

int fa_vec_ne11_bucket(int64_t ne11) {
    for (int i = 0; i < (int) std::size(FA_VEC_NE11_BUCKETS); ++i) {
        if (ne11 < FA_VEC_NE11_BUCKETS[i]) {
            return i;
        }
    }
    return (int) std::size(FA_VEC_NE11_BUCKETS);
}

int fa_vec_ne01_bucket(int64_t ne01) {
    for (int i = 0; i < (int) std::size(FA_VEC_NE01_BUCKETS); ++i) {
        if (ne01 < FA_VEC_NE01_BUCKETS[i]) {
            return i;
        }
    }
    return (int) std::size(FA_VEC_NE01_BUCKETS);
}

// Mirrors the NE of each (dk,dv) baseline instantiation in kernels/fa.metal.
int fa_vec_baseline_ne(int dk, int dv) {
    if (dk == 32 && dv == 32) {
        return 4;
    }
    if (dk == 64 && dv == 64) {
        return 2;
    }
    if (dk == 96 && dv == 96) {
        return 4;
    }
    if (dk == 128 && dv == 128) {
        return 1;
    }
    if (dk == 192 && dv == 192) {
        return 2;
    }
    if (dk == 192 && dv == 128) {
        return 2;
    }
    if (dk == 256 && dv == 256) {
        return 1;
    }
    if (dk == 320 && dv == 256) {
        return 2;
    }
    if (dk == 512 && dv == 512) {
        return 1;
    }
    if (dk == 576 && dv == 512) {
        return 2;
    }
    return 4;  // template default
}

fa_vec_cfg_t fa_vec_baseline_cfg(int dk, int dv) {
    return { 1, (int8_t) fa_vec_baseline_ne(dk, dv) };
}

// M1->7, M2->8, M3->9, M4->9, M5->10 (M5 pending Apple confirmation).
// M3 and M4 are both Apple9, so M3 inherits Apple9 tuned data via the family
// fallback (no independent M3 measurement). GENERIC/unknown -> 0 (never matches).
int fa_vec_device_family(enum ggml_metal_device_id device_id) {
    switch (device_id) {
        case GGML_METAL_DEVICE_M1:
        case GGML_METAL_DEVICE_M1_PRO:
        case GGML_METAL_DEVICE_M1_MAX:
        case GGML_METAL_DEVICE_M1_ULTRA:
            return 7;
        case GGML_METAL_DEVICE_M2:
        case GGML_METAL_DEVICE_M2_PRO:
        case GGML_METAL_DEVICE_M2_MAX:
        case GGML_METAL_DEVICE_M2_ULTRA:
            return 8;
        case GGML_METAL_DEVICE_M3:
        case GGML_METAL_DEVICE_M3_PRO:
        case GGML_METAL_DEVICE_M3_MAX:
        case GGML_METAL_DEVICE_M3_ULTRA:
            return 9;
        case GGML_METAL_DEVICE_M4:
        case GGML_METAL_DEVICE_M4_PRO:
        case GGML_METAL_DEVICE_M4_MAX:
            return 9;
        case GGML_METAL_DEVICE_M5:
        case GGML_METAL_DEVICE_M5_PRO:
        case GGML_METAL_DEVICE_M5_MAX:
        case GGML_METAL_DEVICE_M5_ULTRA:
            return 10;
        case GGML_METAL_DEVICE_GENERIC:
        default:
            return 0;
    }
}

// Per-device tuned configs measured by `test-backend-ops tune --tune-perf`.
// key = { device_id, dtype, dk, dv, ne11_b, ne01_b }; cfg = { Q, NE }.
// ne01_b: 0 = single query row, 1 = >=2 rows. ne11_b: 0..3 over FA_VEC_NE11_BUCKETS.
// Only configs that beat baseline are listed; unlisted buckets fall back to baseline.
// Measured on Apple M4 Max (Apple9). f16 K/V; GQA spec-decode/verify shapes.
constexpr fa_vec_entry_t fa_vec_tuned_table[] = {
    { { GGML_METAL_DEVICE_M4_MAX, GGML_TYPE_F16, 64, 64, 1, 1 },   { 2, 4 } },
    { { GGML_METAL_DEVICE_M4_MAX, GGML_TYPE_F16, 64, 64, 2, 1 },   { 2, 4 } },
    { { GGML_METAL_DEVICE_M4_MAX, GGML_TYPE_F16, 64, 64, 3, 1 },   { 2, 4 } },
    { { GGML_METAL_DEVICE_M4_MAX, GGML_TYPE_F16, 96, 96, 1, 1 },   { 2, 4 } },
    { { GGML_METAL_DEVICE_M4_MAX, GGML_TYPE_F16, 96, 96, 2, 1 },   { 2, 4 } },
    { { GGML_METAL_DEVICE_M4_MAX, GGML_TYPE_F16, 96, 96, 3, 1 },   { 2, 4 } },
    { { GGML_METAL_DEVICE_M4_MAX, GGML_TYPE_F16, 128, 128, 1, 1 }, { 2, 2 } },
    { { GGML_METAL_DEVICE_M4_MAX, GGML_TYPE_F16, 128, 128, 2, 1 }, { 2, 2 } },
    { { GGML_METAL_DEVICE_M4_MAX, GGML_TYPE_F16, 128, 128, 3, 1 }, { 2, 2 } },
    { { GGML_METAL_DEVICE_M4_MAX, GGML_TYPE_F16, 192, 192, 0, 1 }, { 2, 2 } },
    { { GGML_METAL_DEVICE_M4_MAX, GGML_TYPE_F16, 192, 192, 1, 1 }, { 2, 2 } },
    { { GGML_METAL_DEVICE_M4_MAX, GGML_TYPE_F16, 192, 192, 2, 1 }, { 2, 2 } },
    { { GGML_METAL_DEVICE_M4_MAX, GGML_TYPE_F16, 192, 192, 3, 1 }, { 2, 2 } },
    { { GGML_METAL_DEVICE_M4_MAX, GGML_TYPE_F16, 192, 128, 1, 1 }, { 2, 2 } },
    { { GGML_METAL_DEVICE_M4_MAX, GGML_TYPE_F16, 192, 128, 2, 1 }, { 2, 2 } },
    { { GGML_METAL_DEVICE_M4_MAX, GGML_TYPE_F16, 192, 128, 3, 1 }, { 2, 2 } },
    { { GGML_METAL_DEVICE_M4_MAX, GGML_TYPE_F16, 256, 256, 0, 1 }, { 2, 2 } },
    { { GGML_METAL_DEVICE_M4_MAX, GGML_TYPE_F16, 256, 256, 1, 0 }, { 1, 2 } },
    { { GGML_METAL_DEVICE_M4_MAX, GGML_TYPE_F16, 256, 256, 1, 1 }, { 2, 2 } },
    { { GGML_METAL_DEVICE_M4_MAX, GGML_TYPE_F16, 256, 256, 2, 0 }, { 1, 2 } },
    { { GGML_METAL_DEVICE_M4_MAX, GGML_TYPE_F16, 256, 256, 2, 1 }, { 2, 2 } },
    { { GGML_METAL_DEVICE_M4_MAX, GGML_TYPE_F16, 256, 256, 3, 0 }, { 1, 2 } },
    { { GGML_METAL_DEVICE_M4_MAX, GGML_TYPE_F16, 256, 256, 3, 1 }, { 2, 2 } },
    { { GGML_METAL_DEVICE_M4_MAX, GGML_TYPE_F16, 320, 256, 0, 1 }, { 2, 2 } },
    { { GGML_METAL_DEVICE_M4_MAX, GGML_TYPE_F16, 320, 256, 1, 1 }, { 2, 2 } },
    { { GGML_METAL_DEVICE_M4_MAX, GGML_TYPE_F16, 320, 256, 2, 1 }, { 2, 2 } },
    { { GGML_METAL_DEVICE_M4_MAX, GGML_TYPE_F16, 320, 256, 3, 1 }, { 2, 2 } },
};

// Per-family configs, keyed by Apple GPU family in key.device_id (here 9 = Apple9).
// Apple9+ devices (M3/M4/M5) reuse these via the family fallback in fa_vec_pick.
// The 9 is the family value, not a device_id, although it happens to equal the
// GGML_METAL_DEVICE_M3 enumerator; it is only ever matched in this family table.
// Mirrors the M4 Max data as the Apple9 representative (M4 Max is the only Apple9
// device measured here; M3/M5 inherit by family, see PR notes).
constexpr fa_vec_entry_t fa_vec_family_table[] = {
    { { 9, GGML_TYPE_F16, 64, 64, 1, 1 },   { 2, 4 } },
    { { 9, GGML_TYPE_F16, 64, 64, 2, 1 },   { 2, 4 } },
    { { 9, GGML_TYPE_F16, 64, 64, 3, 1 },   { 2, 4 } },
    { { 9, GGML_TYPE_F16, 96, 96, 1, 1 },   { 2, 4 } },
    { { 9, GGML_TYPE_F16, 96, 96, 2, 1 },   { 2, 4 } },
    { { 9, GGML_TYPE_F16, 96, 96, 3, 1 },   { 2, 4 } },
    { { 9, GGML_TYPE_F16, 128, 128, 1, 1 }, { 2, 2 } },
    { { 9, GGML_TYPE_F16, 128, 128, 2, 1 }, { 2, 2 } },
    { { 9, GGML_TYPE_F16, 128, 128, 3, 1 }, { 2, 2 } },
    { { 9, GGML_TYPE_F16, 192, 192, 0, 1 }, { 2, 2 } },
    { { 9, GGML_TYPE_F16, 192, 192, 1, 1 }, { 2, 2 } },
    { { 9, GGML_TYPE_F16, 192, 192, 2, 1 }, { 2, 2 } },
    { { 9, GGML_TYPE_F16, 192, 192, 3, 1 }, { 2, 2 } },
    { { 9, GGML_TYPE_F16, 192, 128, 1, 1 }, { 2, 2 } },
    { { 9, GGML_TYPE_F16, 192, 128, 2, 1 }, { 2, 2 } },
    { { 9, GGML_TYPE_F16, 192, 128, 3, 1 }, { 2, 2 } },
    { { 9, GGML_TYPE_F16, 256, 256, 0, 1 }, { 2, 2 } },
    { { 9, GGML_TYPE_F16, 256, 256, 1, 0 }, { 1, 2 } },
    { { 9, GGML_TYPE_F16, 256, 256, 1, 1 }, { 2, 2 } },
    { { 9, GGML_TYPE_F16, 256, 256, 2, 0 }, { 1, 2 } },
    { { 9, GGML_TYPE_F16, 256, 256, 2, 1 }, { 2, 2 } },
    { { 9, GGML_TYPE_F16, 256, 256, 3, 0 }, { 1, 2 } },
    { { 9, GGML_TYPE_F16, 256, 256, 3, 1 }, { 2, 2 } },
    { { 9, GGML_TYPE_F16, 320, 256, 0, 1 }, { 2, 2 } },
    { { 9, GGML_TYPE_F16, 320, 256, 1, 1 }, { 2, 2 } },
    { { 9, GGML_TYPE_F16, 320, 256, 2, 1 }, { 2, 2 } },
    { { 9, GGML_TYPE_F16, 320, 256, 3, 1 }, { 2, 2 } },
};

static bool         g_override_set = false;
static fa_vec_cfg_t g_override_cfg = { 1, 4 };

void fa_vec_set_override(fa_vec_cfg_t cfg) {
    g_override_cfg = cfg;
    g_override_set = true;
}

void fa_vec_clear_override() {
    g_override_set = false;
}

bool fa_vec_override_active() {
    return g_override_set;
}

static const fa_vec_cfg_t * find(const fa_vec_entry_t * tbl, size_t n, const fa_vec_key_t & k) {
    for (size_t i = 0; i < n; ++i) {
        if (memcmp(&tbl[i].key, &k, sizeof(k)) == 0) {
            return &tbl[i].cfg;
        }
    }
    return nullptr;
}

fa_vec_cfg_t fa_vec_pick(enum ggml_metal_device_id device_id, int dtype, int dk, int dv, int64_t ne11, int64_t ne01) {
    if (g_override_set) {
        return g_override_cfg;
    }

    const fa_vec_cfg_t kBaseline = fa_vec_baseline_cfg(dk, dv);

    fa_vec_key_t k{};  // value-init so memcmp sees no uninitialized padding
    k.dtype  = (int8_t) dtype;
    k.dk     = (int16_t) dk;
    k.dv     = (int16_t) dv;
    k.ne11_b = (int8_t) fa_vec_ne11_bucket(ne11);
    k.ne01_b = (int8_t) fa_vec_ne01_bucket(ne01);

    k.device_id = (int8_t) device_id;
    if (auto * c = find(fa_vec_tuned_table, std::size(fa_vec_tuned_table), k)) {
        return *c;
    }

    const int family = fa_vec_device_family(device_id);
    if (family >= 9) {
        k.device_id = 9;  // Apple9+ reuse the Apple9 family rows
        if (auto * c = find(fa_vec_family_table, std::size(fa_vec_family_table), k)) {
            return *c;
        }
    }

    return kBaseline;
}

}  // namespace ggml_metal_tuning
