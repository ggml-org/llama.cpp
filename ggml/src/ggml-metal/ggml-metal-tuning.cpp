#include "ggml-metal-tuning.h"

#include <cstddef>
#include <cstring>
#include <iterator>

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

static enum ggml_metal_device_id fa_vec_family_representative(int gpu_family) {
    switch (gpu_family) {
        case 9:  return GGML_METAL_DEVICE_M4_MAX;
        default: return GGML_METAL_DEVICE_GENERIC;
    }
}

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

fa_vec_cfg_t fa_vec_pick(enum ggml_metal_device_id device_id, int gpu_family, int dtype, int dk, int dv, int64_t ne11, int64_t ne01) {
    if (g_override_set) {
        return g_override_cfg;
    }

    const fa_vec_cfg_t kBaseline = fa_vec_baseline_cfg(dk, dv);

    fa_vec_key_t k{};
    k.dtype  = (int8_t) dtype;
    k.dk     = (int16_t) dk;
    k.dv     = (int16_t) dv;
    k.ne11_b = (int8_t) fa_vec_ne11_bucket(ne11);
    k.ne01_b = (int8_t) fa_vec_ne01_bucket(ne01);

    k.device_id = (int8_t) device_id;
    if (auto * c = find(fa_vec_tuned_table, std::size(fa_vec_tuned_table), k)) {
        return *c;
    }

    // family fallback: retry the table under the family's representative SKU; none -> baseline
    if (gpu_family > 0) {
        const enum ggml_metal_device_id rep = fa_vec_family_representative(gpu_family);
        if (rep != GGML_METAL_DEVICE_GENERIC) {
            k.device_id = (int8_t) rep;
            if (auto * c = find(fa_vec_tuned_table, std::size(fa_vec_tuned_table), k)) {
                return *c;
            }
        }
    }

    return kBaseline;
}

}  // namespace ggml_metal_tuning
