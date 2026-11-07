#pragma once

#include "ggml.h"

static inline bool ggml_token_mask_valid_rows(const struct ggml_tensor * t) {
    return t && ggml_is_matrix(t) && t->ne[0] > 0 && t->ne[0] <= INT32_MAX && t->ne[1] > 0 &&
        t->nb[0] == sizeof(int32_t) && t->nb[1] % sizeof(int32_t) == 0 &&
        (uint64_t) t->ne[0] <= SIZE_MAX/sizeof(int32_t) &&
        t->nb[1] >= (size_t) t->ne[0]*sizeof(int32_t) &&
        (uint64_t) (t->ne[1] - 1) <= (SIZE_MAX - (size_t) t->ne[0]*sizeof(int32_t))/t->nb[1] &&
        t->ne[1] <= INT64_MAX/t->ne[0] &&
        (uint64_t) t->ne[1] <= SIZE_MAX/sizeof(int32_t)/(uint64_t) t->ne[0];
}

static inline bool ggml_token_mask_valid_layout(
        const struct ggml_tensor * logits, const struct ggml_tensor * ids,
        const struct ggml_tensor * payload, const struct ggml_tensor * desc) {
    return logits && payload && desc &&
        logits->type == GGML_TYPE_F32 && ggml_token_mask_valid_rows(logits) &&
        (!ids || (ids->type == GGML_TYPE_I32 && ggml_are_same_shape(logits, ids) &&
            ggml_token_mask_valid_rows(ids))) &&
        payload->type == GGML_TYPE_I32 && ggml_is_vector(payload) && ggml_is_contiguous(payload) &&
        payload->ne[0] >= 0 && (uint64_t) payload->ne[0] <= SIZE_MAX/sizeof(int32_t) &&
        desc->type == GGML_TYPE_I32 && ggml_token_mask_valid_rows(desc) && desc->ne[0] == 4 &&
        desc->ne[1] == logits->ne[1];
}

#if defined(__CUDACC__) || defined(__HIPCC__)
#define GGML_TOKEN_MASK_HD __host__ __device__
#else
#define GGML_TOKEN_MASK_HD
#endif

GGML_TOKEN_MASK_HD static inline bool ggml_token_mask_valid_desc(
        const int32_t * d, int64_t payload_size, int64_t candidates, bool identity) {
    const int32_t mode = d[0], count = d[1], offset = d[2], vocab = d[3];
    if (mode < GGML_TOKEN_MASK_ALL || mode > GGML_TOKEN_MASK_BITSET ||
            count < 0 || offset < 0 || vocab <= 0 || offset > payload_size ||
            count > payload_size - offset || (identity && candidates > vocab)) {
        return false;
    }
    switch (mode) {
        case GGML_TOKEN_MASK_ALL:
        case GGML_TOKEN_MASK_NONE:
            return count == 0;
        case GGML_TOKEN_MASK_ALLOW_IDS:
        case GGML_TOKEN_MASK_DENY_IDS:
            return count <= vocab;
        case GGML_TOKEN_MASK_BITSET:
            return count == ((int64_t) vocab + 31)/32;
        default:
            return false;
    }
}

GGML_TOKEN_MASK_HD static inline bool ggml_token_mask_valid_payload(const int32_t * d, const int32_t * payload) {
    if (d[0] == GGML_TOKEN_MASK_ALLOW_IDS || d[0] == GGML_TOKEN_MASK_DENY_IDS) {
        int32_t previous = -1;
        for (int64_t i = 0; i < d[1]; ++i) {
            const int32_t id = payload[(int64_t) d[2] + i];
            if (id <= previous || id >= d[3]) {
                return false;
            }
            previous = id;
        }
    }
    return true;
}

GGML_TOKEN_MASK_HD static inline bool ggml_token_mask_allowed(
        const int32_t * d, const int32_t * payload, int32_t id) {
    if (id < 0 || id >= d[3]) {
        return false;
    }
    switch (d[0]) {
        case GGML_TOKEN_MASK_ALL:
            return true;
        case GGML_TOKEN_MASK_NONE:
            return false;
        case GGML_TOKEN_MASK_BITSET:
            return ((uint32_t) payload[(int64_t) d[2] + id/32] & (UINT32_C(1) << (id % 32))) != 0;
        default: {
            bool found = false;
            if (d[1] <= 16) {
                for (int32_t i = 0; i < d[1]; ++i) {
                    found = found || payload[(int64_t) d[2] + i] == id;
                }
            } else {
                int64_t lo = 0, hi = d[1];
                while (lo < hi) {
                    const int64_t mid = lo + (hi - lo)/2;
                    if (payload[(int64_t) d[2] + mid] < id) {
                        lo = mid + 1;
                    } else {
                        hi = mid;
                    }
                }
                found = lo < d[1] && payload[(int64_t) d[2] + lo] == id;
            }
            return d[0] == GGML_TOKEN_MASK_ALLOW_IDS ? found : !found;
        }
    }
}

#undef GGML_TOKEN_MASK_HD
