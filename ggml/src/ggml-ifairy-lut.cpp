#include "ggml-ifairy-lut.h"

#include <stddef.h>

// Skeleton implementation for iFairy 3-weight LUT path.
// Functions currently act as no-op/disabled placeholders to keep build wiring intact.

void ggml_ifairy_lut_init(void) {
    // No global initialization needed for the stub.
}

void ggml_ifairy_lut_free(void) {
    // No global teardown needed for the stub.
}

bool ggml_ifairy_lut_can_mul_mat(const struct ggml_tensor * src0, const struct ggml_tensor * src1, const struct ggml_tensor * dst) {
    (void) src0;
    (void) src1;
    (void) dst;

    // Stub: always return false until the full LUT path is implemented.
    return false;
}

size_t ggml_ifairy_lut_get_wsize(const struct ggml_tensor * src0, const struct ggml_tensor * src1, const struct ggml_tensor * dst, int n_threads) {
    (void) src0;
    (void) src1;
    (void) dst;
    (void) n_threads;

    // Stub: no additional workspace required in placeholder implementation.
    return 0;
}

bool ggml_ifairy_lut_transform_tensor(struct ggml_tensor * tensor, struct ggml_tensor ** index_tensor_out) {
    if (index_tensor_out) {
        *index_tensor_out = NULL;
    }

    (void) tensor;

    // Stub: transformation is not yet implemented.
    return false;
}

#if !(defined(GGML_IFAIRY_ARM_LUT) && defined(__ARM_NEON))
// Stub implementations used when no ARM-specific source is compiled.
void ggml_ifairy_lut_preprocess(int m, int k, int n, const void * act, size_t act_stride, void * lut_scales, void * lut_buf) {
    (void) m;
    (void) k;
    (void) n;
    (void) act;
    (void) act_stride;
    (void) lut_scales;
    (void) lut_buf;
    // Stub: no-op.
}

void ggml_ifairy_lut_qgemm(int m, int k, int n, const void * qweights, const uint8_t * indexes, const void * lut, const void * lut_scales, float * dst, size_t dst_stride) {
    (void) m;
    (void) k;
    (void) n;
    (void) qweights;
    (void) indexes;
    (void) lut;
    (void) lut_scales;
    (void) dst;
    (void) dst_stride;
    // Stub: no-op.
}
#endif
