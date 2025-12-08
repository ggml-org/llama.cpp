#include "ggml-ifairy-lut.h"

// Placeholder for ARM-specific implementations of iFairy 3-weight LUT.
// Real NEON kernels will replace these stubs in subsequent steps.

extern "C" {

void ggml_ifairy_lut_preprocess(int m, int k, int n, const void * act, size_t act_stride, void * lut_scales, void * lut_buf) {
    (void) m;
    (void) k;
    (void) n;
    (void) act;
    (void) act_stride;
    (void) lut_scales;
    (void) lut_buf;
    // ARM NEON implementation TODO
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
    // ARM NEON implementation TODO
}

} // extern "C"

