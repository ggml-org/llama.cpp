#pragma clang diagnostic ignored "-Wunused-variable"
#pragma clang diagnostic ignored "-Wunused-function"
#pragma clang diagnostic ignored "-Wunused-but-set-variable"

#include <hexagon_protos.h>
#include <hexagon_types.h>
#include <math.h>
#include <string.h>

#define GGML_COMMON_DECL_C
#include "ggml-common.h"
#include "htp-ctx.h"
#include "htp-dma.h"
#include "htp-msg.h"
#include "htp-ops.h"
#include "hvx-utils.h"
#include "ops-utils.h"

void hvx_inverse_f32(const uint8_t * restrict src, uint8_t * restrict dst, const int num_elems) {
    int left_over       = num_elems & (VLEN_FP32 - 1);
    int num_elems_whole = num_elems - left_over;

    int unaligned_addr = 0;
    int unaligned_loop = 0;
    if ((0 == htp_is_aligned((void *) src, VLEN)) || (0 == htp_is_aligned((void *) dst, VLEN))) {
        FARF(HIGH, "hvx_inverse_f32: unaligned address in hvx op, possibly slower execution\n");
        unaligned_addr = 1;
    }
    // assert((0 == unaligned_addr) || (0 == num_elems_whole));
    if ((1 == unaligned_addr) && (num_elems_whole != 0)) {
        unaligned_loop = 1;
        FARF(HIGH, "hvx_inverse_f32: unaligned loop in hvx op, possibly slower execution\n");
    }

    if (0 == unaligned_loop) {
        HVX_Vector * p_vec_in  = (HVX_Vector *) src;
        HVX_Vector * p_vec_out = (HVX_Vector *) dst;

        #pragma unroll(4)
        for (int i = 0; i < num_elems_whole; i += VLEN_FP32) {
            *p_vec_out++ = hvx_vec_inverse_fp32(*p_vec_in++);
        }
    } else {
        #pragma unroll(4)
        for (int i = 0; i < num_elems_whole; i += VLEN_FP32) {
            HVX_Vector in                            = *(HVX_UVector *) (src + i * SIZEOF_FP32);
            *(HVX_UVector *) (dst + i * SIZEOF_FP32) = hvx_vec_inverse_fp32(in);
        }
    }

    if (left_over > 0) {
        const float * srcf = (float *) src + num_elems_whole;
        float *       dstf = (float *) dst + num_elems_whole;

        HVX_Vector in  = *(HVX_UVector *) srcf;
        HVX_Vector out = hvx_vec_inverse_fp32(in);

        hvx_vec_store_u((void *) dstf, left_over * SIZEOF_FP32, out);
    }
}
