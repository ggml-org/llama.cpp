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

#if 0
// Reference algo used in hvx-utils
static void fast_sigmoid_f32(const float*  restrict src, float* restrict dst, const int num_elems)
{
    const float c1 = 0.03138777;
    const float c2 = 0.276281267;
    const float c_log2f = 1.442695022;

    int32_t store_ints[32];
    float store_floats[3][32];

    for (int i = 0; i < num_elems; i++)
    {
        float v = src0[i];

        v *= c_log2f*0.5;
        int intPart = (int)v;
        float x = (v - intPart);
        float xx = x * x;
        float v1 = c_log2f + c2 * xx;
        float v2 = x + xx * c1 * x;
        float v3 = (v2 + v1);
        *((int*)&v3) += intPart << 24;
        float v4 = v2 - v1;
        float v5 = v3 - v4;
        float res = v3 / v5;

        dst[i] = res;
    }
}
#endif
