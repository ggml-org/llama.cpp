#include "ggml.h"
#include "quantize.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>

// quantize each group of group_size values of a row to int8, plus the rounding error into residuals for ~15 bits
void ggml_zdnn_quantize_row(const float  * row,
                                  int64_t  ne0,
                                  int64_t  group_size,
                            const float  * channel_scales,
                                  int8_t * quants,
                                  int8_t * residuals,
                                  int64_t  quants_stride,
                                  float  * scales,
                                  int64_t  scales_stride) {

    const int64_t n_groups = (ne0 + group_size - 1) / group_size;

    for (int64_t g = 0; g < n_groups; g++) {
        const float * xg = row + g*group_size;
        const float * cg = channel_scales + g*group_size;
        const int64_t n  = std::min(group_size, ne0 - g*group_size);

        // absolute values order the same as their bits, and an integer max vectorizes without -ffast-math
        uint32_t amax_bits = 0;
        for (int64_t i0 = 0; i0 < n; i0++) {
            const float xc = xg[i0] * cg[i0];

            uint32_t bits;
            memcpy(&bits, &xc, sizeof(bits));
            amax_bits = std::max(amax_bits, bits & 0x7fffffffu);
        }

        float amax;
        memcpy(&amax, &amax_bits, sizeof(amax));

        // the rounding error is at most d/2
        const float d      = amax / 127.0f;
        const float id     = d ? 1.0f/d : 0.0f;
        const float id_res = 254.0f * id;
        scales[g*scales_stride] = d;

        int8_t * qg = quants    + g*quants_stride;
        int8_t * rg = residuals + g*quants_stride;

        for (int64_t i0 = 0; i0 < n; i0++) {
            const float xc = xg[i0] * cg[i0];
            const float v  = roundf(xc * id);

            qg[i0] = (int8_t)v;
            rg[i0] = (int8_t)std::clamp(roundf((xc - v * d) * id_res), -127.0f, 127.0f);
        }

        for (int64_t i0 = n; i0 < group_size; i0++) {
            qg[i0] = 0;
            rg[i0] = 0;
        }
    }
}

void ggml_zdnn_quantize_inputs(int32_t       n_threads,
                         const ggml_tensor * inputs,
                               int64_t       group_size,
                         const float       * channel_scales,
                               int8_t      * quants,
                               int8_t      * residuals,
                               float       * scales) {

    const int64_t ne0 = inputs->ne[0];
    const int64_t ne1 = inputs->ne[1];

    ggml_zdnn_parallel_for(n_threads, ne1, [&](int64_t i1_start, int64_t i1_end) {
        for (int64_t i1 = i1_start; i1 < i1_end; i1++) {
            const float * xi = (const float *)((const char *)inputs->data + i1*inputs->nb[1]);

            ggml_zdnn_quantize_row(xi, ne0, group_size, channel_scales,
                                   quants + i1*group_size, residuals + i1*group_size, ne1*group_size,
                                   scales + i1, ne1);
        }
    });
}
