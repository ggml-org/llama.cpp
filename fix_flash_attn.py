#!/usr/bin/env python3
import re

# Read the file
with open('ggml/src/ggml-cpu/ops.cpp', 'r') as f:
    content = f.read()

# Define the old code to replace
old_code = '''        // If this is the first call (indicated by M == -INFINITY), initialize properly
        if (M == -INFINITY) {
            S = 0.0f;
        }

        float       * VKQ32 = (float       *) params->wdata + ith*(1*DK + 2*DV + CACHE_LINE_SIZE_F32); // FP32 VKQ accumulator
        float       * V32   =                 (VKQ32 + 1*DV); // (temporary) FP32 V buffer
        ggml_fp16_t * VKQ16 = (ggml_fp16_t *) (VKQ32 + 1*DV); // (temporary) FP16 VKQ accumulator
        ggml_fp16_t * Q_q   = (ggml_fp16_t *) (VKQ32 + 2*DV); // (temporary) buffer for Q converted to quantized/FP16

        if (v->type == GGML_TYPE_F16) {
            memset(VKQ16, 0, DV*sizeof(ggml_fp16_t));
        } else {
            memset(VKQ32, 0, DV*sizeof(float));
        }'''

# Define the new code
new_code = '''        // Check if this is a continuation of previous segments
        bool is_continuation = (M != -INFINITY && S > 0.0f);

        float       * VKQ32 = (float       *) params->wdata + ith*(1*DK + 2*DV + CACHE_LINE_SIZE_F32); // FP32 VKQ accumulator
        float       * V32   =                 (VKQ32 + 1*DV); // (temporary) FP32 V buffer
        ggml_fp16_t * VKQ16 = (ggml_fp16_t *) (VKQ32 + 1*DV); // (temporary) FP16 VKQ accumulator
        ggml_fp16_t * Q_q   = (ggml_fp16_t *) (VKQ32 + 2*DV); // (temporary) buffer for Q converted to quantized/FP16

        // Initialize VKQ accumulator - CRITICAL FIX: restore previous accumulated results
        if (v->type == GGML_TYPE_F16) {
            if (is_continuation) {
                // Load previous accumulated result from dst tensor and scale by previous sum S
                const int i1 = iq1;
                const int i2 = iq2;
                const int i3 = iq3;
                float * prev_result = (float *) ((char *) dst->data + (i3*ne2*ne1 + i2 + i1*ne1)*nb1);
                
                // Scale previous result by S and convert to FP16
                for (int64_t d = 0; d < DV; ++d) {
                    VKQ16[d] = GGML_FP32_TO_FP16(prev_result[d] * S);
                }
            } else {
                memset(VKQ16, 0, DV*sizeof(ggml_fp16_t));
                S = 0.0f;
                M = -INFINITY;
            }
        } else {
            if (is_continuation) {
                // Load previous accumulated result from dst tensor and scale by previous sum S
                const int i1 = iq1;
                const int i2 = iq2;
                const int i3 = iq3;
                float * prev_result = (float *) ((char *) dst->data + (i3*ne2*ne1 + i2 + i1*ne1)*nb1);
                
                // Scale previous result by S
                for (int64_t d = 0; d < DV; ++d) {
                    VKQ32[d] = prev_result[d] * S;
                }
            } else {
                memset(VKQ32, 0, DV*sizeof(float));
                S = 0.0f;
                M = -INFINITY;
            }
        }'''

# Replace the code
if old_code in content:
    content = content.replace(old_code, new_code)
    print('Flash attention state fix applied successfully!')
else:
    print('Old code pattern not found. Checking for alternative patterns...')
    # Try to find the memset lines
    if 'memset(VKQ16, 0, DV*sizeof(ggml_fp16_t));' in content and 'memset(VKQ32, 0, DV*sizeof(float));' in content:
        print('Found memset patterns, but full context doesn\'t match.')
        print('Manual fix needed.')

# Write back to file
with open('ggml/src/ggml-cpu/ops.cpp', 'w') as f:
    f.write(content)