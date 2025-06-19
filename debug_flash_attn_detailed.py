#!/usr/bin/env python3
import re

# Read the file
with open('ggml/src/ggml-cpu/ops.cpp', 'r') as f:
    content = f.read()

# Find the line where we restore previous results and add debug output
debug_lines = '''        // Initialize VKQ accumulator - CRITICAL FIX: restore previous accumulated results
        if (v->type == GGML_TYPE_F16) {
            if (is_continuation) {
                // Load previous accumulated result from dst tensor and scale by previous sum S
                const int i1 = iq1;
                const int i2 = iq2;
                const int i3 = iq3;
                float * prev_result = (float *) ((char *) dst->data + (i3*ne2*ne1 + i2 + i1*ne1)*nb1);
                
                printf("[DEBUG] Continuation detected for head %d, pos %d: M=%.6f, S=%.6f\\n", iq2, iq1, M, S);
                printf("[DEBUG] Previous result first 4 values: %.6f %.6f %.6f %.6f\\n", 
                       prev_result[0], prev_result[1], prev_result[2], prev_result[3]);
                
                // Scale previous result by S and convert to FP16
                for (int64_t d = 0; d < DV; ++d) {
                    VKQ16[d] = GGML_FP32_TO_FP16(prev_result[d] * S);
                }
                
                printf("[DEBUG] Restored VKQ first 4 values: %.6f %.6f %.6f %.6f\\n", 
                       GGML_FP16_TO_FP32(VKQ16[0]), GGML_FP16_TO_FP32(VKQ16[1]), 
                       GGML_FP16_TO_FP32(VKQ16[2]), GGML_FP16_TO_FP32(VKQ16[3]));
            } else {
                printf("[DEBUG] First segment for head %d, pos %d: initializing to zero\\n", iq2, iq1);
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
                
                printf("[DEBUG] Continuation detected for head %d, pos %d: M=%.6f, S=%.6f\\n", iq2, iq1, M, S);
                printf("[DEBUG] Previous result first 4 values: %.6f %.6f %.6f %.6f\\n", 
                       prev_result[0], prev_result[1], prev_result[2], prev_result[3]);
                
                // Scale previous result by S
                for (int64_t d = 0; d < DV; ++d) {
                    VKQ32[d] = prev_result[d] * S;
                }
                
                printf("[DEBUG] Restored VKQ first 4 values: %.6f %.6f %.6f %.6f\\n", 
                       VKQ32[0], VKQ32[1], VKQ32[2], VKQ32[3]);
            } else {
                printf("[DEBUG] First segment for head %d, pos %d: initializing to zero\\n", iq2, iq1);
                memset(VKQ32, 0, DV*sizeof(float));
                S = 0.0f;
                M = -INFINITY;
            }
        }'''

old_debug_lines = '''        // Initialize VKQ accumulator - CRITICAL FIX: restore previous accumulated results
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
if old_debug_lines in content:
    content = content.replace(old_debug_lines, debug_lines)
    print('Debug output added successfully!')
else:
    print('Old code pattern not found for debug output.')

# Write back to file
with open('ggml/src/ggml-cpu/ops.cpp', 'w') as f:
    f.write(content)