/*
 * ggml-qfx16-metal.metal — QFX16 GPU Inference Kernel
 * Author: Derek Hinch | QomputeAI 2026
 *
 * Each threadgroup processes one row (QFX16 blocks → dot product).
 * The 256KB BF16→Float LUT is stored in device memory (cached by GPU L2).
 * Each thread processes a segment of the delta stream sequentially
 * (required due to prefix-sum dependency), then reduces via threadgroup.
 */
#include <metal_stdlib>
using namespace metal;

/* Block structure must match C definition */
struct block_qfx16 {
    uint8_t  anchor;
    uint16_t stream_len;
    uint8_t  delta_stream[512];
};

/* Kernel: dot product of one QFX16 block against float input segment */
kernel void ggml_qfx16_dot_metal(
    device const block_qfx16 *weights [[buffer(0)]],
    device const float       *input   [[buffer(1)]],
    device float             *output  [[buffer(2)]],
    device const float       *bf16_lut [[buffer(3)]],  /* 256KB LUT in device mem */
    constant uint            &n_blocks [[buffer(4)]],
    constant uint            &block_size [[buffer(5)]],
    uint tid [[thread_position_in_grid]],
    uint tgid [[threadgroup_position_in_grid]]
) {
    /* Each thread processes one block */
    if (tid >= n_blocks) return;

    device const block_qfx16 *blk = &weights[tid];
    uint input_offset = tid * block_size;

    float dot = 0.0f;
    uint8_t state = blk->anchor;
    uint sp = 0;
    uint wi = 0;
    int half = 1;  /* anchor is first high byte */
    uint8_t high = state;

    while (sp < blk->stream_len && wi < block_size) {
        uint8_t byte_val = blk->delta_stream[sp++];

        if (byte_val == 0xFF && sp < blk->stream_len) {
            uint8_t count = blk->delta_stream[sp++];
            if (count == 0) {
                /* Literal 0xFF */
                state = (state + 0xFF) & 0xFF;
                if (half == 0) { high = state; half = 1; }
                else {
                    uint16_t word = ((uint16_t)high << 8) | state;
                    dot += bf16_lut[word] * input[input_offset + wi];
                    wi++;
                    half = 0;
                }
            } else {
                /* Zero run */
                uint16_t word = ((uint16_t)high << 8) | state;
                float w_val = bf16_lut[word];
                for (uint8_t r = 0; r < count && wi < block_size; r++) {
                    if (half == 0) { high = state; half = 1; }
                    else {
                        dot += w_val * input[input_offset + wi];
                        wi++;
                        half = 0;
                    }
                }
            }
        } else {
            state = (state + byte_val) & 0xFF;
            if (half == 0) { high = state; half = 1; }
            else {
                uint16_t word = ((uint16_t)high << 8) | state;
                dot += bf16_lut[word] * input[input_offset + wi];
                wi++;
                half = 0;
            }
        }
    }

    output[tid] = dot;
}

/* Reduction kernel: sum partial block results for a row */
kernel void ggml_qfx16_reduce(
    device const float *partials [[buffer(0)]],
    device float       *result   [[buffer(1)]],
    constant uint      &n_partials [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid != 0) return;
    float sum = 0.0f;
    for (uint i = 0; i < n_partials; i++) {
        sum += partials[i];
    }
    result[0] = sum;
}
