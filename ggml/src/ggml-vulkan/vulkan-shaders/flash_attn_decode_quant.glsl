// q8_0/q4_0 block unpack for the decode kernels (K in phase 1, V in phase 2). Strides arrive as
// block strides, so callers scale by KV_BLOCK_SIZE to get flat element offsets.

#define KV_BLOCK_SIZE 32

#if defined(KV_Q8_0)
struct kv_block_t {
    float16_t d;
    int16_t qs[16];
};
#else
struct kv_block_t {
    float16_t d;
    uint16_t qs[8];
};
#endif

layout (binding = 1) readonly buffer KV_QUANT_BLOCKS {kv_block_t kvBlocks[];};

vec4 dequant_kv_4(uint flatElem) {
    const uint ib  = flatElem / KV_BLOCK_SIZE;
    const uint iqs = flatElem % KV_BLOCK_SIZE;
#if defined(KV_Q8_0)
    const i8vec2 v0 = unpack8(int32_t(kvBlocks[ib].qs[iqs / 2    ])).xy;
    const i8vec2 v1 = unpack8(int32_t(kvBlocks[ib].qs[iqs / 2 + 1])).xy;
    return float(kvBlocks[ib].d) * vec4(v0.x, v0.y, v1.x, v1.y);
#else
    uint vui_lo = uint(kvBlocks[ib].qs[(iqs & 0xF) / 2 + 0]);
    uint vui_hi = uint(kvBlocks[ib].qs[(iqs & 0xF) / 2 + 1]);
    const uint shift = (iqs & 0x10) >> 2;
    vui_lo >>= shift;
    vui_hi >>= shift;
    const vec4 nibbles = vec4(vui_lo & 0xF, (vui_lo >> 8) & 0xF, vui_hi & 0xF, (vui_hi >> 8) & 0xF);
    return float(kvBlocks[ib].d) * (nibbles - vec4(8.0f));
#endif
}
