#ifdef DST_Q8_0
#define BLOCK_SIZE 32u
#define BLOCK_BYTES 34u
#define QS_WORDS 8u
#elif defined(DST_Q4_0)
#define BLOCK_SIZE 32u
#define BLOCK_BYTES 18u
#define QS_WORDS 4u
#endif

@group(0) @binding(0)
var<storage, read_write> src: array<f32>;

@group(0) @binding(1)
var<storage, read_write> idx: array<u32>;

@group(0) @binding(2)
#ifdef PAIR_BLOCKS
var<storage, read_write> dst: array<u32>;
#else
var<storage, read_write> dst: array<atomic<u32>>;
#endif

#ifdef I64_IDX
@group(0) @binding(3)
var<storage, read_write> error: atomic<u32>;
#define PARAMS_BINDING 4
#else
#define PARAMS_BINDING 3
#endif

struct Params {
    offset_src: u32, // in elements
    offset_idx: u32, // in elements
    offset_dst: u32, // in blocks

    // Strides (in elements / blocks)
    stride_src1: u32,
    stride_src2: u32,
    stride_src3: u32,

    stride_idx0: u32,
    stride_idx1: u32,
    stride_idx2: u32,

    stride_dst1: u32,
    stride_dst2: u32,
    stride_dst3: u32,

    // Shape of src
    ne0: u32,
    n_rows: u32,
    ne2: u32,
    ne3: u32,

    // Shape of idx
    idx1: u32,
    idx2: u32,
};

@group(0) @binding(PARAMS_BINDING)
var<uniform> params: Params;

// if the quantization type is unaligned and there are an odd number of blocks per row, we need to store atomically
#ifndef PAIR_BLOCKS
fn merge_store_dst_word(word_idx: u32, mask: u32, bits: u32) {
    loop {
        let old = atomicLoad(&dst[word_idx]);
        let merged = (old & ~mask) | (bits & mask);
        let result = atomicCompareExchangeWeak(&dst[word_idx], old, merged);
        if (result.exchanged) {
            return;
        }
    }
}
#else
fn merge_store_dst_word(word_idx: u32, mask: u32, bits: u32) {
    let old = dst[word_idx];
    dst[word_idx] = (old & ~mask) | (bits & mask);
}
#endif

fn store_u16(dst_word_idx: u32, block_byte_offset: u32, byte_offset: u32, value: u32) {
    let total_byte_offset = block_byte_offset + byte_offset;
    let word_idx = dst_word_idx + total_byte_offset / 4u;
    let shift = (total_byte_offset & 2u) * 8u;
    let mask = 0xFFFFu << shift;
    merge_store_dst_word(word_idx, mask, (value & 0xFFFFu) << shift);
}

fn store_u32(dst_word_idx: u32, block_byte_offset: u32, byte_offset: u32, value: u32) {
    let total_byte_offset = block_byte_offset + byte_offset;
    let word_idx = dst_word_idx + total_byte_offset / 4u;
    let shift = (total_byte_offset & 3u) * 8u;

    if (shift == 0u) {
#ifdef PAIR_BLOCKS
        dst[word_idx] = value;
#else
        atomicStore(&dst[word_idx], value);
#endif
        return;
    }

    let lo_mask = 0xFFFFFFFFu << shift;
    let hi_mask = (1u << shift) - 1u;
    merge_store_dst_word(word_idx, lo_mask, value << shift);
    merge_store_dst_word(word_idx + 1u, hi_mask, value >> (32u - shift));
}

fn quantize_block_params(src_block: u32) -> vec2<f32> {
#ifdef DST_Q8_0
    var amax = 0.0;
    for (var j: u32 = 0u; j < BLOCK_SIZE; j++) {
        amax = max(amax, abs(src[src_block + j]));
    }

    let d = amax / 127.0;
    let id = select(0.0, 1.0 / d, d > 0.0);
    return vec2(d, id);
#elif defined(DST_Q4_0)
    var amax = 0.0;
    var max_val = 0.0;
    for (var j: u32 = 0u; j < BLOCK_SIZE; j++) {
        let v = src[src_block + j];
        let av = abs(v);
        if (amax < av) {
            amax = av;
            max_val = v;
        }
    }

    let d = max_val / -8.0;
    let id = select(0.0, 1.0 / d, d != 0.0);
    return vec2(d, id);
#endif
}

fn quantize_block_word(src_block: u32, j: u32, id: f32) -> u32 {
#ifdef DST_Q8_0
    let base = src_block + j * 4u;
    return (u32(i32(round(src[base + 0u] * id)) & 0xFF) << 0u) |
           (u32(i32(round(src[base + 1u] * id)) & 0xFF) << 8u) |
           (u32(i32(round(src[base + 2u] * id)) & 0xFF) << 16u) |
           (u32(i32(round(src[base + 3u] * id)) & 0xFF) << 24u);
#elif defined(DST_Q4_0)
    var packed_q = 0u;
    for (var k: u32 = 0u; k < 4u; k++) {
        let x0 = src[src_block + j * 4u + k] * id;
        let x1 = src[src_block + 16u + j * 4u + k] * id;
        let q0 = u32(clamp(i32(x0 + 8.5), 0, 15));
        let q1 = u32(clamp(i32(x1 + 8.5), 0, 15));
        packed_q |= (q0 & 0xFu) << (8u * k);
        packed_q |= (q1 & 0xFu) << (8u * k + 4u);
    }
    return packed_q;
#endif
}

fn quantize_block(src_block: u32, dst_word_idx: u32, block_byte_offset: u32) {
    let params = quantize_block_params(src_block);
    let d = params.x;
    let id = params.y;
    let packed_d = pack2x16float(vec2(d, 0.0)) & 0xFFFFu;
    store_u16(dst_word_idx, block_byte_offset, 0u, packed_d);

    for (var j: u32 = 0u; j < QS_WORDS; j++) {
        store_u32(dst_word_idx, block_byte_offset, 2u + j * 4u, quantize_block_word(src_block, j, id));
    }
}

@compute @workgroup_size(WG_SIZE)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let blocks_per_row = params.ne0 / BLOCK_SIZE;
#ifdef PAIR_BLOCKS
    let blocks_per_invocation = 2u;
#else
    let blocks_per_invocation = 1u;
#endif
    let invocations_per_row = blocks_per_row / blocks_per_invocation;
    let total_invocations = params.ne3 * params.ne2 * params.n_rows * invocations_per_row;
    if (gid.x >= total_invocations) {
        return;
    }

    var i = gid.x / invocations_per_row;
    let block_in_row = (gid.x % invocations_per_row) * blocks_per_invocation;

    let i_src3 = i / (params.ne2 * params.n_rows);
    i = i % (params.ne2 * params.n_rows);
    let i_src2 = i / params.n_rows;
    let i_src1 = i % params.n_rows;

    let i_idx2 = i_src3 % params.idx2;
    let i_idx1 = i_src2 % params.idx1;
    let i_idx0 = i_src1;

#ifdef I64_IDX
    let idx_high = (params.offset_idx + i_idx0 * params.stride_idx0 + i_idx1 * params.stride_idx1 + i_idx2 * params.stride_idx2) * 2u;
    let idx_val = idx[idx_high];
    let idx_low_val = idx[idx_high + 1u];

    if (idx_low_val != 0u) {
        atomicStore(&error, 1u);
        return;
    }
#else
    let idx_i = params.offset_idx + i_idx0 * params.stride_idx0 + i_idx1 * params.stride_idx1 + i_idx2 * params.stride_idx2;
    let idx_val = idx[idx_i];
#endif

    let dst_row_blocks = params.offset_dst + idx_val * params.stride_dst1 + i_src2 * params.stride_dst2 + i_src3 * params.stride_dst3;
    let src_row = params.offset_src + i_src1 * params.stride_src1 + i_src2 * params.stride_src2 + i_src3 * params.stride_src3;
    let src_block = src_row + block_in_row * BLOCK_SIZE;
    let dst_block_byte = (dst_row_blocks + block_in_row) * BLOCK_BYTES;

    let dst_word_idx = dst_block_byte / 4u;
#ifdef PAIR_BLOCKS
    quantize_block(src_block, dst_word_idx, 0u);
    quantize_block(src_block + BLOCK_SIZE, dst_word_idx, BLOCK_BYTES);
#else
    quantize_block(src_block, dst_word_idx, dst_block_byte & 3u);
#endif
}
