enable f16;

fn get_byte(value: u32, index: u32) -> u32 {
    return (value >> (index * 8)) & 0xFF;
}

fn get_byte_i32(value: u32, index: u32) -> i32 {
    return bitcast<i32>(((value >> (index * 8)) & 0xFF) << 24) >> 24;
}

struct q4_0 {
    d: f16,
    qs: array<f16, 8>
};

@group(0) @binding(0)
var<storage, read_write> src: array<q4_0>;

@group(0) @binding(1)
var<storage, read_write> idx: array<i32>;

@group(0) @binding(2)
var<storage, read_write> dst: array<f32>;

//@group(0) @binding(3)
//var<storage, read_write> debug: array<f32>;

struct Params {
    offset_src: u32, // in elements
    offset_idx: u32, // in elements
    offset_dst: u32, // in elements

    // Strides (in elements)
    stride_src1: u32,
    stride_src2: u32,
    stride_src3: u32,

    stride_idx0: u32,
    stride_idx1: u32,
    stride_idx2: u32,

    stride_dst1: u32,
    stride_dst2: u32,
    stride_dst3: u32,

    // Shape of dst
    ne0: u32,
    n_rows: u32,
    ne2: u32,
    ne3: u32,

    // Shape of idx
    idx1: u32,
    idx2: u32,
};

@group(0) @binding(3)
var<uniform> params: Params;

override wg_size: u32;
@compute @workgroup_size(wg_size)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    //debug[0] = 42.0f;
    if (gid.x >= params.n_rows * params.ne2 * params.ne3) {
        return;
    }
    var i = gid.x;
    let i_dst3 = i / (params.ne2 * params.n_rows);

    i = i % (params.ne2 * params.n_rows);
    let i_dst2 = i / params.n_rows;
    let i_dst1 = i % params.n_rows;

    let i_idx2 = i_dst3 % params.idx2;
    let i_idx1 = i_dst2 % params.idx1;
    let i_idx0 = i_dst1;

    let i_idx = params.offset_idx + i_idx0 * params.stride_idx0 + i_idx1 * params.stride_idx1 + i_idx2 * params.stride_idx2;

    let idx_val = u32(idx[i_idx]);

    let i_src_row = params.offset_src + idx_val * params.stride_src1 + i_dst2 * params.stride_src2 + i_dst3 * params.stride_src3;
    let i_dst_row = params.offset_dst + i_dst1 * params.stride_dst1 + i_dst2 * params.stride_dst2 + i_dst3 * params.stride_dst3;

    for (var i: u32 = 0; i < params.ne0/32; i++) {
        let block_q4_0 = src[i_src_row + i];
        let d = f32(block_q4_0.d);
            let q_packed = bitcast<u32>(vec2(block_q4_0.qs[0], block_q4_0.qs[1]));
            for (var k: u32 = 0; k < 4; k++) {
                let q_byte = get_byte(q_packed, k);
                let q_hi = (f32((q_byte >> 4) & 0xF) - 8.0f) * d;
                let q_lo = (f32(q_byte & 0xF) - 8.0f) * d;
                let dst_offset = i_dst_row + i * 32 + k;
                dst[dst_offset] = q_lo;
                dst[dst_offset + 16] = q_hi;
            }
            let q_packed1 = bitcast<u32>(vec2(block_q4_0.qs[2], block_q4_0.qs[3]));
            for (var k: u32 = 0; k < 4; k++) {
                let q_byte = get_byte(q_packed1, k);
                let q_hi = (f32((q_byte >> 4) & 0xF) - 8.0f) * d;
                let q_lo = (f32(q_byte & 0xF) - 8.0f) * d;
                let dst_offset = i_dst_row + i * 32 + k + 4;
                dst[dst_offset] = q_lo;
                dst[dst_offset + 16] = q_hi;
            }
            let q_packed2 = bitcast<u32>(vec2(block_q4_0.qs[4], block_q4_0.qs[5]));
            for (var k: u32 = 0; k < 4; k++) {
                let q_byte = get_byte(q_packed2, k);
                let q_hi = (f32((q_byte >> 4) & 0xF) - 8.0f) * d;
                let q_lo = (f32(q_byte & 0xF) - 8.0f) * d;
                let dst_offset = i_dst_row + i * 32 + k + 8;
                dst[dst_offset] = q_lo;
                dst[dst_offset + 16] = q_hi;
            }
            let q_packed3 = bitcast<u32>(vec2(block_q4_0.qs[6], block_q4_0.qs[7]));
            for (var k: u32 = 0; k < 4; k++) {
                let q_byte = get_byte(q_packed3, k);
                let q_hi = (f32((q_byte >> 4) & 0xF) - 8.0f) * d;
                let q_lo = (f32(q_byte & 0xF) - 8.0f) * d;
                let dst_offset = i_dst_row + i * 32 + k + 12;
                dst[dst_offset] = q_lo;
                dst[dst_offset + 16] = q_hi;
            }
    }
}