enable f16;

@group(0) @binding(0)
var<storage, read_write> src0: array<f32>;

@group(0) @binding(1)
var<storage, read_write> src1: array<f32>;

@group(0) @binding(2)
var<storage, read_write> dst: array<f32>;

struct Params {
    ne: u32,

    stride_src0_0: u32,
    stride_src0_1: u32,
    stride_src0_2: u32,
    stride_src0_3: u32,

    stride_src1_0: u32,
    stride_src1_1: u32,
    stride_src1_2: u32,
    stride_src1_3: u32,

    stride_dst_0: u32,
    stride_dst_1: u32,
    stride_dst_2: u32,
    stride_dst_3: u32,

    a_ne0: u32,
    a_ne1: u32,
    a_ne2: u32,
    a_ne3: u32,

    b_ne0: u32,
    b_ne1: u32,
    b_ne2: u32,
    b_ne3: u32,

    // offsets in elements
    offset_src0: u32,
    offset_src1: u32,
    offset_dst: u32,
};

@group(0) @binding(3)
var<uniform> params: Params;

override wg_size: u32;
@compute @workgroup_size(wg_size)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.ne) {
        return;
    }

    // i = thread id, ranges from 0 --> total ne - 1 
    // represents the position in the flat array a we are adding with array b
    var i = gid.x;  

    // given the index of linear a, we want to compute the 4d index [a_i0, a_i1, a_i2, a_i3]
    // we need this because tensor a and b are different shapes 
    // so the same linear index won't work for b, and we can only compute b's linear index from the 4d index of a
 
    let a_i3 = i / (params.a_ne2 * params.a_ne1 * params.a_ne0);
    i = i % (params.a_ne2 * params.a_ne1 * params.a_ne0);

    let a_i2 = i / (params.a_ne1 * params.a_ne0);
    i = i % (params.a_ne1 * params.a_ne0);

    let a_i1 = i / params.a_ne0;

    let a_i0 = i % params.a_ne0;


    // handle repetition of b 
        // index loops back to the beginning and repeats after elements are exhausted = modulo
    let b_i0 = a_i0 % params.b_ne0;
    let b_i1 = a_i1 % params.b_ne1;
    let b_i2 = a_i2 % params.b_ne2;
    let b_i3 = a_i3 % params.b_ne3;


    // compute index for position in b's flat array
    let src1_idx = b_i0 * params.stride_src1_0 +
                b_i1 * params.stride_src1_1 +
                b_i2 * params.stride_src1_2 +
                b_i3 * params.stride_src1_3;

    // actual addition operation, now that the indexes are all figured out
    // ensuring that the offsets are included
    // gid.x used for flat indexing into dst and a, since variable i was modified during calcs
    dst[params.offset_dst + gid.x] = src0[params.offset_src0 + gid.x] + src1[params.offset_src1 + src1_idx];
}
