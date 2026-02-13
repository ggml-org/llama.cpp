enable f16;

struct Params {
    ne: u32,

    // offsets in elements
    offset_src0: u32,
    offset_src1: u32,
    offset_dst: u32,

    stride_src1_0: u32,
    stride_src1_1: u32,
    stride_src1_2: u32,
    stride_src1_3: u32,

    a_ne0: u32,
    a_ne1: u32,
    a_ne2: u32,

    b_ne0: u32,
    b_ne1: u32,
    b_ne2: u32,
    b_ne3: u32,
};

fn src1_index(_i: u32) -> u32 {
    var i = _i;
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
    return b_i0 * params.stride_src1_0 +
           b_i1 * params.stride_src1_1 +
           b_i2 * params.stride_src1_2 +
           b_i3 * params.stride_src1_3;
}

#ifdef TYPE_F32
#define DataType f32
#endif
#ifdef TYPE_F16
#define DataType f16
#endif

@group(0) @binding(0)
var<storage, read_write> src0: array<DataType>;

@group(0) @binding(1)
var<storage, read_write> src1 : array<DataType>;

#ifdef INPLACE
@group(0) @binding(2)
var<uniform> params: Params;

#elif defined(OVERLAP)
@group(0) @binding(2)
var<uniform> params: Params;

#else
@group(0) @binding(2)
var<storage, read_write> dst: array<DataType>;

@group(0) @binding(3)
var<uniform> params: Params;
#endif

fn op(a: DataType, b: DataType) -> DataType {
#ifdef OP_ADD
    return a + b;
#elif defined(OP_SUB)
    return a - b;
#elif defined(OP_MUL)
    return a * b;
#elif defined(OP_DIV)
    return a / b;
#endif
}

fn update(dst_i: u32, src0_i: u32, src1_i: u32){
    let result = op(src0[src0_i], src1[src1_i]);

#ifdef INPLACE
    src0[dst_i] = result;
#elif defined(OVERLAP)
    src1[dst_i] = result;
#else
    dst[dst_i] = result;
#endif
}

@compute @workgroup_size(WG_SIZE)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x < params.ne) {
        update(params.offset_dst + gid.x, params.offset_src0 + gid.x, params.offset_src1 + src1_index(gid.x));
    }
}
