enable f16;

struct Params {
    ne: u32,

    offset_src0: u32,
    offset_dst: u32,

    stride_src0_0: u32,
    stride_src0_1: u32,
    stride_src0_2: u32,
    stride_src0_3: u32,

    a_ne0: u32,
    a_ne1: u32,
    a_ne2: u32,
    a_ne3: u32,

    ne0: u32,
    ne1: u32,
    ne2: u32,
};

#ifdef TYPE_F32
#define DataType f32
#endif
#ifdef TYPE_I32
#define DataType i32
#endif
#ifdef TYPE_I16
// same size (16-bit) is sufficient for repeat
#define DataType f16
#endif

@group(0) @binding(0)
var<storage, read_write> src0: array<DataType>;

@group(0) @binding(1)
var<storage, read_write> dst: array<DataType>;

@group(0) @binding(2)
var<uniform> params: Params;

@compute @workgroup_size(WG_SIZE)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x < params.ne) {
        var i = gid.x;
        let i3 = i / (params.ne2 * params.ne1 * params.ne0);
        i = i % (params.ne2 * params.ne1 * params.ne0);
        let i2 = i / (params.ne1 * params.ne0);
        i = i % (params.ne1 * params.ne0);
        let i1 = i / params.ne0;
        let i0 = i % params.ne0;

        let a_i0 = i0 % params.a_ne0;
        let a_i1 = i1 % params.a_ne1;
        let a_i2 = i2 % params.a_ne2;
        let a_i3 = i3 % params.a_ne3;

        let a_index = a_i0 * params.stride_src0_0 +
                           a_i1 * params.stride_src0_1 +
                           a_i2 * params.stride_src0_2 +
                           a_i3 * params.stride_src0_3;

        dst[params.offset_dst + gid.x] = src0[params.offset_src0 + a_index];
    }
}
