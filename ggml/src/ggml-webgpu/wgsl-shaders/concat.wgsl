struct Params {
    ne: u32,

    offset_src0: u32,
    offset_src1: u32,
    offset_dst: u32,

    stride_src0_0: u32,
    stride_src0_1: u32,
    stride_src0_2: u32,
    stride_src0_3: u32,

    stride_src1_0: u32,
    stride_src1_1: u32,
    stride_src1_2: u32,
    stride_src1_3: u32,

    ne0: u32,
    ne1: u32,
    ne2: u32,
    ne3: u32,

    dim: u32,
    src0_nedim: u32
};

#ifdef TYPE_F32
#define DataType f32
#endif
#ifdef TYPE_I32
#define DataType i32
#endif

@group(0) @binding(0)
var<storage, read_write> src0: array<DataType>;

@group(0) @binding(1)
var<storage, read_write> src1 : array<DataType>;

@group(0) @binding(2)
var<storage, read_write> dst: array<DataType>;

@group(0) @binding(3)
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

        var ni = array<u32, 4>(i0, i1, i2, i3);

        if (ni[params.dim] < params.src0_nedim) {
            let src_i = ni[0] * params.stride_src0_0 +
                             ni[1] * params.stride_src0_1 +
                             ni[2] * params.stride_src0_2 +
                             ni[3] * params.stride_src0_3;
            dst[params.offset_dst + gid.x] = src0[params.offset_src0 + src_i];
        } else {
            ni[params.dim] -= params.src0_nedim;
            let src_i = ni[0] * params.stride_src1_0 +
                             ni[1] * params.stride_src1_1 +
                             ni[2] * params.stride_src1_2 +
                             ni[3] * params.stride_src1_3;
            dst[params.offset_dst + gid.x] = src1[params.offset_src1 + src_i];
        }
    }
}
