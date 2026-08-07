struct Params {
    offset_src0: u32,
    offset_src1: u32,
    offset_ids: u32,
    offset_dst: u32,

    nb01: u32,
    nb02: u32,
    nb11: u32,
    nb20: u32,
    nb21: u32,

    ne0: u32,
    ne1: u32,
    ne2: u32,
};

@group(0) @binding(0) var<storage, read_write> src0: array<f32>; // [n_embd, n_experts_used, n_token]
@group(0) @binding(1) var<storage, read_write> src1: array<f32>; // [n_embd, n_experts]
@group(0) @binding(2) var<storage, read_write> ids:  array<i32>; // [n_experts_used, n_token]

#ifdef INPLACE

@group(0) @binding(3)
var<uniform> params: Params;

#else

@group(0) @binding(3)
var<storage, read_write> dst: array<f32>;

@group(0) @binding(4)
var<uniform> params: Params;

#endif

@compute @workgroup_size(WG_SIZE)
fn main(@builtin(workgroup_id) wg_id: vec3<u32>,
        @builtin(num_workgroups) num_wg: vec3<u32>,
        @builtin(local_invocation_id) local_id: vec3<u32>) {

    let wg_linear = wg_id.x + wg_id.y * num_wg.x;

    if (wg_linear < params.ne1 * params.ne2) {
        let thread_id = local_id.x;
        let i2 = wg_linear / params.ne1;
        let i1 = wg_linear % params.ne1;

        let i11 = u32(ids[params.offset_ids + i1 * params.nb20 + i2 * params.nb21]);

        let src0_row = params.offset_src0 + i1 * params.nb01 + i2 * params.nb02;
        let src1_row = params.offset_src1 + i11 * params.nb11;
        let dst_row = params.offset_dst + i1 * params.ne0 + i2 * (params.ne0 * params.ne1);

        for (var i = thread_id;i < params.ne0; i += WG_SIZE) {
#ifdef INPLACE
            src0[src0_row + i] = src0[src0_row + i] + src1[src1_row + i];
#else
            dst[dst_row + i] = src0[src0_row + i] + src1[src1_row + i];
#endif
        }
    }

}
