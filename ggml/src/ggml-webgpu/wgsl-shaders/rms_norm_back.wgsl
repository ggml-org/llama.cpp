// grad of rms_norm, src0 is the gradient and src1 is the forward input

struct Params {
    ne0: u32,

    offset_grad: u32,
    offset_x: u32,
    offset_dst: u32,

    eps: f32,
};

@group(0) @binding(0)
var<storage, read_write> grad: array<f32>;

@group(0) @binding(1)
var<storage, read_write> x: array<f32>;

#ifdef INPLACE
#define PARAMS_BINDING 2
#else
#ifdef OVERLAP
#define PARAMS_BINDING 2
#else
@group(0) @binding(2)
var<storage, read_write> dst_buf: array<f32>;
#define PARAMS_BINDING 3
#endif
#endif

@group(0) @binding(PARAMS_BINDING)
var<uniform> params: Params;

// dst can share a binding with one of the sources, WebGPU forbids aliasing writable bindings
#ifdef INPLACE
fn store_dst(i: u32, val: f32) {
    grad[i] = val;
}
#else
#ifdef OVERLAP
fn store_dst(i: u32, val: f32) {
    x[i] = val;
}
#else
fn store_dst(i: u32, val: f32) {
    dst_buf[i] = val;
}
#endif
#endif

var<workgroup> scratch_xx: array<f32, WG_SIZE>;
var<workgroup> scratch_xg: array<f32, WG_SIZE>;

@compute @workgroup_size(WG_SIZE)
fn main(@builtin(workgroup_id)        wid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
    let row = wid.x;

    let i_grad_row = params.offset_grad + row * params.ne0;
    let i_x_row    = params.offset_x + row * params.ne0;
    let i_dst_row  = params.offset_dst + row * params.ne0;

    var sum_xx = 0.0f;
    var sum_xg = 0.0f;
    for (var col = lid.x; col < params.ne0; col += WG_SIZE) {
        let gi = grad[i_grad_row + col];
        let xi = x[i_x_row + col];
        sum_xx += xi * xi;
        sum_xg += xi * gi;
    }

    scratch_xx[lid.x] = sum_xx;
    scratch_xg[lid.x] = sum_xg;
    workgroupBarrier();

    var offset: u32 = WG_SIZE / 2;
    while (offset > 0) {
        if (lid.x < offset) {
            scratch_xx[lid.x] += scratch_xx[lid.x + offset];
            scratch_xg[lid.x] += scratch_xg[lid.x + offset];
        }
        offset = offset / 2;
        workgroupBarrier();
    }
    sum_xx = scratch_xx[0];
    sum_xg = scratch_xg[0];

    let mean    = sum_xx / f32(params.ne0);
    let scale_g = inverseSqrt(mean + params.eps);
    let scale_x = -scale_g * sum_xg / (sum_xx + f32(params.ne0) * params.eps);

    for (var col = lid.x; col < params.ne0; col += WG_SIZE) {
        store_dst(i_dst_row + col, scale_g * grad[i_grad_row + col] + scale_x * x[i_x_row + col]);
    }
}
