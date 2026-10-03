struct Params {
    ne: u32,

    offset_grad: u32,
    offset_src1: u32,
    offset_dst: u32,
};

@group(0) @binding(0)
var<storage, read_write> grad: array<f32>;

@group(0) @binding(1)
var<storage, read_write> src1: array<f32>;

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
    src1[i] = val;
}
#else
fn store_dst(i: u32, val: f32) {
    dst_buf[i] = val;
}
#endif
#endif

@compute @workgroup_size(WG_SIZE)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups)       num_wg: vec3<u32>) {
    let i = gid.x + (num_wg.x * u32(WG_SIZE)) * gid.y;
    if (i >= params.ne) {
        return;
    }

    let xi = src1[params.offset_src1 + i];
    let s  = 1.0 / (1.0 + exp(-xi));

    store_dst(params.offset_dst + i, grad[params.offset_grad + i] * (s + xi * s * (1.0 - s)));
}
