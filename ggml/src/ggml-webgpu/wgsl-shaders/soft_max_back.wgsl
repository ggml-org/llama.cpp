// grad of soft_max: y is the forward output, x is not needed

struct Params {
    ne0: u32,

    offset_grad: u32,
    offset_y: u32,
    offset_dst: u32,

    scale: f32,
};

@group(0) @binding(0)
var<storage, read_write> grad: array<f32>;

@group(0) @binding(1)
var<storage, read_write> y: array<f32>;

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
    y[i] = val;
}
#else
fn store_dst(i: u32, val: f32) {
    dst_buf[i] = val;
}
#endif
#endif

var<workgroup> scratch: array<f32, WG_SIZE>;

@compute @workgroup_size(WG_SIZE)
fn main(@builtin(workgroup_id)          wid: vec3<u32>,
        @builtin(local_invocation_id)   lid: vec3<u32>) {
    let row = wid.x;

    let i_grad_row = params.offset_grad + row * params.ne0;
    let i_y_row    = params.offset_y + row * params.ne0;
    let i_dst_row  = params.offset_dst + row * params.ne0;

    var sum = 0.0f;
    for (var col = lid.x; col < params.ne0; col += WG_SIZE) {
        sum += y[i_y_row + col] * grad[i_grad_row + col];
    }

    scratch[lid.x] = sum;
    workgroupBarrier();

    var offset: u32 = WG_SIZE / 2;
    while (offset > 0) {
        if (lid.x < offset) {
            scratch[lid.x] += scratch[lid.x + offset];
        }
        offset = offset / 2;
        workgroupBarrier();
    }
    let dot_yg = scratch[0];

    for (var col = lid.x; col < params.ne0; col += WG_SIZE) {
        store_dst(i_dst_row + col, params.scale * (grad[i_grad_row + col] - dot_yg) * y[i_y_row + col]);
    }
}
