@group(0) @binding(0)
var<storage, read_write> a: array<f32>;

struct Params {
    offset: u32, // in elements

    // Strides (in elements)
    stride1: u32,
    stride2: u32,
    stride3: u32,

    // Shape
    ne0: u32,
    ne1: u32,
    ne2: u32,
    ne3: u32,

    eps: u32
};

@group(0) @binding(1)
var<uniform> params: Params;

override wg_size: u32;
@compute @workgroup_size(wg_size)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.ne1 * params.ne2 * params.ne3) {
        return;
    }

    // one thread per row
    var i = gid.x;
    let i3 = i / (params.ne2 * params.ne1);
    i = i % (params.ne2 * params.ne1);
    let i2 = i / params.ne1;
    let i1 = i % params.ne1;
    let i_row = params.offset + i3 * params.stride3 + i2 * params.stride2 + i1 * params.stride1;

    var sum = 0.0f;
    for (var j: u32 = 0; j < params.ne0; j++) {
        sum += a[i_row + j] * a[i_row + j];
    }
    let eps = bitcast<f32>(params.eps);
    let scale = 1.0/sqrt(sum/f32(params.ne0) + eps);
    for (var j: u32 = 0; j < params.ne0; j++) {
        a[i_row + j] = scale * a[i_row + j];
    }
}
