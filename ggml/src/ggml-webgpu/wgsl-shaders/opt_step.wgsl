struct Params {
    ne: u32,

    offset_x: u32,
    offset_grad: u32,
#ifdef ADAMW
    offset_gm: u32,
    offset_gv: u32,
#endif
    offset_opt: u32,
};

@group(0) @binding(0)
var<storage, read_write> x: array<f32>;

@group(0) @binding(1)
var<storage, read_write> grad: array<f32>;

#ifdef ADAMW
@group(0) @binding(2)
var<storage, read_write> gm: array<f32>;

@group(0) @binding(3)
var<storage, read_write> gv: array<f32>;

@group(0) @binding(4)
var<storage, read_write> opt: array<f32>;

@group(0) @binding(5)
var<uniform> params: Params;
#else
@group(0) @binding(2)
var<storage, read_write> opt: array<f32>;

@group(0) @binding(3)
var<uniform> params: Params;
#endif

@compute @workgroup_size(WG_SIZE)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups)       num_wg: vec3<u32>) {
    let i = gid.x + (num_wg.x * u32(WG_SIZE)) * gid.y;
    if (i >= params.ne) {
        return;
    }

    let i_x  = params.offset_x + i;
    let gi   = grad[params.offset_grad + i];
    let o    = params.offset_opt;

    let alpha = opt[o];

#ifdef ADAMW
    let beta1  = opt[o + 1];
    let beta2  = opt[o + 2];
    let eps    = opt[o + 3];
    let wd     = opt[o + 4];
    let beta1h = opt[o + 5];
    let beta2h = opt[o + 6];

    let i_gm = params.offset_gm + i;
    let i_gv = params.offset_gv + i;

    let gmi = gm[i_gm] * beta1 + gi * (1.0 - beta1);
    let gvi = gv[i_gv] * beta2 + gi * gi * (1.0 - beta2);

    gm[i_gm] = gmi;
    gv[i_gv] = gvi;

    let mh = gmi * beta1h;
    let vh = sqrt(gvi * beta2h) + eps;

    x[i_x] = x[i_x] * (1.0 - alpha * wd) - alpha * mh / vh;
#else
    let keep = 1.0 - alpha * opt[o + 1];

    x[i_x] = x[i_x] * keep - alpha * gi;
#endif
}
