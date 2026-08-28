@group(0) @binding(0)
var<storage, read_write> src: array<f32>;

@group(0) @binding(1)
var<storage, read_write> dst: array<f32>;

struct Params {
    offset_src: u32, // in elements
    offset_dst: u32, // in elements
    ne0: u32,
};

@group(0) @binding(2)
var<uniform> params: Params;

var<workgroup> shared_sum: array<f32, WG_SIZE>;

@compute @workgroup_size(WG_SIZE)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
    let row_idx = params.offset_src + wid.x * params.ne0;
    let elems = (params.ne0 + WG_SIZE - 1) / WG_SIZE;
    var local_sum: f32 = 0.0;
    for (var col = lid.x * elems; col < (lid.x + 1) * elems && col < params.ne0; col ++) {
        local_sum += src[row_idx + col];
    }
    shared_sum[lid.x] = local_sum;
    workgroupBarrier();

    // upsweep
    var offset = 1u;
    while (offset < WG_SIZE) {
        let idx = (lid.x + 1) * offset * 2 - 1;
        if (idx < WG_SIZE) {
            shared_sum[idx] = shared_sum[idx] + shared_sum[idx - offset];
        }
        workgroupBarrier();
        offset <<= 1;
    }

    // set last to 0 for exclusive sum
    if (lid.x == 0) {
        shared_sum[WG_SIZE - 1] = 0.0;
    }
    workgroupBarrier();

    // downsweep
    offset = WG_SIZE >> 1;
    while (offset > 0) {
        let idx = (lid.x + 1) * offset * 2 - 1;
        if (idx < WG_SIZE) {
            let t = shared_sum[idx - offset];
            shared_sum[idx - offset] = shared_sum[idx];
            shared_sum[idx] = shared_sum[idx] + t;
        }
        workgroupBarrier();
        offset = offset >> 1;
    }

    // shared_sum[lid] is exclusive prefix sum up to this thread.
    var running_sum = shared_sum[lid.x];
    for (var col = lid.x * elems; col < (lid.x + 1) * elems && col < params.ne0; col ++) {
        running_sum += src[row_idx + col];
        dst[params.offset_dst + wid.x * params.ne0 + col] = running_sum;
    }
}
