enable f16;

@group(0) @binding(0)
var<storage, read_write> src: array<f32>;

@group(0) @binding(1)
var<storage, read_write> dst: array<f16>;

struct Params {
    ne: u32, // number of elements
    src_offset: u32, // src offset in bytes
    dst_offset: u32 // dst offset in bytes
};

@group(0) @binding(2)
var<uniform> params: Params;

override wg_size: u32;
const elems_per_thread: u32 = 4;

@compute @workgroup_size(wg_size)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x * elems_per_thread;
    // chunked loop
    for (var j: u32 = 0u; j < elems_per_thread; j = j + 1u) {
        let i = idx + j;
        if (i < params.ne) {
            // Convert f32 to f16
            dst[dst_offset/2 + i] = f16(src[src_offset/4 + i]);
        }
    }
}
