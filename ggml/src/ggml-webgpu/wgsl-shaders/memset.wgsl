@group(0) @binding(0)
var<storage, read_write> output_buffer: array<u32>;

struct Params {
    offset: u32, // in bytes
    size: u32,   // in bytes
    value: u32,  // 4 8-bit values, which are either repeating (memset_tensor) or may be separate (cleaning up unaligned set_tensor operations)
};

@group(0) @binding(1)
var<uniform> params: Params;

override wg_size: u32;

@compute @workgroup_size(wg_size)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(num_workgroups) num_wg: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {

    let i = (wid.x + wid.y * num_wg.x) * wg_size + lid.x;
    let start = params.offset;
    let end = params.offset + params.size;
    let id = (start >> 2u) + i;
    let id_u32 = id * 4;

    if (id_u32 >= end) {
        return;
    }

    let lo = max(start, id_u32) - id_u32;
    let hi = min(end, id_u32 + 4u) - id_u32;
    var mask = 0xFFFFFFFFu << (8u * lo);
    if (hi < 4u) {
        mask &= (1u << (8u * hi)) - 1u;
    }

    if (mask == 0xFFFFFFFFu) {
        output_buffer[id] = params.value;
    } else {
        let existing = output_buffer[id];
        output_buffer[id] = (existing & ~(mask)) | (params.value & mask);
    }
}
