// memset.wgsl
@group(0) @binding(0)
var<storage, read_write> output_buffer: array<u32>;

struct Params {
    offset: u32, // in bytes
    size: u32,   // in bytes
    value: u32,  // four identical values
};

@group(0) @binding(1)
var<uniform> params: Params;

// TODO: figure out workgroup size
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x * 4u;
    let start = params.offset;
    let end = params.offset + params.size;

    // Each thread writes one u32 (4 bytes)
    let byte_index = start + i;
    if (byte_index + 4u <= end) {
        output_buffer[(byte_index >> 2u)] = params.value;
    } else {
        // Handle tail (unaligned)
        for (var j: u32 = 0u; j < 4u; j = j + 1u) {
            let idx = byte_index + j;
            if (idx < end) {
                let word_idx = idx >> 2u;
                let byte_offset = (idx & 3u) * 8u;
                let mask = ~(0xffu << byte_offset);
                let existing = output_buffer[word_idx];
                output_buffer[word_idx] = (existing & mask) | (params.value & 0xffu) << byte_offset;
            }
        }
    }
}
