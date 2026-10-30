@group(0) @binding(0)
var<storage, read_write> output_buffer: array<atomic<u32>>;

@group(0) @binding(1)
var<storage, read> src_buffer: array<u32>;

@group(0) @binding(2)
var<storage, read> descs: array<vec4<u32>>;

override wg_size: u32;

@compute @workgroup_size(wg_size)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(num_workgroups) num_wg: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {

    let w = wid.y * num_wg.x + wid.x;
    
    if (w >= arrayLength(&descs)) {
        return;
    }

    let dst_i     = descs[w].x / 4u;
    let src_i     = descs[w].y;
    let lane    = descs[w].x & 3u;
    let end     = lane + descs[w].z;
    let n_words = (end + 3u) / 4u;

    for (var i = lid.x; i < n_words; i += wg_size) {
        // [lo, hi) is the byte range of this word that belongs to the descs
        let lo    = select(0u, lane, i == 0u);
        let hi    = min(end - i * 4u, 4u);
        let mask  = select((1u << (hi * 8u)) - 1u, 0xffffffffu, hi == 4u) & (0xffffffffu << (lo * 8u));
        let value = src_buffer[src_i + i];
        if (mask == 0xffffffffu) {
            atomicStore(&output_buffer[dst_i + i], value);
        } else {
            atomicAnd(&output_buffer[dst_i + i], ~mask);
            atomicOr(&output_buffer[dst_i + i], value & mask);
        }
    }
}
