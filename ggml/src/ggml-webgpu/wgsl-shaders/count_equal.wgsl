struct Params {
    offset_src0: u32,
    offset_src1: u32,
    offset_dst: u32,
    ne: u32,
};

#ifdef SRC_OVERLAP
@group(0) @binding(0)
var<storage, read_write> merged_src: array<i32>;

@group(0) @binding(1)
var<storage, read_write> dst: array<atomic<u32>>;

@group(0) @binding(2)
var<uniform> params: Params;
#else
@group(0) @binding(0)
var<storage, read_write> src0: array<i32>;

@group(0) @binding(1)
var<storage, read_write> src1: array<i32>;

@group(0) @binding(2)
var<storage, read_write> dst: array<atomic<u32>>;

@group(0) @binding(3)
var<uniform> params: Params;
#endif

var<workgroup> shared_count: array<u32, WG_SIZE>;

@compute @workgroup_size(WG_SIZE)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(num_workgroups) num_wg: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
    var count = 0u;
    let i = (wid.x + wid.y * num_wg.x) * WG_SIZE + lid.x;
    if (i < params.ne) {
#ifdef SRC_OVERLAP
        count += select(0u, 1u, merged_src[params.offset_src0 + i] == merged_src[params.offset_src1 + i]);
#else
        count += select(0u, 1u, src0[params.offset_src0 + i] == src1[params.offset_src1 + i]);
#endif
    }
    shared_count[lid.x] = count;
    workgroupBarrier();
    for (var offset: u32 = WG_SIZE >> 1; offset > 0; offset >>= 1) {
        if (lid.x < offset) {
            shared_count[lid.x] += shared_count[lid.x + offset];
        }
        workgroupBarrier();
    }
    if (lid.x == 0) {
        atomicAdd(&dst[params.offset_dst], shared_count[0]);
    }
}
