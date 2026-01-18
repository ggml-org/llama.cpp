@group(0) @binding(0)
var<storage, read_write> src: array<f32>;

@group(0) @binding(1)
var<storage, read_write> dst: array<i32>;

struct Params {
    offset_src: u32, // in elements
    offset_dst: u32, // in elements

    stride_src1: u32,
    stride_src2: u32,
    stride_src3: u32,

    stride_dst1: u32,
    stride_dst2: u32,
    stride_dst3: u32,

    // src/dst dimensions
    src_ne0: u32,
    ne1: u32,
    ne2: u32,

    ne0: u32,
    top_k: u32,

    npr: u32,   // tiles per row
    nrows: u32
};

@group(0) @binding(2)
var<uniform> params: Params;

var<workgroup> shmem_idx: array<u32, WG_SIZE>;

#if ORDER == 0
#define EXTREME_VALUE 1e30
#define SWAP_COMPARE_UP >
#define SWAP_COMPARE_DOWN <
#else
#define EXTREME_VALUE -1e30
#define SWAP_COMPARE_UP <
#define SWAP_COMPARE_DOWN >
#endif

@compute @workgroup_size(WG_SIZE)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(num_workgroups) num_wg: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
    let linear = wid.x + wid.y * num_wg.x;
    // guard against overprovisioned workgroups
    if (linear >= params.npr * params.nrows) {
        return;
    }
    let tile = linear % params.npr;
    var row = linear / params.npr;
    let i3 = row / (params.ne2 * params.ne1);
    row = row % (params.ne2 * params.ne1);
    let i2 = row / params.ne1;
    let i1 = row % params.ne1;

    let row_base = params.offset_src +
        i1 * params.stride_src1 +
        i2 * params.stride_src2 +
        i3 * params.stride_src3;

    let tile_base = tile * WG_SIZE;
    let idx = tile_base + lid.x;
    shmem_idx[lid.x] = select(params.src_ne0, idx, idx < params.src_ne0);
    workgroupBarrier();

    var k = 2u;
    while (k <= WG_SIZE) {
        var j = k >> 1;
        while (j > 0) {
            let ixj = lid.x ^ j;
            if (ixj > lid.x) {
                let dir_up = (lid.x & k) == 0;
                let a_idx = shmem_idx[lid.x];
                let b_idx = shmem_idx[ixj];
                let a_val = select(EXTREME_VALUE, src[row_base + a_idx], a_idx < params.src_ne0);
                let b_val = select(EXTREME_VALUE, src[row_base + b_idx], b_idx < params.src_ne0);
                let should_swap = select(
                    (a_val SWAP_COMPARE_DOWN b_val),
                    (a_val SWAP_COMPARE_UP b_val),
                    dir_up);
                if (should_swap) {
                    shmem_idx[lid.x] = b_idx;
                    shmem_idx[ixj] = a_idx;
                }
            }
            workgroupBarrier();
            j >>= 1;
        }
        k <<= 1;
    }

    let out_idx = tile * params.top_k + lid.x;
    if (out_idx < params.ne0 && lid.x < params.top_k) {
        let row_dst = params.offset_dst +
            i1 * params.stride_dst1 +
            i2 * params.stride_dst2 +
            i3 * params.stride_dst3;
        dst[row_dst + out_idx] = i32(shmem_idx[lid.x]);
    }
}
