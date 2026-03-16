#ifndef BLOCK_SIZE
#define BLOCK_SIZE 32u
#endif

#ifndef TOKENS_PER_WG
#define TOKENS_PER_WG 8u
#endif

alias SRC_TYPE = f32;

@group(0) @binding(0)
var<storage, read> src0: array<SRC_TYPE>;

@group(0) @binding(1)
var<storage, read> src1: array<SRC_TYPE>;

@group(0) @binding(2)
var<storage, read_write> dst: array<f32>;

struct Params {
    offset_src0: u32,
    offset_src1: u32,
    offset_dst: u32,

    stride_src01: u32,
    stride_src02: u32,
    stride_src11: u32,

    stride_dst0: u32,
    stride_dst1: u32,
    stride_dst2: u32,

    nc: u32,
    nr: u32,
    n_t: u32,
    n_s: u32,
    token_tiles: u32,
};

@group(0) @binding(3)
var<uniform> params: Params;

@compute @workgroup_size(BLOCK_SIZE, TOKENS_PER_WG)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i1 = gid.x;
    let tile_y = gid.y / TOKENS_PER_WG;
    let local_token = gid.y % TOKENS_PER_WG;
    let i3 = tile_y / params.token_tiles;
    let token_tile = tile_y % params.token_tiles;
    let i2 = token_tile * TOKENS_PER_WG + local_token;

    if (i1 >= params.nr || i2 >= params.n_t || i3 >= params.n_s) {
        return;
    }

    let src0_base = params.offset_src0 + i3 * params.stride_src02 + i2 + i1 * params.stride_src01;
    let src1_base = params.offset_src1 + i1 * params.stride_src11;

    var sum = 0.0;

#ifdef VECTORIZED
    sum =
        src0[src0_base + 0u] * src1[src1_base + 0u] +
        src0[src0_base + 1u] * src1[src1_base + 1u] +
        src0[src0_base + 2u] * src1[src1_base + 2u] +
        src0[src0_base + 3u] * src1[src1_base + 3u];
#else
    for (var i0 = 0u; i0 < params.nc; i0++) {
        sum += src0[src0_base + i0] * src1[src1_base + i0];
    }
#endif

    let dst_idx = params.offset_dst + i3 * params.stride_dst2 + i2 * params.stride_dst1 + i1 * params.stride_dst0;
    dst[dst_idx] = sum;
}
