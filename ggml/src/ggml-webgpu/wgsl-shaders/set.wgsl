#ifdef TYPE_I32
#define TYPE i32
#else
#define TYPE f32
#endif

#ifndef INPLACE
@group(0) @binding(0)
var<storage, read_write> src0: array<TYPE>;
#define SRC1_BINDING 1
#else
#define SRC1_BINDING 0
#endif

#define DST_BINDING SRC1_BINDING + 1
#define PARAMS_BINDING SRC1_BINDING + 2

@group(0) @binding(SRC1_BINDING)
var<storage, read_write> src1: array<TYPE>;

@group(0) @binding(DST_BINDING)
var<storage, read_write> dst: array<TYPE>;

struct Params {
    ne: u32,
    offset_src0: u32,
    offset_src1: u32,
    offset_view: u32,

    stride_src10: u32,
    stride_src11: u32,
    stride_src12: u32,
    stride_src13: u32,

    stride_dst10: u32,
    stride_dst11: u32,
    stride_dst12: u32,
    stride_dst13: u32,

    src1_ne0: u32,
    src1_ne1: u32,
    src1_ne2: u32,
    src1_ne3: u32,
};

@group(0) @binding(PARAMS_BINDING)
var<uniform> params: Params;

fn decode_src1_coords(idx: u32) -> vec4<u32> {
    var i = idx;
    let plane = params.src1_ne2 * params.src1_ne1 * params.src1_ne0;
    let i3 = i / plane;
    i = i % plane;
    let row = params.src1_ne1 * params.src1_ne0;
    let i2 = i / row;
    i = i % row;
    let i1 = i / params.src1_ne0;
    let i0 = i % params.src1_ne0;
    return vec4<u32>(i0, i1, i2, i3);
}

fn decode_view_coords(rel: u32) -> vec4<u32> {
    let i3 = rel / params.stride_dst13;
    let rem3 = rel % params.stride_dst13;
    let i2 = rem3 / params.stride_dst12;
    let rem2 = rem3 % params.stride_dst12;
    let i1 = rem2 / params.stride_dst11;
    let i0 = rem2 % params.stride_dst11;
    return vec4<u32>(i0, i1, i2, i3);
}

fn view_rel_from_coords(coords: vec4<u32>) -> u32 {
    return coords.x * params.stride_dst10 + coords.y * params.stride_dst11 +
           coords.z * params.stride_dst12 + coords.w * params.stride_dst13;
}

fn src1_idx_from_coords(coords: vec4<u32>) -> u32 {
    return coords.x * params.stride_src10 + coords.y * params.stride_src11 +
           coords.z * params.stride_src12 + coords.w * params.stride_src13;
}

fn in_set_view(rel: u32, coords: vec4<u32>) -> bool {
    return view_rel_from_coords(coords) == rel;
}

@compute @workgroup_size(WG_SIZE)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.ne) {
        return;
    }

#ifdef INPLACE
    let coords = decode_src1_coords(gid.x);

    let src1_idx = params.offset_src1 + src1_idx_from_coords(coords);
    let dst_idx = params.offset_view + view_rel_from_coords(coords);

    dst[dst_idx] = src1[src1_idx];
#else
    let rel = select(params.ne, gid.x - params.offset_view, gid.x >= params.offset_view);
    let coords = decode_view_coords(rel);

    if (rel < params.stride_dst13 * params.src1_ne3 && in_set_view(rel, coords)) {
        dst[gid.x] = src1[params.offset_src1 + src1_idx_from_coords(coords)];
    } else {
        dst[gid.x] = src0[params.offset_src0 + gid.x];
    }
#endif
}
