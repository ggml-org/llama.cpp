@group(0) @binding(0)
#ifdef VEC4
var<storage, read_write> src: array<vec4<f32>>;
#define VEC_SIZE 4
#else
var<storage, read_write> src: array<f32>;
#define VEC_SIZE 1
#endif

@group(0) @binding(1)
var<storage, read_write> dst: array<i32>;

struct Params {
    offset_src: u32, // in elements
    offset_dst: u32, // in elements
    ne0: u32,
};

@group(0) @binding(2)
var<uniform> params: Params;

const FLOAT_MIN: f32 = -1.0e9;

struct Pair {
    value: f32,
    index: i32
};

var<workgroup> shared_max: array<Pair, WG_SIZE>;

@compute @workgroup_size(WG_SIZE)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
    let row_idx = params.offset_src + wid.x * params.ne0;
    var local_pair = Pair(FLOAT_MIN, -1);
#ifdef VEC4
    for (var col = lid.x; col < params.ne0/VEC_SIZE; col += WG_SIZE) {
        let vec_val = src[row_idx / VEC_SIZE + col];
        for (var v = 0u; v < VEC_SIZE; v++) {
            let val = vec_val[v];
            if (val >= local_pair.value) {
                local_pair = Pair(val, i32(col * VEC_SIZE + v));
            }
        }
    }
#else
    for (var col = lid.x; col < params.ne0; col += WG_SIZE) {
        if (src[row_idx + col] >= local_pair.value) {
            local_pair = Pair(src[row_idx + col], i32(col));
        }
    }
#endif
    shared_max[lid.x] = local_pair;
    workgroupBarrier();
    var offset: u32 = WG_SIZE >> 1;
    while (offset > 0) {
        if (lid.x < offset) {
            let a = shared_max[lid.x];
            let b = shared_max[lid.x + offset];
            if (b.value > a.value) {
                shared_max[lid.x] = b;
            } else if (b.value == a.value && b.index > a.index) {
                shared_max[lid.x] = b;
            }
        }
        workgroupBarrier();
        offset >>= 1;
    }
    if (lid.x == 0u) {
        dst[params.offset_dst + wid.x] = shared_max[0].index;
    }
}
