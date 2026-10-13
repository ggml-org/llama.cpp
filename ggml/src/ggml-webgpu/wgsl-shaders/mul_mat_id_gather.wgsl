enable f16;

struct MulMatIdGatherParams {
    offset_ids: u32,

    n_expert: u32,
    n_expert_used: u32,
    n_tokens: u32,

    stride_ids_1: u32,
};

@group(0) @binding(0) var<storage, read_write> ids: array<i32>;        // [n_expert_used, n_tokens]
@group(0) @binding(1) var<storage, read_write> ids_slot_pos: array<u32>; // [n_expert_used * n_tokens]
@group(0) @binding(2) var<storage, read_write> ids_tiles: array<u32>; // [n_tiles, (expert, first slot, n slots) * n_tiles]

@group(0) @binding(3) var<uniform> params: MulMatIdGatherParams;

// per expert: the slot count, replaced by the first slot of the expert in ids_slot_pos by the prefix sum, then advanced as its write cursor
var<workgroup> ids_count: array<atomic<u32>, N_EXPERT>;

// launched as a single workgroup, where the threads process the elements of ids in parallel with atomics
@compute @workgroup_size(WG_SIZE)
fn main(@builtin(local_invocation_id) local_id: vec3<u32>) {

    let thread_id = local_id.x;
    let ids_slots_num = params.n_expert_used * params.n_tokens;

    for (var e = thread_id; e < N_EXPERT; e += WG_SIZE) {
        atomicStore(&ids_count[e], 0u);
    }

    workgroupBarrier();

    // count the slots of each expert
    for (var i = thread_id; i < ids_slots_num; i += WG_SIZE) {
        let row = i / params.n_expert_used;
        let col = i % params.n_expert_used;
        let expert = u32(ids[params.offset_ids + row * params.stride_ids_1 + col]);
        atomicAdd(&ids_count[expert], 1u);
    }

    workgroupBarrier();

    // exclusive prefix sum over the counts, and emit one tile descriptor per `TILE_N_SLOTS` slots of each expert
    if (thread_id == 0u) {
        var base: u32 = 0;
        var tile: u32 = 0;

        for (var e: u32 = 0; e < N_EXPERT; e += 1) {
            let n = atomicLoad(&ids_count[e]);

            atomicStore(&ids_count[e], base);

            for (var k: u32 = 0; k < n; k += TILE_N_SLOTS) {
                ids_tiles[1 + 3 * tile]     = e;
                ids_tiles[1 + 3 * tile + 1] = base + k;
                ids_tiles[1 + 3 * tile + 2] = min(n - k, TILE_N_SLOTS);
                tile += 1;
            }

            base += n;
        }

        ids_tiles[0] = tile;
    }

    workgroupBarrier();

    // store the position in ids tensor to ids_slot_pos
    for (var i = thread_id; i < ids_slots_num; i += WG_SIZE) {
        let row = i / params.n_expert_used;
        let col = i % params.n_expert_used;
        let expert = u32(ids[params.offset_ids + row * params.stride_ids_1 + col]);
        let pos = atomicAdd(&ids_count[expert], 1u);
        ids_slot_pos[pos] = i;
    }
}
