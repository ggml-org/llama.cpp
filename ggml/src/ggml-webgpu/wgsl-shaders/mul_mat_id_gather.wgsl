enable f16;

struct MulMatIdGatherParams {
    offset_ids: u32,
    offset_dst: u32,

    n_expert: u32,
    n_expert_used: u32,
    n_tokens: u32,

    stride_ids_1: u32,

    m: u32,
};

@group(0) @binding(0) var<storage, read_write> ids: array<i32>;        // [n_expert_used, n_tokens]
@group(0) @binding(1) var<storage, read_write> global_gathered_expert_used: array<u32>; // [n_expert][n_tokens]
@group(0) @binding(2) var<storage, read_write> global_gathered_tokens: array<u32>; // [n_expert][n_tokens]
@group(0) @binding(3) var<storage, read_write> gathered_count_ids: array<u32>; // [n_expert]
@group(0) @binding(4) var<storage, read_write> dst: array<f32>;              // [m, n_expert_used, n_tokens]

@group(0) @binding(5) var<uniform> params: MulMatIdGatherParams;

var<workgroup> count:atomic<u32>;

@compute @workgroup_size(WG_SIZE)
fn main(@builtin(workgroup_id) wg_id: vec3<u32>,
        @builtin(num_workgroups) num_wg: vec3<u32>,
        @builtin(local_invocation_id) local_id: vec3<u32>) {

    let thread_id = local_id.x;
    let own_expert = wg_id.x; // the expert assigned to this workgroup

    if (thread_id == 0u) {
        atomicStore(&count, 0);
    }

    workgroupBarrier();

    for (var i = thread_id;i < params.n_expert_used * params.n_tokens;i += WG_SIZE) {
        let row = i / params.n_expert_used;
        let col = i % params.n_expert_used;
        let expert = i32(ids[params.offset_ids + row * params.stride_ids_1 + col]);
        if (expert != -1 && own_expert == u32(expert)) {
            let pos = atomicAdd(&count, 1u);
            let gathered_id = own_expert * params.n_tokens + pos;
            global_gathered_expert_used[gathered_id] = col;
            global_gathered_tokens[gathered_id] = row;
        }
    }

    workgroupBarrier();

    if (thread_id == 0u) {
        gathered_count_ids[own_expert] = atomicLoad(&count);
    }

    // the matmul writes only the rows that an expert owns, so zero the rows of the skipped slots
    // each workgroup takes a strided share of the slots
    for (var idx = own_expert;idx < params.n_expert_used * params.n_tokens;idx += num_wg.x) {
        let row = idx / params.n_expert_used;
        let col = idx % params.n_expert_used;

        if (i32(ids[params.offset_ids + row * params.stride_ids_1 + col]) != -1) {
            continue;
        }

        let dst_row = params.offset_dst + (row * params.n_expert_used + col) * params.m;
        for (var i = thread_id;i < params.m;i += WG_SIZE) {
            dst[dst_row + i] = 0.0f;
        }
    }
}
