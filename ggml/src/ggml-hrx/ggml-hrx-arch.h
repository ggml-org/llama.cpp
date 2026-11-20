#pragma once

#include "ggml.h"

#include <cstdint>

enum ggml_hrx_matrix_op {
    GGML_HRX_MATRIX_OP_NONE,
    GGML_HRX_MATRIX_OP_WMMA_GFX11,
    GGML_HRX_MATRIX_OP_WMMA_GFX12, // The operand layout is different from gfx11. The WMMA motifs have gfx12 providers.
};

// ggml_mul_mat_vec_xs.
struct ggml_hrx_mul_mat_vec_tune {
    int64_t max_cols;     // At most 8. More input columns use the tiled kernels.
    int64_t xs_max_k;     // At most 4096. The vec_xs kernel keeps 16 superblocks of activations in registers.
    int64_t xs_wg_per_cu;
    int64_t xs_depth;     // At most 4. Pipeline depth of the vec_xs row passes.
};

// Op dispatch values for one architecture.
struct ggml_hrx_tune {
    int nodes_per_submit;   // Nodes between stream flushes.
    int nodes_first_submit; // The interval starts here and doubles up to nodes_per_submit.
    // get_tensor_async copies through mapped host memory instead of hrx_stream_copy_d2h.
    bool mapped_readback;

    // Index: the weight tensor type (src0).
    ggml_hrx_mul_mat_vec_tune mul_mat_vec[GGML_TYPE_COUNT];

    int64_t elementwise_wg_size; // Multiple of 64, at most 1024.
    int64_t norm_wg_size;        // Multiple of 64, at most 1024. Also for soft_max.
    // At most 16. Workgroup limit for each CU of the grid-stride copy kernels. 0 disables these kernels.
    int64_t stream_wg_per_cu;

    int64_t rms_norm_rows_max;  // At most 2048. Longer rows are not supported.
    int64_t rms_norm_lanes_max; // 16, 32 or 64. Lanes for each row.

    int64_t fa_target_workgroups;
    int64_t fa_many_tiles;
    int64_t fa_long_kv_subblocks;
    int64_t fa_mid_kv_subblocks;
    int64_t fa_small_batch;
    int64_t fa_split_waves;
    int64_t fa_min_split_subblocks;
    int64_t fa_max_splits;
};

struct ggml_hrx_arch {
    const char *          name;        // Architecture that the device reports.
    const char *          loom_target; // Loom AMDGPU target of the kernels.
    bool                  wave32;
    bool                  wave64;
    ggml_hrx_matrix_op    matrix_op;
    uint32_t              lds_bytes;         // Local memory limit of one workgroup.
    uint32_t              max_workgroups_yz; // Workgroup count limit in y and z.
    const ggml_hrx_tune * tune;
};

// Returns nullptr if the backend does not support the architecture.
const ggml_hrx_arch * ggml_hrx_arch_find(const char * name);
