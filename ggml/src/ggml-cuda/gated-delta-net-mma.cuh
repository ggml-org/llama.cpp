#pragma once

#include "common.cuh"

#include <cstdint>

struct ggml_cuda_gdn_mma_args {
    const float * q;
    const float * k;
    const float * v;
    const float * g;
    const float * beta;
    const float * state;
    float *       dst;
    float *       state_out;

    int64_t H;
    int64_t H_k;
    int64_t n_tokens;
    int64_t n_seqs;
    int64_t rq3;
    int64_t sq1;
    int64_t sq2;
    int64_t sq3;
    int64_t sv1;
    int64_t sv2;
    int64_t sv3;
    int64_t sb1;
    int64_t sb2;
    int64_t sb3;
    float   scale;
    bool    eligible;

    // ReplaySSM, a batch that does not record: commit n_commit[2 seq] recorded updates of rec to the state first
    const float *   rec      = nullptr;
    int64_t         rec_seq  = 0;
    const int32_t * n_commit = nullptr;
    int             R        = 0;

    // the initial state read from cache rows (the fused gather, ggml_cuda_gdn_state_rows_plan): row state_ids[seq]
    // of state, rows state_row_stride floats apart
    const int32_t * state_ids        = nullptr;
    int64_t         state_row_stride = 0;
};

// whether the device's MMA path reads its initial state from cache rows, so the gather before it can be fused
bool ggml_cuda_gdn_mma_reads_state_rows(int device);

// device scratch the MMA path needs for these arguments (0 when it does not take them); the op reserves it after dst
size_t ggml_cuda_gdn_mma_scratch_size(int device, const ggml_cuda_gdn_mma_args & args);

// scratch: ggml_cuda_gdn_mma_scratch_size(device, args) bytes
bool ggml_cuda_gdn_mma_launch(void * scratch, int device, const ggml_cuda_gdn_mma_args & args, cudaStream_t stream);
