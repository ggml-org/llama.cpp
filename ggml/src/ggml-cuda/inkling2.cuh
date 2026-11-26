#pragma once

#include "common.cuh"

// fused op chains of the Inkling 2 graph

// largest conv width and stack depth the fused kernels handle
#define GGML_CUDA_SHORT_CONV_MAX_K   8
#define GGML_CUDA_DEPTH_MIX_MAX_ROWS 16

// the most tokens the fused loop controller handles, larger batches use the matrix multiplications
#define GGML_CUDA_LOOP_CTRL_MAX_TOKENS 8

// situ(g, u) = (s1*tanh(s0*g + b0) + b1)*sigmoid(g)*(s3*tanh(s2*u + b2) + b3), with s and b of the four SCALE nodes
void ggml_cuda_op_situ(ggml_backend_cuda_context & ctx, const ggml_tensor * gate, const ggml_tensor * up,
        const float * s, const float * b, ggml_tensor * dst);

// causal short conv: x = [past; cur^T], y = conv(x, w) (+ input-dependent taps per head), dst = s*(cur + y) + b (+ res)
// the K - 1 columns of x that end at token Ts - 1 - slot are the conv state of slot, for the last kw slots (kw = 1 without rollback)
// each thread does `run` tokens of a sequence in order, run == Ts lets dst be cur
struct ggml_cuda_short_conv_args {
    const float * past;    // [K - 1, C, S], strides in elements
    int64_t       past_s1;
    int64_t       past_s2;
    const float * cur;     // [C, Ts*S], contiguous
    const float * w;       // [K, C], contiguous
    const float * dyn;     // [K*H, Ts*S] taps or nullptr
    int64_t       dyn_s1;
    int           n_heads;
    const float * res;     // [C, Ts*S], contiguous, or nullptr
    float       * dst;     // [C, Ts*S], contiguous
    float       * state;    // [K - 1, slot, C, S], strides in elements
    int64_t       state_slot_s;
    int64_t       state_ch_s;
    int64_t       state_seq_s;
    int           kw;
    int           run;
    int           C;
    int           K;
    int           Ts;
    int           S;
    float         scale;
    float         bias;
};

void ggml_cuda_op_short_conv(ggml_backend_cuda_context & ctx, const ggml_cuda_short_conv_args & args);

// depth mix: states = [stack; rms_norm(cur)], p = softmax(scale*(states . query)), dst = p . states, written as [Q, n, T]
// norm_cur false: cur is already normalized
void ggml_cuda_op_depth_mix(ggml_backend_cuda_context & ctx, const ggml_tensor * stack, const ggml_tensor * cur,
        bool norm_cur, float eps, const ggml_tensor * query, float scale, ggml_tensor * dst);

// loop controller: gate = w_down . [rms_norm(cur); rms_norm(msg)], dst = cur - tanh(w_forget . gate)*cur + tanh(w_gain . gate)*msg
void ggml_cuda_op_loop_ctrl(ggml_backend_cuda_context & ctx, const ggml_tensor * cur, const ggml_tensor * msg, float eps,
        const ggml_tensor * w_down, const ggml_tensor * w_gain, const ggml_tensor * w_forget, ggml_tensor * gate, ggml_tensor * dst);
