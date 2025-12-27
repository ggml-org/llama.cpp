/**
 * mtmd-tts-code2wav.cpp - Code2Wav vocoder for Qwen3-Omni TTS
 *
 * Converts 16-codebook audio tokens to 24kHz waveform using HiFi-GAN architecture.
 *
 * Architecture:
 *   1. Codebook embedding sum (16 codebooks -> 1024 dim, averaged)
 *   2. Pre-transformer: 8 layers with sliding window causal attention + LayerScale
 *   3. ConvNeXt upsample: 2 blocks (4x total)
 *   4. HiFi-GAN decoder: 4 stages (8x, 5x, 4x, 3x = 480x total) with Snake activations
 *   5. Final conv + clamp -> mono audio
 *
 * Total upsampling: 4 * 480 = 1920x (1 codec frame = 1920 audio samples)
 *
 * Extracted from tools/qwen3omni-tts/main.cpp
 */

#include "mtmd-tts.h"
#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"

// Internal headers for model tensor access
#include "llama-model.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>

// Forward declaration for external linkage
std::vector<float> mtmd_code2wav_run(
    const llama_model * model,
    const std::vector<std::vector<int>> & all_codebook_tokens,
    bool verbose,
    bool cpu_only);
#include <fstream>
#include <algorithm>

// =============================================================================
// Constants
// =============================================================================

// Code2Wav uses eps=1e-5 for RMSNorm (from HF config.rms_norm_eps)
static constexpr float C2W_RMS_NORM_EPS = 1e-5f;

// Pre-transformer sliding window attention size (from HF config.sliding_window)
static constexpr int C2W_SLIDING_WINDOW = 72;

// Chunked decode constants (CUDA IM2COL kernel has grid y-dimension limit)
static const int CODE2WAV_CHUNK_SIZE = 25;        // 25 new frames per chunk
static const int CODE2WAV_LEFT_CONTEXT = 5;       // 5 context frames
static const int CODE2WAV_TOTAL_UPSAMPLE = 1920;  // 4 (ConvNeXt) x 480 (HiFi-GAN)

// =============================================================================
// Code2Wav Context
// =============================================================================

struct mtmd_code2wav_context {
    ggml_backend_t backend_cpu;
    std::vector<ggml_backend_t> backends;
    std::vector<ggml_backend_buffer_type_t> backend_bufts;
    ggml_backend_sched_t sched;
    std::vector<uint8_t> buf_compute_meta;
    int max_nodes;

    mtmd_code2wav_context() : backend_cpu(nullptr), sched(nullptr), max_nodes(32768) {}
    ~mtmd_code2wav_context() {
        if (sched) {
            ggml_backend_sched_free(sched);
        }
        for (auto backend : backends) {
            if (backend && backend != backend_cpu) {
                ggml_backend_free(backend);
            }
        }
        if (backend_cpu) {
            ggml_backend_free(backend_cpu);
        }
    }
};

// =============================================================================
// Helper Functions
// =============================================================================

// Copy tensor data to CPU vector (handles quantized tensors)
static bool copy_tensor_to_cpu(const ggml_tensor * tensor, std::vector<float> & out) {
    if (!tensor) return false;

    int64_t n = ggml_nelements(tensor);
    int64_t nbytes = ggml_nbytes(tensor);
    out.resize(n);

    // Check if tensor is f32
    if (tensor->type == GGML_TYPE_F32) {
        // Direct copy for f32
        ggml_backend_tensor_get(tensor, out.data(), 0, nbytes);
    } else {
        // For other types, need to dequantize
        // First copy raw bytes, then dequantize
        std::vector<uint8_t> raw(nbytes);
        ggml_backend_tensor_get(tensor, raw.data(), 0, nbytes);

        // Dequantize to float
        const ggml_type_traits * traits = ggml_get_type_traits(tensor->type);
        if (traits && traits->to_float) {
            traits->to_float(raw.data(), out.data(), n);
        } else {
            fprintf(stderr, "Warning: Cannot dequantize tensor type %d\n", (int)tensor->type);
            return false;
        }
    }
    return true;
}

// Build Snake activation (SnakeBeta): f(x) = x + (1/(exp(beta) + eps)) * sin^2(x * exp(alpha))
// Reference: HuggingFace SnakeBeta from modeling_qwen3_omni_moe.py
// alpha/beta are stored in log-scale, must exponentiate before use
static ggml_tensor * build_snake(ggml_context * ctx, ggml_tensor * x,
                                  ggml_tensor * alpha, ggml_tensor * beta) {
    if (!alpha) return x;

    // Exponentiate alpha and beta (they're stored in log-scale)
    ggml_tensor * exp_alpha = ggml_exp(ctx, alpha);
    ggml_tensor * exp_beta = beta ? ggml_exp(ctx, beta) : nullptr;

    // sin(x * exp(alpha))
    ggml_tensor * scaled_x = ggml_mul(ctx, x, exp_alpha);
    ggml_tensor * sin_val = ggml_sin(ctx, scaled_x);

    // sin^2(x * exp(alpha))
    ggml_tensor * sin2 = ggml_sqr(ctx, sin_val);

    // (1/(exp(beta) + epsilon)) * sin^2(...)
    // Match HuggingFace: beta + no_div_by_zero where no_div_by_zero = 1e-9
    ggml_tensor * term;
    if (exp_beta) {
        ggml_tensor * beta_safe = ggml_scale_bias(ctx, exp_beta, 1.0f, 1e-9f);
        term = ggml_div(ctx, sin2, beta_safe);
    } else {
        term = sin2;
    }

    // x + (1/(exp(beta) + eps)) * sin^2(x * exp(alpha))
    return ggml_add(ctx, x, term);
}

// Build RMSNorm: x / sqrt(mean(x^2) + eps) * weight
static ggml_tensor * build_rms_norm(ggml_context * ctx, ggml_tensor * x,
                                     ggml_tensor * weight, float eps = 1e-6f) {
    x = ggml_rms_norm(ctx, x, eps);
    return ggml_mul(ctx, x, weight);
}

// Build causal 1D convolution with left-only padding
// For causal conv: effective_kernel = kernel + (kernel-1) * (dilation-1)
// Causal padding = effective_kernel - stride (all on LEFT)
static ggml_tensor * build_causal_conv1d(ggml_context * ctx, ggml_tensor * kernel,
                                          ggml_tensor * input, int dilation) {
    int kernel_size = kernel->ne[0];
    int effective_kernel = kernel_size + (kernel_size - 1) * (dilation - 1);
    int causal_pad = effective_kernel - 1;  // stride=1

    // Pad LEFT side only
    ggml_tensor * padded = ggml_pad_ext(ctx, input, causal_pad, 0, 0, 0, 0, 0, 0, 0);

    // Conv with padding=0 since we already padded
    return ggml_conv_1d(ctx, kernel, padded, 1, 0, dilation);
}

// =============================================================================
// Initialize Code2Wav Context
// =============================================================================

static bool mtmd_code2wav_init(mtmd_code2wav_context * ctx, bool cpu_only) {
    // Try to get GPU backend first
    if (!cpu_only) {
        ggml_backend_t backend_gpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU, nullptr);
        if (backend_gpu) {
            ctx->backends.push_back(backend_gpu);
            ctx->backend_bufts.push_back(ggml_backend_get_default_buffer_type(backend_gpu));
            fprintf(stderr, "Code2Wav: using GPU backend (%s)\n", ggml_backend_name(backend_gpu));
        }
    } else {
        fprintf(stderr, "Code2Wav: CPU-only mode\n");
    }

    // Get CPU backend as fallback
    ctx->backend_cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    if (!ctx->backend_cpu) {
        fprintf(stderr, "Error: Failed to initialize CPU backend\n");
        return false;
    }
    ctx->backends.push_back(ctx->backend_cpu);
    ctx->backend_bufts.push_back(ggml_backend_get_default_buffer_type(ctx->backend_cpu));

    // Create scheduler with all backends
    ctx->sched = ggml_backend_sched_new(
        ctx->backends.data(),
        ctx->backend_bufts.data(),
        ctx->backends.size(),
        ctx->max_nodes,
        false,  // parallel
        true    // op_offload
    );

    if (!ctx->sched) {
        fprintf(stderr, "Error: Failed to create backend scheduler\n");
        return false;
    }

    // Allocate compute meta buffer
    ctx->buf_compute_meta.resize(ctx->max_nodes * ggml_tensor_overhead() + ggml_graph_overhead());

    return true;
}

// =============================================================================
// Build Code2Wav Compute Graph
// =============================================================================

static ggml_tensor * build_code2wav_graph(
        ggml_context * ctx,
        ggml_cgraph * gf,
        const llama_model * model,
        ggml_tensor * input_embd,  // [c2w_n_embd, n_frames]
        int n_frames,
        bool verbose,
        ggml_tensor ** out_attn_mask) {

    const int c2w_n_embd = 1024;
    const int c2w_n_head = 16;
    const int c2w_head_dim = 64;
    const int c2w_n_layer = 8;
    const int c2w_up_n_block = 2;
    const int upsample_rates[] = {8, 5, 4, 3};
    const int c2w_dec_n_stage = 4;
    const int c2w_dec_n_resblk = 3;

    ggml_tensor * cur = input_embd;
    int seq_len = n_frames;

    // Build sliding window causal mask for pre-transformer (window=72)
    ggml_tensor * c2w_attn_mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, seq_len, seq_len);
    ggml_set_name(c2w_attn_mask, "c2w_attn_mask");
    ggml_set_input(c2w_attn_mask);
    if (out_attn_mask) {
        *out_attn_mask = c2w_attn_mask;
    }

    // =========================================================================
    // Pre-transformer: 8 layers with sliding window causal attention
    // =========================================================================
    if (verbose) {
        printf("  Building pre-transformer (%d layers)...\n", c2w_n_layer);
    }

    for (int il = 0; il < c2w_n_layer && il < (int)model->c2w_pre_layers.size(); ++il) {
        const auto & layer = model->c2w_pre_layers[il];
        ggml_tensor * inpSA = cur;

        // Attention norm (RMSNorm)
        if (layer.attn_norm) {
            cur = build_rms_norm(ctx, cur, layer.attn_norm, C2W_RMS_NORM_EPS);
        }

        // Self-attention
        if (layer.wq && layer.wk && layer.wv && layer.wo) {
            // Q/K/V projections
            ggml_tensor * Qcur = ggml_mul_mat(ctx, layer.wq, cur);
            ggml_tensor * Kcur = ggml_mul_mat(ctx, layer.wk, cur);
            ggml_tensor * Vcur = ggml_mul_mat(ctx, layer.wv, cur);

            // Reshape for multi-head
            Qcur = ggml_reshape_3d(ctx, Qcur, c2w_head_dim, c2w_n_head, seq_len);
            Kcur = ggml_reshape_3d(ctx, Kcur, c2w_head_dim, c2w_n_head, seq_len);
            Vcur = ggml_reshape_3d(ctx, Vcur, c2w_head_dim, c2w_n_head, seq_len);

            // Apply RoPE
            ggml_tensor * pos_f32 = ggml_arange(ctx, 0, seq_len, 1);
            ggml_tensor * pos = ggml_cast(ctx, pos_f32, GGML_TYPE_I32);
            ggml_set_name(pos, "rope_pos");

            Qcur = ggml_rope_ext(ctx, Qcur, pos, nullptr, c2w_head_dim, GGML_ROPE_TYPE_NEOX,
                                 2048, 10000.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
            Kcur = ggml_rope_ext(ctx, Kcur, pos, nullptr, c2w_head_dim, GGML_ROPE_TYPE_NEOX,
                                 2048, 10000.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);

            // Permute for attention
            Qcur = ggml_permute(ctx, Qcur, 0, 2, 1, 3);
            Kcur = ggml_permute(ctx, Kcur, 0, 2, 1, 3);
            Vcur = ggml_permute(ctx, Vcur, 0, 2, 1, 3);
            Qcur = ggml_cont(ctx, Qcur);
            Kcur = ggml_cont(ctx, Kcur);
            Vcur = ggml_cont(ctx, Vcur);

            // Compute attention scores
            ggml_tensor * KQ = ggml_mul_mat(ctx, Kcur, Qcur);

            // Scale + sliding window mask + softmax
            float scale = 1.0f / sqrtf((float)c2w_head_dim);
            KQ = ggml_soft_max_ext(ctx, KQ, c2w_attn_mask, scale, 0.0f);

            // Attention output
            ggml_tensor * Vt = ggml_permute(ctx, Vcur, 1, 0, 2, 3);
            Vt = ggml_cont(ctx, Vt);
            ggml_tensor * KQV = ggml_mul_mat(ctx, Vt, KQ);

            // Reshape back
            KQV = ggml_permute(ctx, KQV, 0, 2, 1, 3);
            KQV = ggml_cont(ctx, KQV);
            KQV = ggml_reshape_2d(ctx, KQV, c2w_n_embd, seq_len);

            // Output projection
            cur = ggml_mul_mat(ctx, layer.wo, KQV);
        }

        // LayerScale for attention
        if (layer.attn_scale) {
            cur = ggml_mul(ctx, cur, layer.attn_scale);
        }

        // Residual
        cur = ggml_add(ctx, cur, inpSA);
        ggml_tensor * ffn_inp = cur;

        // FFN norm
        if (layer.ffn_norm) {
            cur = build_rms_norm(ctx, cur, layer.ffn_norm, C2W_RMS_NORM_EPS);
        }

        // FFN (SwiGLU)
        if (layer.ffn_gate && layer.ffn_up && layer.ffn_down) {
            ggml_tensor * gate = ggml_mul_mat(ctx, layer.ffn_gate, cur);
            ggml_tensor * up = ggml_mul_mat(ctx, layer.ffn_up, cur);
            gate = ggml_silu(ctx, gate);
            cur = ggml_mul(ctx, gate, up);
            cur = ggml_mul_mat(ctx, layer.ffn_down, cur);
        }

        // LayerScale for FFN
        if (layer.ffn_scale) {
            cur = ggml_mul(ctx, cur, layer.ffn_scale);
        }

        // Residual
        cur = ggml_add(ctx, cur, ffn_inp);
    }

    // Pre-transformer output norm
    if (model->c2w_pre_output_norm) {
        cur = build_rms_norm(ctx, cur, model->c2w_pre_output_norm, C2W_RMS_NORM_EPS);
    }

    // =========================================================================
    // ConvNeXt Upsample: 2 blocks (4x total)
    // =========================================================================
    if (verbose) {
        printf("  Building ConvNeXt upsample (%d blocks)...\n", c2w_up_n_block);
    }

    for (int ib = 0; ib < c2w_up_n_block && ib < (int)model->c2w_up_blocks.size(); ++ib) {
        const auto & block = model->c2w_up_blocks[ib];

        // Transpose convolution for 2x upsampling
        if (block.conv) {
            cur = ggml_cont(ctx, ggml_transpose(ctx, cur));
            cur = ggml_conv_transpose_1d(ctx, block.conv, cur, 2, 0, 1);
            cur = ggml_cont(ctx, ggml_transpose(ctx, cur));

            if (block.conv_bias) {
                ggml_tensor * bias_2d = ggml_reshape_2d(ctx, block.conv_bias, block.conv_bias->ne[0], 1);
                cur = ggml_add(ctx, cur, bias_2d);
            }

            seq_len *= 2;
        }

        // Save input for residual
        ggml_tensor * convnext_residual = cur;

        // Depthwise conv with causal padding
        if (block.dwconv) {
            cur = ggml_cont(ctx, ggml_transpose(ctx, cur));
            int kernel_size = 7;
            int left_pad = kernel_size - 1;
            cur = ggml_pad_ext(ctx, cur, left_pad, 0, 0, 0, 0, 0, 0, 0);

            ggml_tensor * kernel = ggml_permute(ctx, block.dwconv, 2, 1, 0, 3);
            kernel = ggml_cont(ctx, kernel);
            cur = ggml_conv_1d_dw(ctx, kernel, cur, 1, 0, 1);

            cur = ggml_cont(ctx, ggml_transpose(ctx, cur));

            if (block.dwconv_bias) {
                ggml_tensor * bias_2d = ggml_reshape_2d(ctx, block.dwconv_bias, block.dwconv_bias->ne[0], 1);
                cur = ggml_add(ctx, cur, bias_2d);
            }
        }

        // LayerNorm
        if (block.norm) {
            cur = ggml_norm(ctx, cur, 1e-6f);
            ggml_tensor * norm_2d = ggml_reshape_2d(ctx, block.norm, block.norm->ne[0], 1);
            cur = ggml_mul(ctx, cur, norm_2d);

            if (block.norm_bias) {
                ggml_tensor * bias_2d = ggml_reshape_2d(ctx, block.norm_bias, block.norm_bias->ne[0], 1);
                cur = ggml_add(ctx, cur, bias_2d);
            }
        }

        // Pointwise conv 1 + GELU
        if (block.pwconv1) {
            cur = ggml_mul_mat(ctx, block.pwconv1, cur);
            if (block.pwconv1_bias) {
                ggml_tensor * bias_2d = ggml_reshape_2d(ctx, block.pwconv1_bias, block.pwconv1_bias->ne[0], 1);
                cur = ggml_add(ctx, cur, bias_2d);
            }
            cur = ggml_gelu(ctx, cur);
        }

        // Pointwise conv 2
        if (block.pwconv2) {
            cur = ggml_mul_mat(ctx, block.pwconv2, cur);
            if (block.pwconv2_bias) {
                ggml_tensor * bias_2d = ggml_reshape_2d(ctx, block.pwconv2_bias, block.pwconv2_bias->ne[0], 1);
                cur = ggml_add(ctx, cur, bias_2d);
            }
        }

        // LayerScale (gamma)
        if (block.gamma) {
            cur = ggml_mul(ctx, cur, block.gamma);
        }

        // Residual
        cur = ggml_add(ctx, cur, convnext_residual);
    }

    // =========================================================================
    // HiFi-GAN Decoder: 4 stages (480x total)
    // =========================================================================
    if (verbose) {
        printf("  Building HiFi-GAN decoder (%d stages)...\n", c2w_dec_n_stage);
    }

    // Initial conv: 1024 -> 1536
    if (model->c2w_dec_conv_in) {
        cur = ggml_cont(ctx, ggml_transpose(ctx, cur));
        cur = build_causal_conv1d(ctx, model->c2w_dec_conv_in, cur, 1);
        cur = ggml_cont(ctx, ggml_transpose(ctx, cur));
    }

    if (model->c2w_dec_conv_in_b) {
        ggml_tensor * bias_2d = ggml_reshape_2d(ctx, model->c2w_dec_conv_in_b,
                                                model->c2w_dec_conv_in_b->ne[0], 1);
        cur = ggml_add(ctx, cur, bias_2d);
    }

    // 4 upsample stages
    for (int stage = 0; stage < c2w_dec_n_stage && stage < (int)model->c2w_dec_blocks.size(); ++stage) {
        const auto & dec_block = model->c2w_dec_blocks[stage];
        int rate = upsample_rates[stage];

        // Outer Snake activation
        cur = build_snake(ctx, cur, dec_block.snake_alpha, dec_block.snake_beta);

        // Transpose conv for upsampling
        if (dec_block.upsample) {
            cur = ggml_cont(ctx, ggml_transpose(ctx, cur));
            cur = ggml_conv_transpose_1d(ctx, dec_block.upsample, cur, rate, 0, 1);
            cur = ggml_cont(ctx, ggml_transpose(ctx, cur));

            // Trim output (CausalTransConvNet)
            int trim = rate;
            int64_t out_seq = cur->ne[1];
            if (out_seq > 2 * trim) {
                cur = ggml_view_2d(ctx, cur, cur->ne[0], out_seq - 2*trim,
                                   cur->nb[1], trim * cur->nb[1]);
                cur = ggml_cont(ctx, cur);
            }
            seq_len = seq_len * rate - 2 * trim;

            if (dec_block.upsample_bias) {
                ggml_tensor * bias_2d = ggml_reshape_2d(ctx, dec_block.upsample_bias,
                                                        dec_block.upsample_bias->ne[0], 1);
                cur = ggml_add(ctx, cur, bias_2d);
            }
        }

        // 3 residual blocks per stage with dilations (1, 3, 9)
        const int dilations[] = {1, 3, 9};
        for (int rb = 0; rb < c2w_dec_n_resblk; ++rb) {
            int flat_idx = stage * c2w_dec_n_resblk + rb;
            if (flat_idx >= (int)model->c2w_dec_res_blks.size()) break;

            const auto & res_blk = model->c2w_dec_res_blks[flat_idx];
            ggml_tensor * residual = cur;
            int dilation = dilations[rb];

            // Snake1
            cur = build_snake(ctx, cur, res_blk.act1_alpha, res_blk.act1_beta);

            // Conv1 with dilation
            if (res_blk.conv1) {
                cur = ggml_cont(ctx, ggml_transpose(ctx, cur));
                cur = build_causal_conv1d(ctx, res_blk.conv1, cur, dilation);
                cur = ggml_cont(ctx, ggml_transpose(ctx, cur));
                if (res_blk.conv1_bias) {
                    ggml_tensor * bias_2d = ggml_reshape_2d(ctx, res_blk.conv1_bias,
                                                            res_blk.conv1_bias->ne[0], 1);
                    cur = ggml_add(ctx, cur, bias_2d);
                }
            } else if (res_blk.conv) {
                cur = ggml_cont(ctx, ggml_transpose(ctx, cur));
                cur = build_causal_conv1d(ctx, res_blk.conv, cur, dilation);
                cur = ggml_cont(ctx, ggml_transpose(ctx, cur));
            }

            // Snake2
            cur = build_snake(ctx, cur, res_blk.act2_alpha, res_blk.act2_beta);

            // Conv2 (kernel=1)
            if (res_blk.conv2) {
                cur = ggml_cont(ctx, ggml_transpose(ctx, cur));
                cur = ggml_conv_1d_ph(ctx, res_blk.conv2, cur, 1, 1);
                cur = ggml_cont(ctx, ggml_transpose(ctx, cur));
                if (res_blk.conv2_bias) {
                    ggml_tensor * bias_2d = ggml_reshape_2d(ctx, res_blk.conv2_bias,
                                                            res_blk.conv2_bias->ne[0], 1);
                    cur = ggml_add(ctx, cur, bias_2d);
                }
            }

            // Residual connection
            cur = ggml_add(ctx, cur, residual);
        }
    }

    // Final Snake activation
    if (model->c2w_dec_final_snake_a && model->c2w_dec_final_snake_b) {
        cur = build_snake(ctx, cur, model->c2w_dec_final_snake_a, model->c2w_dec_final_snake_b);
    }

    // Final conv: 96 -> 1
    if (model->c2w_dec_conv_out) {
        cur = ggml_cont(ctx, ggml_transpose(ctx, cur));
        cur = build_causal_conv1d(ctx, model->c2w_dec_conv_out, cur, 1);
        cur = ggml_cont(ctx, ggml_transpose(ctx, cur));
        if (model->c2w_dec_conv_out_b) {
            ggml_tensor * bias_2d = ggml_reshape_2d(ctx, model->c2w_dec_conv_out_b,
                                                    model->c2w_dec_conv_out_b->ne[0], 1);
            cur = ggml_add(ctx, cur, bias_2d);
        }
    }

    // Clamp to [-1, 1]
    cur = ggml_clamp(ctx, cur, -1.0f, 1.0f);

    // Ensure contiguous buffer and flatten to 1D for output
    int64_t total_elements = ggml_nelements(cur);
    cur = ggml_cont(ctx, cur);
    cur = ggml_reshape_1d(ctx, cur, total_elements);
    ggml_set_name(cur, "c2w_output");
    ggml_set_output(cur);

    // Mark as output
    ggml_build_forward_expand(gf, cur);

    return cur;
}

// =============================================================================
// Run Code2Wav (Single Chunk)
// =============================================================================

static std::vector<float> run_code2wav_chunk(
    const llama_model * model,
    const std::vector<std::vector<int>> & all_codebook_tokens,
    bool verbose,
    bool cpu_only) {

    const int c2w_n_embd = 1024;
    const int n_frames = all_codebook_tokens.size();
    const int n_codebooks = 16;

    // Total upsampling: 4 (ConvNeXt) x 480 (HiFi-GAN) = 1920
    const int total_upsample = 4 * 8 * 5 * 4 * 3;
    const int n_samples = n_frames * total_upsample;

    if (verbose) {
        printf("Code2Wav: %d frames -> %d samples (%.2f sec @ 24kHz)\n",
               n_frames, n_samples, n_samples / 24000.0f);
    }

    // Initialize context
    mtmd_code2wav_context c2w_ctx;
    if (!mtmd_code2wav_init(&c2w_ctx, cpu_only)) {
        fprintf(stderr, "Error: Failed to initialize Code2Wav context\n");
        return std::vector<float>(n_samples, 0.0f);
    }

    // Step 1: Embed codebook tokens and sum (on CPU)
    std::vector<float> embd_data;
    if (!copy_tensor_to_cpu(model->c2w_code_embd, embd_data)) {
        fprintf(stderr, "Error: Failed to copy Code2Wav embedding\n");
        return std::vector<float>(n_samples, 0.0f);
    }

    // Sum embeddings for each frame
    std::vector<float> input_data(n_frames * c2w_n_embd, 0.0f);
    for (int f = 0; f < n_frames; ++f) {
        for (int cb = 0; cb < n_codebooks; ++cb) {
            int token = all_codebook_tokens[f][cb];
            int vocab_idx = cb * 2048 + (token % 2048);
            for (int i = 0; i < c2w_n_embd; ++i) {
                input_data[f * c2w_n_embd + i] += embd_data[vocab_idx * c2w_n_embd + i];
            }
        }
    }

    // Average across codebooks (HuggingFace: .mean(1))
    for (int f = 0; f < n_frames; ++f) {
        for (int i = 0; i < c2w_n_embd; ++i) {
            input_data[f * c2w_n_embd + i] /= (float)n_codebooks;
        }
    }

    // Step 2: Create ggml context
    struct ggml_init_params params = {
        /*.mem_size   =*/ c2w_ctx.buf_compute_meta.size(),
        /*.mem_buffer =*/ c2w_ctx.buf_compute_meta.data(),
        /*.no_alloc   =*/ true,
    };
    ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        fprintf(stderr, "Error: Failed to create ggml context\n");
        return std::vector<float>(n_samples, 0.0f);
    }

    // Create graph
    ggml_cgraph * gf = ggml_new_graph_custom(ctx, c2w_ctx.max_nodes, false);

    // Create input tensor
    ggml_tensor * input_tensor = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, c2w_n_embd, n_frames);
    ggml_set_name(input_tensor, "c2w_input");
    ggml_set_input(input_tensor);

    // Build compute graph
    ggml_tensor * attn_mask = nullptr;
    ggml_tensor * output = build_code2wav_graph(ctx, gf, model, input_tensor, n_frames, verbose, &attn_mask);

    // Step 3: Allocate graph
    ggml_backend_sched_reset(c2w_ctx.sched);
    if (!ggml_backend_sched_alloc_graph(c2w_ctx.sched, gf)) {
        fprintf(stderr, "Error: Failed to allocate graph\n");
        ggml_free(ctx);
        return std::vector<float>(n_samples, 0.0f);
    }

    // Step 4: Set input data
    ggml_backend_tensor_set(input_tensor, input_data.data(), 0, input_data.size() * sizeof(float));

    // Fill sliding window attention mask
    if (attn_mask) {
        std::vector<float> mask_data(n_frames * n_frames);
        for (int q = 0; q < n_frames; q++) {
            for (int k = 0; k < n_frames; k++) {
                int idx = q * n_frames + k;
                bool masked = (k > q) || (q - k >= C2W_SLIDING_WINDOW);
                mask_data[idx] = masked ? -INFINITY : 0.0f;
            }
        }
        ggml_backend_tensor_set(attn_mask, mask_data.data(), 0, mask_data.size() * sizeof(float));
    }

    // Step 5: Compute graph
    if (ggml_backend_sched_graph_compute(c2w_ctx.sched, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "Error: Graph computation failed\n");
        ggml_free(ctx);
        return std::vector<float>(n_samples, 0.0f);
    }

    // Step 6: Extract output
    int64_t output_size = ggml_nelements(output);
    std::vector<float> audio(output_size);
    ggml_backend_tensor_get(output, audio.data(), 0, output_size * sizeof(float));

    // Replace NaN/Inf with 0
    for (size_t i = 0; i < audio.size(); i++) {
        if (std::isnan(audio[i]) || std::isinf(audio[i])) {
            audio[i] = 0.0f;
        }
    }

    // Normalize audio to use full dynamic range
    float max_abs = 0.0f;
    for (size_t i = 0; i < audio.size(); i++) {
        float abs_val = std::abs(audio[i]);
        if (abs_val > max_abs) max_abs = abs_val;
    }
    if (max_abs > 1e-6f) {
        float scale = 0.9f / max_abs;
        for (size_t i = 0; i < audio.size(); i++) {
            audio[i] *= scale;
        }
    }

    ggml_free(ctx);
    return audio;
}

// =============================================================================
// Public API: Run Code2Wav with Chunked Decode
// =============================================================================

std::vector<float> mtmd_code2wav_run(
    const llama_model * model,
    const std::vector<std::vector<int>> & all_codebook_tokens,
    bool verbose,
    bool cpu_only) {

    const int n_frames = (int)all_codebook_tokens.size();

    // For small inputs, process directly
    if (n_frames <= CODE2WAV_CHUNK_SIZE) {
        return run_code2wav_chunk(model, all_codebook_tokens, verbose, cpu_only);
    }

    // Chunked decode for large inputs
    if (verbose) {
        printf("Code2Wav: Using chunked decode (%d frames, chunk_size=%d)\n",
               n_frames, CODE2WAV_CHUNK_SIZE);
    }

    std::vector<float> full_wav;
    int start_idx = 0;
    int chunk_num = 0;

    while (start_idx < n_frames) {
        int end_idx = std::min(start_idx + CODE2WAV_CHUNK_SIZE, n_frames);
        int context = (start_idx > CODE2WAV_LEFT_CONTEXT) ? CODE2WAV_LEFT_CONTEXT : start_idx;

        // Extract chunk with context
        std::vector<std::vector<int>> chunk(
            all_codebook_tokens.begin() + start_idx - context,
            all_codebook_tokens.begin() + end_idx
        );

        if (verbose) {
            printf("  Chunk %d: frames [%d-%d) with %d context frames\n",
                   chunk_num, start_idx - context, end_idx, context);
        }

        // Process chunk
        std::vector<float> wav_chunk = run_code2wav_chunk(model, chunk, false, cpu_only);

        // Discard context portion and append
        int discard_samples = context * CODE2WAV_TOTAL_UPSAMPLE;
        if (discard_samples < (int)wav_chunk.size()) {
            full_wav.insert(full_wav.end(),
                wav_chunk.begin() + discard_samples,
                wav_chunk.end());
        }

        start_idx = end_idx;
        chunk_num++;
    }

    if (verbose) {
        printf("Code2Wav: Generated %zu total samples from %d chunks\n",
               full_wav.size(), chunk_num);
    }

    return full_wav;
}

// =============================================================================
// Public API: Estimate Output Samples
// =============================================================================

int mtmd_tts_estimate_samples(int n_codec_tokens) {
    // Code2Wav upsamples by 1920x (4 * 8 * 5 * 4 * 3)
    return n_codec_tokens * CODE2WAV_TOTAL_UPSAMPLE;
}

// =============================================================================
// Public API: Write WAV File
// =============================================================================

bool mtmd_tts_write_wav(const char * path, const float * samples, int n_samples, int sample_rate) {
    FILE * f = fopen(path, "wb");
    if (!f) {
        fprintf(stderr, "Error: Cannot create WAV file: %s\n", path);
        return false;
    }

    // Write RIFF header
    fwrite("RIFF", 1, 4, f);
    uint32_t file_size = 36 + n_samples * sizeof(int16_t);
    fwrite(&file_size, sizeof(file_size), 1, f);
    fwrite("WAVE", 1, 4, f);

    // Write fmt chunk
    fwrite("fmt ", 1, 4, f);
    uint32_t fmt_size = 16;
    fwrite(&fmt_size, sizeof(fmt_size), 1, f);
    uint16_t audio_format = 1;  // PCM
    fwrite(&audio_format, sizeof(audio_format), 1, f);
    uint16_t num_channels = 1;  // Mono
    fwrite(&num_channels, sizeof(num_channels), 1, f);
    uint32_t sr = sample_rate;
    fwrite(&sr, sizeof(sr), 1, f);
    uint32_t byte_rate = sample_rate * sizeof(int16_t);
    fwrite(&byte_rate, sizeof(byte_rate), 1, f);
    uint16_t block_align = sizeof(int16_t);
    fwrite(&block_align, sizeof(block_align), 1, f);
    uint16_t bits_per_sample = 16;
    fwrite(&bits_per_sample, sizeof(bits_per_sample), 1, f);

    // Write data chunk
    fwrite("data", 1, 4, f);
    uint32_t data_size = n_samples * sizeof(int16_t);
    fwrite(&data_size, sizeof(data_size), 1, f);

    // Convert float [-1, 1] to int16
    for (int i = 0; i < n_samples; ++i) {
        float s = samples[i];
        if (s > 1.0f) s = 1.0f;
        if (s < -1.0f) s = -1.0f;
        int16_t sample = static_cast<int16_t>(s * 32767.0f);
        fwrite(&sample, sizeof(sample), 1, f);
    }

    fclose(f);
    return true;
}

// =============================================================================
// Public API: Check TTS Support
// =============================================================================

bool mtmd_tts_supported(const llama_model * model) {
    // Check for Code2Wav tensors
    return model->c2w_code_embd != nullptr &&
           model->c2w_dec_conv_in != nullptr &&
           model->c2w_dec_conv_out != nullptr;
}
