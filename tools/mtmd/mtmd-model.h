#pragma once

#include "ggml.h"
#include "ggml-backend.h"
#include "mtmd-impl.h"

#include <algorithm>
#include <array>
#include <map>
#include <vector>
#include <unordered_set>
#include <cstdint>
#include <cmath>

enum mtmd_modality {
    MTMD_MODALITY_VISION,
    MTMD_MODALITY_AUDIO,
    MTMD_MODALITY_GEN_AUDIO,
};

// gen-side voice reference (chatterbox): speech tokens, mel-rate features and the speaker vector
struct mtmd_gen_ref {
    std::vector<int32_t> tokens;
    std::vector<float>   feat;
    std::vector<float>   spk;
};

// a predefined voice: backbone rows, plus the gen-side reference when the gen stack needs one
struct mtmd_gen_voice {
    std::vector<float> rows;
    mtmd_gen_ref       ref;
};

enum ffn_op_type {
    FFN_GELU,
    FFN_GELU_ERF,
    FFN_SILU,
    FFN_GELU_QUICK,
    FFN_RELU_SQR,
};

enum norm_type {
    NORM_TYPE_NORMAL,
    NORM_TYPE_RMS,
};

enum patch_merge_type {
    PATCH_MERGE_FLAT,
    PATCH_MERGE_SPATIAL_UNPAD,
};

// the low bits select the algo, Pillow-compatible (matching PIL.Image.resize output)
// RESIZE_ALGO_FLAG_TORCH matches torchvision's antialiased resize of uint8 images on the CPU (int16 weights), bilinear and bicubic only
enum resize_algo {
    RESIZE_ALGO_BILINEAR,
    RESIZE_ALGO_BICUBIC,
    RESIZE_ALGO_LANCZOS,

    RESIZE_ALGO_FLAG_TORCH = 1 << 8,
};

// Padding style for img_tool::resize
//   PAD_NONE    - no padding; direct resize to target dimensions
//   PAD_CEIL    - aspect-preserving pad (default)
//   PAD_NEAREST - aspect-preserving pad with nearest-integer rounding (Pillow byte-parity)
enum pad_style {
    PAD_NONE,
    PAD_CEIL,
    PAD_NEAREST,
};

struct mtmd_hparams {
    int32_t image_size = 0;
    int32_t patch_size = 0;
    int32_t n_embd = 0;
    int32_t n_ff = 0;
    int32_t projection_dim = 0;
    int32_t n_head = 0;
    int32_t n_head_kv = 0;
    // 0 = derive from n_embd; set when qkv width != n_embd
    int32_t n_embd_head = 0;
    int32_t n_layer = 0;
    int32_t n_merge = 1; // number of patch merges **per-side**

    // for preprocessor
    int32_t image_longest_edge = 0;
    int32_t image_min_pixels = -1;
    int32_t image_max_pixels = -1;
    resize_algo image_resize_algo = RESIZE_ALGO_BICUBIC;
    pad_style image_resize_pad = PAD_CEIL; // padding style when resizing
    bool image_round_half_even = false;    // smart_resize rounds .5 to even, like Python round() in the HF processor
    bool image_floor_before_align = false;
    std::array<uint8_t, 3> image_pad_color = {0, 0, 0};

    // (preprocessor) for llava-uhd style models
    std::vector<mtmd_image_size> image_res_candidates;
    int32_t preproc_min_tiles = 0;
    int32_t preproc_max_tiles = 0;
    int32_t max_slice_nums = 9; // llava-uhd slice cap; per-model, carried in the GGUF
    int32_t preproc_tile_size = 0; // local tile size (deepseek-ocr, lfm2)
    float preproc_max_pixels_tolerance = 1.0f;
    resize_algo image_resize_algo_rf = RESIZE_ALGO_BICUBIC;
    resize_algo image_resize_algo_ov = RESIZE_ALGO_BICUBIC;
    pad_style image_pad_rf = PAD_CEIL;  // padding style for the refined image (e.g. llava-1.6)
    pad_style image_pad_ov = PAD_NONE;  // padding style for the overview image (e.g. llava-1.6)
    std::array<uint8_t, 3> image_pad_color_rf = {0, 0, 0}; // padding color for refined image
    std::array<uint8_t, 3> image_pad_color_ov = {0, 0, 0}; // padding color for overview image

    float image_mean[3];
    float image_std[3];

    // for models using dynamic image size, we need to have a smaller image size to warmup
    // otherwise, user will get OOM every time they load the model
    int32_t warmup_image_size = 0;
    int32_t warmup_audio_size = 3000;

    ffn_op_type ffn_op = FFN_GELU;

    patch_merge_type mm_patch_merge_type = PATCH_MERGE_FLAT;

    float eps = 1e-6;
    float rope_theta = 0.0;

    float swiglu_clamp = 0.0f;
    int32_t n_expert_used = 0;
    std::vector<int32_t> feature_layers;
    int32_t attn_window_size = 0;
    int32_t n_wa_pattern = 0;
    std::unordered_set<int32_t> wa_layer_indexes; // explicit layer indexes that use full attention (for irregular patterns like YoutuVL)
    std::vector<int32_t> wa_pattern_mode; // mimovl: per-layer window-attention mode

    // deepseek4v: resize solver caps the LLM token count of the aligner grid
    int32_t dsv4_max_n_token  = 0;
    int32_t dsv4_max_wh_ratio = 0;

    // deepseek-ocr (sam)
    int32_t sam_n_layer = 0;
    int32_t sam_n_head  = 0;
    int32_t sam_n_embd  = 0;

    // Granite4 Vision
    std::vector<int32_t> proj_spatial_offsets;
    int32_t downsample_query_side;
    int32_t downsample_window_side;

    // Muse Glimmer vision (per-block sparse-window pattern, learned pos-emb, patch-temporal)
    // NOTE: these perhaps shouldn't have the architecture prefix
    int32_t muse_glimmer_patch_temporal = 0;
    int32_t muse_glimmer_sparse_factor  = 0;

    // audio
    int32_t n_mel_bins = 0; // whisper preprocessor
    int32_t proj_stack_factor = 0; // ultravox
    int32_t subsampling_factor = 0; // parakeet

    int32_t audio_chunk_size           = 0;
    int32_t audio_conv_kernel_size     = 0;
    int32_t audio_max_pos_emb          = 0;
    int32_t audio_proj_window_size     = 0;
    int32_t audio_proj_downsample_rate = 0;
    int32_t audio_proj_head_count      = 0;

    // audio pyramid (muse-spark): widths, folded samples and blocks per stage
    std::vector<int32_t> audio_hidden_sizes;
    std::vector<int32_t> audio_patch_sizes;
    std::vector<int32_t> audio_depths;

    int32_t audio_samples_per_token() const {
        int64_t n = 1;
        for (int32_t p : audio_patch_sizes) {
            n *= p;
            GGML_ASSERT(n >= 1 && n <= (1 << 20));
        }
        return (int32_t) n;
    }

    // audio-to-mel preprocessor params
    int32_t audio_chunk_len   = -1; // in seconds
    int32_t audio_sample_rate = -1;
    int32_t audio_n_fft       = -1;
    int32_t audio_window_len  = -1;
    int32_t audio_hop_len     = -1;

    // parakeet
    std::vector<float> mel_filters;
    std::vector<float> window;

    // nemotron transcribe
    int32_t audio_rope_dims = 0;
    int32_t spk_n_embd      = 0;
    int32_t spk_n_layer     = 0;
    int32_t spk_n_head      = 0;

    // mimo-audio-tokenizer: residual vector quantizer
    int32_t rvq_num_quantizers = 0;
    std::vector<int32_t> rvq_codebook_size; // per-quantizer bin count (ragged, e.g. 1024/1024/256/128x17)

    // threshold for the "out_eos_score" graph output
    float gen_eos_threshold = 0.0f;

    // name of the weight variant, some pipelines tune themselves on it
    std::string gen_model_variant;

    // sampling defaults of the generated audio: greedy, no top-k, no repetition penalty
    float   gen_temp           = 0.0f;
    int32_t gen_top_k          = 0;
    float   gen_penalty_repeat = 1.0f;
    int32_t gen_penalty_last_n = 0;

    // pocket-tts
    static constexpr int32_t pockettts_max_spk_seconds = 30;
    int32_t seanet_n_stage    = 0;
    std::vector<int32_t> seanet_ratios; // encoder order (reversed compared to the config)
    int32_t mimi_downsample   = 0;      // encoder frame rate / model frame rate
    int32_t mimi_tfm_context  = 0;      // attention window of the mimi transformers, in frames
    int32_t flow_n_step       = 1;      // lsd_decode steps

    // qwen3tts code2wav
    int32_t wav_tfm_n_layer      = 0;
    int32_t wav_tfm_n_embd       = 0;
    int32_t wav_tfm_n_ff         = 0;
    int32_t wav_tfm_n_head       = 0;
    int32_t wav_tfm_n_head_kv    = 0;
    float   wav_tfm_eps          = 1e-5f;
    float   wav_tfm_rope_theta   = 10000.0f;
    int32_t wav_upsample_n_block = 0;
    int32_t wav_dac_n_block      = 0;
    int32_t wav_dac_n_res        = 0;
    int32_t wav_tfm_swa          = 0; // pre_transformer's KV cache size, in frames
    int32_t wav_frame_stack      = 0; // lfm3a codec: decoder positions per frame

    // mimo-v2.5: LLM-side connector (input_local_transformer)
    int32_t audio_local_n_layer = 0;
    int32_t audio_local_group_size = 0;

    // legacy
    bool has_llava_projector = false;
    int minicpmv_version = 0;
    int32_t minicpmv_query_num = 0;         // MiniCPM-V query number
    int32_t insert_layer_id   = 0;          // MiniCPM-V 4.6 ViT merger insertion layer

    // custom value provided by user, can be undefined if not set
    int32_t custom_image_min_tokens = -1;
    int32_t custom_image_max_tokens = -1;

    void set_image_resize_algo(resize_algo algo, bool torch = false) {
        GGML_ASSERT(!(torch && algo == RESIZE_ALGO_LANCZOS) && "torchvision has no lanczos");
        image_resize_algo = static_cast<resize_algo>(torch ? algo | RESIZE_ALGO_FLAG_TORCH : algo);
    }

    void set_limit_image_tokens(int n_tokens_min, int n_tokens_max) {
        const int patch_area = patch_size * patch_size * n_merge * n_merge;
        image_min_pixels = (custom_image_min_tokens > 0 ? custom_image_min_tokens : n_tokens_min) * patch_area;
        image_max_pixels = (custom_image_max_tokens > 0 ? custom_image_max_tokens : n_tokens_max) * patch_area;
        warmup_image_size = static_cast<int>(std::sqrt(image_max_pixels));
    }

    // used by longest_edge preprocessor (no model-specific value for min/max tokens)
    void set_limit_image_tokens() {
        const int patch_area = patch_size * patch_size * n_merge * n_merge;
        if (custom_image_min_tokens > 0) {
            image_min_pixels = custom_image_min_tokens * patch_area;
        }
        if (custom_image_max_tokens > 0) {
            image_max_pixels = custom_image_max_tokens * patch_area;
        }
    }

    void set_warmup_n_tokens(int n_tokens) {
        int n_tok_per_side = static_cast<int>(std::sqrt(n_tokens));
        GGML_ASSERT(n_tok_per_side * n_tok_per_side == n_tokens && "n_tokens must be n*n");
        // do not warmup with more tokens than the max allowed
        if (custom_image_max_tokens > 0 && n_tokens > custom_image_max_tokens) {
            n_tok_per_side = std::max(1, static_cast<int>(std::sqrt(custom_image_max_tokens)));
        }
        warmup_image_size = n_tok_per_side * patch_size * n_merge;
    }
    // sam vit deepseek-ocr
    std::vector<int32_t> global_attn_indices() const {
        return {  2,  5,  8, 11 };
    }
    bool is_global_attn(int32_t layer) const {
        const auto indices = global_attn_indices();

        for (const auto & idx : indices) {
            if (layer == idx) {
                return true;
            }
        }

        return false;
    }

    bool is_feature_layer(int32_t layer) const {
        return std::find(feature_layers.begin(), feature_layers.end(), layer) != feature_layers.end();
    }
};

struct mtmd_layer {
    // layernorm 1 (or layer input norm, or pre-attention norm)
    ggml_tensor * ln_1_w = nullptr;
    ggml_tensor * ln_1_b = nullptr;

    // attention
    ggml_tensor * k_w = nullptr;
    ggml_tensor * k_b = nullptr;
    ggml_tensor * q_w = nullptr;
    ggml_tensor * q_b = nullptr;
    ggml_tensor * v_w = nullptr;
    ggml_tensor * v_b = nullptr;
    ggml_tensor * qkv_w = nullptr;
    ggml_tensor * qkv_b = nullptr;

    ggml_tensor * o_w = nullptr;
    ggml_tensor * o_b = nullptr;

    ggml_tensor * attn_sinks = nullptr;

    ggml_tensor * k_norm = nullptr;
    ggml_tensor * q_norm = nullptr;
    ggml_tensor * k_norm_b = nullptr;
    ggml_tensor * q_norm_b = nullptr;

    // lfm3: per-head sigmoid gate on the attention output
    ggml_tensor * attn_gate_w = nullptr;

    ggml_tensor * attn_post_norm_w = nullptr;

    ggml_tensor * ff_up_w = nullptr;
    ggml_tensor * ff_up_b = nullptr;
    ggml_tensor * ff_gate_w = nullptr;
    ggml_tensor * ff_gate_b = nullptr;
    ggml_tensor * ff_down_w = nullptr;
    ggml_tensor * ff_down_b = nullptr;

    // MoE FFN (dots3note vision pyramid blocks)
    ggml_tensor * ff_gate_inp_w  = nullptr;
    ggml_tensor * ff_gate_exps_w = nullptr;
    ggml_tensor * ff_up_exps_w   = nullptr;
    ggml_tensor * ff_down_exps_w = nullptr;
    ggml_tensor * ff_exp_probs_b = nullptr;

    // layernorm 2 (or pre-FFN norm)
    ggml_tensor * ln_2_w = nullptr;
    ggml_tensor * ln_2_b = nullptr;

    ggml_tensor * ff_post_norm_w = nullptr;

    // depthwise causal conv block (muse-spark audio), conv_norm_w is declared below
    ggml_tensor * conv_w = nullptr;
    ggml_tensor * conv_b = nullptr;

    // layer scale (no bias)
    ggml_tensor * ls_1_w   = nullptr;
    ggml_tensor * ls_2_w   = nullptr;
    ggml_tensor * ls_out_w = nullptr; // gemma4

    // qwen3vl deepstack merger
    ggml_tensor * deepstack_norm_w = nullptr;
    ggml_tensor * deepstack_norm_b = nullptr;
    ggml_tensor * deepstack_fc1_w = nullptr;
    ggml_tensor * deepstack_fc1_b = nullptr;
    ggml_tensor * deepstack_fc2_w = nullptr;
    ggml_tensor * deepstack_fc2_b = nullptr;

    // lfm3vl conv layers
    ggml_tensor * conv2d_in_proj_w  = nullptr;
    ggml_tensor * conv2d_spatial_w  = nullptr; // may be nullptr
    ggml_tensor * conv2d_out_proj_w = nullptr;

    // sam rel_pos
    ggml_tensor * rel_pos_w = nullptr;
    ggml_tensor * rel_pos_h = nullptr;
    // lfm2
    ggml_tensor * ff_norm_w     = nullptr;
    ggml_tensor * ff_norm_b     = nullptr;
    ggml_tensor * ff_norm_1_w   = nullptr;
    ggml_tensor * ff_norm_1_b   = nullptr;
    ggml_tensor * ff_up_1_w     = nullptr;
    ggml_tensor * ff_up_1_b     = nullptr;
    ggml_tensor * ff_down_1_w   = nullptr;
    ggml_tensor * ff_down_1_b   = nullptr;
    ggml_tensor * pos_bias_u    = nullptr;
    ggml_tensor * pos_bias_v    = nullptr;
    ggml_tensor * norm_conv_w   = nullptr;
    ggml_tensor * norm_conv_b   = nullptr;
    ggml_tensor * linear_pos_w  = nullptr;

    ggml_tensor * conv_norm_w    = nullptr;
    ggml_tensor * conv_norm_b    = nullptr;
    ggml_tensor * conv_norm_mean = nullptr;  // parakeet
    ggml_tensor * conv_norm_var  = nullptr;  // parakeet
    ggml_tensor * conv_dw_w      = nullptr;
    ggml_tensor * conv_dw_b      = nullptr;
    ggml_tensor * conv_pw1_w     = nullptr;
    ggml_tensor * conv_pw1_b     = nullptr;
    ggml_tensor * conv_pw2_w     = nullptr;
    ggml_tensor * conv_pw2_b     = nullptr;

    // gemma4 audio conformer per-layer
    ggml_tensor * attn_pre_norm_w   = nullptr;
    ggml_tensor * attn_k_rel_w      = nullptr;
    ggml_tensor * per_dim_scale_w   = nullptr;
    ggml_tensor * per_dim_k_scale_w = nullptr;
    ggml_tensor * ff_post_norm_1_w  = nullptr;

    // granite_speech conformer per-layer
    ggml_tensor * attn_rel_pos_emb = nullptr;

    // granite_speech qformer cross-attention
    ggml_tensor * cross_attn_q_w    = nullptr;
    ggml_tensor * cross_attn_q_b    = nullptr;
    ggml_tensor * cross_attn_k_w    = nullptr;
    ggml_tensor * cross_attn_k_b    = nullptr;
    ggml_tensor * cross_attn_v_w    = nullptr;
    ggml_tensor * cross_attn_v_b    = nullptr;
    ggml_tensor * cross_attn_o_w    = nullptr;
    ggml_tensor * cross_attn_o_b    = nullptr;
    ggml_tensor * cross_attn_norm_w = nullptr;
    ggml_tensor * cross_attn_norm_b = nullptr;

    // qwen3tts speaker encoder: SE-Res2Net block, tdnn1/tdnn2 reuse conv_pw1_w/b and conv_pw2_w/b above
    ggml_tensor * se_conv1_w = nullptr;
    ggml_tensor * se_conv1_b = nullptr;
    ggml_tensor * se_conv2_w = nullptr;
    ggml_tensor * se_conv2_b = nullptr;
    std::vector<ggml_tensor *> res2_conv_w; // Res2Net hierarchical branches
    std::vector<ggml_tensor *> res2_conv_b;

    bool has_deepstack() const {
        return deepstack_fc1_w != nullptr;
    }
};

// Expanded MobileNetV5 block structure for Gemma3n vision encoder
struct mobilenetv5_block {
    // Stage 0 (Edge Residual)
    ggml_tensor * s0_conv_exp_w = nullptr;
    ggml_tensor * s0_bn1_w      = nullptr;
    ggml_tensor * s0_conv_pwl_w = nullptr;
    ggml_tensor * s0_bn2_w      = nullptr;

    // Stage 1+ (Universal Inverted Residual)
    ggml_tensor * dw_start_w    = nullptr;
    ggml_tensor * dw_start_bn_w = nullptr;

    ggml_tensor * pw_exp_w      = nullptr;
    ggml_tensor * pw_exp_bn_w   = nullptr;

    ggml_tensor * dw_mid_w      = nullptr;
    ggml_tensor * dw_mid_bn_w   = nullptr;

    ggml_tensor * pw_proj_w     = nullptr;
    ggml_tensor * pw_proj_bn_w  = nullptr;

    ggml_tensor * layer_scale_w = nullptr;

    // Attention (MQA) components
    ggml_tensor * attn_q_w = nullptr;
    ggml_tensor * attn_k_w = nullptr;
    ggml_tensor * attn_v_w = nullptr;
    ggml_tensor * attn_o_w = nullptr;

    // Optional downsampling/norm in attention
    ggml_tensor * attn_k_dw_w   = nullptr;
    ggml_tensor * attn_k_norm_w = nullptr;
    ggml_tensor * attn_v_dw_w   = nullptr;
    ggml_tensor * attn_v_norm_w = nullptr;

    // Block norm (often present in attention blocks)
    ggml_tensor * attn_norm_w   = nullptr;
};

struct yasa2_block {
    ggml_tensor * dw_w  = nullptr;
    ggml_tensor * dw_b  = nullptr;
    ggml_tensor * ln_w  = nullptr;
    ggml_tensor * ln_b  = nullptr;
    ggml_tensor * pw1_w = nullptr;
    ggml_tensor * pw1_b = nullptr;
    ggml_tensor * grn_w = nullptr;
    ggml_tensor * grn_b = nullptr;
    ggml_tensor * pw2_w = nullptr;
    ggml_tensor * pw2_b = nullptr;
};

struct yasa2_stage {
    ggml_tensor * down_ln_w   = nullptr;
    ggml_tensor * down_ln_b   = nullptr;
    ggml_tensor * down_conv_w = nullptr;
    ggml_tensor * down_conv_b = nullptr;
    std::vector<yasa2_block> blocks;
};

// QFormer projector block for models with 1 (or more) QFormer projectors
// Granite Speech, Granite4 Vision
struct qf_block {
    ggml_tensor * qf_proj_query       = nullptr;
    ggml_tensor * qf_proj_norm_w      = nullptr;
    ggml_tensor * qf_proj_norm_b      = nullptr;
    ggml_tensor * qf_proj_linear_w    = nullptr;
    ggml_tensor * qf_proj_linear_b    = nullptr;
    ggml_tensor * qf_proj_post_norm_w = nullptr;
    ggml_tensor * qf_proj_post_norm_b = nullptr;
    ggml_tensor * qf_proj_img_pos     = nullptr; // Vision only
    std::vector<mtmd_layer> qf_proj_layers;
};

// pocket-tts SEANet stack, used in both directions:
// encoder = conv_in -> per stage (residual unit, strided conv) -> conv_out
// decoder = conv_in -> per stage (strided convtr, residual unit) -> conv_out
struct mtmd_seanet {
    // one residual unit: ELU -> dilated conv -> ELU -> pointwise conv, added to the input
    struct stage {
        ggml_tensor * res_conv1_w = nullptr;
        ggml_tensor * res_conv1_b = nullptr;
        ggml_tensor * res_conv2_w = nullptr;
        ggml_tensor * res_conv2_b = nullptr;
        ggml_tensor * scale_conv_w = nullptr; // strided conv (encoder) or convtr (decoder)
        ggml_tensor * scale_conv_b = nullptr;
    };

    ggml_tensor * conv_in_w  = nullptr;
    ggml_tensor * conv_in_b  = nullptr;
    ggml_tensor * conv_out_w = nullptr;
    ggml_tensor * conv_out_b = nullptr;
    std::vector<stage> stages;
};

// pocket-tts flow-matching decoder (SimpleMLPAdaLN)
struct mtmd_flow_net {
    // AdaLN res block: in_ln -> modulate -> Linear -> SiLU -> Linear, gated residual
    struct block {
        ggml_tensor * norm_w = nullptr;
        ggml_tensor * norm_b = nullptr;
        ggml_tensor * up_w   = nullptr;
        ggml_tensor * up_b   = nullptr;
        ggml_tensor * down_w = nullptr;
        ggml_tensor * down_b = nullptr;
        ggml_tensor * ada_w  = nullptr; // -> shift, scale, gate
        ggml_tensor * ada_b  = nullptr;
    };

    // timestep embedder: cos/sin(t * freqs) -> Linear -> SiLU -> Linear -> RMSNorm
    struct time_embd {
        ggml_tensor * freqs  = nullptr;
        ggml_tensor * up_w   = nullptr;
        ggml_tensor * up_b   = nullptr;
        ggml_tensor * down_w = nullptr;
        ggml_tensor * down_b = nullptr;
        ggml_tensor * norm   = nullptr; // RMSNorm alpha
    };

    ggml_tensor * input_proj_w = nullptr;
    ggml_tensor * input_proj_b = nullptr;
    ggml_tensor * cond_embd_w  = nullptr;
    ggml_tensor * cond_embd_b  = nullptr;
    ggml_tensor * final_ada_w  = nullptr; // -> shift, scale
    ggml_tensor * final_ada_b  = nullptr;
    ggml_tensor * final_proj_w = nullptr;
    ggml_tensor * final_proj_b = nullptr;
    std::vector<time_embd> time;
    std::vector<block> blocks;
};

// qwen3tts code2wav: RVQ codes -> raw PCM
struct mtmd_code2wav {
    // "upsample" stage: one ConvNeXt block plus the causal ConvTranspose1d before it
    struct upsample_block {
        ggml_tensor * conv_w   = nullptr; // causal ConvTranspose1d, 2x
        ggml_tensor * conv_b   = nullptr;
        ggml_tensor * dwconv_w = nullptr; // depthwise causal conv, k=7
        ggml_tensor * dwconv_b = nullptr;
        ggml_tensor * norm_w   = nullptr; // LayerNorm
        ggml_tensor * norm_b   = nullptr;
        ggml_tensor * pw1_w    = nullptr; // pointwise expand
        ggml_tensor * pw1_b    = nullptr;
        ggml_tensor * pw2_w    = nullptr; // pointwise project
        ggml_tensor * pw2_b    = nullptr;
        ggml_tensor * gamma    = nullptr; // layer scale
    };

    // one DAC residual unit: SnakeBeta -> dilated causal conv -> SnakeBeta -> pointwise causal conv
    struct dac_res {
        ggml_tensor * act1_alpha = nullptr;
        ggml_tensor * act1_beta  = nullptr;
        ggml_tensor * conv1_w    = nullptr;
        ggml_tensor * conv1_b    = nullptr;
        ggml_tensor * act2_alpha = nullptr;
        ggml_tensor * act2_beta  = nullptr;
        ggml_tensor * conv2_w    = nullptr;
        ggml_tensor * conv2_b    = nullptr;
    };

    // one DAC upsample block (SnakeBeta -> causal ConvTranspose1d -> 3 residual units)
    struct dac_block {
        ggml_tensor * snake_alpha = nullptr;
        ggml_tensor * snake_beta  = nullptr;
        ggml_tensor * conv_w      = nullptr; // causal ConvTranspose1d
        ggml_tensor * conv_b      = nullptr;
        std::vector<dac_res> res;
    };

    // quantizer: RVQ codebook decode
    ggml_tensor * quant_first_in_w  = nullptr; // semantic RVQ, in_proj (1x1 conv, loaded as 2D)
    ggml_tensor * quant_first_out_w = nullptr;
    ggml_tensor * quant_first_out_b = nullptr;
    ggml_tensor * quant_first_cb_w  = nullptr; // codebook (1 layer)
    ggml_tensor * quant_rest_in_w   = nullptr; // acoustic RVQ
    ggml_tensor * quant_rest_out_w  = nullptr;
    ggml_tensor * quant_rest_cb_w   = nullptr; // codebooks, merged 3D [15, vocab, dim]

    ggml_tensor * pre_conv_w = nullptr;
    ggml_tensor * pre_conv_b = nullptr;

    ggml_tensor * tfm_in_proj_w     = nullptr;
    ggml_tensor * tfm_in_proj_b     = nullptr;
    ggml_tensor * tfm_out_proj_w    = nullptr;
    ggml_tensor * tfm_out_proj_b    = nullptr;
    ggml_tensor * tfm_output_norm_w = nullptr;
    std::vector<mtmd_layer> tfm_layers; // reuses the generic block fields (ln_1/attn/ln_2/ffn/ls_1/ls_2)

    std::vector<upsample_block> upsample;

    ggml_tensor * dac_entry_w = nullptr;
    ggml_tensor * dac_entry_b = nullptr;
    std::vector<dac_block> dac;
    ggml_tensor * dac_post_snake_alpha = nullptr;
    ggml_tensor * dac_post_snake_beta  = nullptr;
    ggml_tensor * dac_post_conv_w      = nullptr;
    ggml_tensor * dac_post_conv_b      = nullptr;
};

struct mtmd_chatterbox {
    // espnet rel-pos conformer layer of the flow encoder
    struct enc_layer {
        ggml_tensor * norm_mha_w = nullptr;
        ggml_tensor * norm_mha_b = nullptr;
        ggml_tensor * attn_q_w = nullptr;
        ggml_tensor * attn_q_b = nullptr;
        ggml_tensor * attn_k_w = nullptr;
        ggml_tensor * attn_k_b = nullptr;
        ggml_tensor * attn_v_w = nullptr;
        ggml_tensor * attn_v_b = nullptr;
        ggml_tensor * attn_pos_w = nullptr;      // linear_pos, no bias
        ggml_tensor * attn_pos_bias_u = nullptr;
        ggml_tensor * attn_pos_bias_v = nullptr;
        ggml_tensor * attn_out_w = nullptr;
        ggml_tensor * attn_out_b = nullptr;
        ggml_tensor * norm_ff_w = nullptr;
        ggml_tensor * norm_ff_b = nullptr;
        ggml_tensor * ffn_1_w = nullptr;
        ggml_tensor * ffn_1_b = nullptr;
        ggml_tensor * ffn_2_w = nullptr;
        ggml_tensor * ffn_2_b = nullptr;
    };

    // causal conv block of the estimator (conv k3 left padded + layer norm)
    struct causal_block {
        ggml_tensor * conv_w = nullptr;
        ggml_tensor * conv_b = nullptr;
        ggml_tensor * norm_w = nullptr;
        ggml_tensor * norm_b = nullptr;
    };

    // time conditioned resnet of the estimator
    struct resnet {
        causal_block block1;
        causal_block block2;
        ggml_tensor * mlp_w = nullptr;      // time projection (mlp.1)
        ggml_tensor * mlp_b = nullptr;
        ggml_tensor * res_conv_w = nullptr;
        ggml_tensor * res_conv_b = nullptr;
    };

    // diffusers style transformer block of the estimator
    struct tfm_block {
        ggml_tensor * norm1_w = nullptr;
        ggml_tensor * norm1_b = nullptr;
        ggml_tensor * attn_q_w = nullptr;   // no bias on qkv
        ggml_tensor * attn_k_w = nullptr;
        ggml_tensor * attn_v_w = nullptr;
        ggml_tensor * attn_out_w = nullptr; // to_out.0
        ggml_tensor * attn_out_b = nullptr;
        ggml_tensor * norm3_w = nullptr;
        ggml_tensor * norm3_b = nullptr;
        ggml_tensor * ff_in_w = nullptr;    // ff.net.0.proj
        ggml_tensor * ff_in_b = nullptr;
        ggml_tensor * ff_out_w = nullptr;   // ff.net.2
        ggml_tensor * ff_out_b = nullptr;
    };

    // one estimator stage: resnet, transformer stack, boundary causal conv (null on mid stages)
    struct est_stage {
        resnet res;
        std::vector<tfm_block> tfm;
        ggml_tensor * conv_w = nullptr;
        ggml_tensor * conv_b = nullptr;
    };

    // snake resblock unit of the hift vocoder
    struct hift_res_unit {
        ggml_tensor * act1_alpha = nullptr;
        ggml_tensor * act2_alpha = nullptr;
        ggml_tensor * conv1_w = nullptr;
        ggml_tensor * conv1_b = nullptr;
        ggml_tensor * conv2_w = nullptr;
        ggml_tensor * conv2_b = nullptr;
    };
    struct hift_res {
        std::vector<hift_res_unit> units;   // dilations 1/3/5 on conv1
    };

    // one hift upsample stage: conv transpose, source injection, 3 resblocks
    struct hift_up {
        ggml_tensor * up_w = nullptr;
        ggml_tensor * up_b = nullptr;
        ggml_tensor * source_down_w = nullptr;
        ggml_tensor * source_down_b = nullptr;
        hift_res source_res;
        hift_res res_0;
        hift_res res_1;
        hift_res res_2;
    };

    // f0 predictor conv of the hift vocoder
    struct f0_conv {
        ggml_tensor * w = nullptr;
        ggml_tensor * b = nullptr;
    };

    // s3 tokenizer attention block (neox rope on q/k, fsmn memory on v)
    struct s3tok_block {
        ggml_tensor * attn_ln_w = nullptr;
        ggml_tensor * attn_ln_b = nullptr;
        ggml_tensor * attn_q_w = nullptr;
        ggml_tensor * attn_q_b = nullptr;
        ggml_tensor * attn_k_w = nullptr;   // no bias
        ggml_tensor * attn_v_w = nullptr;
        ggml_tensor * attn_v_b = nullptr;
        ggml_tensor * fsmn_w = nullptr;
        ggml_tensor * attn_out_w = nullptr;
        ggml_tensor * attn_out_b = nullptr;
        ggml_tensor * mlp_ln_w = nullptr;
        ggml_tensor * mlp_ln_b = nullptr;
        ggml_tensor * mlp_in_w = nullptr;
        ggml_tensor * mlp_in_b = nullptr;
        ggml_tensor * mlp_out_w = nullptr;
        ggml_tensor * mlp_out_b = nullptr;
    };

    // folded batchnorm (w/b stay null on the affine free variant)
    struct bn {
        ggml_tensor * mean = nullptr;
        ggml_tensor * var  = nullptr;
        ggml_tensor * w    = nullptr;
        ggml_tensor * b    = nullptr;
    };

    // fcm residual 2d block of the speaker encoder, the shortcut is null on stride 1 blocks
    struct spk_res2d {
        ggml_tensor * conv1_w = nullptr;
        bn bn1;
        ggml_tensor * conv2_w = nullptr;
        bn bn2;
        ggml_tensor * shortcut_w = nullptr;
        bn shortcut_bn;
    };

    // cam dense tdnn layer of the speaker encoder
    struct spk_cam_layer {
        bn nl1_bn;
        ggml_tensor * linear1_w = nullptr;
        bn nl2_bn;
        ggml_tensor * local_w = nullptr;    // cam_layer.linear_local
        ggml_tensor * ctx1_w = nullptr;     // cam_layer.linear1
        ggml_tensor * ctx1_b = nullptr;
        ggml_tensor * ctx2_w = nullptr;     // cam_layer.linear2
        ggml_tensor * ctx2_b = nullptr;
    };
    struct spk_cam_block {
        std::vector<spk_cam_layer> layers;
        bn transit_bn;
        ggml_tensor * transit_w = nullptr;
    };

    // flow encoder
    ggml_tensor * input_embedding_w = nullptr;  // flow.input_embedding
    ggml_tensor * embed_linear_w = nullptr;     // fenc.embed.out.0
    ggml_tensor * embed_linear_b = nullptr;
    ggml_tensor * embed_norm_w = nullptr;       // fenc.embed.out.1
    ggml_tensor * embed_norm_b = nullptr;
    ggml_tensor * pre_conv1_w = nullptr;        // pre_lookahead_layer
    ggml_tensor * pre_conv1_b = nullptr;
    ggml_tensor * pre_conv2_w = nullptr;
    ggml_tensor * pre_conv2_b = nullptr;
    std::vector<enc_layer> enc;
    ggml_tensor * up_conv_w = nullptr;          // fenc.up_layer.conv
    ggml_tensor * up_conv_b = nullptr;
    ggml_tensor * up_embed_linear_w = nullptr;  // fenc.up_embed.out.0
    ggml_tensor * up_embed_linear_b = nullptr;
    ggml_tensor * up_embed_norm_w = nullptr;    // fenc.up_embed.out.1
    ggml_tensor * up_embed_norm_b = nullptr;
    std::vector<enc_layer> up_enc;
    ggml_tensor * after_norm_w = nullptr;
    ggml_tensor * after_norm_b = nullptr;
    ggml_tensor * encoder_proj_w = nullptr;     // flow.encoder_proj
    ggml_tensor * encoder_proj_b = nullptr;

    // cfm estimator
    ggml_tensor * time_mlp_1_w = nullptr;
    ggml_tensor * time_mlp_1_b = nullptr;
    ggml_tensor * time_mlp_2_w = nullptr;
    ggml_tensor * time_mlp_2_b = nullptr;
    ggml_tensor * time_embed_mixer_w = nullptr; // meanflow variant only
    est_stage est_down;
    std::vector<est_stage> est_mid;
    est_stage est_up;
    causal_block est_final_block;
    ggml_tensor * est_final_proj_w = nullptr;
    ggml_tensor * est_final_proj_b = nullptr;

    // hift vocoder
    ggml_tensor * hift_pre_w = nullptr;
    ggml_tensor * hift_pre_b = nullptr;
    std::vector<hift_up> hift_ups;
    ggml_tensor * hift_post_w = nullptr;
    ggml_tensor * hift_post_b = nullptr;
    std::vector<f0_conv> f0_condnet;
    ggml_tensor * f0_classifier_w = nullptr;
    ggml_tensor * f0_classifier_b = nullptr;

    // s3 tokenizer
    ggml_tensor * s3tok_conv1_w = nullptr;
    ggml_tensor * s3tok_conv1_b = nullptr;
    ggml_tensor * s3tok_conv2_w = nullptr;
    ggml_tensor * s3tok_conv2_b = nullptr;
    std::vector<s3tok_block> s3tok_blocks;
    ggml_tensor * s3tok_down_w = nullptr;       // quantizer._codebook.project_down
    ggml_tensor * s3tok_down_b = nullptr;

    // CAMPPlus speaker encoder
    ggml_tensor * spk_conv1_w = nullptr;
    bn spk_bn1;
    spk_res2d spk_layer1_0;
    spk_res2d spk_layer1_1;
    spk_res2d spk_layer2_0;
    spk_res2d spk_layer2_1;
    ggml_tensor * spk_conv2_w = nullptr;
    bn spk_bn2;
    ggml_tensor * spk_tdnn_w = nullptr;
    bn spk_tdnn_bn;
    spk_cam_block spk_block1;
    spk_cam_block spk_block2;
    spk_cam_block spk_block3;
    bn spk_out_bn;
    ggml_tensor * spk_dense_w = nullptr;
    bn spk_dense_bn;
    ggml_tensor * spk_affine_w = nullptr;
    ggml_tensor * spk_affine_b = nullptr;
};

// drama3 tower: extra tensors around the generic block
struct mtmd_drama3_layer {
    ggml_tensor * enorm_w    = nullptr;
    ggml_tensor * hnorm_w    = nullptr;
    ggml_tensor * eh_proj_w  = nullptr; // [code embd; hidden] -> hidden
    ggml_tensor * embd_w     = nullptr;
    ggml_tensor * out_norm_w = nullptr;
};

struct mtmd_vocos {
    ggml_tensor * input_w        = nullptr;
    ggml_tensor * input_b        = nullptr;
    ggml_tensor * norm_w         = nullptr;
    ggml_tensor * norm_b         = nullptr;
    ggml_tensor * output_norm_w  = nullptr;
    ggml_tensor * output_norm_b  = nullptr;
    ggml_tensor * output_w       = nullptr;
    ggml_tensor * output_b       = nullptr;
    std::vector<mtmd_code2wav::upsample_block> blocks;
};

// the weights and hparams of one modality of an mmproj file, read-only once loaded
struct mtmd_model {
    mtmd_modality modality = MTMD_MODALITY_VISION;
    projector_type proj_type = PROJECTOR_TYPE_MLP;
    mtmd_hparams hparams;

    // runtime requirements declared by the model, applied by mtmd_subcontext
    int max_nodes        = 8192;
    bool flash_attn_auto = true; // false: MTMD_FLASH_ATTN_TYPE_AUTO resolves to disabled

    // weights
    ggml_context_ptr        ctx_data;
    ggml_backend_buffer_ptr buf;
    std::map<ggml_backend_dev_t, size_t> mem_weights;

    // embeddings
    ggml_tensor * class_embedding = nullptr;
    ggml_tensor * patch_embeddings_0 = nullptr;
    ggml_tensor * patch_embeddings_1 = nullptr;  // additional patch kernel; interpretation is model-specific (Qwen2VL, Nemotron Omni)
    ggml_tensor * patch_bias = nullptr;
    ggml_tensor * position_embeddings = nullptr;
    ggml_tensor * rope_freqs = nullptr;
    ggml_tensor * norm_embd_w = nullptr;
    ggml_tensor * norm_embd_b = nullptr;

    // "indexed" patch embedding norms
    ggml_tensor * patch_norm_1_w = nullptr;
    ggml_tensor * patch_norm_1_b = nullptr;
    ggml_tensor * patch_norm_2_w = nullptr;
    ggml_tensor * patch_norm_2_b = nullptr;
    ggml_tensor * patch_norm_3_w = nullptr;
    ggml_tensor * patch_norm_3_b = nullptr;

    ggml_tensor * pre_ln_w = nullptr;
    ggml_tensor * pre_ln_b = nullptr;

    std::vector<mtmd_layer> layers;

    // one patch embedder per audio pyramid stage (muse-spark)
    struct audio_patch {
        ggml_tensor * proj_w = nullptr;
        ggml_tensor * proj_b = nullptr;
        ggml_tensor * norm_w = nullptr;
    };
    std::vector<audio_patch> a_patches;

    int32_t n_deepstack_layers = 0; // used by Qwen3-VL, calculated from mtmd_layer

    ggml_tensor * post_ln_w;
    ggml_tensor * post_ln_b;

    ggml_tensor * mm_fc_w;
    ggml_tensor * mm_fc_b;
    ggml_tensor * mm_ffn_up_w = nullptr;
    ggml_tensor * mm_ffn_up_b = nullptr;
    ggml_tensor * mm_ffn_gate_w = nullptr;
    ggml_tensor * mm_ffn_gate_b = nullptr;
    ggml_tensor * mm_ffn_down_w = nullptr;
    ggml_tensor * mm_ffn_down_b = nullptr;
    ggml_tensor * mm_post_norm_w = nullptr;
    ggml_tensor * mm_post_norm_b = nullptr;

    // LLaVA projection
    ggml_tensor * mm_input_norm_w = nullptr;
    ggml_tensor * mm_input_norm_b = nullptr;
    ggml_tensor * mm_0_w = nullptr;
    ggml_tensor * mm_0_b = nullptr;
    ggml_tensor * mm_2_w = nullptr;
    ggml_tensor * mm_2_b = nullptr;
    ggml_tensor * mm_merger_fc1_w = nullptr;   // minimax-m3
    ggml_tensor * mm_merger_fc1_b = nullptr;
    ggml_tensor * mm_merger_fc2_w = nullptr;
    ggml_tensor * mm_merger_fc2_b = nullptr;

    ggml_tensor * image_newline = nullptr;
    ggml_tensor * view_seperator = nullptr;
    ggml_tensor * tok_embd_sep = nullptr; // [n_embd_text, n_sep] rows of the text model tok_embd (MiniCPM-V 4.7)


    // Yi type models with mlp+normalization projection
    ggml_tensor * mm_1_w = nullptr; // Yi type models have 0, 1, 3, 4
    ggml_tensor * mm_1_b = nullptr;
    ggml_tensor * mm_3_w = nullptr;
    ggml_tensor * mm_3_b = nullptr;
    ggml_tensor * mm_4_w = nullptr;
    ggml_tensor * mm_4_b = nullptr;
    ggml_tensor * mm_5_w = nullptr;
    ggml_tensor * mm_5_b = nullptr;
    ggml_tensor * mm_6_w = nullptr;
    ggml_tensor * mm_6_b = nullptr;

    // GLMV-Edge projection
    ggml_tensor * mm_model_adapter_conv_w = nullptr;
    ggml_tensor * mm_model_adapter_conv_b = nullptr;

    // MobileVLM projection
    ggml_tensor * mm_model_mlp_1_w = nullptr;
    ggml_tensor * mm_model_mlp_1_b = nullptr;
    ggml_tensor * mm_model_mlp_3_w = nullptr;
    ggml_tensor * mm_model_mlp_3_b = nullptr;
    ggml_tensor * mm_model_block_1_block_0_0_w = nullptr;
    ggml_tensor * mm_model_block_1_block_0_1_w = nullptr;
    ggml_tensor * mm_model_block_1_block_0_1_b = nullptr;
    ggml_tensor * mm_model_block_1_block_1_fc1_w = nullptr;
    ggml_tensor * mm_model_block_1_block_1_fc1_b = nullptr;
    ggml_tensor * mm_model_block_1_block_1_fc2_w = nullptr;
    ggml_tensor * mm_model_block_1_block_1_fc2_b = nullptr;
    ggml_tensor * mm_model_block_1_block_2_0_w = nullptr;
    ggml_tensor * mm_model_block_1_block_2_1_w = nullptr;
    ggml_tensor * mm_model_block_1_block_2_1_b = nullptr;
    ggml_tensor * mm_model_block_2_block_0_0_w = nullptr;
    ggml_tensor * mm_model_block_2_block_0_1_w = nullptr;
    ggml_tensor * mm_model_block_2_block_0_1_b = nullptr;
    ggml_tensor * mm_model_block_2_block_1_fc1_w = nullptr;
    ggml_tensor * mm_model_block_2_block_1_fc1_b = nullptr;
    ggml_tensor * mm_model_block_2_block_1_fc2_w = nullptr;
    ggml_tensor * mm_model_block_2_block_1_fc2_b = nullptr;
    ggml_tensor * mm_model_block_2_block_2_0_w = nullptr;
    ggml_tensor * mm_model_block_2_block_2_1_w = nullptr;
    ggml_tensor * mm_model_block_2_block_2_1_b = nullptr;

    // MobileVLM_V2 projection
    ggml_tensor * mm_model_mlp_0_w = nullptr;
    ggml_tensor * mm_model_mlp_0_b = nullptr;
    ggml_tensor * mm_model_mlp_2_w = nullptr;
    ggml_tensor * mm_model_mlp_2_b = nullptr;
    ggml_tensor * mm_model_peg_0_w = nullptr;
    ggml_tensor * mm_model_peg_0_b = nullptr;

    // MINICPMV projection
    ggml_tensor * mm_model_pos_embed_k = nullptr;
    ggml_tensor * mm_model_query = nullptr;
    ggml_tensor * mm_model_proj   = nullptr;
    ggml_tensor * mm_model_proj_b = nullptr;
    ggml_tensor * mm_model_kv_proj = nullptr;
    ggml_tensor * mm_model_attn_q_w = nullptr;
    ggml_tensor * mm_model_attn_q_b = nullptr;
    ggml_tensor * mm_model_attn_k_w = nullptr;
    ggml_tensor * mm_model_attn_k_b = nullptr;
    ggml_tensor * mm_model_attn_v_w = nullptr;
    ggml_tensor * mm_model_attn_v_b = nullptr;
    ggml_tensor * mm_model_attn_o_w = nullptr;
    ggml_tensor * mm_model_attn_o_b = nullptr;
    ggml_tensor * mm_model_ln_q_w = nullptr;
    ggml_tensor * mm_model_ln_q_b = nullptr;
    ggml_tensor * mm_model_ln_kv_w = nullptr;
    ggml_tensor * mm_model_ln_kv_b = nullptr;
    ggml_tensor * mm_model_ln_post_w = nullptr;
    ggml_tensor * mm_model_ln_post_b = nullptr;

    // MiniCPM-V 4.6 ViT merger (window self-attention + ViT MLP downsample)
    ggml_tensor * vit_merger_ln1_w     = nullptr;
    ggml_tensor * vit_merger_ln1_b     = nullptr;
    ggml_tensor * vit_merger_attn_q_w  = nullptr;
    ggml_tensor * vit_merger_attn_q_b  = nullptr;
    ggml_tensor * vit_merger_attn_k_w  = nullptr;
    ggml_tensor * vit_merger_attn_k_b  = nullptr;
    ggml_tensor * vit_merger_attn_v_w  = nullptr;
    ggml_tensor * vit_merger_attn_v_b  = nullptr;
    ggml_tensor * vit_merger_attn_o_w  = nullptr;
    ggml_tensor * vit_merger_attn_o_b  = nullptr;
    ggml_tensor * vit_merger_ds_ln_w   = nullptr;
    ggml_tensor * vit_merger_ds_ln_b   = nullptr;
    ggml_tensor * vit_merger_ds_up_w   = nullptr;
    ggml_tensor * vit_merger_ds_up_b   = nullptr;
    ggml_tensor * vit_merger_ds_down_w = nullptr;
    ggml_tensor * vit_merger_ds_down_b = nullptr;

    // gemma3
    ggml_tensor * mm_input_proj_w = nullptr;
    ggml_tensor * mm_soft_emb_norm_w = nullptr;

    // mobilenetv5 for gemma3n
    std::vector<mobilenetv5_block> mobilenet_blocks;
    std::vector<int> mobilenet_stage_ends;
    ggml_tensor * mobilenet_stem_conv_w = nullptr;
    ggml_tensor * mobilenet_stem_conv_b = nullptr;
    ggml_tensor * mobilenet_stem_norm_w = nullptr;
    ggml_tensor * mm_post_proj_norm_w = nullptr;

    // Multi-Scale Fusion Adapter (MSFA) components
    ggml_tensor * msfa_concat_conv_w = nullptr;
    ggml_tensor * msfa_concat_norm_w = nullptr;
    ggml_tensor * msfa_ffn_expand_w = nullptr;
    ggml_tensor * msfa_ffn_project_w = nullptr;
    ggml_tensor * msfa_ffn_expand_bn = nullptr;
    ggml_tensor * msfa_ffn_project_bn = nullptr;

    // yasa2
    ggml_tensor * yasa_patch_w = nullptr;
    ggml_tensor * yasa_patch_b = nullptr;
    ggml_tensor * yasa_patch_ln_w = nullptr;
    ggml_tensor * yasa_patch_ln_b = nullptr;
    ggml_tensor * yasa_backbone_ln_w = nullptr;
    ggml_tensor * yasa_backbone_ln_b = nullptr;
    ggml_tensor * yasa_vision_pos_embed = nullptr;
    std::vector<yasa2_stage> yasa_stages;

    // lfm3vl
    ggml_tensor * lfm3vl_embedding_norm_w = nullptr;

    // pixtral, glm4v
    ggml_tensor * token_embd_img_break = nullptr;

    // deepseek4v sentinel embeddings (image_newline is reused for IMAGE_NEW_LINE)
    ggml_tensor * token_embd_img_start = nullptr;
    ggml_tensor * token_embd_img_end   = nullptr;
    ggml_tensor * token_embd_img_pad   = nullptr;
    ggml_tensor * mm_patch_merger_w = nullptr;
    ggml_tensor * mm_patch_merger_b = nullptr;

    // ultravox / whisper encoder
    ggml_tensor * conv1d_1_w = nullptr;
    ggml_tensor * conv1d_1_b = nullptr;
    ggml_tensor * conv1d_2_w = nullptr;
    ggml_tensor * conv1d_2_b = nullptr;
    ggml_tensor * conv_out_w = nullptr;
    ggml_tensor * conv_out_b = nullptr;
    ggml_tensor * mm_norm_pre_w = nullptr;
    ggml_tensor * mm_norm_pre_b = nullptr;
    ggml_tensor * mm_norm_mid_w = nullptr;

    // mimo-audio-tokenizer: post-transformer downsample + RVQ codebook
    ggml_tensor * downsample_conv_w = nullptr; // no bias
    ggml_tensor * downsample_norm_w = nullptr;
    ggml_tensor * downsample_norm_b = nullptr;
    ggml_tensor * rvq_codebook = nullptr; // merged 3D [n_q, max_bins, dim]

    // mimo-v2.5: text-side RVQ code embedding ("text codebook")
    ggml_tensor * mm_a_code_embd = nullptr; // merged 3D [n_channels, vocab, dim]

    // mimo-v2.5: LLM-side connector (input_local_transformer, separate from the
    // audio_tokenizer's own encoder `layers`)
    std::vector<mtmd_layer> mm_a_local_layers;
    ggml_tensor * mm_a_local_norm_w = nullptr;

    // qwen3a
    ggml_tensor * conv2d_1_w = nullptr;
    ggml_tensor * conv2d_1_b = nullptr;
    ggml_tensor * conv2d_2_w = nullptr;
    ggml_tensor * conv2d_2_b = nullptr;
    ggml_tensor * conv2d_3_w = nullptr;
    ggml_tensor * conv2d_3_b = nullptr;

    // qwen3tts speaker encoder (ECAPA-TDNN)
    // reused tensors: stem conv is conv1d_1_w/b, feature aggregation is conv_out_w/b, output proj is mm_fc_w/b
    ggml_tensor * spk_asp_attn_w = nullptr;
    ggml_tensor * spk_asp_attn_b = nullptr;
    ggml_tensor * spk_asp_tdnn_w = nullptr;
    ggml_tensor * spk_asp_tdnn_b = nullptr;

    // qwen3tts code_predictor
    ggml_tensor * gen_code_proj_in_w  = nullptr; // small_to_mtp_projection
    ggml_tensor * gen_code_proj_in_b  = nullptr;
    ggml_tensor * gen_code_embd_w     = nullptr; // per-codebook embedding, merged 3D
    ggml_tensor * gen_code_head_w     = nullptr; // per-codebook output head, merged 3D
    ggml_tensor * gen_code_out_embd_w = nullptr; // codebook-0 embedding, fed back into the talker
    ggml_tensor * gen_code_norm_w     = nullptr; // final norm

    // lfm3a depthformer (MaskGIT)
    ggml_tensor * gen_code_head_norm_w = nullptr; // per-codebook norm before the tied head, merged 2D
    ggml_tensor * gen_code_mask_embd   = nullptr;

    // qwen3tts code2wav: RVQ codes -> raw PCM
    mtmd_code2wav c2w;
    std::vector<mtmd_drama3_layer> drama3_layers;

    // pocket-tts: SEANet stack, shared by the encoder (speaker path) and the decoder (gen path)
    mtmd_seanet seanet;

    // pocket-tts: voice latent -> backbone embd (speaker path)
    ggml_tensor * spk_proj_w      = nullptr;
    ggml_tensor * downsample_w    = nullptr;

    // pocket-tts: flow-matching decoder, backbone hidden state -> next latent
    mtmd_flow_net flow;
    ggml_tensor * gen_out_eos_w   = nullptr;
    ggml_tensor * gen_out_eos_b   = nullptr;
    ggml_tensor * gen_input_lin_w = nullptr; // latent -> backbone embd
    ggml_tensor * gen_emb_mean    = nullptr;
    ggml_tensor * gen_emb_std     = nullptr;
    ggml_tensor * gen_quant_out_w = nullptr; // latent -> decoder dim
    ggml_tensor * gen_upsample_w  = nullptr; // depthwise convtr, frame rate -> encoder frame rate
    std::vector<mtmd_layer> gen_tfm_layers; // mimi decoder_transformer

    // chatterbox flow encoder, cfm estimator, hift vocoder, s3 tokenizer and CAMPPlus speaker encoder
    mtmd_chatterbox cbx;

    // chatterbox tensors read on the host, see cbx_read_tensor()
    std::map<std::string, ggml_tensor *> cbx_tensors;
    // predefined voices, no rows if the voice is named in the prompt (lfm3a)
    std::map<std::string, mtmd_gen_voice> gen_voices;

    // soprano: Vocos decoder, hidden states -> STFT coefficients
    mtmd_vocos vocos;

    // cogvlm
    ggml_tensor * mm_post_fc_norm_w = nullptr;
    ggml_tensor * mm_post_fc_norm_b = nullptr;
    ggml_tensor * mm_h_to_4h_w = nullptr;
    ggml_tensor * mm_gate_w = nullptr;
    ggml_tensor * mm_4h_to_h_w = nullptr;
    ggml_tensor * mm_boi = nullptr;
    ggml_tensor * mm_eoi = nullptr;

    // hunyuanvl perceiver
    ggml_tensor * mm_pre_norm_w  = nullptr;
    ggml_tensor * mm_img_begin   = nullptr;
    ggml_tensor * mm_img_end     = nullptr;

    // deepseek ocr sam
    ggml_tensor * patch_embed_proj_w = nullptr;
    ggml_tensor * patch_embed_proj_b = nullptr;
    ggml_tensor * pos_embed          = nullptr;

    ggml_tensor * neck_0_w;
    ggml_tensor * neck_1_w;
    ggml_tensor * neck_1_b;
    ggml_tensor * neck_2_w;
    ggml_tensor * neck_3_w;
    ggml_tensor * neck_3_b;
    ggml_tensor * net_2;
    ggml_tensor * net_3;

    int32_t n_sam_layers = 12; // used by deepseek-ocr sam encoder

    std::vector<mtmd_layer> sam_layers;

    // nemotron transcribe: speaker expert, Sortformer head, fusion
    std::vector<mtmd_layer> spk_layers;
    ggml_tensor * spk_inp_proj_w     = nullptr;
    ggml_tensor * spk_pre_ln_w       = nullptr;
    ggml_tensor * spk_pre_ln_b       = nullptr;
    ggml_tensor * spk_post_ln_w      = nullptr;
    ggml_tensor * spk_post_ln_b      = nullptr;
    ggml_tensor * spk_head_proj_w    = nullptr;
    ggml_tensor * spk_head_proj_b    = nullptr;
    ggml_tensor * spk_head_up_w      = nullptr;
    ggml_tensor * spk_head_up_b      = nullptr;
    ggml_tensor * spk_head_fc1_w     = nullptr;
    ggml_tensor * spk_head_fc1_b     = nullptr;
    ggml_tensor * spk_head_fc2_w     = nullptr;
    ggml_tensor * spk_head_fc2_b     = nullptr;
    ggml_tensor * fusion_asr_norm_w  = nullptr;
    ggml_tensor * fusion_asr_norm_b  = nullptr;
    ggml_tensor * fusion_spk_norm_w  = nullptr;
    ggml_tensor * fusion_spk_norm_b  = nullptr;
    ggml_tensor * fusion_spk_kernel  = nullptr;
    std::vector<float> spk_sil_emb;

    // deepseek-ocr-2
    ggml_tensor * resample_query_768 = nullptr;
    ggml_tensor * resample_query_1024 = nullptr;

    // lfm2 audio
    std::array<ggml_tensor *, 7> pre_encode_conv_X_w = {nullptr};
    std::array<ggml_tensor *, 7> pre_encode_conv_X_b = {nullptr};
    ggml_tensor * pre_encode_out_w = nullptr;
    ggml_tensor * pre_encode_out_b = nullptr;

    // gemma4
    ggml_tensor * std_bias = nullptr;
    ggml_tensor * std_scale = nullptr;
    // Gemma4ClippableLinear
    struct clamp_info {
        float inp_max;
        float inp_min;
        float out_max;
        float out_min;
    };
    std::map<std::string, clamp_info> clamp_info_map;

    // gemma4 audio conformer
    std::array<ggml_tensor *, 2> sscp_conv_w = {nullptr};
    std::array<ggml_tensor *, 2> sscp_conv_b = {nullptr};
    std::array<ggml_tensor *, 2> sscp_norm_w = {nullptr};
    ggml_tensor * sscp_inp_proj_w = nullptr;
    ggml_tensor * sscp_inp_proj_b = nullptr;
    ggml_tensor * audio_out_proj_w = nullptr;
    ggml_tensor * audio_out_proj_b = nullptr;

    // granite_speech encoder
    ggml_tensor * inp_proj_w    = nullptr;
    ggml_tensor * inp_proj_b    = nullptr;
    ggml_tensor * ctc_out_w     = nullptr;
    ggml_tensor * ctc_out_b     = nullptr;
    ggml_tensor * ctc_out_mid_w = nullptr;
    ggml_tensor * ctc_out_mid_b = nullptr;
    // qformer projector(s)
    std::vector<qf_block> qf_proj_blocks;

    bool audio_has_avgpool() const {
        return proj_type == PROJECTOR_TYPE_QWEN2A
            || proj_type == PROJECTOR_TYPE_VOXTRAL
            || proj_type == PROJECTOR_TYPE_MUSIC_FLAMINGO;
    }

    bool audio_has_stack_frames() const {
        return proj_type == PROJECTOR_TYPE_ULTRAVOX
            || proj_type == PROJECTOR_TYPE_VOXTRAL
            || proj_type == PROJECTOR_TYPE_MERALION;
    }

    // this should be equal to the embedding dimension of the text model
    int n_mmproj_embd() const;

    int n_output_tokens(const mtmd_image_f32 * img) const;

    // for M-RoPE, this will be the number of token positions in X and Y directions
    // for other models, X will be the total number of tokens and Y will be 1
    int n_output_tokens_x(const mtmd_image_f32 * img) const;
    int n_output_tokens_y(const mtmd_image_f32 * img) const;

    // max number of output tokens per image, -1 if not dynamic size
    int image_max_tokens() const;

    // TODO @ngxson : this is no longer correct with mtmd_batch API
    int n_temporal_merge() const;

    // read a chatterbox tensor by name as F32, returns the element count (0 if not found), out may be null to query the size
    size_t cbx_read_tensor(const char * name, float * out, size_t n_max) const;
};

struct mtmd_model_params {
    // custom value provided by user, -1 if not set
    int32_t image_min_tokens = -1;
    int32_t image_max_tokens = -1;
    bool no_alloc = false; // only create the tensors, do not read their data
    mtmd_progress_callback progress_callback = nullptr;
    void * progress_callback_user_data = nullptr;
};

// the modalities of an mmproj file, a missing one is nullptr
struct mtmd_models {
    std::unique_ptr<mtmd_model> v;
    std::unique_ptr<mtmd_model> a;
    std::unique_ptr<mtmd_model> gen_a;
};

// the weights are allocated in buft, throws on error
mtmd_models mtmd_model_load(const char * fname, const mtmd_model_params & params, ggml_backend_buffer_type_t buft);

struct mtmd_model_caps {
    bool has_vision;
    bool has_audio;
    bool has_gen_audio;
};

// read the modalities of an mmproj file without loading its tensors
mtmd_model_caps mtmd_model_get_caps(const char * fname);
