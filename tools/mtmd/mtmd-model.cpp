#include "mtmd-impl.h"
#include "mtmd-model.h"
#include "models/models.h"

#include "ggml.h"
#include "ggml-cpp.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "gguf.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <random>
#include <stdexcept>
#include <unordered_set>
#include <vector>
#include <cinttypes>
#include <limits>
#include <array>
#include <functional>
#include <float.h>

//
// mtmd_model_loader
//

struct mtmd_model_loader {
    ggml_context_ptr ctx_meta;
    gguf_context_ptr ctx_gguf;

    std::string fname;

    size_t model_size = 0; // in bytes

    bool has_vision    = false;
    bool has_audio     = false;
    bool has_gen_audio = false;

    mtmd_progress_callback progress_callback = nullptr;
    void * progress_callback_user_data = nullptr;

    mtmd_model_loader(const char * fname,
            bool skip_tensors = false,
            mtmd_progress_callback progress_cb = nullptr,
            void * progress_user_data = nullptr)
        : fname(fname),
          progress_callback(progress_cb),
          progress_callback_user_data(progress_user_data) {
        struct ggml_context * meta = nullptr;

        struct gguf_init_params params = {
            /*.no_alloc = */ true,
            /*.ctx      = */ &meta,
        };

        ctx_gguf = gguf_context_ptr(gguf_init_from_file(fname, params));
        if (!ctx_gguf.get()) {
            throw std::runtime_error(string_format("%s: failed to load mmproj model from %s. Does this file exist?\n", __func__, fname));
        }

        ctx_meta.reset(meta);

        const int n_tensors = gguf_get_n_tensors(ctx_gguf.get());

        // print gguf info
        {
            std::string name;
            get_string(KEY_NAME, name, false);
            std::string description;
            get_string(KEY_DESCRIPTION, description, false);
            LOG_INF("%s: model name:   %s\n",  __func__, name.c_str());
            LOG_INF("%s: description:  %s\n",  __func__, description.c_str());
            LOG_INF("%s: GGUF version: %d\n",  __func__, gguf_get_version(ctx_gguf.get()));
            LOG_INF("%s: alignment:    %zu\n", __func__, gguf_get_alignment(ctx_gguf.get()));
            LOG_INF("%s: n_tensors:    %d\n",  __func__, n_tensors);
            LOG_INF("%s: n_kv:         %d\n",  __func__, (int)gguf_get_n_kv(ctx_gguf.get()));
            LOG_INF("\n");
        }

        // modalities
        {
            get_bool(KEY_HAS_VISION_ENC,    has_vision,    false);
            get_bool(KEY_HAS_AUDIO_ENC,     has_audio,     false);
            get_bool(KEY_HAS_GEN_AUDIO_ENC, has_gen_audio, false);

            if (has_vision) {
                LOG_INF("%s: has vision encoder\n", __func__);
            }
            if (has_audio) {
                LOG_INF("%s: has audio encoder\n", __func__);
            }
            if (has_gen_audio) {
                LOG_INF("%s: has audio generation (gen) encoder\n", __func__);
            }
        }

        // tensors
        if (!skip_tensors) {
            for (int i = 0; i < n_tensors; ++i) {
                const char * name = gguf_get_tensor_name(ctx_gguf.get(), i);
                const size_t offset = gguf_get_tensor_offset(ctx_gguf.get(), i);
                enum ggml_type type = gguf_get_tensor_type(ctx_gguf.get(), i);
                ggml_tensor * cur = ggml_get_tensor(meta, name);
                size_t tensor_size = ggml_nbytes(cur);
                model_size += tensor_size;
                LOG_DBG("%s: tensor[%d]: n_dims = %d, name = %s, tensor_size=%zu, offset=%zu, shape:[%" PRIu64 ", %" PRIu64 ", %" PRIu64 ", %" PRIu64 "], type = %s\n",
                    __func__, i, ggml_n_dims(cur), cur->name, tensor_size, offset, cur->ne[0], cur->ne[1], cur->ne[2], cur->ne[3], ggml_type_name(type));
            }
        }
    }

    void load_hparams(mtmd_model & model, mtmd_modality modality) {
        auto & hparams = model.hparams;
        std::string log_ffn_op; // for logging

        // sanity check
        if (modality == MTMD_MODALITY_VISION) {
            GGML_ASSERT(has_vision);
        } else if (modality == MTMD_MODALITY_AUDIO) {
            GGML_ASSERT(has_audio);
        } else if (modality == MTMD_MODALITY_GEN_AUDIO) {
            GGML_ASSERT(has_gen_audio);
        }
        model.modality = modality;


        // projector type
        std::string proj_type;
        {
            // default key
            get_string(KEY_PROJ_TYPE, proj_type, false);

            // for models with mixed modalities
            if (proj_type.empty()) {
                if (modality == MTMD_MODALITY_VISION) {
                    get_string(KEY_VISION_PROJ_TYPE, proj_type, false);
                } else if (modality == MTMD_MODALITY_AUDIO) {
                    get_string(KEY_AUDIO_PROJ_TYPE, proj_type, false);
                } else if (modality == MTMD_MODALITY_GEN_AUDIO) {
                    get_string(KEY_GEN_AUDIO_PROJ_TYPE, proj_type, false);
                } else {
                    GGML_ABORT("unknown modality");
                }
            }

            model.proj_type = projector_type_from_string(proj_type);

            if (model.proj_type == PROJECTOR_TYPE_UNKNOWN) {
                throw std::runtime_error(string_format("%s: unknown projector type: %s\n", __func__, proj_type.c_str()));
            }

            // correct arch for multimodal models (legacy method)
            if (model.proj_type == PROJECTOR_TYPE_QWEN25O) {
                model.proj_type = modality == MTMD_MODALITY_VISION
                                    ? PROJECTOR_TYPE_QWEN25VL
                                    : PROJECTOR_TYPE_QWEN2A;
            }
        }

        const bool is_vision    = model.modality == MTMD_MODALITY_VISION;
        const bool is_audio     = model.modality == MTMD_MODALITY_AUDIO;
        const bool is_gen_audio = model.modality == MTMD_MODALITY_GEN_AUDIO;

        // other hparams
        {
            const char * prefix = is_vision ? "vision" : (is_audio ? "audio" : "gen.audio");
            get_u32(string_format(KEY_N_EMBD,         prefix), hparams.n_embd);
            get_u32(string_format(KEY_N_HEAD,         prefix), hparams.n_head);
            get_u32(string_format(KEY_N_EMBD_HEAD,    prefix), hparams.n_embd_head, false);
            get_u32(string_format(KEY_N_FF,           prefix), hparams.n_ff);
            get_u32(string_format(KEY_N_BLOCK,        prefix), hparams.n_layer);
            get_u32(string_format(KEY_PROJ_DIM,       prefix), hparams.projection_dim);
            get_f32(string_format(KEY_LAYER_NORM_EPS, prefix), hparams.eps);

            // n_head_kv is optional (for GQA), default to n_head
            hparams.n_head_kv = hparams.n_head;
            get_u32(string_format(KEY_N_HEAD_KV, prefix), hparams.n_head_kv, false);

            if (is_vision) {
                get_u32(KEY_IMAGE_SIZE, hparams.image_size);
                get_u32(KEY_PATCH_SIZE, hparams.patch_size);
                get_u32(KEY_MAX_SLICE_NUMS, hparams.max_slice_nums, false);
                get_i32(KEY_MINICPMV_VERSION, hparams.minicpmv_version, false); // legacy
                get_u32(KEY_MINICPMV_QUERY_NUM, hparams.minicpmv_query_num, false);
                if (hparams.minicpmv_query_num == 0) {
                    // Fallback to hardcoded values for legacy models
                    if (hparams.minicpmv_version == 3) {
                        hparams.minicpmv_query_num = 64;
                    } else if (hparams.minicpmv_version == 4) {
                        hparams.minicpmv_query_num = 64;
                    } else if (hparams.minicpmv_version == 5) {
                        hparams.minicpmv_query_num = 64;
                    } else if (hparams.minicpmv_version == 6) {
                        hparams.minicpmv_query_num = 64;
                    } else if (hparams.minicpmv_version == 100045) {
                        hparams.minicpmv_query_num = 64;
                    } else {
                        hparams.minicpmv_query_num = 96;
                    }
                }
            } else if (is_audio) {
                get_u32(KEY_A_NUM_MEL_BINS, hparams.n_mel_bins);
                // some hparams are unused, but still need to set to avoid issues
                hparams.image_size = 0;
                hparams.patch_size = 1;

            } else if (is_gen_audio) {
                // these are unused, but still need to be set to avoid issues
                hparams.image_size = 0;
                hparams.patch_size = 1;
                get_string(KEY_GEN_AUDIO_VARIANT, hparams.gen_model_variant, false);

            } else {
                GGML_ASSERT(false && "unknown modality");
            }

            // for pinpoints, we need to convert it into a list of resolution candidates
            {
                std::vector<int> pinpoints;
                get_arr_int(KEY_IMAGE_GRID_PINPOINTS, pinpoints, false);
                if (pinpoints.size() % 2 != 0) {
                    throw std::runtime_error(string_format("%s: image_grid_pinpoints must have an even number of elements, got %zu\n", __func__, pinpoints.size()));
                }
                if (!pinpoints.empty()) {
                    for (size_t i = 0; i < pinpoints.size(); i += 2) {
                        hparams.image_res_candidates.push_back({
                            pinpoints[i],
                            pinpoints[i+1],
                        });
                    }
                }
            }

            // default warmup value
            hparams.warmup_image_size = hparams.image_size;

            {
                bool use_gelu = false;
                bool use_silu = false;
                get_bool(KEY_USE_GELU, use_gelu, false);
                get_bool(KEY_USE_SILU, use_silu, false);
                if (use_gelu && use_silu) {
                    throw std::runtime_error(string_format("%s: both use_gelu and use_silu are set to true\n", __func__));
                }
                if (use_gelu) {
                    hparams.ffn_op = FFN_GELU;
                    log_ffn_op = "gelu";
                } else if (use_silu) {
                    hparams.ffn_op = FFN_SILU;
                    log_ffn_op = "silu";
                } else {
                    hparams.ffn_op = FFN_GELU_QUICK;
                    log_ffn_op = "gelu_quick";
                }
            }

            {
                std::string mm_patch_merge_type;
                get_string(KEY_MM_PATCH_MERGE_TYPE, mm_patch_merge_type, false);
                if (mm_patch_merge_type == "spatial_unpad") {
                    hparams.mm_patch_merge_type = PATCH_MERGE_SPATIAL_UNPAD;
                }
            }

            if (is_vision) {
                std::vector<float> image_mean;
                std::vector<float> image_std;
                get_arr_f32(KEY_IMAGE_MEAN, image_mean, false);
                get_arr_f32(KEY_IMAGE_STD , image_std, false);
                if (image_mean.size() < 3 || image_std.size() < 3) {
                    throw std::runtime_error(string_format("%s: image_mean/image_std arrays must have at least 3 elements, got %zu and %zu\n", __func__, image_mean.size(), image_std.size()));
                }
                for (int i = 0; i < 3; ++i) {
                    hparams.image_mean[i] = image_mean[i];
                    hparams.image_std[i]  = image_std[i];
                }
            }

            // Load the vision/audio feature layer indices if they are explicitly provided
            // NOTE: gguf conversions should standardize the values of the vision feature layer to be non-negative, since we use -1 to mark values as unset here.
            get_arr_int(string_format(KEY_FEATURE_LAYERS, prefix), hparams.feature_layers, false);
            for (const auto & v : hparams.feature_layers) {
                if (v > (int) hparams.n_layer) {
                    throw std::runtime_error(string_format("%s: feature layer index %d is out of range (n_layer: %d)",
                                                           __func__, v, hparams.n_layer));
                }
            }

            // model-specific params
            switch (model.proj_type) {
                case PROJECTOR_TYPE_MLP:
                case PROJECTOR_TYPE_MLP_NORM:
                case PROJECTOR_TYPE_LDP:
                case PROJECTOR_TYPE_LDPV2:
                case PROJECTOR_TYPE_COGVLM:
                    {
                        hparams.has_llava_projector = model.proj_type != PROJECTOR_TYPE_COGVLM;
                        hparams.image_pad_color     = {122, 116, 104};
                        if (!hparams.image_res_candidates.empty()) {
                            hparams.image_resize_pad  = PAD_CEIL;
                            hparams.image_resize_algo = RESIZE_ALGO_BICUBIC;
                        } else {
                            // llava-1.6 default params
                            hparams.image_pad_ov         = PAD_NONE;
                            hparams.image_pad_rf         = PAD_CEIL;
                            hparams.image_pad_color_rf   = {122, 116, 104};
                        }
                    } break;
                case PROJECTOR_TYPE_GLM_EDGE:
                    {
                        hparams.image_resize_pad  = PAD_CEIL;
                        hparams.image_resize_algo = RESIZE_ALGO_BICUBIC;
                    } break;
                case PROJECTOR_TYPE_MINICPMV:
                    {
                        // use default llava-uhd preprocessing params
                        if (hparams.minicpmv_version == 0) {
                            hparams.minicpmv_version = 2; // default to 2 if not set
                        }
                    } break;
                case PROJECTOR_TYPE_MINICPMV4_6:
                case PROJECTOR_TYPE_MINICPMV4_7:
                case PROJECTOR_TYPE_MINICPM_OCR_1_0:
                    {
                        // MiniCPM-V 4.6/4.7 unified merger projector
                        // ViT merger 2x2 + final merger 2x2 = 4x spatial merge per dimension
                        hparams.n_merge = 4;
                        get_u32(KEY_PROJ_SCALE_FACTOR, hparams.n_merge, false);
                        GGML_ASSERT(hparams.n_merge == 2 || hparams.n_merge == 4);

                        // no padding: the reference stretches the refined image to the target size
                        hparams.image_pad_ov = PAD_NONE;
                        hparams.image_pad_rf = PAD_NONE;

                        // borrow wa_layer_indexes for vit_merger insertion point
                        std::vector<int> wa_layer_indexes_vec;
                        get_arr_int(KEY_WIN_ATTN_LAYER_INDEXES, wa_layer_indexes_vec, false);
                        if (!wa_layer_indexes_vec.empty()) {
                            const int insert_lid = wa_layer_indexes_vec[0];
                            if (insert_lid < 0 || insert_lid >= (int) hparams.n_layer) {
                                throw std::runtime_error(string_format("%s: layer index %d is out of range (n_layer: %d)",
                                                                       __func__, insert_lid, hparams.n_layer));
                            }
                            hparams.insert_layer_id = insert_lid;
                        }
                    } break;
                case PROJECTOR_TYPE_INTERNVL:
                    {
                        // use default llava-uhd preprocessing params
                        // older version of internvl doesn't have min/max tiles, we need to provide default values for them to avoid issues
                        hparams.preproc_min_tiles = 1;
                        hparams.preproc_max_tiles = 12;
                        get_u32(KEY_PROJ_SCALE_FACTOR, hparams.n_merge, false);
                        get_u32(KEY_PREPROC_MIN_TILES, hparams.preproc_min_tiles, false);
                        get_u32(KEY_PREPROC_MAX_TILES, hparams.preproc_max_tiles, false);
                        GGML_ASSERT(hparams.preproc_min_tiles <= hparams.preproc_max_tiles && hparams.preproc_max_tiles < INT32_MAX);
                        set_internvl_dhr_res_candidates(model);
                    } break;
                case PROJECTOR_TYPE_NEMOTRON_V2_VL:
                    {
                        get_u32(KEY_PROJ_SCALE_FACTOR, hparams.n_merge, false);
                    } break;
                case PROJECTOR_TYPE_NEMOTRON_OMNI:
                    {
                        hparams.ffn_op = FFN_GELU_ERF;
                        log_ffn_op = "gelu_erf";
                        hparams.image_resize_pad = PAD_NONE;
                        get_u32(KEY_PROJ_SCALE_FACTOR, hparams.n_merge);
                        get_u32(KEY_IMAGE_MIN_PIXELS, hparams.image_min_pixels);
                        get_u32(KEY_IMAGE_MAX_PIXELS, hparams.image_max_pixels);
                        hparams.warmup_image_size = hparams.image_size;
                    } break;
                case PROJECTOR_TYPE_PARAKEET:
                    {
                        get_u32(KEY_AUDIO_SUBSMPL_FACTOR, hparams.subsampling_factor);
                        GGML_ASSERT(hparams.subsampling_factor == 8 &&
                            "subsampling_factor must match the conv strides in mtmd_graph_parakeet::build()");
                        get_u32(KEY_A_CONV_KERNEL_SIZE,       hparams.audio_conv_kernel_size);
                        GGML_ASSERT(hparams.audio_conv_kernel_size > 0 && hparams.audio_conv_kernel_size % 2 == 1 &&
                            "audio_conv_kernel_size must be a positive odd integer");
                        hparams.audio_chunk_len    = 0;
                        hparams.audio_sample_rate  = 16000;
                        hparams.audio_n_fft        = 512;
                        hparams.audio_window_len   = 400;
                        hparams.audio_hop_len      = 160;
                    } break;
                case PROJECTOR_TYPE_NEMOTRON_TRANSCRIBE:
                    {
                        get_u32(KEY_AUDIO_SUBSMPL_FACTOR, hparams.subsampling_factor);
                        get_u32(KEY_A_ROPE_DIMS,          hparams.audio_rope_dims);
                        get_u32(KEY_A_SPK_N_BLOCK,        hparams.spk_n_layer);
                        get_u32(KEY_A_SPK_N_HEAD,         hparams.spk_n_head);
                        if (hparams.subsampling_factor != 8) {
                            throw std::runtime_error(string_format("%s: unsupported subsampling factor %d\n", __func__, hparams.subsampling_factor));
                        }
                        if (hparams.spk_n_layer <= 0 || hparams.spk_n_layer > (int) gguf_get_n_tensors(ctx_gguf.get())) {
                            throw std::runtime_error(string_format("%s: invalid speaker block count %d\n", __func__, hparams.spk_n_layer));
                        }
                        hparams.audio_chunk_len    = 0;
                        hparams.audio_sample_rate  = 16000;
                        hparams.audio_n_fft        = 512;
                        hparams.audio_window_len   = 400;
                        hparams.audio_hop_len      = 160;
                        // one full window
                        using nt = mtmd_graph_nemotron_transcribe;
                        hparams.warmup_audio_size = (nt::window_left + nt::window_core + nt::window_right) * hparams.subsampling_factor;
                    } break;
                case PROJECTOR_TYPE_IDEFICS3:
                    {
                        // use default llava-uhd preprocessing params
                        hparams.image_resize_algo = RESIZE_ALGO_LANCZOS;
                        get_u32(KEY_PROJ_SCALE_FACTOR, hparams.n_merge, false);
                        get_u32(KEY_PREPROC_IMAGE_SIZE, hparams.image_longest_edge, false);
                        hparams.set_limit_image_tokens();
                    } break;
                case PROJECTOR_TYPE_COHERE2V:
                    {
                        hparams.image_pad_rf = PAD_NONE;
                        get_u32(KEY_PROJ_SCALE_FACTOR, hparams.n_merge);
                        get_u32(KEY_PREPROC_MAX_TILES, hparams.preproc_max_tiles);
                        if (hparams.preproc_max_tiles <= 0 || hparams.preproc_max_tiles > 256) {
                            throw std::runtime_error(string_format("%s: preproc_max_tiles (%d) must be in range [1, 256]\n", __func__, hparams.preproc_max_tiles));
                        }
                    } break;
                case PROJECTOR_TYPE_LFM2:
                case PROJECTOR_TYPE_D1OMNI_V:
                    {
                        // default for older GGUFs
                        std::string resize_algo = "bilinear";
                        get_string(KEY_IMAGE_RESIZE_ALGO, resize_algo, false);
                        // Lfm2VlImageProcessor and d1-omni vision.py resize bilinear and bicubic with torchvision
                        if (resize_algo == "bilinear") {
                            hparams.set_image_resize_algo(RESIZE_ALGO_BILINEAR, true);
                        } else if (resize_algo == "bicubic") {
                            hparams.set_image_resize_algo(RESIZE_ALGO_BICUBIC, true);
                        } else if (resize_algo == "lanczos") {
                            hparams.set_image_resize_algo(RESIZE_ALGO_LANCZOS);
                        } else {
                            throw std::runtime_error(string_format("%s: unsupported image resize algo: %s\n", __func__, resize_algo.c_str()));
                        }
                        hparams.image_resize_algo_rf = hparams.image_resize_algo;
                        hparams.image_resize_algo_ov = hparams.image_resize_algo;
                        // the tiles stretch the image to the grid (Lfm2VlImageProcessor, d1-omni vision.py)
                        hparams.image_pad_rf = PAD_NONE;
                        get_u32(KEY_PROJ_SCALE_FACTOR, hparams.n_merge, false);
                        // defaults for GGUFs converted before tiling params were written
                        // ref: https://huggingface.co/LiquidAI/LFM2.5-VL-1.6B/blob/main/processor_config.json
                        hparams.preproc_min_tiles            = 2;
                        hparams.preproc_max_tiles            = 10;
                        hparams.preproc_tile_size            = 512;
                        hparams.preproc_max_pixels_tolerance = 2.0f;
                        get_u32(KEY_PREPROC_MIN_TILES, hparams.preproc_min_tiles, false);
                        get_u32(KEY_PREPROC_MAX_TILES, hparams.preproc_max_tiles, false);
                        get_u32(KEY_PREPROC_TILE_SIZE, hparams.preproc_tile_size, false);
                        get_f32(KEY_PREPROC_MAX_PIXELS_TOLERANCE, hparams.preproc_max_pixels_tolerance, false);
                        GGML_ASSERT(hparams.preproc_min_tiles >= 1 && hparams.preproc_min_tiles <= hparams.preproc_max_tiles);
                        hparams.set_limit_image_tokens(64, 256);
                    } break;
                case PROJECTOR_TYPE_PHI4:
                    {
                        hparams.n_merge = 1;
                        hparams.image_resize_algo = RESIZE_ALGO_BILINEAR;
                        get_u32(KEY_IMAGE_MIN_PIXELS, hparams.image_min_pixels);
                        get_u32(KEY_IMAGE_MAX_PIXELS, hparams.image_max_pixels);
                        hparams.set_warmup_n_tokens(16*16);
                    } break;
                case PROJECTOR_TYPE_PIXTRAL:
                    {
                        // ref: https://huggingface.co/mistral-community/pixtral-12b/blob/main/preprocessor_config.json
                        // TODO: verify the image_min_tokens
                        hparams.n_merge = 1; // the original pixtral does not use patch merging
                        hparams.image_resize_algo = RESIZE_ALGO_BICUBIC;
                        hparams.rope_theta = 10000.0f;
                        get_u32(KEY_SPATIAL_MERGE_SIZE, hparams.n_merge, false);
                        get_u32(KEY_PREPROC_IMAGE_SIZE, hparams.image_longest_edge, false);
                        if (hparams.image_longest_edge > 0) {
                            const int64_t patch_size = int64_t(hparams.patch_size) * hparams.n_merge;
                            if (patch_size <= 0) {
                                throw std::runtime_error("Pixtral merged patch size must be positive");
                            }
                            const int64_t n_tokens_per_side =
                                (hparams.image_longest_edge + patch_size - 1) / patch_size;
                            if (n_tokens_per_side * patch_size > std::sqrt(std::numeric_limits<int>::max())) {
                                throw std::runtime_error("Pixtral preprocessing image size is too large");
                            }
                            hparams.image_resize_pad         = PAD_NONE;
                            hparams.image_floor_before_align = true;
                            hparams.set_limit_image_tokens(0, n_tokens_per_side * n_tokens_per_side);
                        } else {
                            hparams.set_limit_image_tokens(8, 1024);
                        }
                        hparams.set_warmup_n_tokens(256); // avoid OOM on warmup
                    } break;
                case PROJECTOR_TYPE_LIGHTONOCR:
                    {
                        hparams.n_merge = 1;
                        hparams.image_resize_algo = RESIZE_ALGO_BICUBIC;
                        hparams.rope_theta = 10000.0f;
                        get_u32(KEY_SPATIAL_MERGE_SIZE, hparams.n_merge, false);
                        hparams.image_longest_edge = hparams.image_size;
                        get_u32(KEY_PREPROC_IMAGE_SIZE, hparams.image_longest_edge, false);
                        hparams.set_limit_image_tokens();
                        hparams.set_warmup_n_tokens(256); // avoid OOM on warmup
                    } break;
                case PROJECTOR_TYPE_DOTS_OCR:
                    {
                        hparams.rope_theta = 10000.0f;
                        hparams.image_resize_pad  = PAD_NONE; // the HF processor resizes to the smart_resize size, no padding
                        hparams.image_round_half_even = true; // smart_resize uses Python round()
                        get_u32(KEY_PROJ_SCALE_FACTOR, hparams.n_merge);
                        get_u32(KEY_IMAGE_MIN_PIXELS, hparams.image_min_pixels);
                        get_u32(KEY_IMAGE_MAX_PIXELS, hparams.image_max_pixels);
                        hparams.set_warmup_n_tokens(46*46); // avoid OOM on warmup
                    } break;
                case PROJECTOR_TYPE_DOTS3NOTE_V:
                    {
                        hparams.rope_theta = 10000.0f;
                        hparams.image_resize_algo = RESIZE_ALGO_BICUBIC;
                        get_u32(KEY_SPATIAL_MERGE_SIZE, hparams.n_merge);
                        get_u32(KEY_IMAGE_MIN_PIXELS, hparams.image_min_pixels);
                        get_u32(KEY_IMAGE_MAX_PIXELS, hparams.image_max_pixels);
                        get_u32(KEY_VISION_N_EXPERT_USED, hparams.n_expert_used);
                        hparams.set_warmup_n_tokens(46*46); // avoid OOM on warmup
                    } break;
                case PROJECTOR_TYPE_DOTS3NOTE_A:
                    {
                        hparams.rope_theta = 10000.0f;
                        hparams.audio_chunk_len   = 60; // in seconds
                        hparams.audio_sample_rate = 16000;
                        hparams.audio_n_fft       = 400;
                        hparams.audio_window_len  = 400;
                        hparams.audio_hop_len     = 160;
                    } break;
                case PROJECTOR_TYPE_KIMIVL:
                    {
                        hparams.image_resize_algo = RESIZE_ALGO_BICUBIC;
                        hparams.rope_theta = 10000.0f;
                        get_u32(KEY_PROJ_SCALE_FACTOR, hparams.n_merge, false);
                        // TODO: check kimivl preprocessor for exact values
                        hparams.set_limit_image_tokens(8, 1024);
                        hparams.set_warmup_n_tokens(256); // avoid OOM on warmup
                    } break;
                case PROJECTOR_TYPE_KIMIK25:
                    {
                        hparams.image_resize_algo = RESIZE_ALGO_BICUBIC;
                        hparams.rope_theta = 10000.0f;
                        get_u32(KEY_PROJ_SCALE_FACTOR, hparams.n_merge, false);

                        int min_pixels = 0, max_pixels = 0;
                        get_u32(KEY_IMAGE_MIN_PIXELS, min_pixels, false);
                        get_u32(KEY_IMAGE_MAX_PIXELS, max_pixels, false);
                        if (min_pixels > 0 && max_pixels > 0) {
                            hparams.image_min_pixels = min_pixels;
                            hparams.image_max_pixels = max_pixels;
                            hparams.warmup_image_size = static_cast<int>(std::sqrt(max_pixels));
                        } else {
                            hparams.set_limit_image_tokens(2, 4096);
                        }
                    } break;
                case PROJECTOR_TYPE_DEEPSEEK4V:
                    {
                        hparams.image_resize_algo = RESIZE_ALGO_BICUBIC;
                        hparams.image_pad_color   = {127, 127, 127};
                        hparams.rope_theta = 10000.0f;
                        get_u32(KEY_PROJ_SCALE_FACTOR, hparams.n_merge);
                        get_u32(KEY_IMAGE_MIN_PIXELS,  hparams.image_min_pixels);
                        hparams.dsv4_max_n_token  = 384;
                        hparams.dsv4_max_wh_ratio = 8;
                        const int patch_area = hparams.patch_size * hparams.patch_size * hparams.n_merge * hparams.n_merge;
                        // handle min/max token counts from CLI
                        if (hparams.custom_image_min_tokens > 0) {
                            hparams.image_min_pixels = hparams.custom_image_min_tokens * patch_area;
                        }
                        if (hparams.custom_image_max_tokens > 0) {
                            // the cap is on the whole token block, keep some room for the resize solver
                            hparams.dsv4_max_n_token = std::max(hparams.custom_image_max_tokens, 16);
                        }
                        hparams.image_max_pixels = hparams.dsv4_max_n_token * patch_area;
                        // a small custom max token count also lowers the min-pixel upscale threshold
                        hparams.image_min_pixels = std::min(hparams.image_min_pixels, hparams.image_max_pixels);
                        // avoid OOM on warmup
                        const int warmup_side = (int) std::sqrt((double) std::min(256, hparams.dsv4_max_n_token));
                        hparams.set_warmup_n_tokens(warmup_side * warmup_side);
                    } break;
                case PROJECTOR_TYPE_DEEPSEEK41V:
                    {
                        // ref: load_image in inference/image_processor.py (V4.1), PIL ImageOps.pad on gray 127
                        hparams.image_resize_algo = RESIZE_ALGO_BICUBIC;
                        hparams.image_pad_color   = {127, 127, 127};
                        hparams.rope_theta = 10000.0f;
                        get_u32(KEY_PROJ_SCALE_FACTOR, hparams.n_merge);
                        get_u32(KEY_IMAGE_MIN_PIXELS,  hparams.image_min_pixels);
                        // vision_max_n_token = 1024, vision_max_wh_ratio = null (converter asserts both)
                        hparams.dsv4_max_n_token  = 1024;
                        hparams.dsv4_max_wh_ratio = 0;
                        const int patch_area = hparams.patch_size * hparams.patch_size * hparams.n_merge * hparams.n_merge;
                        if (hparams.custom_image_min_tokens > 0) {
                            hparams.image_min_pixels = hparams.custom_image_min_tokens * patch_area;
                        }
                        if (hparams.custom_image_max_tokens > 0) {
                            // the cap is on the whole token block (newlines + start/end included)
                            hparams.dsv4_max_n_token = std::max(hparams.custom_image_max_tokens, 16);
                        }
                        hparams.image_max_pixels = hparams.dsv4_max_n_token * patch_area;
                        hparams.image_min_pixels = std::min(hparams.image_min_pixels, hparams.image_max_pixels);
                        const int warmup_side = (int) std::sqrt((double) std::min(256, hparams.dsv4_max_n_token));
                        hparams.set_warmup_n_tokens(warmup_side * warmup_side);
                    } break;
                case PROJECTOR_TYPE_GEMMA3:
                    {
                        // default value (used by all model sizes in gemma 3 family)
                        // number of patches for each **side** is reduced by a factor of 4
                        hparams.n_merge = 4;
                        hparams.image_resize_algo = RESIZE_ALGO_BILINEAR;
                        // test model (tinygemma3) has a different value, we optionally read it
                        get_u32(KEY_PROJ_SCALE_FACTOR, hparams.n_merge, false);
                    } break;

                case PROJECTOR_TYPE_GEMMA4V:
                case PROJECTOR_TYPE_GEMMA4UV:
                    {
                        hparams.rope_theta = 100.0f;
                        hparams.ffn_op = FFN_GELU;
                        log_ffn_op = "gelu";
                        hparams.n_merge = 3; // pooling_kernel_size
                        hparams.image_resize_algo = RESIZE_ALGO_BICUBIC;
                        hparams.image_resize_pad  = PAD_NONE;
                        get_u32(KEY_PROJ_SCALE_FACTOR, hparams.n_merge, false);
                        if (model.proj_type == PROJECTOR_TYPE_GEMMA4UV) {
                            // for "unified" variant, we directly use a bigger patch size, because the "token merging" is done directly on conv layer
                            hparams.patch_size = hparams.patch_size * hparams.n_merge;
                            hparams.n_merge = 1;
                        }
                        hparams.set_limit_image_tokens(70, 1120);
                        hparams.set_warmup_n_tokens(256); // avoid OOM on warmup
                    } break;

                case PROJECTOR_TYPE_GEMMA3NV:
                    {
                        // Gemma3n uses MobileNetV5 which produces 256 tokens (16x16)
                        // Similar configuration to Gemma3
                        hparams.n_merge = 1;  // MobileNetV5 handles resizing internally
                        hparams.image_resize_algo = RESIZE_ALGO_BILINEAR;
                        get_u32(KEY_PROJ_SCALE_FACTOR, hparams.n_merge, false);
                    } break;
                case PROJECTOR_TYPE_QWEN2VL:
                case PROJECTOR_TYPE_QWEN25VL:
                case PROJECTOR_TYPE_QWEN3VL:
                case PROJECTOR_TYPE_LING3VL:
                    {
                        hparams.n_merge = 2; // default value for Qwen 2 and 2.5
                        hparams.image_resize_algo = RESIZE_ALGO_BICUBIC;
                        hparams.image_resize_pad  = PAD_NONE; // the HF processor resizes to the smart_resize size, no padding
                        hparams.image_round_half_even = true; // smart_resize uses Python round()
                        get_u32(KEY_SPATIAL_MERGE_SIZE, hparams.n_merge, false);
                        get_u32(KEY_WIN_ATTN_PATTERN, hparams.n_wa_pattern, model.proj_type == PROJECTOR_TYPE_QWEN25VL); // only 2.5 requires it
                        // ref: https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/blob/main/preprocessor_config.json
                        hparams.set_limit_image_tokens(8, 4096);
                        // optional limits of the model, the custom values take precedence
                        if (hparams.custom_image_min_tokens <= 0) {
                            get_u32(KEY_IMAGE_MIN_PIXELS, hparams.image_min_pixels, false);
                        }
                        if (hparams.custom_image_max_tokens <= 0) {
                            get_u32(KEY_IMAGE_MAX_PIXELS, hparams.image_max_pixels, false);
                        }
                        hparams.set_warmup_n_tokens(46*46); // avoid OOM on warmup
                        const int warn_min_pixels = 1024 * hparams.n_merge * hparams.n_merge * hparams.patch_size * hparams.patch_size;
                        if (hparams.image_min_pixels < warn_min_pixels) {
                            LOG_WRN("%s: Qwen-VL models require at minimum 1024 image tokens to function correctly on grounding tasks\n", __func__);
                            LOG_WRN("%s: if you encounter problems with accuracy, try adding --image-min-tokens 1024\n", __func__);
                            LOG_WRN("%s: more info: https://github.com/ggml-org/llama.cpp/issues/16842\n\n", __func__);
                        }
                    } break;
                case PROJECTOR_TYPE_MAGEVL:
                    {
                        hparams.ffn_op = FFN_GELU_ERF;
                        log_ffn_op = "gelu_erf";
                        hparams.image_resize_algo = RESIZE_ALGO_BICUBIC;
                        hparams.image_resize_pad  = PAD_NONE; // the HF processor resizes to the smart_resize size, no padding
                        hparams.image_round_half_even = true; // smart_resize uses Python round()
                        get_u32(KEY_SPATIAL_MERGE_SIZE, hparams.n_merge);
                        get_u32(KEY_IMAGE_MIN_PIXELS, hparams.image_min_pixels);
                        get_u32(KEY_IMAGE_MAX_PIXELS, hparams.image_max_pixels);
                        hparams.set_warmup_n_tokens(46*46); // avoid OOM on warmup
                    } break;
                case PROJECTOR_TYPE_SENSENOVA_U1:
                    {
                        hparams.n_merge = 2;
                        hparams.image_resize_algo = RESIZE_ALGO_BICUBIC;
                        get_u32(KEY_SPATIAL_MERGE_SIZE, hparams.n_merge, false);
                        get_u32(KEY_IMAGE_MIN_PIXELS, hparams.image_min_pixels);
                        get_u32(KEY_IMAGE_MAX_PIXELS, hparams.image_max_pixels);
                        hparams.rope_theta = 10000.0f;
                        hparams.set_limit_image_tokens(16, 4096);
                        if (hparams.n_merge != 2) {
                            throw std::runtime_error(string_format("%s: SenseNova U1 requires spatial_merge_size=2, got %u", __func__, hparams.n_merge));
                        }
                        hparams.set_warmup_n_tokens(16*16);
                    } break;
                case PROJECTOR_TYPE_LFM3VL:
                    {
                        std::string resize_algo;
                        get_string(KEY_IMAGE_RESIZE_ALGO, resize_algo);
                        // Lfm3VlImageProcessor resizes with torchvision
                        if (resize_algo == "bilinear") {
                            hparams.set_image_resize_algo(RESIZE_ALGO_BILINEAR, true);
                        } else if (resize_algo == "bicubic") {
                            hparams.set_image_resize_algo(RESIZE_ALGO_BICUBIC, true);
                        } else {
                            throw std::runtime_error(string_format("%s: unsupported image resize algo: %s\n", __func__, resize_algo.c_str()));
                        }
                        // smart_resize stretches to the aligned size, no letterbox
                        hparams.image_resize_pad = PAD_NONE;
                        get_u32(KEY_SPATIAL_MERGE_SIZE, hparams.n_merge);
                        // token counts and the projector graph assume a 2x2 merge
                        if (hparams.n_merge != 2) {
                            throw std::runtime_error(string_format("%s: unsupported spatial_merge_size %d for lfm3vl, only 2 is supported\n", __func__, hparams.n_merge));
                        }
                        get_u32(string_format(KEY_N_HEAD_KV, "vision"), hparams.n_head_kv);
                        hparams.rope_theta = 1000000.0f; // vision_rope_theta
                        // HF processor size; the max is also capped at 4096 tokens as an OOM guard
                        int min_pixels = 0;
                        int max_pixels = 0;
                        get_u32(KEY_IMAGE_MIN_PIXELS, min_pixels);
                        get_u32(KEY_IMAGE_MAX_PIXELS, max_pixels);
                        if (min_pixels <= 0 || max_pixels < min_pixels) {
                            throw std::runtime_error(string_format("%s: invalid image min/max pixels %d/%d\n", __func__, min_pixels, max_pixels));
                        }
                        const int patch_area = hparams.patch_size * hparams.patch_size * hparams.n_merge * hparams.n_merge;
                        hparams.set_limit_image_tokens(min_pixels / patch_area, std::min(4096, max_pixels / patch_area));
                        hparams.set_warmup_n_tokens(46*46); // avoid OOM on warmup
                    } break;
                case PROJECTOR_TYPE_MINIMAX_M3:
                    {
                        hparams.n_merge = 2; // spatial_merge_size
                        hparams.image_resize_algo = RESIZE_ALGO_BICUBIC;
                        hparams.image_resize_pad  = PAD_NONE;
                        get_u32(KEY_SPATIAL_MERGE_SIZE, hparams.n_merge, false);
                        // n_merge is used as a divisor in mtmd_subcontext::encode()
                        // (gh / n_merge); reject 0 to avoid int div-by-zero (DoS).
                        GGML_ASSERT(hparams.n_merge > 0);
                        hparams.rope_theta = 10000.0f; // vision_config.rope_theta
                        // MiniMax-M3: max_pixels 451584 (=672^2) -> 576 merged tokens (image_seq_length)
                        hparams.set_limit_image_tokens(8, 576);
                        hparams.set_warmup_n_tokens(16*16);
                    } break;
                case PROJECTOR_TYPE_MUSE_SPARK_AUDIO:
                    {
                        hparams.audio_sample_rate = 24000; // mono waveform, no frontend
                        get_u32(KEY_A_CONV_KERNEL_SIZE, hparams.audio_conv_kernel_size, true);
                        get_arr_int(KEY_A_HIDDEN_SIZES, hparams.audio_hidden_sizes, true);
                        get_arr_int(KEY_A_PATCH_SIZES,  hparams.audio_patch_sizes,  true);
                        get_arr_int(KEY_A_DEPTHS,       hparams.audio_depths,       true);

                        // the graph indexes the three arrays by stage and model.layers by the running depth sum
                        const size_t n_stage = hparams.audio_patch_sizes.size();
                        GGML_ASSERT(n_stage > 0);
                        GGML_ASSERT(hparams.audio_hidden_sizes.size() == n_stage);
                        GGML_ASSERT(hparams.audio_depths.size() == n_stage);
                        GGML_ASSERT(hparams.n_mel_bins == 1);
                        GGML_ASSERT(hparams.audio_conv_kernel_size >= 1);

                        int64_t n_blocks = 0;
                        for (size_t i = 0; i < n_stage; ++i) {
                            GGML_ASSERT(hparams.audio_hidden_sizes[i] >= 1);
                            GGML_ASSERT(hparams.audio_depths[i] >= 0);
                            n_blocks += hparams.audio_depths[i];
                        }
                        if (n_blocks != hparams.n_layer) {
                            throw std::runtime_error(string_format("%s: sum of audio depths (%" PRId64 ") does not match n_layer (%d)\n", __func__, n_blocks, hparams.n_layer));
                        }
                        hparams.warmup_audio_size = hparams.audio_samples_per_token();
                    } break;
                case PROJECTOR_TYPE_MUSE_GLIMMER:
                    {
                        hparams.n_merge = 2; // pixel-shuffle downsample after the ViT
                        hparams.image_resize_algo = RESIZE_ALGO_LANCZOS;
                        hparams.rope_theta = 10000.0f;
                        hparams.muse_glimmer_patch_temporal = 2;
                        hparams.muse_glimmer_sparse_factor  = 4; // 3 sparse layers + 1 global, repeating
                        get_u32(KEY_SPATIAL_MERGE_SIZE,  hparams.n_merge,             false);
                        hparams.set_limit_image_tokens(1, 4096);
                        hparams.set_warmup_n_tokens(32*32);
                    } break;
                case PROJECTOR_TYPE_MIMOVL:
                    {
                        hparams.n_merge = 2; // spatial_merge_size
                        hparams.image_resize_algo = RESIZE_ALGO_BICUBIC;
                        hparams.image_resize_pad  = PAD_NONE; // the HF processor resizes to the smart_resize size, no padding
                        hparams.image_round_half_even = true; // smart_resize uses Python round()
                        get_u32(KEY_SPATIAL_MERGE_SIZE, hparams.n_merge, false);
                        get_u32(string_format(KEY_N_HEAD_KV, "vision"), hparams.n_head_kv);
                        // 1D banded sliding-window radius (visual_token_window_size); required
                        get_u32(KEY_ATTN_WINDOW_SIZE, hparams.attn_window_size);
                        std::vector<int> pat;
                        get_arr_int(KEY_WA_PATTERN_MODE, pat, true);
                        GGML_ASSERT((int) pat.size() == hparams.n_layer && "mimovl wa_pattern_mode length must equal n_layer");
                        hparams.wa_pattern_mode.assign(pat.begin(), pat.end());
                        get_u32(KEY_IMAGE_MIN_PIXELS, hparams.image_min_pixels);
                        get_u32(KEY_IMAGE_MAX_PIXELS, hparams.image_max_pixels);
                        hparams.set_warmup_n_tokens(46*46); // avoid OOM on warmup
                    } break;
                case PROJECTOR_TYPE_STEP3VL:
                    {
                        hparams.n_merge = 4; // two stride-2 downsamplers after patching
                        get_u32(KEY_PROJ_SCALE_FACTOR, hparams.n_merge, false);
                        hparams.rope_theta = 10000.0f;
                        get_u32(KEY_PREPROC_IMAGE_SIZE, hparams.image_longest_edge, false);
                        if (hparams.image_longest_edge == 0) {
                            hparams.image_longest_edge = 3024;
                        }
                        // note: the step3vl preprocessor slices based on a fixed window grid, so it does not support custom min/max image tokens
                        hparams.warmup_image_size = hparams.image_size;
                    } break;
                case PROJECTOR_TYPE_YOUTUVL:
                    {
                        hparams.n_merge = 2;
                        hparams.image_resize_algo = RESIZE_ALGO_BILINEAR;
                        hparams.image_resize_pad  = PAD_NONE;
                        get_u32(KEY_SPATIAL_MERGE_SIZE, hparams.n_merge, false);
                        get_u32(KEY_ATTN_WINDOW_SIZE, hparams.attn_window_size, true);
                        std::vector<int> wa_layer_indexes_vec;
                        get_arr_int(KEY_WIN_ATTN_LAYER_INDEXES, wa_layer_indexes_vec, true);
                        for (auto & layer : wa_layer_indexes_vec) {
                            hparams.wa_layer_indexes.insert(layer);
                        }
                        // support max_height * max_width = 8000 * 8000. 8000/16/2 = 250 image tokens
                        hparams.set_limit_image_tokens(1, 62500);
                        hparams.set_warmup_n_tokens(16*16); // avoid OOM on warmup
                    } break;
                case PROJECTOR_TYPE_YASA2:
                    {
                        hparams.ffn_op = FFN_GELU_ERF;
                        log_ffn_op = "gelu_erf";
                        hparams.image_resize_algo = RESIZE_ALGO_BICUBIC;

                        // reka model performs better when the image is stretched to fit
                        // fixed square size (no padding)
                        hparams.image_resize_pad = PAD_NONE;
                    } break;
                case PROJECTOR_TYPE_GLM4V:
                    {
                        hparams.rope_theta = 10000.0f;
                        hparams.n_merge = 2; // default value for GLM4-V
                        hparams.image_resize_algo = RESIZE_ALGO_BICUBIC;
                        get_u32(KEY_SPATIAL_MERGE_SIZE, hparams.n_merge, false);
                        hparams.set_limit_image_tokens(8, 4096);
                        hparams.set_warmup_n_tokens(46*46); // avoid OOM on warmup
                    } break;
                case PROJECTOR_TYPE_GLM5V:
                    {
                        // glm4v tower with clamped SwiGLU, ceil-aligned resize and its own token budget
                        hparams.rope_theta = 10000.0f;
                        hparams.n_merge = 2;
                        hparams.image_resize_algo = RESIZE_ALGO_BICUBIC;
                        get_u32(KEY_SPATIAL_MERGE_SIZE, hparams.n_merge, false);
                        get_f32(KEY_SWIGLU_CLAMP, hparams.swiglu_clamp, true);
                        get_u32(KEY_IMAGE_MIN_PIXELS, hparams.image_min_pixels);
                        get_u32(KEY_IMAGE_MAX_PIXELS, hparams.image_max_pixels);
                        hparams.set_limit_image_tokens();
                        hparams.set_warmup_n_tokens(46*46); // avoid OOM on warmup
                    } break;
                case PROJECTOR_TYPE_LLAMA4:
                    {
                        hparams.rope_theta = 10000.0f;
                        hparams.image_resize_algo = RESIZE_ALGO_BILINEAR;
                        get_u32(KEY_PROJ_SCALE_FACTOR, hparams.n_merge, false);
                        set_llava_uhd_res_candidates(model, 3);
                    } break;
                case PROJECTOR_TYPE_ULTRAVOX:
                case PROJECTOR_TYPE_QWEN2A:
                case PROJECTOR_TYPE_QWEN3A:
                case PROJECTOR_TYPE_GLMA:
                case PROJECTOR_TYPE_VOXTRAL:
                case PROJECTOR_TYPE_MERALION:
                case PROJECTOR_TYPE_MUSIC_FLAMINGO:
                    {
                        bool require_stack = model.proj_type == PROJECTOR_TYPE_ULTRAVOX ||
                                             model.proj_type == PROJECTOR_TYPE_VOXTRAL ||
                                             model.proj_type == PROJECTOR_TYPE_MERALION ||
                                             model.proj_type == PROJECTOR_TYPE_GLMA;
                        get_u32(KEY_A_PROJ_STACK_FACTOR, hparams.proj_stack_factor, require_stack);
                        hparams.ffn_op = FFN_GELU_ERF;
                        log_ffn_op = "gelu_erf"; // temporary solution for logging

                        // audio preprocessing params
                        hparams.audio_chunk_len    = 30; // in seconds
                        hparams.audio_sample_rate  = 16000;
                        hparams.audio_n_fft        = 400;
                        hparams.audio_window_len   = 400;
                        hparams.audio_hop_len      = 160;
                    } break;
                case PROJECTOR_TYPE_VOXTRAL_RT:
                    {
                        get_u32(KEY_A_PROJ_STACK_FACTOR, hparams.proj_stack_factor);
                        get_u32(KEY_A_ATTN_WINDOW_SIZE,  hparams.attn_window_size);
                        hparams.rope_theta = 1000000.0f; // params.json
                        hparams.ffn_op = FFN_SILU;
                        log_ffn_op = "silu";

                        // audio preprocessing params
                        hparams.audio_sample_rate  = 16000;
                        hparams.audio_n_fft        = 400;
                        hparams.audio_window_len   = 400;
                        hparams.audio_hop_len      = 160;
                        // 2 context frames + 8 s of new frames, see mtmd_subcontext::enc_state_init()
                        hparams.warmup_audio_size  = 2 + 2*400;
                    } break;
                case PROJECTOR_TYPE_MIMO_AUDIO:
                    {
                        get_u32(KEY_A_RVQ_NUM_QUANTIZERS, hparams.rvq_num_quantizers, false);
                        get_arr_int(KEY_A_RVQ_CODEBOOK_SIZE, hparams.rvq_codebook_size, false);
                        if (hparams.rvq_num_quantizers <= 0) {
                            throw std::runtime_error(string_format("%s: mimo_audio: missing %s\n", __func__, KEY_A_RVQ_NUM_QUANTIZERS));
                        }
                        if ((int) hparams.rvq_codebook_size.size() != hparams.rvq_num_quantizers) {
                            throw std::runtime_error(string_format(
                                "%s: mimo_audio: %s length (%zu) must equal %s (%d)\n", __func__,
                                KEY_A_RVQ_CODEBOOK_SIZE, hparams.rvq_codebook_size.size(),
                                KEY_A_RVQ_NUM_QUANTIZERS, hparams.rvq_num_quantizers));
                        }
                        hparams.ffn_op = FFN_GELU_ERF; // PyTorch F.gelu default (approximate="none")
                        hparams.rope_theta = 10000.0f;

                        // audio preprocessing params (mel spectrogram)
                        hparams.audio_sample_rate = 24000;
                        hparams.audio_n_fft       = 960;
                        hparams.audio_window_len  = 960;
                        hparams.audio_hop_len     = 240;

                        get_u32(KEY_A_ATTN_WINDOW_SIZE, hparams.attn_window_size);
                        std::vector<int> wa_pattern;
                        get_arr_int(KEY_A_WA_PATTERN_MODE, wa_pattern, true);
                        if ((int) wa_pattern.size() != hparams.n_layer) {
                            throw std::runtime_error(string_format(
                                "%s: mimo_audio: %s length (%zu) must equal n_layer (%d)\n", __func__,
                                KEY_A_WA_PATTERN_MODE, wa_pattern.size(), hparams.n_layer));
                        }
                        hparams.wa_pattern_mode.assign(wa_pattern.begin(), wa_pattern.end());

                        get_u32(KEY_A_LOCAL_BLOCK_COUNT, hparams.audio_local_n_layer);
                        get_u32(KEY_A_LOCAL_GROUP_SIZE, hparams.audio_local_group_size);
                        if (hparams.audio_local_group_size <= 0) {
                            throw std::runtime_error(string_format(
                                "%s: mimo_audio: %s must be > 0\n", __func__, KEY_A_LOCAL_GROUP_SIZE));
                        }
                    } break;
                case PROJECTOR_TYPE_QWEN3TTS_SPKENC:
                    {
                        // ECAPA-TDNN speaker encoder, mel front-end uses the Slaney default (fmin=0, fmax=sr/2)
                        hparams.audio_sample_rate = 24000;
                        hparams.audio_n_fft       = 1024;
                        hparams.audio_window_len  = 1024;
                        hparams.audio_hop_len     = 256;
                    } break;
                case PROJECTOR_TYPE_CHATTERBOX_SPKENC:
                    {
                        // kaldi fbank front-end of CAMPPlus: 25 ms / 10 ms framing, 512-point spectrum
                        hparams.audio_sample_rate = 16000;
                        hparams.audio_n_fft       = 512;
                        hparams.audio_window_len  = 400;
                        hparams.audio_hop_len     = 160;
                    } break;
                case PROJECTOR_TYPE_LFM3A_GEN:
                    {
                        // TODO: CUDA has no flash attention kernel for the depthformer head dim 32, enable it once it does
                        model.flash_attn_auto = false;

                        // fixed by LiquidCodecConfig, head counts come from the tensors
                        hparams.rope_theta = 1000000.0f;

                        hparams.wav_tfm_n_layer    = 8;
                        hparams.wav_tfm_n_embd     = 512;
                        hparams.wav_tfm_eps        = 1e-5f;
                        hparams.wav_tfm_rope_theta = 1000000.0f;
                        hparams.wav_tfm_swa        = 30; // decoder_window
                        hparams.wav_frame_stack    = 6;  // stft_frame_stack, STFT frames per codec frame
                        // the ISTFT runs on the host, see mtmd_gen_audio_process()
                        hparams.audio_sample_rate  = 24000;
                        hparams.audio_n_fft        = 1280;
                        hparams.audio_window_len   = 1280;
                        hparams.audio_hop_len      = 320;

                        // speech-code sampling, the converter writes the model card values; else the LFM3-Omni card
                        hparams.gen_temp           = 0.4f;
                        hparams.gen_top_k          = 0;
                        hparams.gen_penalty_repeat = 1.2f;
                        hparams.gen_penalty_last_n = 64;
                        get_f32(KEY_GEN_AUDIO_TEMP,           hparams.gen_temp,           false);
                        get_i32(KEY_GEN_AUDIO_TOP_K,          hparams.gen_top_k,          false);
                        get_f32(KEY_GEN_AUDIO_PENALTY_REPEAT, hparams.gen_penalty_repeat, false);
                        get_i32(KEY_GEN_AUDIO_PENALTY_LAST_N, hparams.gen_penalty_last_n, false);
                    } break;
                case PROJECTOR_TYPE_QWEN3TTS_GEN:
                    {
                        // TODO: hardcoded for now, read from code_predictor_config instead
                        hparams.rope_theta = 1000000.0f;

                        // code2wav params
                        hparams.wav_tfm_n_layer      = 8;
                        hparams.wav_tfm_n_embd       = 512;
                        hparams.wav_tfm_n_ff         = 1024;
                        hparams.wav_tfm_n_head       = 16;
                        hparams.wav_tfm_n_head_kv    = 16;
                        hparams.wav_tfm_eps          = 1e-5f;
                        hparams.wav_tfm_rope_theta   = 10000.0f;
                        hparams.wav_upsample_n_block = 2;
                        hparams.wav_dac_n_block      = 4;
                        hparams.wav_dac_n_res        = 3;
                        // matches the reference decoder's sliding_window (speech_tokenizer/config.json)
                        hparams.wav_tfm_swa = 72;
                    } break;
                case PROJECTOR_TYPE_DRAMA3_GEN:
                    {
                        // Fish DAC decoder, codec.pth has no config
                        hparams.audio_sample_rate    = 44100;
                        hparams.wav_tfm_n_layer      = 8;
                        hparams.wav_tfm_n_embd       = 1024;
                        hparams.wav_tfm_n_ff         = 3072;
                        hparams.wav_tfm_n_head       = 16;
                        hparams.wav_tfm_n_head_kv    = 16;
                        hparams.wav_tfm_eps          = 1e-5f;
                        hparams.wav_tfm_rope_theta   = 10000.0f;
                        hparams.wav_tfm_swa          = 128;
                        hparams.wav_upsample_n_block = 2;
                        hparams.wav_dac_n_block      = 4;
                        hparams.wav_dac_n_res        = 3;
                    } break;
                case PROJECTOR_TYPE_POCKETTTS_SPKENC:
                case PROJECTOR_TYPE_POCKETTTS_GEN:
                    {
                        // mimi front-end takes the raw waveform, no mel
                        hparams.audio_sample_rate = 24000;
                        // seanet ratios are [6,5,4] in the config, the encoder reverses them
                        hparams.seanet_ratios   = { 4, 5, 6 };
                        hparams.seanet_n_stage  = (int32_t) hparams.seanet_ratios.size();
                        hparams.mimi_downsample = 16;
                        // matches the reference transformer's "context"
                        hparams.mimi_tfm_context = 250;
                        hparams.rope_theta       = 10000.0f;
                        // flow_lm defaults, see pocket_tts/default_parameters.py
                        hparams.flow_n_step       = 1;
                        hparams.gen_eos_threshold = -4.0f;
                    } break;
                case PROJECTOR_TYPE_CHATTERBOX:
                    {
                        // the classic cfm solver unrolls up to 10 steps x 2 cfg estimator evaluations in one graph
                        model.max_nodes = 65536;

                        // s3gen output rate
                        hparams.audio_sample_rate = 24000;
                    } break;
                case PROJECTOR_TYPE_SOPRANO:
                    {
                        // SopranoDecoder defaults, n_fft comes from the head tensor
                        hparams.audio_sample_rate = 32000;
                        hparams.audio_hop_len     = 512;
                    } break;
                case PROJECTOR_TYPE_PADDLEOCR:
                    {
                        hparams.n_merge = 2;
                        hparams.image_resize_algo = RESIZE_ALGO_BICUBIC;
                        get_u32(KEY_IMAGE_MIN_PIXELS, hparams.image_min_pixels);
                        get_u32(KEY_IMAGE_MAX_PIXELS, hparams.image_max_pixels);

                        hparams.set_warmup_n_tokens(28*28); // avoid OOM on warmup
                    } break;
                case PROJECTOR_TYPE_DEEPSEEKOCR:
                case PROJECTOR_TYPE_DEEPSEEKOCR2:
                    {
                        hparams.patch_size = 16;
                        hparams.image_size = 1024;
                        hparams.warmup_image_size = 1024;
                        hparams.image_resize_algo = RESIZE_ALGO_BICUBIC;
                        hparams.image_pad_color = {127, 127, 127};

                        get_u32(KEY_SAM_N_BLOCK, hparams.sam_n_layer, true);
                        get_u32(KEY_SAM_N_HEAD, hparams.sam_n_head, true);
                        get_u32(KEY_SAM_N_EMBD, hparams.sam_n_embd, true);
                        get_u32(KEY_ATTN_WINDOW_SIZE, hparams.attn_window_size, true);
                        hparams.preproc_min_tiles = 2;
                        if (model.proj_type == PROJECTOR_TYPE_DEEPSEEKOCR) {
                            hparams.preproc_max_tiles = 9;
                            hparams.preproc_tile_size = 640;
                            // the CLIP/ViT body runs its layernorms at 1e-5 (the SAM stage uses 1e-6)
                            hparams.eps = 1e-5f;
                        }
                        if (model.proj_type == PROJECTOR_TYPE_DEEPSEEKOCR2) {
                            hparams.preproc_max_tiles = 6;
                            hparams.preproc_tile_size = 768;
                            // qwen2 encoder is GQA, requires KEY_N_HEAD_KV
                            get_u32(string_format(KEY_N_HEAD_KV, "vision"), hparams.n_head_kv);
                        }
                        // unlimited-ocr shares the v1 projector but tiles up to 32
                        get_u32(KEY_PREPROC_MIN_TILES, hparams.preproc_min_tiles, false);
                        get_u32(KEY_PREPROC_MAX_TILES, hparams.preproc_max_tiles, false);
                        GGML_ASSERT(hparams.preproc_min_tiles >= 0
                                    && hparams.preproc_min_tiles <= hparams.preproc_max_tiles
                                    && hparams.preproc_max_tiles <= 256);
                     } break;
                case PROJECTOR_TYPE_HUNYUANVL:
                    {
                        hparams.n_merge = 2;
                        hparams.image_resize_algo = RESIZE_ALGO_LANCZOS;
                        hparams.image_resize_pad = PAD_NONE;
                        hparams.ffn_op = FFN_GELU;
                        hparams.set_limit_image_tokens(256, 16384);
                        get_u32(KEY_SPATIAL_MERGE_SIZE, hparams.n_merge, false);
                        get_u32(KEY_IMAGE_MIN_PIXELS, hparams.image_min_pixels, false);
                        get_u32(KEY_IMAGE_MAX_PIXELS, hparams.image_max_pixels, false);
                        hparams.set_warmup_n_tokens(32*32);
                    } break;
                case PROJECTOR_TYPE_LFM2A:
                case PROJECTOR_TYPE_D1OMNI_A:
                    {
                        // audio preprocessing params
                        hparams.audio_chunk_len        = 1; // in seconds
                        hparams.audio_sample_rate      = 16000;
                        hparams.audio_n_fft            = 512;
                        hparams.audio_window_len       = 400;
                        hparams.audio_hop_len          = 160;
                    } break;
                case PROJECTOR_TYPE_LFM3A:
                    {
                        get_u32(KEY_A_ATTN_WINDOW_SIZE, hparams.attn_window_size);
                        if (hparams.attn_window_size <= 0) {
                            throw std::runtime_error(string_format("%s: invalid audio attention window %d\n", __func__, hparams.attn_window_size));
                        }
                        hparams.rope_theta = 1000000.0f; // checked by the converter
                        // audio preprocessing params, see mtmd_audio_preprocessor_lfm3a
                        hparams.audio_chunk_len        = 0;
                        hparams.audio_sample_rate      = 16000;
                        hparams.audio_n_fft            = 400;
                        hparams.audio_window_len       = 400;
                        hparams.audio_hop_len          = 160;
                    } break;
                case PROJECTOR_TYPE_EXAONE4_5:
                    {
                        hparams.n_merge = 2;
                        hparams.image_resize_pad  = PAD_NONE; // the HF processor resizes to the smart_resize size, no padding
                        hparams.image_round_half_even = true; // smart_resize uses Python round()
                        get_u32(KEY_SPATIAL_MERGE_SIZE, hparams.n_merge, false);
                        get_u32(KEY_WIN_ATTN_PATTERN, hparams.n_wa_pattern, false);
                        get_u32(KEY_IMAGE_MIN_PIXELS, hparams.image_min_pixels);
                        get_u32(KEY_IMAGE_MAX_PIXELS, hparams.image_max_pixels);
                        hparams.set_warmup_n_tokens(46 * 46);
                        if (hparams.rope_theta <= 0.0f) {
                            hparams.rope_theta = 10000.0f;
                        }
                        get_u32(string_format(KEY_N_HEAD_KV, "vision"), hparams.n_head_kv);
                    } break;
                case PROJECTOR_TYPE_GEMMA4A:
                    {
                        // Gemma4 feature_extraction_gemma4.py:
                        // frame_length_ms=20 -> 320 samples, n_fft=512, hop=10ms -> 160
                        hparams.audio_chunk_len        = 0;  // no fixed-length padding
                        hparams.audio_sample_rate      = 16000;
                        hparams.audio_n_fft            = 512;
                        hparams.audio_window_len       = 320;  // 20ms frame (NOT 25ms/400)
                        hparams.audio_hop_len          = 160;
                        // due to a mistake in the original conversion code, rms_norm_eps is set to a wrong value
                        // since all gemma4a models use 1e-6, we just hardcode it here to avoid re-conversion
                        hparams.eps = 1e-6f;
                    } break;
                case PROJECTOR_TYPE_GEMMA4UA:
                    {
                        // Encoder-free: raw 16 kHz waveform chunked into 640-sample frames.
                        hparams.audio_chunk_len   = 0;
                        hparams.audio_sample_rate = 16000;
                        hparams.eps               = 1e-6f;
                        hparams.n_mel_bins        = 640;
                    } break;
                case PROJECTOR_TYPE_GRANITE_SPEECH:
                    {
                        hparams.audio_chunk_len        = 0;
                        hparams.audio_sample_rate      = 16000;
                        hparams.audio_n_fft            = 512;
                        hparams.audio_window_len       = 400;
                        hparams.audio_hop_len          = 160;
                        get_u32(KEY_A_CHUNK_SIZE,           hparams.audio_chunk_size);
                        // context_size is squared for the attn_dists/mask buffers; cap to prevent int32 overflow
                        // (legitimate values are small, e.g. 12-200; 8192^2 = 67M still fits int32)
                        GGML_ASSERT(hparams.audio_chunk_size > 0 && hparams.audio_chunk_size <= 8192);
                        get_u32(KEY_A_CONV_KERNEL_SIZE,     hparams.audio_conv_kernel_size);
                        get_u32(KEY_A_MAX_POS_EMB,          hparams.audio_max_pos_emb);
                        get_u32(KEY_A_PROJ_WINDOW_SIZE,     hparams.audio_proj_window_size);
                        get_u32(KEY_A_PROJ_DOWNSAMPLE_RATE, hparams.audio_proj_downsample_rate);
                        get_u32(KEY_A_PROJ_HEAD_COUNT,      hparams.audio_proj_head_count);
                        // NOTE: feature layers loaded above in common path
                    } break;
                case PROJECTOR_TYPE_JANUS_PRO:
                    {
                        hparams.image_pad_color   = {127, 127, 127};
                        hparams.image_resize_algo = RESIZE_ALGO_BICUBIC;
                    } break;
                case PROJECTOR_TYPE_GRANITE4_VISION:
                    {
                        // SigLIP tower.
                        hparams.image_resize_algo = RESIZE_ALGO_BICUBIC;
                        hparams.image_resize_pad = PAD_CEIL;

                        // NOTE: feature_layers loaded in common path as optional
                        get_arr_int(KEY_PROJ_SPATIAL_OFFSETS, hparams.proj_spatial_offsets);
                        if (hparams.feature_layers.size() != hparams.proj_spatial_offsets.size()) {
                            throw std::runtime_error(string_format("%s: feature_layers.size() %d != proj_spatial_offsets.size() %d",
                                                                   hparams.feature_layers.size(), hparams.proj_spatial_offsets.size()));
                        }

                        get_u32(KEY_PROJ_SAMPLE_QUERY_SIDE,  hparams.downsample_query_side);
                        get_u32(KEY_PROJ_SAMPLE_WINDOW_SIDE, hparams.downsample_window_side);
                        hparams.warmup_image_size = hparams.image_size;
                    } break;
                default:
                    throw std::runtime_error(string_format("%s: unknown vision projector type %s\n", __func__, proj_type.c_str()));
            }

            // sanity check
            {
                if (hparams.image_size < 0) {
                    // note: some models having hparams.image_size == 0, which means the image size is dynamic
                    throw std::runtime_error(string_format("%s: image_size (%d) cannot be negative\n", __func__, hparams.image_size));
                }
                if (hparams.image_size > 8192) {
                    // cap prevents int32 overflow in n_patches = (image_size/patch_size)^2
                    throw std::runtime_error(string_format("%s: image_size (%d) is too large (max 8192)\n", __func__, hparams.image_size));
                }
                if (hparams.patch_size <= 0 || hparams.patch_size >= 65536) {
                    throw std::runtime_error(string_format("%s: patch_size (%d) must be positive and less than 65536\n", __func__, hparams.patch_size));
                }
                if (hparams.n_embd <= 0) {
                    throw std::runtime_error(string_format("%s: n_embd (%d) must be greater than 0\n", __func__, hparams.n_embd));
                }
                if (hparams.image_max_pixels < hparams.image_min_pixels) {
                    throw std::runtime_error(string_format("%s: image_max_pixels (%d) is less than image_min_pixels (%d)\n", __func__, hparams.image_max_pixels, hparams.image_min_pixels));
                }
                if (hparams.n_merge <= 0 || hparams.n_merge >= 65536) {
                    throw std::runtime_error(string_format("%s: n_merge (%d) must be greater than 0 and less than 65536\n", __func__, hparams.n_merge));
                }
                if (hparams.attn_window_size > 4096) {
                    throw std::runtime_error(string_format("%s: attn_window_size (%d) is too large (max 4096)\n", __func__, hparams.attn_window_size));
                }
            }

            LOG_INF("%s: projector:          %s\n", __func__, proj_type.c_str());
            LOG_INF("%s: n_embd:             %d\n", __func__, hparams.n_embd);
            LOG_INF("%s: n_head:             %d\n", __func__, hparams.n_head);
            LOG_INF("%s: n_ff:               %d\n", __func__, hparams.n_ff);
            LOG_INF("%s: n_layer:            %d\n", __func__, hparams.n_layer);
            LOG_INF("%s: ffn_op:             %s\n", __func__, log_ffn_op.c_str());
            LOG_INF("%s: projection_dim:     %d\n", __func__, hparams.projection_dim);
            if (is_vision) {
                LOG_INF("\n--- vision hparams ---\n");
                LOG_INF("%s: image_size:         %d\n", __func__, hparams.image_size);
                LOG_INF("%s: patch_size:         %d\n", __func__, hparams.patch_size);
                LOG_INF("%s: has_llava_proj:     %d\n", __func__, hparams.has_llava_projector);
                LOG_INF("%s: minicpmv_version:   %d\n", __func__, hparams.minicpmv_version);
                LOG_INF("%s: n_merge:            %d\n", __func__, hparams.n_merge);
                LOG_INF("%s: n_wa_pattern: %d\n", __func__, hparams.n_wa_pattern);
                if (!hparams.wa_layer_indexes.empty()) {
                    LOG_INF("%s: wa_layer_indexes:  ", __func__);
                    for (auto & layer : hparams.wa_layer_indexes) {
                        LOG_INF("%d ", layer);
                    }
                    LOG_INF("\n");
                }
                if (hparams.image_min_pixels > 0) {
                    LOG_INF("%s: image_min_pixels:   %d%s\n", __func__, hparams.image_min_pixels, hparams.custom_image_min_tokens > 0 ? " (custom value)" : "");
                }
                if (hparams.image_max_pixels > 0) {
                    LOG_INF("%s: image_max_pixels:   %d%s\n", __func__, hparams.image_max_pixels, hparams.custom_image_max_tokens > 0 ? " (custom value)" : "");
                }
                if (hparams.preproc_max_tiles > 0) {
                    LOG_INF("%s: preproc_tiles:      %d - %d\n", __func__, hparams.preproc_min_tiles, hparams.preproc_max_tiles);
                }
            } else if (is_audio) {
                LOG_INF("\n--- audio hparams ---\n");
                LOG_INF("%s: n_mel_bins:         %d\n", __func__, hparams.n_mel_bins);
                LOG_INF("%s: proj_stack_factor:  %d\n", __func__, hparams.proj_stack_factor);
                LOG_INF("%s: audio_chunk_len:    %d\n", __func__, hparams.audio_chunk_len);
                LOG_INF("%s: audio_sample_rate:  %d\n", __func__, hparams.audio_sample_rate);
                LOG_INF("%s: audio_n_fft:        %d\n", __func__, hparams.audio_n_fft);
                LOG_INF("%s: audio_window_len:   %d\n", __func__, hparams.audio_window_len);
                LOG_INF("%s: audio_hop_len:      %d\n", __func__, hparams.audio_hop_len);

                // GEMMA4UA is encoder-free: it uses n_mel_bins as a raw-waveform frame size (640) and has no FFT/filterbank, so the mel-range and FFT
                // checks below do not apply to it.
                // pocket-tts is encoder-free in the same sense: mimi convolves the raw waveform
                // Muse Spark is encoder-free in the same sense: its conv pyramid reads the raw waveform, one channel takes the place of the mel bins
                const bool fft_based = model.proj_type != PROJECTOR_TYPE_GEMMA4UA &&
                                       model.proj_type != PROJECTOR_TYPE_MUSE_SPARK_AUDIO &&
                                       model.proj_type != PROJECTOR_TYPE_POCKETTTS_SPKENC;

                // Validate audio hparams loaded from GGUF metadata
                if (hparams.n_mel_bins <= 0 || (fft_based && hparams.n_mel_bins > 256)) {
                    throw std::runtime_error(string_format("%s: n_mel_bins (%d) must be in range [1, 256]\n", __func__, hparams.n_mel_bins));
                }
                if (fft_based && (hparams.audio_sample_rate <= 0 || hparams.audio_n_fft <= 0 || hparams.audio_hop_len <= 0 || hparams.audio_window_len <= 0)) {
                    throw std::runtime_error(string_format("%s: audio hparams invalid: sample_rate=%d n_fft=%d window_len=%d hop_len=%d\n",
                        __func__, hparams.audio_sample_rate, hparams.audio_n_fft, hparams.audio_window_len, hparams.audio_hop_len));
                }
            }
            LOG_INF("\n");
            LOG_INF("%s: model size:         %.2f MiB\n", __func__, model_size / 1024.0 / 1024.0);
            LOG_INF("%s: metadata size:      %.2f MiB\n", __func__, ggml_get_mem_size(ctx_meta.get()) / 1024.0 / 1024.0);
        }
    }

    void load_tensors(mtmd_model & model, ggml_backend_buffer_type_t buft, bool no_alloc) {
        auto & hparams = model.hparams;
        std::map<std::string, size_t> tensor_offset;
        std::vector<ggml_tensor *> tensors_to_load;

        auto fin = open_ifstream_binary(fname);
        if (!fin) {
            throw std::runtime_error(string_format("%s: failed to open %s\n", __func__, fname.c_str()));
        }

        // TODO @ngxson : support both audio and video in the future
        const char * prefix = model.modality == MTMD_MODALITY_AUDIO ? "a"
                             : model.modality == MTMD_MODALITY_GEN_AUDIO ? "a.gen.code"
                             : "v";

        // get offsets
        for (int64_t i = 0; i < gguf_get_n_tensors(ctx_gguf.get()); ++i) {
            const char * name = gguf_get_tensor_name(ctx_gguf.get(), i);
            tensor_offset[name] = gguf_get_data_offset(ctx_gguf.get()) + gguf_get_tensor_offset(ctx_gguf.get(), i);
        }

        // create data context
        struct ggml_init_params params = {
            /*.mem_size =*/ static_cast<size_t>(gguf_get_n_tensors(ctx_gguf.get()) + 1) * ggml_tensor_overhead(),
            /*.mem_buffer =*/ NULL,
            /*.no_alloc =*/ true,
        };
        model.ctx_data.reset(ggml_init(params));
        if (!model.ctx_data) {
            throw std::runtime_error(string_format("%s: failed to init ggml context\n", __func__));
        }

        // helper function
        std::unordered_set<std::string> loaded_tensor_names;
        auto get_tensor = [&](const std::string & name, bool required = true) {
            // Each tensor should only be loaded once; duplicates indicate a bug
            if (loaded_tensor_names.count(name)) {
                throw std::runtime_error(string_format("%s: tensor already loaded: %s\n", __func__, name.c_str()));
            }
            ggml_tensor * cur = ggml_get_tensor(ctx_meta.get(), name.c_str());
            if (!cur && required) {
                throw std::runtime_error(string_format("%s: unable to find tensor %s\n", __func__, name.c_str()));
            }
            if (cur) {
                tensors_to_load.push_back(cur);
                ggml_tensor * data_tensor = ggml_dup_tensor(model.ctx_data.get(), cur);
                ggml_set_name(data_tensor, cur->name);
                loaded_tensor_names.insert(name);
                cur = data_tensor;
                // add to weight memory counter
                model.mem_weights[ggml_backend_buft_get_device(buft)] += ggml_nbytes(cur);
            }
            return cur;
        };

        // pocket-tts: the encoder and the decoder share the same layout, only the prefix differs
        auto load_seanet = [&](mtmd_seanet & seanet, bool is_decoder) {
            const char * conv_in  = is_decoder ? TN_A_GEN_WAV_SEANET_CONV_IN    : TN_A_SEANET_CONV_IN;
            const char * conv_out = is_decoder ? TN_A_GEN_WAV_SEANET_CONV_OUT   : TN_A_SEANET_CONV_OUT;
            const char * res1     = is_decoder ? TN_A_GEN_WAV_SEANET_RES_CONV1  : TN_A_SEANET_RES_CONV1;
            const char * res2     = is_decoder ? TN_A_GEN_WAV_SEANET_RES_CONV2  : TN_A_SEANET_RES_CONV2;
            const char * scale    = is_decoder ? TN_A_GEN_WAV_SEANET_SCALE_CONV : TN_A_SEANET_SCALE_CONV;

            seanet.conv_in_w  = get_tensor(string_format(conv_in,  "weight"));
            seanet.conv_in_b  = get_tensor(string_format(conv_in,  "bias"));
            seanet.conv_out_w = get_tensor(string_format(conv_out, "weight"));
            seanet.conv_out_b = get_tensor(string_format(conv_out, "bias"));

            seanet.stages.resize(hparams.seanet_n_stage);
            for (int i = 0; i < hparams.seanet_n_stage; i++) {
                auto & stage = seanet.stages[i];
                stage.res_conv1_w  = get_tensor(string_format(res1,  i, "weight"));
                stage.res_conv1_b  = get_tensor(string_format(res1,  i, "bias"));
                stage.res_conv2_w  = get_tensor(string_format(res2,  i, "weight"));
                stage.res_conv2_b  = get_tensor(string_format(res2,  i, "bias"));
                stage.scale_conv_w = get_tensor(string_format(scale, i, "weight"));
                stage.scale_conv_b = get_tensor(string_format(scale, i, "bias"));
            }
        };

        auto get_vector = [&](const std::string & name) {
            std::vector<float> result;
            auto it = tensor_offset.find(name);
            if (it == tensor_offset.end()) {
                return result;
            }

            const int64_t idx = gguf_find_tensor(ctx_gguf.get(), name.c_str());
            if (idx < 0) {
                throw std::runtime_error(string_format("%s: failed to find tensor %s\n", __func__, name.c_str()));
            }

            if (const auto type = gguf_get_tensor_type(ctx_gguf.get(), idx); type != GGML_TYPE_F32) {
                throw std::runtime_error(string_format("%s: %s must be %s, was %s\n", __func__,
                            name.c_str(), ggml_type_name(GGML_TYPE_F32), ggml_type_name(type)));
            }

            const size_t n_bytes = gguf_get_tensor_size(ctx_gguf.get(), idx);
            if (n_bytes == 0) {
                throw std::runtime_error(string_format("%s: tensor %s is empty\n", __func__, name.c_str()));
            }

            const size_t n_elems = n_bytes / sizeof(float);
            result.resize(n_elems);
            fin.seekg(it->second, std::ios::beg);
            fin.read(reinterpret_cast<char*>(result.data()), n_bytes);
            return result;
        };

        auto get_scalar = [&](const std::string & name, float default_val) {
            auto v = get_vector(name);
            if (v.empty()) {
                return default_val;
            }
            if (v.size() != 1) {
                throw std::runtime_error(string_format("%s: expected scalar tensor '%s' but got %d elements\n",
                            __func__, name.c_str(), (int) v.size()));
            }

            return v[0];
        };

        model.class_embedding = get_tensor(TN_CLASS_EMBD, false);

        model.pre_ln_w = get_tensor(string_format(TN_LN_PRE, prefix, "weight"), false);
        model.pre_ln_b = get_tensor(string_format(TN_LN_PRE, prefix, "bias"),   false);

        model.post_ln_w = get_tensor(string_format(TN_LN_POST, prefix, "weight"), false);
        model.post_ln_b = get_tensor(string_format(TN_LN_POST, prefix, "bias"),   false);

        model.patch_bias = get_tensor(TN_PATCH_BIAS, false);
        model.patch_embeddings_0 = get_tensor(TN_PATCH_EMBD,   false);
        model.patch_embeddings_1 = get_tensor(TN_PATCH_EMBD_1, false);

        model.norm_embd_w = get_tensor(string_format(TN_NORM_EMBD, "weight"), false);
        model.norm_embd_b = get_tensor(string_format(TN_NORM_EMBD, "bias"),   false);

        model.position_embeddings = get_tensor(string_format(TN_POS_EMBD, prefix), false);

        // lfm3vl
        model.lfm3vl_embedding_norm_w = get_tensor(string_format(TN_LFM3VL_EMBEDDING_NORM, prefix, "weight"), false);

        const bool has_standard_layers = (
            model.proj_type != PROJECTOR_TYPE_GEMMA3NV &&
            model.proj_type != PROJECTOR_TYPE_QWEN3TTS_SPKENC &&
            model.proj_type != PROJECTOR_TYPE_POCKETTTS_GEN &&
            model.proj_type != PROJECTOR_TYPE_SOPRANO &&
            model.proj_type != PROJECTOR_TYPE_MUSE_SPARK_AUDIO &&
            model.proj_type != PROJECTOR_TYPE_CHATTERBOX &&
            model.proj_type != PROJECTOR_TYPE_CHATTERBOX_SPKENC);

        // layers
        const int n_layers_to_load = has_standard_layers ? hparams.n_layer : 0;
        model.layers.resize(n_layers_to_load);
        for (int il = 0; il < n_layers_to_load; ++il) {
            auto & layer = model.layers[il];
            // lfm3vl / lfm3a conv layers have no attention tensors
            const bool o_w_required = model.proj_type != PROJECTOR_TYPE_LFM3VL
                                   && model.proj_type != PROJECTOR_TYPE_LFM3A;
            layer.k_w    = get_tensor(string_format(TN_ATTN_K,      prefix, il, "weight"), false);
            layer.q_w    = get_tensor(string_format(TN_ATTN_Q,      prefix, il, "weight"), false);
            layer.v_w    = get_tensor(string_format(TN_ATTN_V,      prefix, il, "weight"), false);
            layer.o_w    = get_tensor(string_format(TN_ATTN_OUTPUT, prefix, il, "weight"), o_w_required);
            layer.qkv_w  = get_tensor(string_format(TN_ATTN_QKV,    prefix, il, "weight"), false);
            layer.k_norm = get_tensor(string_format(TN_ATTN_K_NORM, prefix, il, "weight"), false);
            layer.q_norm = get_tensor(string_format(TN_ATTN_Q_NORM, prefix, il, "weight"), false);
            layer.ln_1_w = get_tensor(string_format(TN_LN_1,        prefix, il, "weight"), false);
            layer.ln_2_w = get_tensor(string_format(TN_LN_2,        prefix, il, "weight"), false);
            layer.ls_1_w        = get_tensor(string_format(TN_LS_1,         prefix, il, "weight"), false); // no bias
            layer.ls_2_w        = get_tensor(string_format(TN_LS_2,         prefix, il, "weight"), false); // no bias
            layer.ls_out_w      = get_tensor(string_format(TN_LS_OUT,        prefix, il, "weight"), false); // no bias
            layer.attn_post_norm_w = get_tensor(string_format(TN_ATTN_POST_NORM, prefix, il, "weight"), false); // no bias
            layer.ff_post_norm_w   = get_tensor(string_format(TN_FFN_POST_NORM,  prefix, il, "weight"), false); // no bias

            layer.k_b    = get_tensor(string_format(TN_ATTN_K,      prefix, il, "bias"), false);
            layer.q_b    = get_tensor(string_format(TN_ATTN_Q,      prefix, il, "bias"), false);
            layer.v_b    = get_tensor(string_format(TN_ATTN_V,      prefix, il, "bias"), false);
            layer.o_b    = get_tensor(string_format(TN_ATTN_OUTPUT, prefix, il, "bias"), false);
            layer.qkv_b  = get_tensor(string_format(TN_ATTN_QKV,    prefix, il, "bias"), false);
            layer.ln_1_b = get_tensor(string_format(TN_LN_1,        prefix, il, "bias"), false);
            layer.ln_2_b = get_tensor(string_format(TN_LN_2,        prefix, il, "bias"), false);

            // MoE ffn (dots3note vision pyramid blocks); replaces the dense ffn when present
            layer.ff_gate_inp_w  = get_tensor(string_format(TN_FFN_GATE_INP,  prefix, il, "weight"), false);
            layer.ff_gate_exps_w = get_tensor(string_format(TN_FFN_GATE_EXPS, prefix, il, "weight"), false);
            layer.ff_up_exps_w   = get_tensor(string_format(TN_FFN_UP_EXPS,   prefix, il, "weight"), false);
            layer.ff_down_exps_w = get_tensor(string_format(TN_FFN_DOWN_EXPS, prefix, il, "weight"), false);
            layer.ff_exp_probs_b = get_tensor(string_format(TN_FFN_EXP_PROBS_B, prefix, il, "weight"), false);
            const bool is_moe = layer.ff_gate_exps_w != nullptr;

            // ffn
            layer.ff_up_w   = get_tensor(string_format(TN_FFN_UP,   prefix, il, "weight"), !is_moe);
            layer.ff_up_b   = get_tensor(string_format(TN_FFN_UP,   prefix, il, "bias"),   false);
            layer.ff_gate_w = get_tensor(string_format(TN_FFN_GATE, prefix, il, "weight"), false);
            layer.ff_gate_b = get_tensor(string_format(TN_FFN_GATE, prefix, il, "bias"),   false);
            layer.ff_down_w = get_tensor(string_format(TN_FFN_DOWN, prefix, il, "weight"), !is_moe);
            layer.ff_down_b = get_tensor(string_format(TN_FFN_DOWN, prefix, il, "bias"),   false);

            // mimovl per-head attention sink bias
            layer.attn_sinks = get_tensor(string_format(TN_ATTN_SINKS, prefix, il), false);

            // qwen3vl deepstack layer
            layer.deepstack_norm_w = get_tensor(string_format(TN_DEEPSTACK_NORM, il, "weight"), false);
            layer.deepstack_norm_b = get_tensor(string_format(TN_DEEPSTACK_NORM, il, "bias"), false);
            layer.deepstack_fc1_w  = get_tensor(string_format(TN_DEEPSTACK_FC1,  il, "weight"), false);
            layer.deepstack_fc1_b  = get_tensor(string_format(TN_DEEPSTACK_FC1,  il, "bias"), false);
            layer.deepstack_fc2_w  = get_tensor(string_format(TN_DEEPSTACK_FC2,  il, "weight"), false);
            layer.deepstack_fc2_b  = get_tensor(string_format(TN_DEEPSTACK_FC2,  il, "bias"), false);
            if (layer.has_deepstack()) {
                model.n_deepstack_layers++;
            }

            // lfm3vl conv layers
            layer.conv2d_in_proj_w  = get_tensor(string_format(TN_LFM3VL_CONV2D_IN_PROJ,  prefix, il, "weight"), false);
            layer.conv2d_spatial_w  = get_tensor(string_format(TN_LFM3VL_CONV2D_SPATIAL,  prefix, il, "weight"), false);
            layer.conv2d_out_proj_w = get_tensor(string_format(TN_LFM3VL_CONV2D_OUT_PROJ, prefix, il, "weight"), false);

            // some models already exported with legacy (incorrect) naming which is quite messy, let's fix it here
            // note: Qwen model converted from the old surgery script has n_ff = 0, so we cannot use n_ff to check!
            bool is_ffn_swapped = (
                    // only old models need this fix
                    model.proj_type == PROJECTOR_TYPE_MLP
                    || model.proj_type == PROJECTOR_TYPE_MLP_NORM
                    || model.proj_type == PROJECTOR_TYPE_LDP
                    || model.proj_type == PROJECTOR_TYPE_LDPV2
                    || model.proj_type == PROJECTOR_TYPE_QWEN2VL
                    || model.proj_type == PROJECTOR_TYPE_QWEN25VL
                    || model.proj_type == PROJECTOR_TYPE_EXAONE4_5
                    || model.proj_type == PROJECTOR_TYPE_GLM_EDGE
                    || model.proj_type == PROJECTOR_TYPE_GEMMA3
                    || model.proj_type == PROJECTOR_TYPE_IDEFICS3
                    || model.proj_type == PROJECTOR_TYPE_MINICPMV
                    || model.proj_type == PROJECTOR_TYPE_MINICPMV4_6
                    || model.proj_type == PROJECTOR_TYPE_MINICPMV4_7
                    || model.proj_type == PROJECTOR_TYPE_MINICPM_OCR_1_0
                ) && layer.ff_up_w && layer.ff_down_w && layer.ff_down_w->ne[0] == hparams.n_embd;
            if (is_ffn_swapped) {
                // swap up and down weights
                ggml_tensor * tmp = layer.ff_up_w;
                layer.ff_up_w = layer.ff_down_w;
                layer.ff_down_w = tmp;
                // swap up and down biases
                tmp = layer.ff_up_b;
                layer.ff_up_b = layer.ff_down_b;
                layer.ff_down_b = tmp;
                if (il == 0) {
                    LOG_WRN("%s: ffn up/down are swapped\n", __func__);
                }
            }
        }

        // code2wav: RVQ codes -> raw PCM, drama3 has no quant/pre_conv/tfm projections
        auto load_code2wav = [&](bool with_proj) {
            auto & c2w = model.c2w;

            c2w.quant_first_in_w  = get_tensor(string_format(TN_A_GEN_WAV_QUANT_FIRST_IN,  "weight"), with_proj);
            c2w.quant_first_out_w = get_tensor(string_format(TN_A_GEN_WAV_QUANT_FIRST_OUT, "weight"), with_proj);
            c2w.quant_first_cb_w  = get_tensor(string_format(TN_A_GEN_WAV_QUANT_FIRST_CB,  "weight"));
            c2w.quant_rest_in_w   = get_tensor(string_format(TN_A_GEN_WAV_QUANT_REST_IN,   "weight"), with_proj);
            c2w.quant_rest_out_w  = get_tensor(string_format(TN_A_GEN_WAV_QUANT_REST_OUT,  "weight"), with_proj);
            c2w.quant_rest_cb_w   = get_tensor(string_format(TN_A_GEN_WAV_QUANT_REST_CB,   "weight"));

            c2w.pre_conv_w = get_tensor(string_format(TN_A_GEN_WAV_PRE_CONV, "weight"), with_proj);
            c2w.pre_conv_b = get_tensor(string_format(TN_A_GEN_WAV_PRE_CONV, "bias"),   with_proj);

            c2w.tfm_in_proj_w     = get_tensor(string_format(TN_A_GEN_WAV_TFM_IN_PROJ,  "weight"), with_proj);
            c2w.tfm_in_proj_b     = get_tensor(string_format(TN_A_GEN_WAV_TFM_IN_PROJ,  "bias"), with_proj);
            c2w.tfm_out_proj_w    = get_tensor(string_format(TN_A_GEN_WAV_TFM_OUT_PROJ, "weight"), with_proj);
            c2w.tfm_out_proj_b    = get_tensor(string_format(TN_A_GEN_WAV_TFM_OUT_PROJ, "bias"), with_proj);
            c2w.tfm_output_norm_w = get_tensor(string_format(TN_A_GEN_WAV_TFM_OUT_NORM, "weight"));

            // loaded manually, the generic model.layers loop is taken by code_predictor
            c2w.tfm_layers.resize(hparams.wav_tfm_n_layer);
            for (int il = 0; il < hparams.wav_tfm_n_layer; il++) {
                auto & layer = c2w.tfm_layers[il];
                const char * p = "a.gen.wav.tfm";
                layer.q_w      = get_tensor(string_format(TN_ATTN_Q,      p, il, "weight"));
                layer.k_w      = get_tensor(string_format(TN_ATTN_K,      p, il, "weight"));
                layer.v_w      = get_tensor(string_format(TN_ATTN_V,      p, il, "weight"));
                layer.o_w      = get_tensor(string_format(TN_ATTN_OUTPUT, p, il, "weight"));
                layer.ln_1_w   = get_tensor(string_format(TN_LN_1,        p, il, "weight"));
                layer.ln_2_w   = get_tensor(string_format(TN_LN_2,        p, il, "weight"));
                layer.ls_1_w   = get_tensor(string_format(TN_LS_1,        p, il, "weight"));
                layer.ls_2_w   = get_tensor(string_format(TN_LS_2,        p, il, "weight"));
                layer.ff_gate_w = get_tensor(string_format(TN_FFN_GATE,   p, il, "weight"));
                layer.ff_up_w   = get_tensor(string_format(TN_FFN_UP,     p, il, "weight"));
                layer.ff_down_w = get_tensor(string_format(TN_FFN_DOWN,   p, il, "weight"));
            }

            // upsample: 2x (causal ConvTranspose1d + ConvNeXt block)
            c2w.upsample.resize(hparams.wav_upsample_n_block);
            for (int il = 0; il < hparams.wav_upsample_n_block; il++) {
                auto & up = c2w.upsample[il];
                up.conv_w   = get_tensor(string_format(TN_A_GEN_WAV_UP_CONV,   il, "weight"));
                up.conv_b   = get_tensor(string_format(TN_A_GEN_WAV_UP_CONV,   il, "bias"));
                up.dwconv_w = get_tensor(string_format(TN_A_GEN_WAV_UP_DWCONV, il, "weight"));
                up.dwconv_b = get_tensor(string_format(TN_A_GEN_WAV_UP_DWCONV, il, "bias"));
                up.norm_w   = get_tensor(string_format(TN_A_GEN_WAV_UP_NORM,   il, "weight"));
                up.norm_b   = get_tensor(string_format(TN_A_GEN_WAV_UP_NORM,   il, "bias"));
                up.pw1_w    = get_tensor(string_format(TN_A_GEN_WAV_UP_PW1,    il, "weight"));
                up.pw1_b    = get_tensor(string_format(TN_A_GEN_WAV_UP_PW1,    il, "bias"));
                up.pw2_w    = get_tensor(string_format(TN_A_GEN_WAV_UP_PW2,    il, "weight"));
                up.pw2_b    = get_tensor(string_format(TN_A_GEN_WAV_UP_PW2,    il, "bias"));
                up.gamma    = get_tensor(string_format(TN_A_GEN_WAV_UP_GAMMA,  il));
            }

            // DAC decoder: conv_pre + n upsample blocks (each with n_res residual units) + conv_post
            c2w.dac_entry_w = get_tensor(string_format(TN_A_GEN_WAV_DAC_ENTRY, "weight"));
            c2w.dac_entry_b = get_tensor(string_format(TN_A_GEN_WAV_DAC_ENTRY, "bias"));

            c2w.dac.resize(hparams.wav_dac_n_block);
            for (int il = 0; il < hparams.wav_dac_n_block; il++) {
                auto & blk = c2w.dac[il];
                blk.snake_alpha = get_tensor(string_format(TN_A_GEN_WAV_DAC_SNAKE, il, "alpha"));
                blk.snake_beta  = get_tensor(string_format(TN_A_GEN_WAV_DAC_SNAKE, il, "beta"));
                blk.conv_w      = get_tensor(string_format(TN_A_GEN_WAV_DAC_CONV,  il, "weight"));
                blk.conv_b      = get_tensor(string_format(TN_A_GEN_WAV_DAC_CONV,  il, "bias"));

                blk.res.resize(hparams.wav_dac_n_res);
                for (int ir = 0; ir < hparams.wav_dac_n_res; ir++) {
                    auto & res = blk.res[ir];
                    res.act1_alpha = get_tensor(string_format(TN_A_GEN_WAV_DAC_RES_ACT1,  il, ir, "alpha"));
                    res.act1_beta  = get_tensor(string_format(TN_A_GEN_WAV_DAC_RES_ACT1,  il, ir, "beta"));
                    res.conv1_w    = get_tensor(string_format(TN_A_GEN_WAV_DAC_RES_CONV1, il, ir, "weight"));
                    res.conv1_b    = get_tensor(string_format(TN_A_GEN_WAV_DAC_RES_CONV1, il, ir, "bias"));
                    res.act2_alpha = get_tensor(string_format(TN_A_GEN_WAV_DAC_RES_ACT2,  il, ir, "alpha"));
                    res.act2_beta  = get_tensor(string_format(TN_A_GEN_WAV_DAC_RES_ACT2,  il, ir, "beta"));
                    res.conv2_w    = get_tensor(string_format(TN_A_GEN_WAV_DAC_RES_CONV2, il, ir, "weight"));
                    res.conv2_b    = get_tensor(string_format(TN_A_GEN_WAV_DAC_RES_CONV2, il, ir, "bias"));
                }
            }

            c2w.dac_post_snake_alpha = get_tensor(string_format(TN_A_GEN_WAV_DAC_POST_SNAKE, "alpha"));
            c2w.dac_post_snake_beta  = get_tensor(string_format(TN_A_GEN_WAV_DAC_POST_SNAKE, "beta"));
            c2w.dac_post_conv_w      = get_tensor(string_format(TN_A_GEN_WAV_DAC_POST_CONV,  "weight"));
            c2w.dac_post_conv_b      = get_tensor(string_format(TN_A_GEN_WAV_DAC_POST_CONV,  "bias"));
        };

        switch (model.proj_type) {
            case PROJECTOR_TYPE_MLP:
            case PROJECTOR_TYPE_MLP_NORM:
                {
                    // LLaVA projection
                    model.mm_0_w = get_tensor(string_format(TN_LLAVA_PROJ, 0, "weight"), false);
                    model.mm_0_b = get_tensor(string_format(TN_LLAVA_PROJ, 0, "bias"), false);
                    // Yi-type llava
                    model.mm_1_w = get_tensor(string_format(TN_LLAVA_PROJ, 1, "weight"), false);
                    model.mm_1_b = get_tensor(string_format(TN_LLAVA_PROJ, 1, "bias"), false);
                    // missing in Yi-type llava
                    model.mm_2_w = get_tensor(string_format(TN_LLAVA_PROJ, 2, "weight"), false);
                    model.mm_2_b = get_tensor(string_format(TN_LLAVA_PROJ, 2, "bias"), false);
                    // Yi-type llava
                    model.mm_3_w = get_tensor(string_format(TN_LLAVA_PROJ, 3, "weight"), false);
                    model.mm_3_b = get_tensor(string_format(TN_LLAVA_PROJ, 3, "bias"), false);
                    model.mm_4_w = get_tensor(string_format(TN_LLAVA_PROJ, 4, "weight"), false);
                    model.mm_4_b = get_tensor(string_format(TN_LLAVA_PROJ, 4, "bias"), false);
                    if (model.mm_3_w) {
                        // TODO: this is a hack to support Yi-type llava
                        model.proj_type = PROJECTOR_TYPE_MLP_NORM;
                    }
                    model.image_newline = get_tensor(TN_IMAGE_NEWLINE, false);
                } break;
            case PROJECTOR_TYPE_LDP:
                {
                    // MobileVLM projection
                    model.mm_model_mlp_1_w = get_tensor(string_format(TN_MVLM_PROJ_MLP, 1, "weight"));
                    model.mm_model_mlp_1_b = get_tensor(string_format(TN_MVLM_PROJ_MLP, 1, "bias"));
                    model.mm_model_mlp_3_w = get_tensor(string_format(TN_MVLM_PROJ_MLP, 3, "weight"));
                    model.mm_model_mlp_3_b = get_tensor(string_format(TN_MVLM_PROJ_MLP, 3, "bias"));
                    model.mm_model_block_1_block_0_0_w = get_tensor(string_format(TN_MVLM_PROJ_BLOCK, 1, 0, "0.weight"));
                    model.mm_model_block_1_block_0_1_w = get_tensor(string_format(TN_MVLM_PROJ_BLOCK, 1, 0, "1.weight"));
                    model.mm_model_block_1_block_0_1_b = get_tensor(string_format(TN_MVLM_PROJ_BLOCK, 1, 0, "1.bias"));
                    model.mm_model_block_1_block_1_fc1_w = get_tensor(string_format(TN_MVLM_PROJ_BLOCK, 1, 1, "fc1.weight"));
                    model.mm_model_block_1_block_1_fc1_b = get_tensor(string_format(TN_MVLM_PROJ_BLOCK, 1, 1, "fc1.bias"));
                    model.mm_model_block_1_block_1_fc2_w = get_tensor(string_format(TN_MVLM_PROJ_BLOCK, 1, 1, "fc2.weight"));
                    model.mm_model_block_1_block_1_fc2_b = get_tensor(string_format(TN_MVLM_PROJ_BLOCK, 1, 1, "fc2.bias"));
                    model.mm_model_block_1_block_2_0_w = get_tensor(string_format(TN_MVLM_PROJ_BLOCK, 1, 2, "0.weight"));
                    model.mm_model_block_1_block_2_1_w = get_tensor(string_format(TN_MVLM_PROJ_BLOCK, 1, 2, "1.weight"));
                    model.mm_model_block_1_block_2_1_b = get_tensor(string_format(TN_MVLM_PROJ_BLOCK, 1, 2, "1.bias"));
                    model.mm_model_block_2_block_0_0_w = get_tensor(string_format(TN_MVLM_PROJ_BLOCK, 2, 0, "0.weight"));
                    model.mm_model_block_2_block_0_1_w = get_tensor(string_format(TN_MVLM_PROJ_BLOCK, 2, 0, "1.weight"));
                    model.mm_model_block_2_block_0_1_b = get_tensor(string_format(TN_MVLM_PROJ_BLOCK, 2, 0, "1.bias"));
                    model.mm_model_block_2_block_1_fc1_w = get_tensor(string_format(TN_MVLM_PROJ_BLOCK, 2, 1, "fc1.weight"));
                    model.mm_model_block_2_block_1_fc1_b = get_tensor(string_format(TN_MVLM_PROJ_BLOCK, 2, 1, "fc1.bias"));
                    model.mm_model_block_2_block_1_fc2_w = get_tensor(string_format(TN_MVLM_PROJ_BLOCK, 2, 1, "fc2.weight"));
                    model.mm_model_block_2_block_1_fc2_b = get_tensor(string_format(TN_MVLM_PROJ_BLOCK, 2, 1, "fc2.bias"));
                    model.mm_model_block_2_block_2_0_w = get_tensor(string_format(TN_MVLM_PROJ_BLOCK, 2, 2, "0.weight"));
                    model.mm_model_block_2_block_2_1_w = get_tensor(string_format(TN_MVLM_PROJ_BLOCK, 2, 2, "1.weight"));
                    model.mm_model_block_2_block_2_1_b = get_tensor(string_format(TN_MVLM_PROJ_BLOCK, 2, 2, "1.bias"));
                } break;
            case PROJECTOR_TYPE_LDPV2:
                {
                    // MobilVLM_V2 projection
                    model.mm_model_mlp_0_w = get_tensor(string_format(TN_MVLM_PROJ_MLP, 0, "weight"));
                    model.mm_model_mlp_0_b = get_tensor(string_format(TN_MVLM_PROJ_MLP, 0, "bias"));
                    model.mm_model_mlp_2_w = get_tensor(string_format(TN_MVLM_PROJ_MLP, 2, "weight"));
                    model.mm_model_mlp_2_b = get_tensor(string_format(TN_MVLM_PROJ_MLP, 2, "bias"));
                    model.mm_model_peg_0_w = get_tensor(string_format(TN_MVLM_PROJ_PEG, 0, "weight"));
                    model.mm_model_peg_0_b = get_tensor(string_format(TN_MVLM_PROJ_PEG, 0, "bias"));
                } break;
            case PROJECTOR_TYPE_MINICPMV:
                {
                    // model.mm_model_pos_embed = get_tensor(new_clip->ctx_data, TN_MINICPMV_POS_EMBD);
                    model.mm_model_pos_embed_k = get_tensor(TN_MINICPMV_POS_EMBD_K);
                    model.mm_model_query = get_tensor(TN_MINICPMV_QUERY);
                    model.mm_model_proj = get_tensor(TN_MINICPMV_PROJ);
                    model.mm_model_kv_proj = get_tensor(TN_MINICPMV_KV_PROJ);
                    model.mm_model_attn_q_w = get_tensor(string_format(TN_MINICPMV_ATTN, "q", "weight"));
                    model.mm_model_attn_k_w = get_tensor(string_format(TN_MINICPMV_ATTN, "k", "weight"));
                    model.mm_model_attn_v_w = get_tensor(string_format(TN_MINICPMV_ATTN, "v", "weight"));
                    model.mm_model_attn_q_b = get_tensor(string_format(TN_MINICPMV_ATTN, "q", "bias"));
                    model.mm_model_attn_k_b = get_tensor(string_format(TN_MINICPMV_ATTN, "k", "bias"));
                    model.mm_model_attn_v_b = get_tensor(string_format(TN_MINICPMV_ATTN, "v", "bias"));
                    model.mm_model_attn_o_w = get_tensor(string_format(TN_MINICPMV_ATTN, "out", "weight"));
                    model.mm_model_attn_o_b = get_tensor(string_format(TN_MINICPMV_ATTN, "out", "bias"));
                    model.mm_model_ln_q_w = get_tensor(string_format(TN_MINICPMV_LN, "q", "weight"));
                    model.mm_model_ln_q_b = get_tensor(string_format(TN_MINICPMV_LN, "q", "bias"));
                    model.mm_model_ln_kv_w = get_tensor(string_format(TN_MINICPMV_LN, "kv", "weight"));
                    model.mm_model_ln_kv_b = get_tensor(string_format(TN_MINICPMV_LN, "kv", "bias"));
                    model.mm_model_ln_post_w = get_tensor(string_format(TN_MINICPMV_LN, "post", "weight"));
                    model.mm_model_ln_post_b = get_tensor(string_format(TN_MINICPMV_LN, "post", "bias"));
                } break;
            case PROJECTOR_TYPE_MINICPMV4_6:
            case PROJECTOR_TYPE_MINICPMV4_7:
            case PROJECTOR_TYPE_MINICPM_OCR_1_0:
                {
                    const bool merger_required = hparams.n_merge == 4;
                    auto get_merger_tensor = [&](const std::string & name, bool required = true) {
                        return get_tensor(name, merger_required && required);
                    };

                    // ViT merger: window self-attention
                    model.vit_merger_ln1_w     = get_merger_tensor(string_format(TN_VIT_MERGER_LN1, "weight"));
                    model.vit_merger_ln1_b     = get_merger_tensor(string_format(TN_VIT_MERGER_LN1, "bias"));
                    model.vit_merger_attn_q_w  = get_merger_tensor(string_format(TN_VIT_MERGER_ATTN_Q, "weight"));
                    model.vit_merger_attn_q_b  = get_merger_tensor(string_format(TN_VIT_MERGER_ATTN_Q, "bias"), false);
                    model.vit_merger_attn_k_w  = get_merger_tensor(string_format(TN_VIT_MERGER_ATTN_K, "weight"));
                    model.vit_merger_attn_k_b  = get_merger_tensor(string_format(TN_VIT_MERGER_ATTN_K, "bias"), false);
                    model.vit_merger_attn_v_w  = get_merger_tensor(string_format(TN_VIT_MERGER_ATTN_V, "weight"));
                    model.vit_merger_attn_v_b  = get_merger_tensor(string_format(TN_VIT_MERGER_ATTN_V, "bias"), false);
                    model.vit_merger_attn_o_w  = get_merger_tensor(string_format(TN_VIT_MERGER_ATTN_O, "weight"));
                    model.vit_merger_attn_o_b  = get_merger_tensor(string_format(TN_VIT_MERGER_ATTN_O, "bias"), false);
                    // ViT merger: MLP downsample
                    model.vit_merger_ds_ln_w   = get_merger_tensor(string_format(TN_VIT_MERGER_DS_LN, "weight"));
                    model.vit_merger_ds_ln_b   = get_merger_tensor(string_format(TN_VIT_MERGER_DS_LN, "bias"));
                    model.vit_merger_ds_up_w   = get_merger_tensor(string_format(TN_VIT_MERGER_DS_UP, "weight"));
                    model.vit_merger_ds_up_b   = get_merger_tensor(string_format(TN_VIT_MERGER_DS_UP, "bias"), false);
                    model.vit_merger_ds_down_w = get_merger_tensor(string_format(TN_VIT_MERGER_DS_DOWN, "weight"));
                    model.vit_merger_ds_down_b = get_merger_tensor(string_format(TN_VIT_MERGER_DS_DOWN, "bias"), false);
                    // Final Merger (DownsampleMLP)
                    model.mm_input_norm_w = get_tensor(TN_MM_INP_NORM);
                    model.mm_input_norm_b = get_tensor(TN_MM_INP_NORM_B, false);
                    model.mm_ffn_up_w     = get_tensor(string_format(TN_MM_UP,   "weight"));
                    model.mm_ffn_up_b     = get_tensor(string_format(TN_MM_UP,   "bias"), false);
                    model.mm_ffn_down_w   = get_tensor(string_format(TN_MM_DOWN, "weight"));
                    model.mm_ffn_down_b   = get_tensor(string_format(TN_MM_DOWN, "bias"), false);
                    model.tok_embd_sep    = get_tensor(TN_TOK_EMBD_SEP, model.proj_type == PROJECTOR_TYPE_MINICPMV4_7);
                } break;
            case PROJECTOR_TYPE_GLM_EDGE:
                {
                    model.mm_model_adapter_conv_w = get_tensor(string_format(TN_GLM_ADAPER_CONV, "weight"));
                    model.mm_model_adapter_conv_b = get_tensor(string_format(TN_GLM_ADAPER_CONV, "bias"));
                    model.mm_model_mlp_0_w = get_tensor(string_format(TN_GLM_ADAPTER_LINEAR, "weight"));
                    model.mm_model_ln_q_w = get_tensor(string_format(TN_GLM_ADAPTER_NORM_1, "weight"));
                    model.mm_model_ln_q_b = get_tensor(string_format(TN_GLM_ADAPTER_NORM_1, "bias"));
                    model.mm_model_mlp_1_w = get_tensor(string_format(TN_GLM_ADAPTER_D_H_2_4H, "weight"));
                    model.mm_model_mlp_2_w = get_tensor(string_format(TN_GLM_ADAPTER_GATE, "weight"));
                    model.mm_model_mlp_3_w = get_tensor(string_format(TN_GLM_ADAPTER_D_4H_2_H, "weight"));
                    model.mm_boi = get_tensor(string_format(TN_TOK_GLM_BOI));
                    model.mm_eoi = get_tensor(string_format(TN_TOK_GLM_EOI));
                } break;
            case PROJECTOR_TYPE_QWEN2VL:
            case PROJECTOR_TYPE_QWEN25VL:
            case PROJECTOR_TYPE_EXAONE4_5:
                {
                    model.mm_0_w = get_tensor(string_format(TN_LLAVA_PROJ, 0, "weight"));
                    model.mm_0_b = get_tensor(string_format(TN_LLAVA_PROJ, 0, "bias"));
                    model.mm_1_w = get_tensor(string_format(TN_LLAVA_PROJ, 2, "weight"));
                    model.mm_1_b = get_tensor(string_format(TN_LLAVA_PROJ, 2, "bias"));
                } break;
            case PROJECTOR_TYPE_MAGEVL:
                {
                    // d_head rope factors: the first half drives the even outputs, the second half the odd ones
                    model.rope_freqs = get_tensor(TN_ROPE_FREQS);
                    const int d_head = hparams.n_head > 0 ? hparams.n_embd / hparams.n_head : 0;
                    if (d_head <= 0 || d_head * hparams.n_head != hparams.n_embd || d_head % 32 != 0 ||
                            model.rope_freqs->type != GGML_TYPE_F32 || ggml_nelements(model.rope_freqs) != d_head) {
                        throw std::runtime_error(string_format("%s: hparams do not match the tensor shapes\n", __func__));
                    }
                    model.mm_0_w = get_tensor(string_format(TN_LLAVA_PROJ, 0, "weight"));
                    model.mm_0_b = get_tensor(string_format(TN_LLAVA_PROJ, 0, "bias"));
                    model.mm_1_w = get_tensor(string_format(TN_LLAVA_PROJ, 2, "weight"));
                    model.mm_1_b = get_tensor(string_format(TN_LLAVA_PROJ, 2, "bias"));
                } break;
            case PROJECTOR_TYPE_QWEN3VL:
                {
                    model.mm_0_w = get_tensor(string_format(TN_LLAVA_PROJ, 0, "weight"));
                    model.mm_0_b = get_tensor(string_format(TN_LLAVA_PROJ, 0, "bias"));
                    model.mm_1_w = get_tensor(string_format(TN_LLAVA_PROJ, 2, "weight"));
                    model.mm_1_b = get_tensor(string_format(TN_LLAVA_PROJ, 2, "bias"));
                } break;
            case PROJECTOR_TYPE_LFM3VL:
                {
                    model.mm_input_norm_w = get_tensor(TN_MM_INP_NORM);
                    model.mm_input_norm_b = get_tensor(TN_MM_INP_NORM_B);
                    model.mm_0_w = get_tensor(string_format(TN_LLAVA_PROJ, 0, "weight"));
                    model.mm_0_b = get_tensor(string_format(TN_LLAVA_PROJ, 0, "bias"));
                    model.mm_1_w = get_tensor(string_format(TN_LLAVA_PROJ, 2, "weight"));
                    model.mm_1_b = get_tensor(string_format(TN_LLAVA_PROJ, 2, "bias"));
                    // each layer must be either a conv block or a complete attention block
                    for (size_t il = 0; il < model.layers.size(); ++il) {
                        const auto & layer = model.layers[il];
                        const bool is_conv = layer.conv2d_in_proj_w != nullptr;
                        const bool is_attn = layer.q_w && layer.k_w && layer.v_w && layer.o_w;
                        if (is_conv == is_attn) {
                            throw std::runtime_error(string_format("%s: lfm3vl layer %zu is neither a complete conv nor a complete attention block\n", __func__, il));
                        }
                    }
                } break;
            case PROJECTOR_TYPE_LING3VL:
                {
                    model.mm_input_norm_w = get_tensor(TN_MM_INP_NORM);        // merger.norm
                    model.mm_input_norm_b = get_tensor(TN_MM_INP_NORM_B);     // merger.norm
                    model.mm_0_w = get_tensor(string_format(TN_LLAVA_PROJ, 0, "weight"));  // linear_proj.0
                    model.mm_0_b = get_tensor(string_format(TN_LLAVA_PROJ, 0, "bias"));
                    model.mm_1_w = get_tensor(string_format(TN_LLAVA_PROJ, 2, "weight"));  // linear_proj.2
                    model.mm_1_b = get_tensor(string_format(TN_LLAVA_PROJ, 2, "bias"));
                } break;
            case PROJECTOR_TYPE_MIMOVL:
                {
                    model.mm_0_w = get_tensor(string_format(TN_LLAVA_PROJ, 0, "weight"));
                    model.mm_0_b = get_tensor(string_format(TN_LLAVA_PROJ, 0, "bias"), false);
                    model.mm_1_w = get_tensor(string_format(TN_LLAVA_PROJ, 2, "weight"));
                    model.mm_1_b = get_tensor(string_format(TN_LLAVA_PROJ, 2, "bias"), false);
                } break;
            case PROJECTOR_TYPE_MINIMAX_M3:
                {
                    // per-patch MLP: mm.1 -> gelu -> mm.2
                    model.mm_1_w = get_tensor(string_format(TN_LLAVA_PROJ, 1, "weight"));
                    model.mm_1_b = get_tensor(string_format(TN_LLAVA_PROJ, 1, "bias"));
                    model.mm_2_w = get_tensor(string_format(TN_LLAVA_PROJ, 2, "weight"));
                    model.mm_2_b = get_tensor(string_format(TN_LLAVA_PROJ, 2, "bias"));
                    // 2x2 merge MLP: mm.merge.fc1 -> gelu -> mm.merge.fc2
                    model.mm_merger_fc1_w = get_tensor(string_format(TN_MM_MERGER_FC1, "weight"));
                    model.mm_merger_fc1_b = get_tensor(string_format(TN_MM_MERGER_FC1, "bias"));
                    model.mm_merger_fc2_w = get_tensor(string_format(TN_MM_MERGER_FC2, "weight"));
                    model.mm_merger_fc2_b = get_tensor(string_format(TN_MM_MERGER_FC2, "bias"));
                } break;
            case PROJECTOR_TYPE_MUSE_SPARK_AUDIO:
                {
                    model.layers.resize(hparams.n_layer);
                    model.a_patches.resize(hparams.audio_patch_sizes.size());
                    for (size_t i = 0; i < model.a_patches.size(); ++i) {
                        auto & pe = model.a_patches[i];
                        pe.proj_w = get_tensor(string_format(TN_A_PATCH,      (int) i, "weight"));
                        pe.proj_b = get_tensor(string_format(TN_A_PATCH,      (int) i, "bias"));
                        pe.norm_w = get_tensor(string_format(TN_A_PATCH_NORM, (int) i, "weight"));
                        if (pe.proj_w->ne[1] != hparams.audio_hidden_sizes[i]) {
                            throw std::runtime_error(string_format("%s: audio stage %zu projection has width %" PRId64 ", expected %d\n", __func__, i, pe.proj_w->ne[1], hparams.audio_hidden_sizes[i]));
                        }
                    }

                    for (int il = 0; il < hparams.n_layer; ++il) {
                        auto & layer = model.layers[il];
                        layer.conv_w      = get_tensor(string_format(TN_CONV1D, il, "weight"));
                        layer.conv_b      = get_tensor(string_format(TN_CONV1D, il, "bias"));
                        layer.conv_norm_w = get_tensor(string_format(TN_A_CONV1D_NORM, il, "weight"));
                        layer.ff_up_w     = get_tensor(string_format(TN_FFN_UP,   "a", il, "weight"));
                        layer.ff_up_b     = get_tensor(string_format(TN_FFN_UP,   "a", il, "bias"));
                        layer.ff_down_w   = get_tensor(string_format(TN_FFN_DOWN, "a", il, "weight"));
                        layer.ff_down_b   = get_tensor(string_format(TN_FFN_DOWN, "a", il, "bias"));
                    }

                    // pre_ln and post_ln come from the generic path above, where they are optional
                    if (!model.pre_ln_w || !model.post_ln_w) {
                        throw std::runtime_error(string_format("%s: missing pre_ln or post_ln weight\n", __func__));
                    }
                    model.mm_fc_w   = get_tensor(string_format(TN_MM_AUDIO_FC, "weight"));
                    model.mm_fc_b   = get_tensor(string_format(TN_MM_AUDIO_FC, "bias"));
                } break;
            case PROJECTOR_TYPE_MUSE_GLIMMER:
                {
                    // 3-linear MLP: fc -> erf-GELU -> proj -> erf-GELU -> vision_proj (into LLM residual dim)
                    model.mm_0_w = get_tensor(string_format(TN_LLAVA_PROJ, 0, "weight"));
                    model.mm_1_w = get_tensor(string_format(TN_LLAVA_PROJ, 1, "weight"));
                    model.mm_2_w = get_tensor(string_format(TN_LLAVA_PROJ, 2, "weight"));
                } break;
            case PROJECTOR_TYPE_STEP3VL:
                {
                    model.mm_0_w     = get_tensor(string_format(TN_LLAVA_PROJ, 0, "weight"));
                    model.mm_0_b     = get_tensor(string_format(TN_LLAVA_PROJ, 0, "bias"), false);
                    model.mm_1_w     = get_tensor(string_format(TN_LLAVA_PROJ, 1, "weight"));
                    model.mm_1_b     = get_tensor(string_format(TN_LLAVA_PROJ, 1, "bias"), false);
                    model.mm_model_proj = get_tensor(string_format(TN_MM_PROJECTOR, "weight"));
                } break;
            case PROJECTOR_TYPE_YOUTUVL:
                {
                    model.mm_input_norm_w = get_tensor(TN_MM_INP_NORM);        // merger.ln_q (RMS norm)
                    model.mm_0_w = get_tensor(string_format(TN_LLAVA_PROJ, 0, "weight"));  // merger.mlp.0
                    model.mm_0_b = get_tensor(string_format(TN_LLAVA_PROJ, 0, "bias"));
                    model.mm_1_w = get_tensor(string_format(TN_LLAVA_PROJ, 2, "weight"));  // merger.mlp.2
                    model.mm_1_b = get_tensor(string_format(TN_LLAVA_PROJ, 2, "bias"));
                } break;
            case PROJECTOR_TYPE_YASA2:
                {
                    // reuse tensors already loaded by the common section
                    // (TN_PATCH_EMBD and TN_PATCH_BIAS have the same tensor names)
                    GGML_ASSERT(model.patch_embeddings_0 && "yasa2 requires v.patch_embd.weight");
                    model.yasa_patch_w = model.patch_embeddings_0;
                    model.yasa_patch_b = model.patch_bias;
                    model.yasa_patch_ln_w = get_tensor(TN_YASA_PATCH_LN_W, false);
                    model.yasa_patch_ln_b = get_tensor(TN_YASA_PATCH_LN_B, false);
                    model.yasa_backbone_ln_w = get_tensor(TN_YASA_BACKBONE_LN_W, false);
                    model.yasa_backbone_ln_b = get_tensor(TN_YASA_BACKBONE_LN_B, false);
                    model.yasa_vision_pos_embed = get_tensor(TN_YASA_POS_EMBD, false);
                    model.mm_0_w = get_tensor(string_format(TN_LLAVA_PROJ, 0, "weight"));
                    model.mm_0_b = get_tensor(string_format(TN_LLAVA_PROJ, 0, "bias"), false);
                    model.mm_2_w = get_tensor(string_format(TN_LLAVA_PROJ, 2, "weight"));
                    model.mm_2_b = get_tensor(string_format(TN_LLAVA_PROJ, 2, "bias"), false);

                    model.yasa_stages.clear();
                    for (int s = 0; ; ++s) {
                        yasa2_stage stage;
                        stage.down_ln_w   = get_tensor(string_format(TN_YASA_STAGE_DOWN_LN, s, "weight"), false);
                        stage.down_ln_b   = get_tensor(string_format(TN_YASA_STAGE_DOWN_LN, s, "bias"), false);
                        stage.down_conv_w = get_tensor(string_format(TN_YASA_STAGE_DOWN_CONV, s, "weight"), false);
                        stage.down_conv_b = get_tensor(string_format(TN_YASA_STAGE_DOWN_CONV, s, "bias"), false);

                        for (int bi = 0; ; ++bi) {
                            yasa2_block blk;
                            blk.dw_w = get_tensor(string_format(TN_YASA_STAGE_BLK, s, bi, "dw", "weight"), false);
                            if (!blk.dw_w) {
                                break;
                            }
                            blk.dw_b  = get_tensor(string_format(TN_YASA_STAGE_BLK, s, bi, "dw", "bias"), false);
                            blk.ln_w  = get_tensor(string_format(TN_YASA_STAGE_BLK, s, bi, "ln", "weight"), false);
                            blk.ln_b  = get_tensor(string_format(TN_YASA_STAGE_BLK, s, bi, "ln", "bias"), false);
                            blk.pw1_w = get_tensor(string_format(TN_YASA_STAGE_BLK, s, bi, "pw1", "weight"), false);
                            blk.pw1_b = get_tensor(string_format(TN_YASA_STAGE_BLK, s, bi, "pw1", "bias"), false);
                            blk.grn_w = get_tensor(string_format(TN_YASA_STAGE_BLK, s, bi, "grn", "weight"), false);
                            blk.grn_b = get_tensor(string_format(TN_YASA_STAGE_BLK, s, bi, "grn", "bias"), false);
                            blk.pw2_w = get_tensor(string_format(TN_YASA_STAGE_BLK, s, bi, "pw2", "weight"), false);
                            blk.pw2_b = get_tensor(string_format(TN_YASA_STAGE_BLK, s, bi, "pw2", "bias"), false);
                            stage.blocks.push_back(blk);
                        }

                        if (!stage.down_conv_w && stage.blocks.empty()) {
                            break;
                        }
                        model.yasa_stages.push_back(std::move(stage));
                    }
                } break;
            case PROJECTOR_TYPE_GLM4V:
            case PROJECTOR_TYPE_GLM5V:
                {
                    model.mm_fc_w        = get_tensor(string_format(TN_MM_PROJECTOR, "weight"));
                    model.mm_ffn_up_w    = get_tensor(string_format(TN_MM_UP,        "weight"));
                    model.mm_ffn_up_b    = get_tensor(string_format(TN_MM_UP,        "bias"), false);
                    model.mm_ffn_gate_w  = get_tensor(string_format(TN_MM_GATE,      "weight"));
                    model.mm_ffn_gate_b  = get_tensor(string_format(TN_MM_GATE,      "bias"), false);
                    model.mm_ffn_down_w  = get_tensor(string_format(TN_MM_DOWN,      "weight"));
                    model.mm_ffn_down_b  = get_tensor(string_format(TN_MM_DOWN,      "bias"), false);
                    model.mm_post_norm_w = get_tensor(string_format(TN_MM_POST_NORM, "weight"));
                    model.mm_post_norm_b = get_tensor(string_format(TN_MM_POST_NORM, "bias"), false);
                    model.mm_patch_merger_w = get_tensor(string_format(TN_MM_PATCH_MERGER, "weight"));
                    model.mm_patch_merger_b = get_tensor(string_format(TN_MM_PATCH_MERGER, "bias"));
                } break;
            case PROJECTOR_TYPE_GEMMA3:
                {
                    model.mm_input_proj_w = get_tensor(TN_MM_INP_PROJ);
                    model.mm_soft_emb_norm_w = get_tensor(TN_MM_SOFT_EMB_N);
                } break;
            case PROJECTOR_TYPE_GEMMA4V:
                {
                    model.mm_input_proj_w = get_tensor(TN_MM_INP_PROJ);
                    model.std_bias  = get_tensor(TN_STD_BIAS,  false);
                    model.std_scale = get_tensor(TN_STD_SCALE, false);
                    // load scalar for Gemma4ClippableLinear
                    for (auto * tensor : tensors_to_load) {
                        std::string name = tensor->name;
                        if (string_ends_with(name, ".weight")) {
                            std::string name_inp_max = name;
                            std::string name_inp_min = name;
                            std::string name_out_max = name;
                            std::string name_out_min = name;
                            string_replace_all(name_inp_max, ".weight", ".input_max");
                            string_replace_all(name_inp_min, ".weight", ".input_min");
                            string_replace_all(name_out_max, ".weight", ".output_max");
                            string_replace_all(name_out_min, ".weight", ".output_min");
                            model.clamp_info_map[name] = {
                                get_scalar(name_inp_max, FLT_MAX),
                                get_scalar(name_inp_min, -FLT_MAX),
                                get_scalar(name_out_max, FLT_MAX),
                                get_scalar(name_out_min, -FLT_MAX)
                            };
                        }
                    }
                } break;
            case PROJECTOR_TYPE_GEMMA4UV:
                {
                    model.mm_input_proj_w = get_tensor(TN_MM_INP_PROJ);
                    model.patch_norm_1_w = get_tensor(string_format(TN_PATCH_NORM, 1, "weight"));
                    model.patch_norm_1_b = get_tensor(string_format(TN_PATCH_NORM, 1, "bias"));
                    model.patch_norm_2_w = get_tensor(string_format(TN_PATCH_NORM, 2, "weight"));
                    model.patch_norm_2_b = get_tensor(string_format(TN_PATCH_NORM, 2, "bias"));
                    model.patch_norm_3_w = get_tensor(string_format(TN_PATCH_NORM, 3, "weight")); // pos_norm
                    model.patch_norm_3_b = get_tensor(string_format(TN_PATCH_NORM, 3, "bias"));   // pos_norm
                } break;
            case PROJECTOR_TYPE_GEMMA3NV:
                {
                    model.mobilenet_stem_conv_w = get_tensor(TN_MNV5_STEM_CONV, false);
                    model.mobilenet_stem_conv_b = get_tensor(TN_MNV5_STEM_BIAS, false);
                    model.mobilenet_stem_norm_w = get_tensor(TN_MNV5_STEM_BN, false);

                    model.msfa_ffn_expand_w  = get_tensor(TN_MNV5_MSFA_FFN_EXP_W, false);
                    model.msfa_ffn_expand_bn = get_tensor(TN_MNV5_MSFA_FFN_EXP_BN, false); // Consume BN if present but likely folded
                    model.msfa_ffn_project_w = get_tensor(TN_MNV5_MSFA_FFN_PROJ_W, false);
                    model.msfa_ffn_project_bn = get_tensor(TN_MNV5_MSFA_FFN_PROJ_BN, false);

                    model.msfa_concat_norm_w = get_tensor(TN_MNV5_MSFA_NORM, false);

                    // Dynamically load blocks stage by stage
                    for (int stage = 0; stage < 4; ++stage) {
                        int blocks_found_in_stage = 0;

                        for (int blk_idx = 0; ; ++blk_idx) {
                            bool found_block = false;
                            mobilenetv5_block block;

                            // 1. Check for Edge Residual (S0)
                            block.s0_conv_exp_w = get_tensor(string_format(TN_MNV5_BLK_S0_EXP_W, stage, blk_idx), false);
                            if (block.s0_conv_exp_w) {
                                found_block = true;
                                block.s0_bn1_w      = get_tensor(string_format(TN_MNV5_BLK_S0_BN1_W, stage, blk_idx), false);
                                block.s0_conv_pwl_w = get_tensor(string_format(TN_MNV5_BLK_S0_PWL_W, stage, blk_idx), false);
                                block.s0_bn2_w      = get_tensor(string_format(TN_MNV5_BLK_S0_BN2_W, stage, blk_idx), false);
                            }
                            // 2. Check for UIR (Universal Inverted Residual)
                            else {
                                // Check for dw_start OR pw_exp (some UIR blocks skip dw_start)
                                block.dw_start_w = get_tensor(string_format(TN_MNV5_BLK_DW_START_W, stage, blk_idx), false);
                                block.pw_exp_w   = get_tensor(string_format(TN_MNV5_BLK_PW_EXP_W, stage, blk_idx), false);

                                if (block.dw_start_w || block.pw_exp_w) {
                                    found_block = true;
                                    if (block.dw_start_w) {
                                        block.dw_start_bn_w = get_tensor(string_format(TN_MNV5_BLK_DW_START_BN, stage, blk_idx), false);
                                    }
                                    if (block.pw_exp_w) {
                                        block.pw_exp_bn_w   = get_tensor(string_format(TN_MNV5_BLK_PW_EXP_BN, stage, blk_idx), false);
                                    }
                                    block.dw_mid_w      = get_tensor(string_format(TN_MNV5_BLK_DW_MID_W, stage, blk_idx), false);
                                    if (block.dw_mid_w) {
                                        block.dw_mid_bn_w   = get_tensor(string_format(TN_MNV5_BLK_DW_MID_BN, stage, blk_idx), false);
                                    }
                                    block.pw_proj_w     = get_tensor(string_format(TN_MNV5_BLK_PW_PROJ_W, stage, blk_idx), false);
                                    if (block.pw_proj_w) {
                                        block.pw_proj_bn_w  = get_tensor(string_format(TN_MNV5_BLK_PW_PROJ_BN, stage, blk_idx), false);
                                    }
                                    block.layer_scale_w = get_tensor(string_format(TN_MNV5_BLK_LAYER_SCALE, stage, blk_idx), false);
                                }
                            }

                            // 3. Check for Attention (MQA)
                            // Even if UIR/Edge check failed, this might be a pure attention block
                            ggml_tensor* attn_q_check = get_tensor(string_format(TN_MNV5_ATTN_Q_W, stage, blk_idx), false);
                            if (attn_q_check) {
                                found_block = true;
                                block.attn_q_w = attn_q_check;
                                block.attn_k_w = get_tensor(string_format(TN_MNV5_ATTN_K_W, stage, blk_idx), false);
                                block.attn_v_w = get_tensor(string_format(TN_MNV5_ATTN_V_W, stage, blk_idx), false);
                                block.attn_o_w = get_tensor(string_format(TN_MNV5_ATTN_O_W, stage, blk_idx), false);
                                block.attn_k_dw_w   = get_tensor(string_format(TN_MNV5_ATTN_K_DW, stage, blk_idx), false);
                                block.attn_k_norm_w = get_tensor(string_format(TN_MNV5_ATTN_K_NORM, stage, blk_idx), false);
                                block.attn_v_dw_w   = get_tensor(string_format(TN_MNV5_ATTN_V_DW, stage, blk_idx), false);
                                block.attn_v_norm_w = get_tensor(string_format(TN_MNV5_ATTN_V_NORM, stage, blk_idx), false);
                                block.attn_norm_w   = get_tensor(string_format(TN_MNV5_ATTN_NORM, stage, blk_idx), false);
                                // Note: Attention blocks also have layer_scale, load it if not already loaded by UIR check
                                if (!block.layer_scale_w) {
                                    block.layer_scale_w = get_tensor(string_format(TN_MNV5_BLK_LAYER_SCALE, stage, blk_idx), false);
                                }
                            }

                            if (found_block) {
                                model.mobilenet_blocks.push_back(block);
                                blocks_found_in_stage++;
                            } else {
                                // End of blocks for this stage
                                break;
                            }
                        }

                        // Track where this stage ends in the flat vector
                        if (blocks_found_in_stage > 0) {
                            model.mobilenet_stage_ends.push_back(model.mobilenet_blocks.size() - 1);
                            LOG_INF("%s: Stage %d ended at global block index %zu\n", __func__, stage, model.mobilenet_blocks.size() - 1);
                        }
                    }
                    model.mm_input_proj_w = get_tensor(TN_MM_INP_PROJ);
                    model.mm_soft_emb_norm_w = get_tensor(TN_MM_SOFT_EMB_N);
                } break;
            case PROJECTOR_TYPE_IDEFICS3:
                {
                    model.mm_fc_w = get_tensor(string_format(TN_MM_PROJECTOR, "weight"));
                } break;
            case PROJECTOR_TYPE_COHERE2V:
                {
                    model.mm_1_w = get_tensor(string_format(TN_LLAVA_PROJ, 1, "weight"));
                    model.mm_1_b = get_tensor(string_format(TN_LLAVA_PROJ, 1, "bias"));
                    model.mm_2_w = get_tensor(string_format(TN_LLAVA_PROJ, 2, "weight"));
                    model.mm_2_b = get_tensor(string_format(TN_LLAVA_PROJ, 2, "bias"));
                } break;
            case PROJECTOR_TYPE_LFM2:
            case PROJECTOR_TYPE_D1OMNI_V:
                {
                    model.mm_input_norm_w = get_tensor(TN_MM_INP_NORM, false);
                    model.mm_input_norm_b = get_tensor(TN_MM_INP_NORM_B, false);
                    model.mm_1_w = get_tensor(string_format(TN_LLAVA_PROJ, 1, "weight"));
                    model.mm_1_b = get_tensor(string_format(TN_LLAVA_PROJ, 1, "bias"));
                    model.mm_2_w = get_tensor(string_format(TN_LLAVA_PROJ, 2, "weight"));
                    model.mm_2_b = get_tensor(string_format(TN_LLAVA_PROJ, 2, "bias"));
                } break;
            case PROJECTOR_TYPE_KIMIVL:
            case PROJECTOR_TYPE_PADDLEOCR:
            case PROJECTOR_TYPE_KIMIK25:
                {
                    model.mm_input_norm_w = get_tensor(TN_MM_INP_NORM);
                    model.mm_input_norm_b = get_tensor(TN_MM_INP_NORM_B);
                    model.mm_1_w = get_tensor(string_format(TN_LLAVA_PROJ, 1, "weight"));
                    model.mm_1_b = get_tensor(string_format(TN_LLAVA_PROJ, 1, "bias"));
                    model.mm_2_w = get_tensor(string_format(TN_LLAVA_PROJ, 2, "weight"));
                    model.mm_2_b = get_tensor(string_format(TN_LLAVA_PROJ, 2, "bias"));
                } break;
            case PROJECTOR_TYPE_DEEPSEEK4V:
                {
                    model.mm_1_w = get_tensor(string_format(TN_LLAVA_PROJ, 1, "weight"));
                    model.mm_1_b = get_tensor(string_format(TN_LLAVA_PROJ, 1, "bias"));
                    model.mm_2_w = get_tensor(string_format(TN_LLAVA_PROJ, 2, "weight"));
                    model.mm_2_b = get_tensor(string_format(TN_LLAVA_PROJ, 2, "bias"));
                    // sentinel token embeddings written into the output block
                    model.image_newline        = get_tensor(TN_IMAGE_NEWLINE);
                    model.token_embd_img_start = get_tensor(TN_TOK_IMG_START);
                    model.token_embd_img_end   = get_tensor(TN_TOK_IMG_END);
                    model.token_embd_img_pad   = get_tensor(TN_TOK_IMG_PAD);
                } break;
            case PROJECTOR_TYPE_DEEPSEEK41V:
                {
                    model.mm_1_w = get_tensor(string_format(TN_LLAVA_PROJ, 1, "weight"));
                    model.mm_1_b = get_tensor(string_format(TN_LLAVA_PROJ, 1, "bias"));
                    model.mm_2_w = get_tensor(string_format(TN_LLAVA_PROJ, 2, "weight"));
                    model.mm_2_b = get_tensor(string_format(TN_LLAVA_PROJ, 2, "bias"));
                    // image_start / image_end / image_newline, written into the output block
                    model.image_newline        = get_tensor(TN_IMAGE_NEWLINE);
                    model.token_embd_img_start = get_tensor(TN_TOK_IMG_START);
                    model.token_embd_img_end   = get_tensor(TN_TOK_IMG_END);
                } break;
            case PROJECTOR_TYPE_PIXTRAL:
                {
                    model.mm_1_w = get_tensor(string_format(TN_LLAVA_PROJ, 1, "weight"));
                    model.mm_1_b = get_tensor(string_format(TN_LLAVA_PROJ, 1, "bias"), false);
                    model.mm_2_w = get_tensor(string_format(TN_LLAVA_PROJ, 2, "weight"));
                    model.mm_2_b = get_tensor(string_format(TN_LLAVA_PROJ, 2, "bias"), false);
                    // [IMG_BREAK] token embedding
                    model.token_embd_img_break = get_tensor(TN_TOK_IMG_BREAK);
                    // for mistral small 3.1
                    model.mm_input_norm_w   = get_tensor(TN_MM_INP_NORM, false);
                    model.mm_patch_merger_w = get_tensor(string_format(TN_MM_PATCH_MERGER, "weight"), false);
                } break;
            case PROJECTOR_TYPE_LIGHTONOCR:
                {
                    model.mm_1_w = get_tensor(string_format(TN_LLAVA_PROJ, 1, "weight"));
                    model.mm_1_b = get_tensor(string_format(TN_LLAVA_PROJ, 1, "bias"), false);
                    model.mm_2_w = get_tensor(string_format(TN_LLAVA_PROJ, 2, "weight"));
                    model.mm_2_b = get_tensor(string_format(TN_LLAVA_PROJ, 2, "bias"), false);
                    model.mm_input_norm_w   = get_tensor(TN_MM_INP_NORM, false);
                    model.mm_patch_merger_w = get_tensor(string_format(TN_MM_PATCH_MERGER, "weight"), false);
                } break;
            case PROJECTOR_TYPE_DOTS_OCR:
            case PROJECTOR_TYPE_DOTS3NOTE_V:
                {
                    model.mm_0_w = get_tensor(string_format(TN_LLAVA_PROJ, 0, "weight"));
                    model.mm_0_b = get_tensor(string_format(TN_LLAVA_PROJ, 0, "bias"));
                    model.mm_2_w = get_tensor(string_format(TN_LLAVA_PROJ, 2, "weight"));
                    model.mm_2_b = get_tensor(string_format(TN_LLAVA_PROJ, 2, "bias"));
                    model.mm_input_norm_w = get_tensor(TN_MM_INP_NORM);
                    model.mm_input_norm_b = get_tensor(TN_MM_INP_NORM_B);
                    // post_trunk_norm: applied after all ViT blocks, before the merger
                    model.post_ln_w = get_tensor(string_format(TN_MM_POST_NORM, "weight"));
                } break;
            case PROJECTOR_TYPE_DOTS3NOTE_A:
                {
                    model.conv2d_1_w = get_tensor(string_format(TN_CONV2D, 1, "weight"));
                    model.conv2d_1_b = get_tensor(string_format(TN_CONV2D, 1, "bias"));
                    model.conv2d_2_w = get_tensor(string_format(TN_CONV2D, 2, "weight"));
                    model.conv2d_2_b = get_tensor(string_format(TN_CONV2D, 2, "bias"));
                    model.conv2d_3_w = get_tensor(string_format(TN_CONV2D, 3, "weight"));
                    model.conv2d_3_b = get_tensor(string_format(TN_CONV2D, 3, "bias"));
                    model.conv_out_w = get_tensor(string_format(TN_CONV_OUT, "weight")); // no bias
                    // adapter: LayerNorm -> Linear -> GELU -> Linear
                    model.mm_norm_pre_w = get_tensor(string_format(TN_MM_NORM_PRE, "weight"));
                    model.mm_norm_pre_b = get_tensor(string_format(TN_MM_NORM_PRE, "bias"));
                    model.mm_1_w = get_tensor(string_format(TN_MM_AUDIO_MLP, 1, "weight"));
                    model.mm_1_b = get_tensor(string_format(TN_MM_AUDIO_MLP, 1, "bias"));
                    model.mm_2_w = get_tensor(string_format(TN_MM_AUDIO_MLP, 3, "weight"));
                    model.mm_2_b = get_tensor(string_format(TN_MM_AUDIO_MLP, 3, "bias"));
                } break;
            case PROJECTOR_TYPE_ULTRAVOX:
                {
                    model.conv1d_1_w = get_tensor(string_format(TN_CONV1D, 1, "weight"));
                    model.conv1d_1_b = get_tensor(string_format(TN_CONV1D, 1, "bias"));
                    model.conv1d_2_w = get_tensor(string_format(TN_CONV1D, 2, "weight"));
                    model.conv1d_2_b = get_tensor(string_format(TN_CONV1D, 2, "bias"));
                    model.mm_1_w = get_tensor(string_format(TN_MM_AUDIO_MLP, 1, "weight"));
                    model.mm_2_w = get_tensor(string_format(TN_MM_AUDIO_MLP, 2, "weight"));
                    model.mm_norm_pre_w = get_tensor(string_format(TN_MM_NORM_PRE, "weight"));
                    model.mm_norm_mid_w = get_tensor(string_format(TN_MM_NORM_MID, "weight"));
                } break;
            case PROJECTOR_TYPE_MERALION:
                {
                    // Whisper encoder conv layers
                    model.conv1d_1_w = get_tensor(string_format(TN_CONV1D, 1, "weight"));
                    model.conv1d_1_b = get_tensor(string_format(TN_CONV1D, 1, "bias"));
                    model.conv1d_2_w = get_tensor(string_format(TN_CONV1D, 2, "weight"));
                    model.conv1d_2_b = get_tensor(string_format(TN_CONV1D, 2, "bias"));
                    // MERaLiON adaptor: 4 linear layers + ln_pre
                    // linear_0 = frame compression (19200->6400) + SiLU
                    // linear_1 = gate_proj (6400->6400) for GLU
                    // linear_2 = pool_proj (6400->6400) for GLU
                    // linear_3 = out_proj  (6400->3584)
                    model.mm_0_w = get_tensor(string_format(TN_MM_AUDIO_MLP, 0, "weight"));
                    model.mm_0_b = get_tensor(string_format(TN_MM_AUDIO_MLP, 0, "bias"));
                    model.mm_1_w = get_tensor(string_format(TN_MM_AUDIO_MLP, 1, "weight"));
                    model.mm_1_b = get_tensor(string_format(TN_MM_AUDIO_MLP, 1, "bias"));
                    model.mm_2_w = get_tensor(string_format(TN_MM_AUDIO_MLP, 2, "weight"));
                    model.mm_2_b = get_tensor(string_format(TN_MM_AUDIO_MLP, 2, "bias"));
                    model.mm_3_w = get_tensor(string_format(TN_MM_AUDIO_MLP, 3, "weight"));
                    model.mm_3_b = get_tensor(string_format(TN_MM_AUDIO_MLP, 3, "bias"));
                    // ln_speech (LayerNorm before adaptor)
                    model.mm_norm_pre_w = get_tensor(string_format(TN_MM_NORM_PRE, "weight"));
                    model.mm_norm_pre_b = get_tensor(string_format(TN_MM_NORM_PRE, "bias"));
                } break;
            case PROJECTOR_TYPE_QWEN2A:
                {
                    model.conv1d_1_w = get_tensor(string_format(TN_CONV1D, 1, "weight"));
                    model.conv1d_1_b = get_tensor(string_format(TN_CONV1D, 1, "bias"));
                    model.conv1d_2_w = get_tensor(string_format(TN_CONV1D, 2, "weight"));
                    model.conv1d_2_b = get_tensor(string_format(TN_CONV1D, 2, "bias"));
                    model.mm_fc_w = get_tensor(string_format(TN_MM_AUDIO_FC, "weight"));
                    model.mm_fc_b = get_tensor(string_format(TN_MM_AUDIO_FC, "bias"));
                } break;
            case PROJECTOR_TYPE_QWEN3A:
                {
                    model.conv2d_1_w = get_tensor(string_format(TN_CONV2D, 1, "weight"));
                    model.conv2d_1_b = get_tensor(string_format(TN_CONV2D, 1, "bias"));
                    model.conv2d_2_w = get_tensor(string_format(TN_CONV2D, 2, "weight"));
                    model.conv2d_2_b = get_tensor(string_format(TN_CONV2D, 2, "bias"));
                    model.conv2d_3_w = get_tensor(string_format(TN_CONV2D, 3, "weight"));
                    model.conv2d_3_b = get_tensor(string_format(TN_CONV2D, 3, "bias"));
                    model.conv_out_w = get_tensor(string_format(TN_CONV_OUT, "weight")); // no bias
                    model.mm_1_w = get_tensor(string_format(TN_MM_AUDIO_MLP, 1, "weight"));
                    model.mm_1_b = get_tensor(string_format(TN_MM_AUDIO_MLP, 1, "bias"));
                    model.mm_2_w = get_tensor(string_format(TN_MM_AUDIO_MLP, 2, "weight"));
                    model.mm_2_b = get_tensor(string_format(TN_MM_AUDIO_MLP, 2, "bias"));
                } break;
            case PROJECTOR_TYPE_MIMO_AUDIO:
                {
                    model.conv1d_1_w = get_tensor(string_format(TN_CONV1D, 1, "weight"));
                    model.conv1d_1_b = get_tensor(string_format(TN_CONV1D, 1, "bias"));
                    model.conv1d_2_w = get_tensor(string_format(TN_CONV1D, 2, "weight"));
                    model.conv1d_2_b = get_tensor(string_format(TN_CONV1D, 2, "bias"));
                    model.downsample_conv_w = get_tensor(string_format(TN_A_DOWNSAMPLE_CONV, "weight"));
                    model.downsample_norm_w = get_tensor(string_format(TN_A_DOWNSAMPLE_NORM, "weight"));
                    model.downsample_norm_b = get_tensor(string_format(TN_A_DOWNSAMPLE_NORM, "bias"));
                    model.rvq_codebook   = get_tensor(string_format(TN_A_RVQ_CODEBOOK, "weight"), false);
                    model.mm_a_code_embd = get_tensor(string_format(TN_MM_A_CODE_EMBD, "weight"), false);
                    if (!model.rvq_codebook || !model.mm_a_code_embd) {
                        throw std::runtime_error(string_format("%s: mimo_audio: missing %s or %s\n", __func__,
                            TN_A_RVQ_CODEBOOK, TN_MM_A_CODE_EMBD));
                    }
                    // hparams.rvq_codebook_size comes from GGUF metadata and is independent of the
                    // tensors' actual shapes - bound it so codebook/code_embd views built from it
                    // (mimo-audio.cpp) can never read past either tensor's allocated bins/vocab.
                    for (int32_t bins : hparams.rvq_codebook_size) {
                        if (bins <= 0 || bins > model.rvq_codebook->ne[1] || bins > model.mm_a_code_embd->ne[1]) {
                            throw std::runtime_error(string_format(
                                "%s: mimo_audio: %s entry (%d) out of range for codebook/code_embd tensors\n",
                                __func__, KEY_A_RVQ_CODEBOOK_SIZE, bins));
                        }
                    }

                    // LLM-side connector: input_local_transformer + projection
                    model.mm_a_local_layers.resize(hparams.audio_local_n_layer);
                    for (int il = 0; il < hparams.audio_local_n_layer; il++) {
                        auto & layer = model.mm_a_local_layers[il];
                        layer.q_w    = get_tensor(string_format(TN_MM_A_LOCAL_ATTN_Q,   il, "weight"));
                        layer.q_b    = get_tensor(string_format(TN_MM_A_LOCAL_ATTN_Q,   il, "bias"));
                        layer.k_w    = get_tensor(string_format(TN_MM_A_LOCAL_ATTN_K,   il, "weight"));
                        layer.k_b    = get_tensor(string_format(TN_MM_A_LOCAL_ATTN_K,   il, "bias"));
                        layer.v_w    = get_tensor(string_format(TN_MM_A_LOCAL_ATTN_V,   il, "weight"));
                        layer.v_b    = get_tensor(string_format(TN_MM_A_LOCAL_ATTN_V,   il, "bias"));
                        layer.o_w    = get_tensor(string_format(TN_MM_A_LOCAL_ATTN_OUT, il, "weight"));
                        layer.ff_gate_w = get_tensor(string_format(TN_MM_A_LOCAL_FFN_GATE, il, "weight"));
                        layer.ff_up_w   = get_tensor(string_format(TN_MM_A_LOCAL_FFN_UP,   il, "weight"));
                        layer.ff_down_w = get_tensor(string_format(TN_MM_A_LOCAL_FFN_DOWN, il, "weight"));
                        layer.ln_1_w = get_tensor(string_format(TN_MM_A_LOCAL_LN1, il, "weight"));
                        layer.ln_2_w = get_tensor(string_format(TN_MM_A_LOCAL_LN2, il, "weight"));
                    }
                    model.mm_a_local_norm_w = get_tensor(string_format(TN_MM_A_LOCAL_NORM, "weight"));

                    model.mm_1_w = get_tensor(string_format(TN_MM_AUDIO_MLP, 1, "weight"));
                    model.mm_2_w = get_tensor(string_format(TN_MM_AUDIO_MLP, 2, "weight"));
                } break;
            case PROJECTOR_TYPE_QWEN3TTS_SPKENC:
                {
                    // stem TDNN (block 0)
                    model.conv1d_1_w = get_tensor(string_format(TN_CONV1D, 0, "weight"));
                    model.conv1d_1_b = get_tensor(string_format(TN_CONV1D, 0, "bias"));

                    // SE-Res2Net blocks (GGUF bid 1..3, one per hparams.n_layer)
                    model.layers.resize(hparams.n_layer);
                    for (int il = 0; il < hparams.n_layer; il++) {
                        auto & layer = model.layers[il];
                        int bid = il + 1;
                        layer.conv_pw1_w = get_tensor(string_format(TN_CONV_PW1, prefix, bid, "weight"));
                        layer.conv_pw1_b = get_tensor(string_format(TN_CONV_PW1, prefix, bid, "bias"));
                        layer.conv_pw2_w = get_tensor(string_format(TN_CONV_PW2, prefix, bid, "weight"));
                        layer.conv_pw2_b = get_tensor(string_format(TN_CONV_PW2, prefix, bid, "bias"));
                        layer.se_conv1_w = get_tensor(string_format(TN_A_SE_CONV1, bid, "weight"));
                        layer.se_conv1_b = get_tensor(string_format(TN_A_SE_CONV1, bid, "bias"));
                        layer.se_conv2_w = get_tensor(string_format(TN_A_SE_CONV2, bid, "weight"));
                        layer.se_conv2_b = get_tensor(string_format(TN_A_SE_CONV2, bid, "bias"));
                        layer.res2_conv_w.resize(7);
                        layer.res2_conv_b.resize(7);
                        for (int xid = 0; xid < 7; xid++) {
                            layer.res2_conv_w[xid] = get_tensor(string_format(TN_A_CONV_RES2, bid, xid, "weight"));
                            layer.res2_conv_b[xid] = get_tensor(string_format(TN_A_CONV_RES2, bid, xid, "bias"));
                        }
                    }

                    // multi-layer feature aggregation
                    model.conv_out_w = get_tensor(string_format(TN_CONV_OUT, "weight"));
                    model.conv_out_b = get_tensor(string_format(TN_CONV_OUT, "bias"));

                    // attentive statistics pooling
                    model.spk_asp_attn_w = get_tensor(string_format(TN_A_ASP_ATTN, "weight"));
                    model.spk_asp_attn_b = get_tensor(string_format(TN_A_ASP_ATTN, "bias"));
                    model.spk_asp_tdnn_w = get_tensor(string_format(TN_A_ASP_TDNN, "weight"));
                    model.spk_asp_tdnn_b = get_tensor(string_format(TN_A_ASP_TDNN, "bias"));

                    // final speaker embedding projection
                    model.mm_fc_w = get_tensor(string_format(TN_MM_AUDIO_FC, "weight"));
                    model.mm_fc_b = get_tensor(string_format(TN_MM_AUDIO_FC, "bias"));
                } break;
            case PROJECTOR_TYPE_POCKETTTS_SPKENC:
                {
                    load_seanet(model.seanet, false);
                    model.downsample_w = get_tensor(string_format(TN_A_DOWNSAMPLE_CONV, "weight"));
                    model.spk_proj_w   = get_tensor(string_format(TN_A_SPEAKER_PROJ, "weight"));
                } break;
            case PROJECTOR_TYPE_POCKETTTS_GEN:
                {
                    auto & flow = model.flow;
                    flow.input_proj_w = get_tensor(string_format(TN_A_GEN_FLOW_INPUT_PROJ, "weight"));
                    flow.input_proj_b = get_tensor(string_format(TN_A_GEN_FLOW_INPUT_PROJ, "bias"));
                    flow.cond_embd_w  = get_tensor(string_format(TN_A_GEN_FLOW_COND_EMBD, "weight"));
                    flow.cond_embd_b  = get_tensor(string_format(TN_A_GEN_FLOW_COND_EMBD, "bias"));
                    flow.final_ada_w  = get_tensor(string_format(TN_A_GEN_FLOW_FINAL_ADA, "weight"));
                    flow.final_ada_b  = get_tensor(string_format(TN_A_GEN_FLOW_FINAL_ADA, "bias"));
                    flow.final_proj_w = get_tensor(string_format(TN_A_GEN_FLOW_FINAL_PROJ, "weight"));
                    flow.final_proj_b = get_tensor(string_format(TN_A_GEN_FLOW_FINAL_PROJ, "bias"));

                    flow.time.resize(2);
                    for (size_t i = 0; i < flow.time.size(); i++) {
                        auto & t = flow.time[i];
                        t.freqs  = get_tensor(string_format(TN_A_GEN_FLOW_TIME_FREQS, (int) i));
                        t.up_w   = get_tensor(string_format(TN_A_GEN_FLOW_TIME_UP,   (int) i, "weight"));
                        t.up_b   = get_tensor(string_format(TN_A_GEN_FLOW_TIME_UP,   (int) i, "bias"));
                        t.down_w = get_tensor(string_format(TN_A_GEN_FLOW_TIME_DOWN, (int) i, "weight"));
                        t.down_b = get_tensor(string_format(TN_A_GEN_FLOW_TIME_DOWN, (int) i, "bias"));
                        t.norm   = get_tensor(string_format(TN_A_GEN_FLOW_TIME_NORM, (int) i));
                    }

                    // one AdaLN block per flow depth, the count is only known from the tensors
                    for (int il = 0; ; il++) {
                        ggml_tensor * probe = get_tensor(string_format(TN_A_GEN_FLOW_BLK_NORM, il, "weight"), false);
                        if (probe == nullptr) {
                            break;
                        }
                        mtmd_flow_net::block blk;
                        blk.norm_w = probe;
                        blk.norm_b = get_tensor(string_format(TN_A_GEN_FLOW_BLK_NORM, il, "bias"));
                        blk.up_w   = get_tensor(string_format(TN_A_GEN_FLOW_BLK_UP,   il, "weight"));
                        blk.up_b   = get_tensor(string_format(TN_A_GEN_FLOW_BLK_UP,   il, "bias"));
                        blk.down_w = get_tensor(string_format(TN_A_GEN_FLOW_BLK_DOWN, il, "weight"));
                        blk.down_b = get_tensor(string_format(TN_A_GEN_FLOW_BLK_DOWN, il, "bias"));
                        blk.ada_w  = get_tensor(string_format(TN_A_GEN_FLOW_BLK_ADA,  il, "weight"));
                        blk.ada_b  = get_tensor(string_format(TN_A_GEN_FLOW_BLK_ADA,  il, "bias"));
                        flow.blocks.push_back(blk);
                    }

                    model.gen_out_eos_w   = get_tensor(string_format(TN_A_GEN_OUT_EOS, "weight"));
                    model.gen_out_eos_b   = get_tensor(string_format(TN_A_GEN_OUT_EOS, "bias"));
                    model.gen_input_lin_w = get_tensor(string_format(TN_A_GEN_INPUT_LINEAR, "weight"));
                    model.gen_emb_mean    = get_tensor(TN_A_GEN_EMB_MEAN);
                    model.gen_emb_std     = get_tensor(TN_A_GEN_EMB_STD);

                    // mimi decoder
                    model.gen_quant_out_w = get_tensor(string_format(TN_A_GEN_WAV_QUANT_OUT, "weight"));
                    model.gen_upsample_w  = get_tensor(string_format(TN_A_GEN_WAV_UPSAMPLE, "weight"));
                    load_seanet(model.seanet, true);
                    model.gen_tfm_layers.resize(hparams.n_layer);
                    for (int il = 0; il < hparams.n_layer; il++) {
                        auto & layer = model.gen_tfm_layers[il];
                        const char * p = "a.gen.wav.tfm";
                        layer.ln_1_w    = get_tensor(string_format(TN_LN_1,        p, il, "weight"));
                        layer.ln_1_b    = get_tensor(string_format(TN_LN_1,        p, il, "bias"));
                        layer.q_w       = get_tensor(string_format(TN_ATTN_Q,      p, il, "weight"));
                        layer.k_w       = get_tensor(string_format(TN_ATTN_K,      p, il, "weight"));
                        layer.v_w       = get_tensor(string_format(TN_ATTN_V,      p, il, "weight"));
                        layer.o_w       = get_tensor(string_format(TN_ATTN_OUTPUT, p, il, "weight"));
                        layer.ls_1_w    = get_tensor(string_format(TN_LS_1,        p, il, "weight"));
                        layer.ln_2_w    = get_tensor(string_format(TN_LN_2,        p, il, "weight"));
                        layer.ln_2_b    = get_tensor(string_format(TN_LN_2,        p, il, "bias"));
                        layer.ff_up_w   = get_tensor(string_format(TN_FFN_UP,      p, il, "weight"));
                        layer.ff_down_w = get_tensor(string_format(TN_FFN_DOWN,    p, il, "weight"));
                        layer.ls_2_w    = get_tensor(string_format(TN_LS_2,        p, il, "weight"));
                    }

                    for (int64_t i = 0; i < gguf_get_n_tensors(ctx_gguf.get()); ++i) {
                        const char * name = gguf_get_tensor_name(ctx_gguf.get(), i);
                        if (strncmp(name, "voice.", 6) != 0) {
                            continue;
                        }
                        const ggml_tensor * t = ggml_get_tensor(ctx_meta.get(), name);
                        if (t->type != GGML_TYPE_F32 || ggml_n_dims(t) > 2 || t->ne[0] != model.gen_input_lin_w->ne[1]) {
                            throw std::runtime_error(string_format("%s: invalid voice tensor %s\n", __func__, name));
                        }
                        auto & embd = model.gen_voices[name + 6].rows;
                        embd.resize(ggml_nelements(t));
                        fin.seekg(tensor_offset.at(name), std::ios::beg);
                        if (!fin.read(reinterpret_cast<char *>(embd.data()), ggml_nbytes(t))) {
                            throw std::runtime_error(string_format("%s: failed to read %s\n", __func__, name));
                        }
                    }
                    if (!model.gen_voices.empty()) {
                        LOG_INF("%s: %zu predefined voices\n", __func__, model.gen_voices.size());
                    }
                } break;
            case PROJECTOR_TYPE_LFM3A_GEN:
                {
                    // must match the "tts" chat template, see conversion/lfm3.py
                    // note for contributor: this is one-off fix, do not repeat this for future models
                    model.gen_voices = { {"female", {}}, {"male", {}} };

                    // depthformer, the generic loop above loaded the a.gen.code.blk.* tensors
                    for (int il = 0; il < hparams.n_layer; ++il) {
                        model.layers[il].attn_gate_w = get_tensor(string_format(TN_ATTN_GATE, prefix, il, "weight"));
                    }
                    model.gen_code_proj_in_w   = get_tensor(string_format(TN_A_GEN_CODE_PROJ_IN,   "weight"));
                    model.gen_code_proj_in_b   = get_tensor(string_format(TN_A_GEN_CODE_PROJ_IN,   "bias"));
                    model.gen_code_embd_w      = get_tensor(string_format(TN_A_GEN_CODE_EMBD,      "weight")); // tied head
                    model.gen_code_out_embd_w  = get_tensor(string_format(TN_A_GEN_CODE_OUT_EMBD,  "weight"));
                    model.gen_code_norm_w      = get_tensor(string_format(TN_A_GEN_CODE_NORM,      "weight"));
                    model.gen_code_head_norm_w = get_tensor(string_format(TN_A_GEN_CODE_HEAD_NORM, "weight"));
                    model.gen_code_mask_embd   = get_tensor(TN_A_GEN_CODE_MASK_EMBD);

                    // codec decoder: FSQ level table, one projection over all codebooks, LFM3 decoder -> polar STFT frames
                    auto & c2w = model.c2w;
                    c2w.quant_first_cb_w  = get_tensor(string_format(TN_A_GEN_WAV_QUANT_FIRST_CB,  "weight"));
                    c2w.quant_first_out_w = get_tensor(string_format(TN_A_GEN_WAV_QUANT_FIRST_OUT, "weight"));
                    c2w.quant_first_out_b = get_tensor(string_format(TN_A_GEN_WAV_QUANT_FIRST_OUT, "bias"));
                    c2w.tfm_in_proj_w     = get_tensor(string_format(TN_A_GEN_WAV_TFM_IN_PROJ,  "weight"));
                    c2w.tfm_in_proj_b     = get_tensor(string_format(TN_A_GEN_WAV_TFM_IN_PROJ,  "bias"));
                    c2w.tfm_out_proj_w    = get_tensor(string_format(TN_A_GEN_WAV_TFM_OUT_PROJ, "weight"));
                    c2w.tfm_out_proj_b    = get_tensor(string_format(TN_A_GEN_WAV_TFM_OUT_PROJ, "bias"));
                    c2w.tfm_output_norm_w = get_tensor(string_format(TN_A_GEN_WAV_TFM_OUT_NORM, "weight"));

                    // each layer is a short conv block or a gated attention block, both with sandwich norms
                    c2w.tfm_layers.resize(hparams.wav_tfm_n_layer);
                    for (int il = 0; il < hparams.wav_tfm_n_layer; il++) {
                        auto & layer = c2w.tfm_layers[il];
                        const char * p = "a.gen.wav.tfm";
                        layer.ln_1_w           = get_tensor(string_format(TN_LN_1,           p, il, "weight"));
                        layer.attn_post_norm_w = get_tensor(string_format(TN_ATTN_POST_NORM, p, il, "weight"));
                        layer.ln_2_w           = get_tensor(string_format(TN_LN_2,           p, il, "weight"));
                        layer.ff_post_norm_w   = get_tensor(string_format(TN_FFN_POST_NORM,  p, il, "weight"));
                        layer.ff_gate_w        = get_tensor(string_format(TN_FFN_GATE,       p, il, "weight"));
                        layer.ff_up_w          = get_tensor(string_format(TN_FFN_UP,         p, il, "weight"));
                        layer.ff_down_w        = get_tensor(string_format(TN_FFN_DOWN,       p, il, "weight"));
                        layer.conv_pw1_w       = get_tensor(string_format(TN_CONV_PW1,       p, il, "weight"), false);
                        layer.conv_dw_w        = get_tensor(string_format(TN_CONV_DW,        p, il, "weight"), false);
                        layer.conv_pw2_w       = get_tensor(string_format(TN_CONV_PW2,       p, il, "weight"), false);
                        layer.q_w              = get_tensor(string_format(TN_ATTN_Q,         p, il, "weight"), false);
                        layer.k_w              = get_tensor(string_format(TN_ATTN_K,         p, il, "weight"), false);
                        layer.v_w              = get_tensor(string_format(TN_ATTN_V,         p, il, "weight"), false);
                        layer.o_w              = get_tensor(string_format(TN_ATTN_OUTPUT,    p, il, "weight"), false);
                        layer.q_norm           = get_tensor(string_format(TN_ATTN_Q_NORM,    p, il, "weight"), false);
                        layer.k_norm           = get_tensor(string_format(TN_ATTN_K_NORM,    p, il, "weight"), false);
                        layer.attn_gate_w      = get_tensor(string_format(TN_ATTN_GATE,      p, il, "weight"), false);
                        const bool is_conv = layer.conv_pw1_w && layer.conv_dw_w && layer.conv_pw2_w;
                        const bool is_attn = layer.q_w && layer.k_w && layer.v_w && layer.o_w && layer.q_norm && layer.k_norm && layer.attn_gate_w;
                        if (is_conv == is_attn) {
                            throw std::runtime_error(string_format("%s: codec layer %d is neither a conv nor an attention block\n", __func__, il));
                        }
                        if (is_attn) {
                            const int d_head = layer.q_norm->ne[0];
                            hparams.wav_tfm_n_head    = layer.q_w->ne[1] / d_head;
                            hparams.wav_tfm_n_head_kv = layer.k_w->ne[1] / d_head;
                        }
                    }
                } break;
            case PROJECTOR_TYPE_QWEN3TTS_GEN:
                {
                    // code_predictor, proj_in is absent when the talker and the predictor share the hidden size
                    model.gen_code_proj_in_w = get_tensor(string_format(TN_A_GEN_CODE_PROJ_IN, "weight"), false);
                    model.gen_code_proj_in_b = get_tensor(string_format(TN_A_GEN_CODE_PROJ_IN, "bias"),   false);
                    model.gen_code_embd_w     = get_tensor(string_format(TN_A_GEN_CODE_EMBD,     "weight"));
                    model.gen_code_head_w     = get_tensor(string_format(TN_A_GEN_CODE_HEAD,     "weight"));
                    model.gen_code_out_embd_w = get_tensor(string_format(TN_A_GEN_CODE_OUT_EMBD, "weight"));
                    model.gen_code_norm_w     = get_tensor(string_format(TN_A_GEN_CODE_NORM,     "weight"));

                    // code2wav: RVQ codes -> raw PCM, lives in the same ctx as code_predictor
                    load_code2wav(true);
                } break;
            case PROJECTOR_TYPE_DRAMA3_GEN:
                {
                    // generic a.gen.code.blk.* tensors are loaded by the loop above
                    model.drama3_layers.resize(hparams.n_layer);
                    for (int il = 0; il < hparams.n_layer; ++il) {
                        auto & tl = model.drama3_layers[il];
                        model.layers[il].attn_gate_w = get_tensor(string_format(TN_ATTN_GATE, prefix, il, "weight"));
                        tl.enorm_w    = get_tensor(string_format(TN_A_GEN_CODE_ENORM,        il, "weight"));
                        tl.hnorm_w    = get_tensor(string_format(TN_A_GEN_CODE_HNORM,        il, "weight"));
                        tl.eh_proj_w  = get_tensor(string_format(TN_A_GEN_CODE_EH_PROJ,      il, "weight"));
                        tl.embd_w     = get_tensor(string_format(TN_A_GEN_CODE_BLK_EMBD,     il, "weight"));
                        tl.out_norm_w = get_tensor(string_format(TN_A_GEN_CODE_BLK_OUT_NORM, il, "weight"));
                    }
                    model.gen_code_embd_w     = get_tensor(string_format(TN_A_GEN_CODE_EMBD,     "weight")); // merged 3D, one table per codebook
                    model.gen_code_head_w     = get_tensor(string_format(TN_A_GEN_CODE_HEAD,     "weight")); // merged 3D, codebooks 1..
                    model.gen_code_out_embd_w = get_tensor(string_format(TN_A_GEN_CODE_OUT_EMBD, "weight")); // backbone rows of the semantic band
                    load_code2wav(false);
                } break;
            case PROJECTOR_TYPE_CHATTERBOX:
                {
                    auto & c = model.cbx;
                    auto has = [&](const std::string & name) {
                        return gguf_find_tensor(ctx_gguf.get(), name.c_str()) >= 0;
                    };
                    auto load_enc_layer = [&](const std::string & p, mtmd_chatterbox::enc_layer & l) {
                        l.norm_mha_w      = get_tensor(p + ".norm_mha.weight");
                        l.norm_mha_b      = get_tensor(p + ".norm_mha.bias");
                        l.attn_q_w        = get_tensor(p + ".self_attn.linear_q.weight");
                        l.attn_q_b        = get_tensor(p + ".self_attn.linear_q.bias");
                        l.attn_k_w        = get_tensor(p + ".self_attn.linear_k.weight");
                        l.attn_k_b        = get_tensor(p + ".self_attn.linear_k.bias");
                        l.attn_v_w        = get_tensor(p + ".self_attn.linear_v.weight");
                        l.attn_v_b        = get_tensor(p + ".self_attn.linear_v.bias");
                        l.attn_pos_w      = get_tensor(p + ".self_attn.linear_pos.weight");
                        l.attn_pos_bias_u = get_tensor(p + ".self_attn.pos_bias_u");
                        l.attn_pos_bias_v = get_tensor(p + ".self_attn.pos_bias_v");
                        l.attn_out_w      = get_tensor(p + ".self_attn.linear_out.weight");
                        l.attn_out_b      = get_tensor(p + ".self_attn.linear_out.bias");
                        l.norm_ff_w       = get_tensor(p + ".norm_ff.weight");
                        l.norm_ff_b       = get_tensor(p + ".norm_ff.bias");
                        l.ffn_1_w         = get_tensor(p + ".feed_forward.w_1.weight");
                        l.ffn_1_b         = get_tensor(p + ".feed_forward.w_1.bias");
                        l.ffn_2_w         = get_tensor(p + ".feed_forward.w_2.weight");
                        l.ffn_2_b         = get_tensor(p + ".feed_forward.w_2.bias");
                    };
                    auto load_causal = [&](const std::string & p, mtmd_chatterbox::causal_block & b) {
                        b.conv_w = get_tensor(p + ".block.0.weight");
                        b.conv_b = get_tensor(p + ".block.0.bias");
                        b.norm_w = get_tensor(p + ".block.2.weight");
                        b.norm_b = get_tensor(p + ".block.2.bias");
                    };
                    auto load_resnet = [&](const std::string & p, mtmd_chatterbox::resnet & r) {
                        load_causal(p + ".block1", r.block1);
                        load_causal(p + ".block2", r.block2);
                        r.mlp_w      = get_tensor(p + ".mlp.1.weight");
                        r.mlp_b      = get_tensor(p + ".mlp.1.bias");
                        r.res_conv_w = get_tensor(p + ".res_conv.weight");
                        r.res_conv_b = get_tensor(p + ".res_conv.bias");
                    };
                    auto load_tfm = [&](const std::string & p, mtmd_chatterbox::tfm_block & b) {
                        b.norm1_w    = get_tensor(p + ".norm1.weight");
                        b.norm1_b    = get_tensor(p + ".norm1.bias");
                        b.attn_q_w   = get_tensor(p + ".attn1.to_q.weight");
                        b.attn_k_w   = get_tensor(p + ".attn1.to_k.weight");
                        b.attn_v_w   = get_tensor(p + ".attn1.to_v.weight");
                        b.attn_out_w = get_tensor(p + ".attn1.to_out.0.weight");
                        b.attn_out_b = get_tensor(p + ".attn1.to_out.0.bias");
                        b.norm3_w    = get_tensor(p + ".norm3.weight");
                        b.norm3_b    = get_tensor(p + ".norm3.bias");
                        b.ff_in_w    = get_tensor(p + ".ff.net.0.proj.weight");
                        b.ff_in_b    = get_tensor(p + ".ff.net.0.proj.bias");
                        b.ff_out_w   = get_tensor(p + ".ff.net.2.weight");
                        b.ff_out_b   = get_tensor(p + ".ff.net.2.bias");
                    };
                    auto load_stage = [&](const std::string & p, mtmd_chatterbox::est_stage & s, bool boundary) {
                        load_resnet(p + ".0", s.res);
                        for (int j = 0; has(p + ".1." + std::to_string(j) + ".norm1.weight"); j++) {
                            mtmd_chatterbox::tfm_block b;
                            load_tfm(p + ".1." + std::to_string(j), b);
                            s.tfm.push_back(b);
                        }
                        if (boundary) {
                            s.conv_w = get_tensor(p + ".2.weight");
                            s.conv_b = get_tensor(p + ".2.bias");
                        }
                    };
                    auto load_hres = [&](const std::string & p, mtmd_chatterbox::hift_res & r) {
                        for (int j = 0; has(p + ".convs1." + std::to_string(j) + ".weight"); j++) {
                            const std::string js = std::to_string(j);
                            mtmd_chatterbox::hift_res_unit u;
                            u.act1_alpha = get_tensor(p + ".activations1." + js + ".alpha");
                            u.act2_alpha = get_tensor(p + ".activations2." + js + ".alpha");
                            u.conv1_w    = get_tensor(p + ".convs1." + js + ".weight");
                            u.conv1_b    = get_tensor(p + ".convs1." + js + ".bias");
                            u.conv2_w    = get_tensor(p + ".convs2." + js + ".weight");
                            u.conv2_b    = get_tensor(p + ".convs2." + js + ".bias");
                            r.units.push_back(u);
                        }
                    };

                    // flow encoder
                    c.input_embedding_w = get_tensor("a.gen.flow.input_embedding.weight");
                    c.spk_affine_w      = get_tensor("a.gen.flow.spk_embed_affine_layer.weight");
                    c.spk_affine_b      = get_tensor("a.gen.flow.spk_embed_affine_layer.bias");
                    c.embed_linear_w    = get_tensor("a.gen.fenc.embed.out.0.weight");
                    c.embed_linear_b    = get_tensor("a.gen.fenc.embed.out.0.bias");
                    c.embed_norm_w      = get_tensor("a.gen.fenc.embed.out.1.weight");
                    c.embed_norm_b      = get_tensor("a.gen.fenc.embed.out.1.bias");
                    c.pre_conv1_w       = get_tensor("a.gen.fenc.pre_lookahead_layer.conv1.weight");
                    c.pre_conv1_b       = get_tensor("a.gen.fenc.pre_lookahead_layer.conv1.bias");
                    c.pre_conv2_w       = get_tensor("a.gen.fenc.pre_lookahead_layer.conv2.weight");
                    c.pre_conv2_b       = get_tensor("a.gen.fenc.pre_lookahead_layer.conv2.bias");
                    for (int i = 0; has("a.gen.fenc.encoders." + std::to_string(i) + ".norm_mha.weight"); i++) {
                        mtmd_chatterbox::enc_layer l;
                        load_enc_layer("a.gen.fenc.encoders." + std::to_string(i), l);
                        c.enc.push_back(l);
                    }
                    c.up_conv_w         = get_tensor("a.gen.fenc.up_layer.conv.weight");
                    c.up_conv_b         = get_tensor("a.gen.fenc.up_layer.conv.bias");
                    c.up_embed_linear_w = get_tensor("a.gen.fenc.up_embed.out.0.weight");
                    c.up_embed_linear_b = get_tensor("a.gen.fenc.up_embed.out.0.bias");
                    c.up_embed_norm_w   = get_tensor("a.gen.fenc.up_embed.out.1.weight");
                    c.up_embed_norm_b   = get_tensor("a.gen.fenc.up_embed.out.1.bias");
                    for (int i = 0; has("a.gen.fenc.up_encoders." + std::to_string(i) + ".norm_mha.weight"); i++) {
                        mtmd_chatterbox::enc_layer l;
                        load_enc_layer("a.gen.fenc.up_encoders." + std::to_string(i), l);
                        c.up_enc.push_back(l);
                    }
                    c.after_norm_w      = get_tensor("a.gen.fenc.after_norm.weight");
                    c.after_norm_b      = get_tensor("a.gen.fenc.after_norm.bias");
                    c.encoder_proj_w    = get_tensor("a.gen.flow.encoder_proj.weight");
                    c.encoder_proj_b    = get_tensor("a.gen.flow.encoder_proj.bias");

                    // cfm estimator
                    c.time_mlp_1_w        = get_tensor("a.gen.est.time_mlp.linear_1.weight");
                    c.time_mlp_1_b        = get_tensor("a.gen.est.time_mlp.linear_1.bias");
                    c.time_mlp_2_w        = get_tensor("a.gen.est.time_mlp.linear_2.weight");
                    c.time_mlp_2_b        = get_tensor("a.gen.est.time_mlp.linear_2.bias");
                    c.time_embed_mixer_w  = get_tensor("a.gen.est.time_embed_mixer.weight", false);
                    load_stage("a.gen.est.down_blocks.0", c.est_down, true);
                    for (int i = 0; has("a.gen.est.mid_blocks." + std::to_string(i) + ".0.block1.block.0.weight"); i++) {
                        mtmd_chatterbox::est_stage s;
                        load_stage("a.gen.est.mid_blocks." + std::to_string(i), s, false);
                        c.est_mid.push_back(std::move(s));
                    }
                    load_stage("a.gen.est.up_blocks.0", c.est_up, true);
                    load_causal("a.gen.est.final_block", c.est_final_block);
                    c.est_final_proj_w    = get_tensor("a.gen.est.final_proj.weight");
                    c.est_final_proj_b    = get_tensor("a.gen.est.final_proj.bias");

                    // hift vocoder
                    c.hift_pre_w  = get_tensor("a.gen.hift.conv_pre.weight");
                    c.hift_pre_b  = get_tensor("a.gen.hift.conv_pre.bias");
                    c.hift_post_w = get_tensor("a.gen.hift.conv_post.weight");
                    c.hift_post_b = get_tensor("a.gen.hift.conv_post.bias");
                    for (int i = 0; has("a.gen.hift.ups." + std::to_string(i) + ".weight"); i++) {
                        const std::string is = std::to_string(i);
                        mtmd_chatterbox::hift_up up;
                        up.up_w          = get_tensor("a.gen.hift.ups." + is + ".weight");
                        up.up_b          = get_tensor("a.gen.hift.ups." + is + ".bias");
                        up.source_down_w = get_tensor("a.gen.hift.source_downs." + is + ".weight");
                        up.source_down_b = get_tensor("a.gen.hift.source_downs." + is + ".bias");
                        load_hres("a.gen.hift.source_resblocks." + is, up.source_res);
                        load_hres("a.gen.hift.resblocks." + std::to_string(3 * i),     up.res_0);
                        load_hres("a.gen.hift.resblocks." + std::to_string(3 * i + 1), up.res_1);
                        load_hres("a.gen.hift.resblocks." + std::to_string(3 * i + 2), up.res_2);
                        c.hift_ups.push_back(std::move(up));
                    }
                    for (int i = 0; has("a.gen.hift.f0_predictor.condnet." + std::to_string(i) + ".weight"); i += 2) {
                        const std::string is = std::to_string(i);
                        mtmd_chatterbox::f0_conv fc;
                        fc.w = get_tensor("a.gen.hift.f0_predictor.condnet." + is + ".weight");
                        fc.b = get_tensor("a.gen.hift.f0_predictor.condnet." + is + ".bias");
                        c.f0_condnet.push_back(fc);
                    }
                    c.f0_classifier_w = get_tensor("a.gen.hift.f0_predictor.classifier.weight");
                    c.f0_classifier_b = get_tensor("a.gen.hift.f0_predictor.classifier.bias");

                    // s3 tokenizer
                    c.s3tok_conv1_w = get_tensor("a.s3tok.encoder.conv1.weight");
                    c.s3tok_conv1_b = get_tensor("a.s3tok.encoder.conv1.bias");
                    c.s3tok_conv2_w = get_tensor("a.s3tok.encoder.conv2.weight");
                    c.s3tok_conv2_b = get_tensor("a.s3tok.encoder.conv2.bias");
                    for (int i = 0; has("a.s3tok.encoder.blocks." + std::to_string(i) + ".attn_ln.weight"); i++) {
                        const std::string p = "a.s3tok.encoder.blocks." + std::to_string(i);
                        mtmd_chatterbox::s3tok_block b;
                        b.attn_ln_w  = get_tensor(p + ".attn_ln.weight");
                        b.attn_ln_b  = get_tensor(p + ".attn_ln.bias");
                        b.attn_q_w   = get_tensor(p + ".attn.query.weight");
                        b.attn_q_b   = get_tensor(p + ".attn.query.bias");
                        b.attn_k_w   = get_tensor(p + ".attn.key.weight");
                        b.attn_v_w   = get_tensor(p + ".attn.value.weight");
                        b.attn_v_b   = get_tensor(p + ".attn.value.bias");
                        b.fsmn_w     = get_tensor(p + ".attn.fsmn_block.weight");
                        b.attn_out_w = get_tensor(p + ".attn.out.weight");
                        b.attn_out_b = get_tensor(p + ".attn.out.bias");
                        b.mlp_ln_w   = get_tensor(p + ".mlp_ln.weight");
                        b.mlp_ln_b   = get_tensor(p + ".mlp_ln.bias");
                        b.mlp_in_w   = get_tensor(p + ".mlp.0.weight");
                        b.mlp_in_b   = get_tensor(p + ".mlp.0.bias");
                        b.mlp_out_w  = get_tensor(p + ".mlp.2.weight");
                        b.mlp_out_b  = get_tensor(p + ".mlp.2.bias");
                        c.s3tok_blocks.push_back(b);
                    }
                    c.s3tok_down_w = get_tensor("a.s3tok.quantizer._codebook.project_down.weight");
                    c.s3tok_down_b = get_tensor("a.s3tok.quantizer._codebook.project_down.bias");

                    // read on the host: embedding tables, source module
                    for (ggml_tensor * t = ggml_get_first_tensor(ctx_meta.get()); t; t = ggml_get_next_tensor(ctx_meta.get(), t)) {
                        const std::string name = t->name;
                        if (name.rfind("a.gen.code.", 0) == 0 || name.rfind("a.gen.t3.", 0) == 0 ||
                            name.rfind("a.gen.hift.m_source.", 0) == 0) {
                            model.cbx_tensors[name] = get_tensor(name);
                        }
                    }

                    // voice.<name> holds the talker rows, voice.<name>.ref_* the flow decoder reference
                    const int64_t n_embd_talker = model.cbx_tensors.at("a.gen.code.out_embd.weight")->ne[0];
                    for (int64_t i = 0; i < gguf_get_n_tensors(ctx_gguf.get()); ++i) {
                        const std::string name = gguf_get_tensor_name(ctx_gguf.get(), i);
                        if (name.rfind("voice.", 0) != 0) {
                            continue;
                        }
                        const ggml_tensor * t = ggml_get_tensor(ctx_meta.get(), name.c_str());
                        std::string voice = name.substr(6);
                        std::vector<float> ids; // the converter stores 1D tensors as F32, ids are exact
                        void * dst = nullptr;
                        bool ok = false;
                        auto take = [&](const char * suffix) {
                            const size_t n = strlen(suffix);
                            if (voice.size() > n && voice.compare(voice.size() - n, n, suffix) == 0) {
                                voice.resize(voice.size() - n);
                                return true;
                            }
                            return false;
                        };
                        if (take(".ref_tokens")) {
                            ids.resize(ggml_nelements(t));
                            dst = ids.data();
                            ok  = t->type == GGML_TYPE_F32;
                        } else if (take(".ref_feat")) {
                            auto & v = model.gen_voices[voice].ref.feat;
                            v.resize(ggml_nelements(t));
                            dst = v.data();
                            ok  = t->type == GGML_TYPE_F32 && t->ne[0] == 80;
                        } else if (take(".ref_spk")) {
                            auto & v = model.gen_voices[voice].ref.spk;
                            v.resize(ggml_nelements(t));
                            dst = v.data();
                            ok  = t->type == GGML_TYPE_F32 && t->ne[0] == 192;
                        } else {
                            auto & v = model.gen_voices[voice].rows;
                            v.resize(ggml_nelements(t));
                            dst = v.data();
                            ok  = t->type == GGML_TYPE_F32 && ggml_n_dims(t) <= 2 && t->ne[0] == n_embd_talker;
                        }
                        if (!ok) {
                            throw std::runtime_error(string_format("%s: invalid voice tensor %s\n", __func__, name.c_str()));
                        }
                        fin.seekg(tensor_offset.at(name), std::ios::beg);
                        if (!fin.read(reinterpret_cast<char *>(dst), ggml_nbytes(t))) {
                            throw std::runtime_error(string_format("%s: failed to read %s\n", __func__, name.c_str()));
                        }
                        if (!ids.empty()) {
                            model.gen_voices[voice].ref.tokens.assign(ids.begin(), ids.end());
                        }
                    }
                    for (const auto & [voice, v] : model.gen_voices) {
                        if (v.rows.empty() || v.ref.tokens.empty() || v.ref.feat.size() != 2 * 80 * v.ref.tokens.size() || v.ref.spk.empty()) {
                            throw std::runtime_error(string_format("%s: incomplete voice %s\n", __func__, voice.c_str()));
                        }
                    }
                } break;
            case PROJECTOR_TYPE_CHATTERBOX_SPKENC:
                {
                    auto & c = model.cbx;
                    auto has = [&](const std::string & name) {
                        return gguf_find_tensor(ctx_gguf.get(), name.c_str()) >= 0;
                    };
                    auto load_bn = [&](const std::string & p, mtmd_chatterbox::bn & n) {
                        n.mean = get_tensor(p + ".running_mean");
                        n.var  = get_tensor(p + ".running_var");
                        n.w    = get_tensor(p + ".weight", false);
                        n.b    = get_tensor(p + ".bias",   false);
                    };
                    auto load_res2d = [&](const std::string & p, mtmd_chatterbox::spk_res2d & r) {
                        r.conv1_w = get_tensor(p + ".conv1.weight");
                        load_bn(p + ".bn1", r.bn1);
                        r.conv2_w = get_tensor(p + ".conv2.weight");
                        load_bn(p + ".bn2", r.bn2);
                        r.shortcut_w = get_tensor(p + ".shortcut.0.weight", false);
                        if (r.shortcut_w) {
                            load_bn(p + ".shortcut.1", r.shortcut_bn);
                        }
                    };
                    auto load_cam_block = [&](const std::string & bp, const std::string & tp, mtmd_chatterbox::spk_cam_block & blk) {
                        for (int li = 1; has(bp + ".tdnnd" + std::to_string(li) + ".linear1.weight"); li++) {
                            const std::string p = bp + ".tdnnd" + std::to_string(li);
                            mtmd_chatterbox::spk_cam_layer l;
                            load_bn(p + ".nonlinear1.batchnorm", l.nl1_bn);
                            l.linear1_w = get_tensor(p + ".linear1.weight");
                            load_bn(p + ".nonlinear2.batchnorm", l.nl2_bn);
                            l.local_w = get_tensor(p + ".cam_layer.linear_local.weight");
                            l.ctx1_w  = get_tensor(p + ".cam_layer.linear1.weight");
                            l.ctx1_b  = get_tensor(p + ".cam_layer.linear1.bias");
                            l.ctx2_w  = get_tensor(p + ".cam_layer.linear2.weight");
                            l.ctx2_b  = get_tensor(p + ".cam_layer.linear2.bias");
                            blk.layers.push_back(l);
                        }
                        load_bn(tp + ".nonlinear.batchnorm", blk.transit_bn);
                        blk.transit_w = get_tensor(tp + ".linear.weight");
                    };

                    c.spk_conv1_w = get_tensor("a.spk.head.conv1.weight");
                    load_bn("a.spk.head.bn1", c.spk_bn1);
                    load_res2d("a.spk.head.layer1.0", c.spk_layer1_0);
                    load_res2d("a.spk.head.layer1.1", c.spk_layer1_1);
                    load_res2d("a.spk.head.layer2.0", c.spk_layer2_0);
                    load_res2d("a.spk.head.layer2.1", c.spk_layer2_1);
                    c.spk_conv2_w = get_tensor("a.spk.head.conv2.weight");
                    load_bn("a.spk.head.bn2", c.spk_bn2);
                    c.spk_tdnn_w = get_tensor("a.spk.xvector.tdnn.linear.weight");
                    load_bn("a.spk.xvector.tdnn.nonlinear.batchnorm", c.spk_tdnn_bn);
                    load_cam_block("a.spk.xvector.block1", "a.spk.xvector.transit1", c.spk_block1);
                    load_cam_block("a.spk.xvector.block2", "a.spk.xvector.transit2", c.spk_block2);
                    load_cam_block("a.spk.xvector.block3", "a.spk.xvector.transit3", c.spk_block3);
                    load_bn("a.spk.xvector.out_nonlinear.batchnorm", c.spk_out_bn);
                    c.spk_dense_w = get_tensor("a.spk.xvector.dense.linear.weight");
                    load_bn("a.spk.xvector.dense.nonlinear.batchnorm", c.spk_dense_bn);

                    // read on the host: voice encoder lstm, conditioning encoder
                    for (ggml_tensor * t = ggml_get_first_tensor(ctx_meta.get()); t; t = ggml_get_next_tensor(ctx_meta.get(), t)) {
                        const std::string name = t->name;
                        if (name.rfind("a.ve.", 0) == 0 || name.rfind("a.cenc.", 0) == 0) {
                            model.cbx_tensors[name] = get_tensor(name);
                        }
                    }
                } break;
            case PROJECTOR_TYPE_SOPRANO:
                {
                    auto & v = model.vocos;
                    v.input_w       = get_tensor(string_format(TN_A_GEN_WAV_INPUT, "weight"));
                    v.input_b       = get_tensor(string_format(TN_A_GEN_WAV_INPUT, "bias"));
                    v.norm_w        = get_tensor(string_format(TN_A_GEN_WAV_NORM, "weight"));
                    v.norm_b        = get_tensor(string_format(TN_A_GEN_WAV_NORM, "bias"));
                    v.output_norm_w = get_tensor(string_format(TN_A_GEN_WAV_OUTPUT_NORM, "weight"));
                    v.output_norm_b = get_tensor(string_format(TN_A_GEN_WAV_OUTPUT_NORM, "bias"));
                    v.output_w      = get_tensor(string_format(TN_A_GEN_WAV_OUTPUT, "weight"));
                    v.output_b      = get_tensor(string_format(TN_A_GEN_WAV_OUTPUT, "bias"));
                    v.blocks.resize(hparams.n_layer);
                    for (int il = 0; il < hparams.n_layer; ++il) {
                        auto & b = v.blocks[il];
                        b.dwconv_w = get_tensor(string_format(TN_A_GEN_WAV_UP_DWCONV, il, "weight"));
                        b.dwconv_b = get_tensor(string_format(TN_A_GEN_WAV_UP_DWCONV, il, "bias"));
                        b.norm_w   = get_tensor(string_format(TN_A_GEN_WAV_UP_NORM, il, "weight"));
                        b.norm_b   = get_tensor(string_format(TN_A_GEN_WAV_UP_NORM, il, "bias"));
                        b.pw1_w    = get_tensor(string_format(TN_A_GEN_WAV_UP_PW1, il, "weight"));
                        b.pw1_b    = get_tensor(string_format(TN_A_GEN_WAV_UP_PW1, il, "bias"));
                        b.pw2_w    = get_tensor(string_format(TN_A_GEN_WAV_UP_PW2, il, "weight"));
                        b.pw2_b    = get_tensor(string_format(TN_A_GEN_WAV_UP_PW2, il, "bias"));
                        b.gamma    = get_tensor(string_format(TN_A_GEN_WAV_UP_GAMMA, il));
                    }
                } break;
            case PROJECTOR_TYPE_VOXTRAL:
            case PROJECTOR_TYPE_VOXTRAL_RT:
                {
                    model.conv1d_1_w = get_tensor(string_format(TN_CONV1D, 1, "weight"));
                    model.conv1d_1_b = get_tensor(string_format(TN_CONV1D, 1, "bias"));
                    model.conv1d_2_w = get_tensor(string_format(TN_CONV1D, 2, "weight"));
                    model.conv1d_2_b = get_tensor(string_format(TN_CONV1D, 2, "bias"));
                    model.mm_1_w = get_tensor(string_format(TN_MM_AUDIO_MLP, 1, "weight"));
                    model.mm_2_w = get_tensor(string_format(TN_MM_AUDIO_MLP, 2, "weight"));
                } break;
            case PROJECTOR_TYPE_MUSIC_FLAMINGO:
                {
                    model.conv1d_1_w = get_tensor(string_format(TN_CONV1D, 1, "weight"));
                    model.conv1d_1_b = get_tensor(string_format(TN_CONV1D, 1, "bias"));
                    model.conv1d_2_w = get_tensor(string_format(TN_CONV1D, 2, "weight"));
                    model.conv1d_2_b = get_tensor(string_format(TN_CONV1D, 2, "bias"));
                    model.mm_1_w = get_tensor(string_format(TN_MM_AUDIO_MLP, 1, "weight"));
                    model.mm_1_b = get_tensor(string_format(TN_MM_AUDIO_MLP, 1, "bias"));
                    model.mm_2_w = get_tensor(string_format(TN_MM_AUDIO_MLP, 2, "weight"));
                    model.mm_2_b = get_tensor(string_format(TN_MM_AUDIO_MLP, 2, "bias"));
                } break;
            case PROJECTOR_TYPE_INTERNVL:
                {
                    model.mm_0_w = get_tensor(string_format(TN_MVLM_PROJ_MLP, 0, "weight"));
                    model.mm_0_b = get_tensor(string_format(TN_MVLM_PROJ_MLP, 0, "bias"));
                    model.mm_1_w = get_tensor(string_format(TN_MVLM_PROJ_MLP, 1, "weight"));
                    model.mm_1_b = get_tensor(string_format(TN_MVLM_PROJ_MLP, 1, "bias"));
                    model.mm_3_w = get_tensor(string_format(TN_MVLM_PROJ_MLP, 3, "weight"));
                    model.mm_3_b = get_tensor(string_format(TN_MVLM_PROJ_MLP, 3, "bias"));
                } break;
            case PROJECTOR_TYPE_NEMOTRON_V2_VL:
            case PROJECTOR_TYPE_NEMOTRON_OMNI:
                {
                    model.mm_0_w = get_tensor(string_format(TN_MVLM_PROJ_MLP, 0, "weight"));
                    model.mm_1_w = get_tensor(string_format(TN_MVLM_PROJ_MLP, 1, "weight"));
                    model.mm_3_w = get_tensor(string_format(TN_MVLM_PROJ_MLP, 3, "weight"));
                } break;
            case PROJECTOR_TYPE_GLMA:
                {
                    model.conv1d_1_w = get_tensor(string_format(TN_CONV1D, 1, "weight"));
                    model.conv1d_1_b = get_tensor(string_format(TN_CONV1D, 1, "bias"));
                    model.conv1d_2_w = get_tensor(string_format(TN_CONV1D, 2, "weight"));
                    model.conv1d_2_b = get_tensor(string_format(TN_CONV1D, 2, "bias"));
                    model.mm_1_w = get_tensor(string_format(TN_MM_AUDIO_MLP, 1, "weight"));
                    model.mm_1_b = get_tensor(string_format(TN_MM_AUDIO_MLP, 1, "bias"));
                    model.mm_2_w = get_tensor(string_format(TN_MM_AUDIO_MLP, 2, "weight"));
                    model.mm_2_b = get_tensor(string_format(TN_MM_AUDIO_MLP, 2, "bias"));
                    model.mm_norm_pre_w = get_tensor(string_format(TN_MM_NORM_PRE, "weight"));
                    model.mm_norm_pre_b = get_tensor(string_format(TN_MM_NORM_PRE, "bias"));
                    model.mm_boi = get_tensor(string_format(TN_TOK_BOI));
                    model.mm_eoi = get_tensor(string_format(TN_TOK_EOI));
                } break;
            case PROJECTOR_TYPE_LLAMA4:
                {
                    model.mm_model_proj    = get_tensor(string_format(TN_MM_PROJECTOR, "weight"));
                    model.mm_model_mlp_1_w = get_tensor(string_format(TN_MVLM_PROJ_MLP, 1, "weight"));
                    model.mm_model_mlp_2_w = get_tensor(string_format(TN_MVLM_PROJ_MLP, 2, "weight"));
                } break;
            case PROJECTOR_TYPE_COGVLM:
                {
                    model.mm_model_proj     = get_tensor(string_format(TN_MM_PROJECTOR, "weight"));
                    model.mm_post_fc_norm_w = get_tensor(string_format(TN_MM_POST_FC_NORM, "weight"));
                    model.mm_post_fc_norm_b = get_tensor(string_format(TN_MM_POST_FC_NORM, "bias"));
                    model.mm_h_to_4h_w      = get_tensor(string_format(TN_MM_H_TO_4H,      "weight"));
                    model.mm_gate_w         = get_tensor(string_format(TN_MM_GATE,         "weight"));
                    model.mm_4h_to_h_w      = get_tensor(string_format(TN_MM_4H_TO_H,      "weight"));
                    model.mm_boi            = get_tensor(TN_TOK_BOI);
                    model.mm_eoi            = get_tensor(TN_TOK_EOI);
                } break;
            case PROJECTOR_TYPE_HUNYUANVL:
                {
                    // proj.0 -> mm.0 (conv1), proj.2 -> mm.2 (conv2), mlp -> mm.model.fc (linear)
                    model.mm_0_w            = get_tensor(string_format(TN_LLAVA_PROJ, 0, "weight"));
                    model.mm_0_b            = get_tensor(string_format(TN_LLAVA_PROJ, 0, "bias"));
                    model.mm_1_w            = get_tensor(string_format(TN_LLAVA_PROJ, 2, "weight"));
                    model.mm_1_b            = get_tensor(string_format(TN_LLAVA_PROJ, 2, "bias"));
                    model.mm_model_proj     = get_tensor(string_format(TN_MM_PROJECTOR, "weight"));
                    model.mm_model_proj_b   = get_tensor(string_format(TN_MM_PROJECTOR, "bias"));
                    model.mm_pre_norm_w     = get_tensor(string_format(TN_MM_PRE_NORM, "weight"));
                    model.mm_post_norm_w    = get_tensor(string_format(TN_MM_POST_NORM, "weight"));
                    model.mm_img_begin      = get_tensor(TN_MM_IMG_BEGIN);
                    model.mm_img_end        = get_tensor(TN_MM_IMG_END);
                    model.image_newline     = get_tensor(TN_IMAGE_NEWLINE);
                    model.view_seperator    = get_tensor(TN_IMAGE_SEPERATOR, false);
                } break;
            case PROJECTOR_TYPE_JANUS_PRO:
                {
                    model.mm_0_w = get_tensor(string_format(TN_LLAVA_PROJ, 0, "weight"));
                    model.mm_0_b = get_tensor(string_format(TN_LLAVA_PROJ, 0, "bias"));
                    model.mm_1_w = get_tensor(string_format(TN_LLAVA_PROJ, 1, "weight"));
                    model.mm_1_b = get_tensor(string_format(TN_LLAVA_PROJ, 1, "bias"));
                } break;
            case PROJECTOR_TYPE_PHI4:
                {
                    model.mm_0_w = get_tensor(string_format(TN_LLAVA_PROJ, 0, "weight"));
                    model.mm_0_b = get_tensor(string_format(TN_LLAVA_PROJ, 0, "bias"));
                    model.mm_2_w = get_tensor(string_format(TN_LLAVA_PROJ, 2, "weight"));
                    model.mm_2_b = get_tensor(string_format(TN_LLAVA_PROJ, 2, "bias"));
                } break;
            case PROJECTOR_TYPE_DEEPSEEKOCR:
            case PROJECTOR_TYPE_DEEPSEEKOCR2:
                {
                    model.pos_embed          = get_tensor(string_format(TN_SAM_POS_EMBD,   "weight"));
                    model.patch_embed_proj_w = get_tensor(string_format(TN_SAM_PATCH_EMBD, "weight"));
                    model.patch_embed_proj_b = get_tensor(string_format(TN_SAM_PATCH_EMBD, "bias"));
                    model.n_sam_layers = hparams.sam_n_layer;
                    model.sam_layers.resize(model.n_sam_layers);
                    for (int il = 0; il < model.n_sam_layers; ++il) {
                        auto & layer    = model.sam_layers[il];
                        layer.qkv_w     = get_tensor(string_format(TN_SAM_ATTN_QKV, il, "weight"));
                        layer.qkv_b     = get_tensor(string_format(TN_SAM_ATTN_QKV, il, "bias"));
                        layer.o_w       = get_tensor(string_format(TN_SAM_ATTN_OUT, il, "weight"));
                        layer.o_b       = get_tensor(string_format(TN_SAM_ATTN_OUT, il, "bias"));
                        layer.ln_1_w    = get_tensor(string_format(TN_SAM_PRE_NORM, il, "weight"));
                        layer.ln_1_b    = get_tensor(string_format(TN_SAM_PRE_NORM, il, "bias"));
                        layer.ln_2_w    = get_tensor(string_format(TN_SAM_POST_NORM, il, "weight"));
                        layer.ln_2_b    = get_tensor(string_format(TN_SAM_POST_NORM, il, "bias"));
                        layer.rel_pos_h = get_tensor(string_format(TN_SAM_ATTN_POS_H, il, "weight"));
                        layer.rel_pos_w = get_tensor(string_format(TN_SAM_ATTN_POS_W, il, "weight"));
                        layer.ff_up_w   = get_tensor(string_format(TN_SAM_FFN_UP, il, "weight"));
                        layer.ff_up_b   = get_tensor(string_format(TN_SAM_FFN_UP, il, "bias"));
                        layer.ff_down_w = get_tensor(string_format(TN_SAM_FFN_DOWN, il, "weight"));
                        layer.ff_down_b = get_tensor(string_format(TN_SAM_FFN_DOWN, il, "bias"));
                    }
                    model.neck_0_w       = get_tensor(string_format(TN_SAM_NECK, 0, "weight"));
                    model.neck_1_b       = get_tensor(string_format(TN_SAM_NECK, 1, "bias"));
                    model.neck_1_w       = get_tensor(string_format(TN_SAM_NECK, 1, "weight"));
                    model.neck_2_w       = get_tensor(string_format(TN_SAM_NECK, 2, "weight"));
                    model.neck_3_b       = get_tensor(string_format(TN_SAM_NECK, 3, "bias"));
                    model.neck_3_w       = get_tensor(string_format(TN_SAM_NECK, 3, "weight"));
                    model.net_2          = get_tensor(string_format(TN_SAM_NET, 2, "weight"));
                    model.net_3          = get_tensor(string_format(TN_SAM_NET, 3, "weight"));
                    model.image_newline  = get_tensor(TN_IMAGE_NEWLINE, false);
                    model.view_seperator = get_tensor(TN_IMAGE_SEPERATOR);
                    model.mm_fc_w        = get_tensor(string_format(TN_MM_PROJECTOR, "weight"));
                    model.mm_fc_b        = get_tensor(string_format(TN_MM_PROJECTOR, "bias"));
                    model.resample_query_768  = get_tensor(string_format(TN_RESMPL_QUERY, 768, "weight"), false);
                    model.resample_query_1024 = get_tensor(string_format(TN_RESMPL_QUERY, 1024, "weight"), false);
                 } break;
            case PROJECTOR_TYPE_GEMMA4A:
                {
                    for (int i = 0; i < 2; i++) {
                        model.sscp_conv_w[i] = get_tensor(string_format(TN_A_CONV1D, i, "weight"));
                        model.sscp_conv_b[i] = get_tensor(string_format(TN_A_CONV1D, i, "bias"), false);
                        model.sscp_norm_w[i] = get_tensor(string_format(TN_A_CONV1D_NORM, i, "weight"), false);
                    }
                    model.sscp_inp_proj_w = get_tensor(string_format(TN_A_INP_PROJ, "weight"));
                    model.sscp_inp_proj_b = get_tensor(string_format(TN_A_INP_PROJ, "bias"), false);
                    model.audio_out_proj_w = get_tensor(string_format(TN_A_OUT_PROJ, "weight"), false);
                    model.audio_out_proj_b = get_tensor(string_format(TN_A_OUT_PROJ, "bias"), false);
                    // audio multimodal embedder (mm.a.* namespace, not mm.*)
                    model.mm_soft_emb_norm_w = get_tensor(string_format(TN_A_MM_SOFT_EMB_N, "weight"), false);
                    model.mm_input_proj_w    = get_tensor(string_format(TN_A_MM_INP_PROJ, "weight"), false);

                    // Per-layer tensors NOT loaded by the generic loop above
                    for (int il = 0; il < hparams.n_layer; ++il) {
                        auto & layer = model.layers[il];

                        // Gemma4 audio conformer-specific tensors
                        layer.ff_norm_w        = get_tensor(string_format(TN_FFN_NORM, prefix, il, "weight"));
                        layer.attn_pre_norm_w  = get_tensor(string_format(TN_A_ATTN_PRE_NORM, prefix, il, "weight"), false);
                        layer.per_dim_scale_w  = get_tensor(string_format(TN_A_PER_DIM_SCALE, prefix, il, "weight"), false);
                        layer.per_dim_k_scale_w = get_tensor(string_format(TN_A_PER_DIM_K_SCALE, prefix, il, "weight"), false);
                        layer.attn_k_rel_w     = get_tensor(string_format(TN_A_ATTN_K_REL, prefix, il, "weight"), false);

                        // Convolution module
                        // Note: conv_norm / norm_conv are swapped in GGUF due to
                        // upstream tensor_mapping.py, so we load them in reverse order
                        layer.norm_conv_w  = get_tensor(string_format(TN_CONV_NORM, prefix, il, "weight"), false);
                        layer.norm_conv_b  = get_tensor(string_format(TN_CONV_NORM, prefix, il, "bias"), false);
                        layer.conv_pw1_w   = get_tensor(string_format(TN_CONV_PW1,  prefix, il, "weight"));
                        layer.conv_pw1_b   = get_tensor(string_format(TN_CONV_PW1,  prefix, il, "bias"), false);
                        layer.conv_dw_w    = get_tensor(string_format(TN_CONV_DW,   prefix, il, "weight"));
                        layer.conv_dw_b    = get_tensor(string_format(TN_CONV_DW,   prefix, il, "bias"), false);
                        layer.conv_norm_w  = get_tensor(string_format(TN_NORM_CONV, prefix, il, "weight"), false);
                        layer.conv_norm_b  = get_tensor(string_format(TN_NORM_CONV, prefix, il, "bias"), false);
                        layer.conv_pw2_w   = get_tensor(string_format(TN_CONV_PW2,  prefix, il, "weight"));
                        layer.conv_pw2_b   = get_tensor(string_format(TN_CONV_PW2,  prefix, il, "bias"), false);

                        // FFN2 (second half-step)
                        layer.ff_norm_1_w      = get_tensor(string_format(TN_FFN_NORM_1, prefix, il, "weight"));
                        layer.ff_up_1_w        = get_tensor(string_format(TN_FFN_UP_1, prefix, il, "weight"));
                        layer.ff_up_1_b        = get_tensor(string_format(TN_FFN_UP_1, prefix, il, "bias"), false);
                        layer.ff_down_1_w      = get_tensor(string_format(TN_FFN_DOWN_1, prefix, il, "weight"));
                        layer.ff_down_1_b      = get_tensor(string_format(TN_FFN_DOWN_1, prefix, il, "bias"), false);
                        layer.ff_post_norm_1_w = get_tensor(string_format(TN_A_FFN_POST_NORM_1, prefix, il, "weight"), false);
                    }

                    // Load clamp info for ClippableLinear AFTER all tensors are loaded
                    for (auto * tensor : tensors_to_load) {
                        std::string name = tensor->name;
                        if (string_ends_with(name, ".weight")) {
                            std::string name_inp_max = name;
                            std::string name_inp_min = name;
                            std::string name_out_max = name;
                            std::string name_out_min = name;
                            string_replace_all(name_inp_max, ".weight", ".input_max");
                            string_replace_all(name_inp_min, ".weight", ".input_min");
                            string_replace_all(name_out_max, ".weight", ".output_max");
                            string_replace_all(name_out_min, ".weight", ".output_min");
                            model.clamp_info_map[name] = {
                                get_scalar(name_inp_max, FLT_MAX),
                                get_scalar(name_inp_min, -FLT_MAX),
                                get_scalar(name_out_max, FLT_MAX),
                                get_scalar(name_out_min, -FLT_MAX)
                            };
                        }
                    }
                } break;
            case PROJECTOR_TYPE_GEMMA4UA:
                {
                    model.mm_input_proj_w = get_tensor(string_format(TN_A_MM_INP_PROJ, "weight"));
                } break;
            case PROJECTOR_TYPE_LFM2A:
            case PROJECTOR_TYPE_D1OMNI_A:
                {
                    for (int i : {0, 2, 3, 5, 6}) {
                        model.pre_encode_conv_X_w[i] = get_tensor(string_format(TN_CONV1D, i, "weight"));
                        model.pre_encode_conv_X_b[i] = get_tensor(string_format(TN_CONV1D, i, "bias"));
                    }
                    model.pre_encode_out_w    = get_tensor(string_format(TN_PRE_ENCODE_OUT, "weight"));
                    model.pre_encode_out_b    = get_tensor(string_format(TN_PRE_ENCODE_OUT, "bias"));

                    model.mm_0_w = get_tensor(string_format(TN_MM_AUDIO_MLP, 0, "weight"));
                    model.mm_0_b = get_tensor(string_format(TN_MM_AUDIO_MLP, 0, "bias"));
                    model.mm_1_w = get_tensor(string_format(TN_MM_AUDIO_MLP, 1, "weight"));
                    model.mm_1_b = get_tensor(string_format(TN_MM_AUDIO_MLP, 1, "bias"));
                    model.mm_3_w = get_tensor(string_format(TN_MM_AUDIO_MLP, 3, "weight"));
                    model.mm_3_b = get_tensor(string_format(TN_MM_AUDIO_MLP, 3, "bias"));

                    // residual block after the projector: norm, down, up
                    if (model.proj_type == PROJECTOR_TYPE_D1OMNI_A) {
                        model.mm_4_w = get_tensor(string_format(TN_MM_AUDIO_MLP, 4, "weight"));
                        model.mm_4_b = get_tensor(string_format(TN_MM_AUDIO_MLP, 4, "bias"));
                        model.mm_5_w = get_tensor(string_format(TN_MM_AUDIO_MLP, 5, "weight"));
                        model.mm_5_b = get_tensor(string_format(TN_MM_AUDIO_MLP, 5, "bias"));
                        model.mm_6_w = get_tensor(string_format(TN_MM_AUDIO_MLP, 6, "weight"));
                        model.mm_6_b = get_tensor(string_format(TN_MM_AUDIO_MLP, 6, "bias"));
                    }

                    for (int il = 0; il < hparams.n_layer; ++il) {
                        auto & layer = model.layers[il];

                        layer.ff_norm_w   = get_tensor(string_format(TN_FFN_NORM,   prefix, il, "weight"));
                        layer.ff_norm_b   = get_tensor(string_format(TN_FFN_NORM,   prefix, il, "bias"));
                        layer.ff_norm_1_w = get_tensor(string_format(TN_FFN_NORM_1, prefix, il, "weight"));
                        layer.ff_norm_1_b = get_tensor(string_format(TN_FFN_NORM_1, prefix, il, "bias"));
                        layer.ff_up_1_w   = get_tensor(string_format(TN_FFN_UP_1,   prefix, il, "weight"));
                        layer.ff_up_1_b   = get_tensor(string_format(TN_FFN_UP_1,   prefix, il, "bias"));
                        layer.ff_down_1_w = get_tensor(string_format(TN_FFN_DOWN_1, prefix, il, "weight"));
                        layer.ff_down_1_b = get_tensor(string_format(TN_FFN_DOWN_1, prefix, il, "bias"));

                        layer.pos_bias_u = get_tensor(string_format(TN_POS_BIAS_U, prefix, il));
                        layer.pos_bias_v = get_tensor(string_format(TN_POS_BIAS_V, prefix, il));

                        layer.norm_conv_w = get_tensor(string_format(TN_NORM_CONV, prefix, il, "weight"));
                        layer.norm_conv_b = get_tensor(string_format(TN_NORM_CONV, prefix, il, "bias"));

                        layer.linear_pos_w = get_tensor(string_format(TN_LINEAR_POS, prefix, il, "weight"));

                        layer.conv_norm_w  = get_tensor(string_format(TN_CONV_NORM, prefix, il, "weight"));
                        layer.conv_norm_b  = get_tensor(string_format(TN_CONV_NORM, prefix, il, "bias"));
                        layer.conv_dw_w    = get_tensor(string_format(TN_CONV_DW,   prefix, il, "weight"));
                        layer.conv_dw_b    = get_tensor(string_format(TN_CONV_DW,   prefix, il, "bias"));
                        layer.conv_pw1_w   = get_tensor(string_format(TN_CONV_PW1,  prefix, il, "weight"));
                        layer.conv_pw1_b   = get_tensor(string_format(TN_CONV_PW1,  prefix, il, "bias"));
                        layer.conv_pw2_w   = get_tensor(string_format(TN_CONV_PW2,  prefix, il, "weight"));
                        layer.conv_pw2_b   = get_tensor(string_format(TN_CONV_PW2,  prefix, il, "bias"));
                    }
                } break;
            case PROJECTOR_TYPE_LFM3A:
                {
                    for (int i : {0, 2, 3, 5, 6}) {
                        model.pre_encode_conv_X_w[i] = get_tensor(string_format(TN_CONV1D, i, "weight"));
                        model.pre_encode_conv_X_b[i] = get_tensor(string_format(TN_CONV1D, i, "bias"));
                    }
                    model.pre_encode_out_w = get_tensor(string_format(TN_PRE_ENCODE_OUT, "weight"));
                    model.pre_encode_out_b = get_tensor(string_format(TN_PRE_ENCODE_OUT, "bias"));

                    model.mm_0_w = get_tensor(string_format(TN_MM_AUDIO_MLP, 0, "weight"));
                    model.mm_0_b = get_tensor(string_format(TN_MM_AUDIO_MLP, 0, "bias"));
                    model.mm_1_w = get_tensor(string_format(TN_MM_AUDIO_MLP, 1, "weight"));
                    model.mm_1_b = get_tensor(string_format(TN_MM_AUDIO_MLP, 1, "bias"));
                    model.mm_3_w = get_tensor(string_format(TN_MM_AUDIO_MLP, 3, "weight"));
                    model.mm_3_b = get_tensor(string_format(TN_MM_AUDIO_MLP, 3, "bias"));

                    if (!model.post_ln_w) {
                        throw std::runtime_error(string_format("%s: missing tensor %s\n", __func__, string_format(TN_LN_POST, prefix, "weight").c_str()));
                    }

                    // each layer is either a short conv block or a complete attention block
                    for (int il = 0; il < hparams.n_layer; ++il) {
                        auto & layer = model.layers[il];
                        layer.conv_pw1_w = get_tensor(string_format(TN_CONV_PW1, prefix, il, "weight"), false);
                        layer.conv_dw_w  = get_tensor(string_format(TN_CONV_DW,  prefix, il, "weight"), false);
                        layer.conv_pw2_w = get_tensor(string_format(TN_CONV_PW2, prefix, il, "weight"), false);
                        const bool is_conv = layer.conv_pw1_w && layer.conv_dw_w && layer.conv_pw2_w;
                        const bool is_attn = layer.q_w && layer.k_w && layer.v_w && layer.o_w && layer.q_norm && layer.k_norm;
                        if (is_conv == is_attn) {
                            throw std::runtime_error(string_format("%s: lfm3a layer %d is neither a complete conv nor a complete attention block\n", __func__, il));
                        }
                    }
                } break;
            case PROJECTOR_TYPE_PARAKEET:
                {

                    hparams.mel_filters = get_vector(TN_MEL_FILTERS);
                    hparams.window      = get_vector(TN_WINDOW);

                    // Subsampling layers (conv1d)
                    for (int i : {0, 2, 3, 5, 6}) {
                        model.pre_encode_conv_X_w[i] = get_tensor(string_format(TN_CONV1D, i, "weight"));
                        model.pre_encode_conv_X_b[i] = get_tensor(string_format(TN_CONV1D, i, "bias"));
                    }
                    model.pre_encode_out_w = get_tensor(string_format(TN_PRE_ENCODE_OUT, "weight"));
                    model.pre_encode_out_b = get_tensor(string_format(TN_PRE_ENCODE_OUT, "bias"));

                    // Projection layers
                    model.mm_norm_pre_w = get_tensor(string_format(TN_MM_NORM_PRE, "weight"), false);
                    model.mm_0_w        = get_tensor(string_format(TN_MM_AUDIO_MLP, 1, "weight"), false);
                    model.mm_1_w        = get_tensor(string_format(TN_MM_AUDIO_MLP, 2, "weight"), false);

                    // Encoder layers
                    for (int il = 0; il < hparams.n_layer; ++il) {
                        auto & layer = model.layers[il];

                        // Attention (from shared above)

                        // Relative position encoding
                        layer.linear_pos_w = get_tensor(string_format(TN_LINEAR_POS, prefix, il, "weight"));
                        layer.pos_bias_u   = get_tensor(string_format(TN_POS_BIAS_U, prefix, il));
                        layer.pos_bias_v   = get_tensor(string_format(TN_POS_BIAS_V, prefix, il));

                        // Convolution module
                        layer.conv_pw1_w     = get_tensor(string_format(TN_CONV_PW1,       prefix, il, "weight"));
                        layer.conv_pw1_b     = get_tensor(string_format(TN_CONV_PW1,       prefix, il, "bias"), false);
                        layer.conv_dw_w      = get_tensor(string_format(TN_CONV_DW,        prefix, il, "weight"));
                        layer.conv_dw_b      = get_tensor(string_format(TN_CONV_DW,        prefix, il, "bias"), false);
                        layer.conv_norm_w    = get_tensor(string_format(TN_CONV_NORM,      prefix, il, "weight"));
                        layer.conv_norm_b    = get_tensor(string_format(TN_CONV_NORM,      prefix, il, "bias"));
                        layer.conv_norm_mean = get_tensor(string_format(TN_CONV_NORM_MEAN, prefix, il));
                        layer.conv_norm_var  = get_tensor(string_format(TN_CONV_NORM_VAR,  prefix, il));
                        layer.conv_pw2_w     = get_tensor(string_format(TN_CONV_PW2,       prefix, il, "weight"));
                        layer.conv_pw2_b     = get_tensor(string_format(TN_CONV_PW2,       prefix, il, "bias"), false);

                        // Feed-forward networks
                        layer.ff_norm_w   = get_tensor(string_format(TN_FFN_NORM,   prefix, il, "weight"));
                        layer.ff_norm_b   = get_tensor(string_format(TN_FFN_NORM,   prefix, il, "bias"));

                        layer.ff_norm_1_w = get_tensor(string_format(TN_FFN_NORM_1, prefix, il, "weight"));
                        layer.ff_norm_1_b = get_tensor(string_format(TN_FFN_NORM_1, prefix, il, "bias"));
                        layer.ff_up_1_w   = get_tensor(string_format(TN_FFN_UP_1,   prefix, il, "weight"));
                        layer.ff_up_1_b   = get_tensor(string_format(TN_FFN_UP_1,   prefix, il, "bias"), false);
                        layer.ff_down_1_w = get_tensor(string_format(TN_FFN_DOWN_1, prefix, il, "weight"));
                        layer.ff_down_1_b = get_tensor(string_format(TN_FFN_DOWN_1, prefix, il, "bias"), false);

                        // Layer norms
                        layer.norm_conv_w = get_tensor(string_format(TN_NORM_CONV, prefix, il, "weight"));
                        layer.norm_conv_b = get_tensor(string_format(TN_NORM_CONV, prefix, il, "bias"));
                    }

                    model.mm_model_mlp_1_w = get_tensor(string_format(TN_MVLM_PROJ_MLP, 0, "weight"));
                    model.mm_model_mlp_2_w = get_tensor(string_format(TN_MVLM_PROJ_MLP, 1, "weight"));
                    model.mm_model_mlp_3_w = get_tensor(string_format(TN_MVLM_PROJ_MLP, 3, "weight"));
                } break;
            case PROJECTOR_TYPE_NEMOTRON_TRANSCRIBE:
                {
                    hparams.mel_filters = get_vector(TN_MEL_FILTERS);
                    hparams.window      = get_vector(TN_WINDOW);

                    // ASR expert: blocks, pre_ln and post_ln come from the generic loader
                    model.inp_proj_w = get_tensor(string_format(TN_INP_PROJ, "weight"));
                    for (int il = 0; il < hparams.n_layer; ++il) {
                        auto & layer = model.layers[il];
                        layer.q_norm_b = get_tensor(string_format(TN_ATTN_Q_NORM, prefix, il, "bias"));
                        layer.k_norm_b = get_tensor(string_format(TN_ATTN_K_NORM, prefix, il, "bias"));
                    }

                    // speaker expert
                    const char * sp = "a.spk";
                    model.spk_inp_proj_w   = get_tensor(string_format(TN_A_SPK_INP_PROJ, "weight"));
                    model.spk_pre_ln_w     = get_tensor(string_format(TN_LN_PRE,  sp, "weight"));
                    model.spk_pre_ln_b     = get_tensor(string_format(TN_LN_PRE,  sp, "bias"));
                    model.spk_post_ln_w    = get_tensor(string_format(TN_LN_POST, sp, "weight"));
                    model.spk_post_ln_b    = get_tensor(string_format(TN_LN_POST, sp, "bias"));
                    model.spk_layers.resize(hparams.spk_n_layer);
                    for (int il = 0; il < hparams.spk_n_layer; ++il) {
                        auto & layer = model.spk_layers[il];
                        layer.ln_1_w    = get_tensor(string_format(TN_LN_1,        sp, il, "weight"));
                        layer.ln_1_b    = get_tensor(string_format(TN_LN_1,        sp, il, "bias"));
                        layer.qkv_w     = get_tensor(string_format(TN_ATTN_QKV,    sp, il, "weight"));
                        layer.o_w       = get_tensor(string_format(TN_ATTN_OUTPUT, sp, il, "weight"));
                        layer.o_b       = get_tensor(string_format(TN_ATTN_OUTPUT, sp, il, "bias"));
                        layer.ln_2_w    = get_tensor(string_format(TN_LN_2,        sp, il, "weight"));
                        layer.ln_2_b    = get_tensor(string_format(TN_LN_2,        sp, il, "bias"));
                        layer.ff_up_w   = get_tensor(string_format(TN_FFN_UP,      sp, il, "weight"));
                        layer.ff_up_b   = get_tensor(string_format(TN_FFN_UP,      sp, il, "bias"));
                        layer.ff_down_w = get_tensor(string_format(TN_FFN_DOWN,    sp, il, "weight"));
                        layer.ff_down_b = get_tensor(string_format(TN_FFN_DOWN,    sp, il, "bias"));
                    }

                    // Sortformer head, fusion
                    model.spk_head_proj_w   = get_tensor(string_format(TN_A_SPK_HEAD, "proj",     "weight"));
                    model.spk_head_proj_b   = get_tensor(string_format(TN_A_SPK_HEAD, "proj",     "bias"));
                    model.spk_head_up_w     = get_tensor(string_format(TN_A_SPK_HEAD, "upsample", "weight"));
                    model.spk_head_up_b     = get_tensor(string_format(TN_A_SPK_HEAD, "upsample", "bias"));
                    model.spk_head_fc1_w    = get_tensor(string_format(TN_A_SPK_HEAD, "fc1",      "weight"));
                    model.spk_head_fc1_b    = get_tensor(string_format(TN_A_SPK_HEAD, "fc1",      "bias"));
                    model.spk_head_fc2_w    = get_tensor(string_format(TN_A_SPK_HEAD, "fc2",      "weight"));
                    model.spk_head_fc2_b    = get_tensor(string_format(TN_A_SPK_HEAD, "fc2",      "bias"));
                    model.fusion_asr_norm_w = get_tensor(string_format(TN_A_FUSION_NORM, "asr_norm", "weight"));
                    model.fusion_asr_norm_b = get_tensor(string_format(TN_A_FUSION_NORM, "asr_norm", "bias"));
                    model.fusion_spk_norm_w = get_tensor(string_format(TN_A_FUSION_NORM, "spk_norm", "weight"));
                    model.fusion_spk_norm_b = get_tensor(string_format(TN_A_FUSION_NORM, "spk_norm", "bias"));
                    model.fusion_spk_kernel = get_tensor(TN_A_FUSION_KERNEL);

                    model.spk_sil_emb = get_vector(TN_A_SPK_SIL_EMB);

                    // the head layouts and the widths the graph and the speaker state use must match the tensors
                    hparams.spk_n_embd = model.spk_inp_proj_w->ne[1];
                    const int64_t d_head     = hparams.n_head     > 0 ? hparams.n_embd     / hparams.n_head     : 0;
                    const int64_t d_head_spk = hparams.spk_n_head > 0 ? hparams.spk_n_embd / hparams.spk_n_head : 0;
                    const int64_t n_spk      = model.spk_head_fc2_w->ne[1];
                    // the graph splits each fused QKV into three views of the layer width
                    bool qkv_ok = true;
                    for (const auto & layer : model.layers) {
                        qkv_ok = qkv_ok && layer.qkv_w && layer.qkv_w->ne[0] == hparams.n_embd && layer.qkv_w->ne[1] == 3 * hparams.n_embd;
                    }
                    for (const auto & layer : model.spk_layers) {
                        qkv_ok = qkv_ok && layer.qkv_w->ne[0] == hparams.spk_n_embd && layer.qkv_w->ne[1] == 3 * hparams.spk_n_embd;
                    }
                    if (!qkv_ok || model.inp_proj_w->ne[1] != hparams.n_embd || d_head <= 0 || d_head * hparams.n_head != hparams.n_embd ||
                            (hparams.n_embd_head > 0 && hparams.n_embd_head != d_head) ||
                            hparams.audio_rope_dims <= 0 || hparams.audio_rope_dims % 2 != 0 || hparams.audio_rope_dims > d_head ||
                            d_head_spk <= 0 || d_head_spk % 2 != 0 || d_head_spk * hparams.spk_n_head != hparams.spk_n_embd ||
                            (int64_t) model.spk_sil_emb.size() != hparams.spk_n_embd || n_spk <= 0 ||
                            model.fusion_spk_kernel->ne[0] != n_spk || model.fusion_spk_kernel->ne[1] != hparams.n_embd ||
                            hparams.mel_filters.size() != (size_t) hparams.n_mel_bins * (hparams.audio_n_fft / 2 + 1) ||
                            hparams.window.size() != (size_t) hparams.audio_window_len) {
                        throw std::runtime_error(string_format("%s: hparams do not match the tensor shapes\n", __func__));
                    }

                    // projection to the text model
                    model.mm_fc_w = get_tensor(string_format(TN_MM_AUDIO_FC, "weight"));
                    model.mm_fc_b = get_tensor(string_format(TN_MM_AUDIO_FC, "bias"));
                } break;
            case PROJECTOR_TYPE_GRANITE_SPEECH:
                {
                    model.inp_proj_w     = get_tensor(string_format(TN_INP_PROJ,    "weight"));
                    model.inp_proj_b     = get_tensor(string_format(TN_INP_PROJ,    "bias"));
                    model.ctc_out_w      = get_tensor(string_format(TN_CTC_OUT,     "weight"));
                    model.ctc_out_b      = get_tensor(string_format(TN_CTC_OUT,     "bias"));
                    model.ctc_out_mid_w  = get_tensor(string_format(TN_CTC_OUT_MID, "weight"));
                    model.ctc_out_mid_b  = get_tensor(string_format(TN_CTC_OUT_MID, "bias"));

                    // per-layer tensors not loaded by the generic loop above
                    for (int il = 0; il < hparams.n_layer; ++il) {
                        auto & layer = model.layers[il];

                        layer.attn_rel_pos_emb = get_tensor(string_format(TN_ATTN_REL_POS_EMB, prefix, il));

                        layer.ff_norm_w   = get_tensor(string_format(TN_FFN_NORM,   prefix, il, "weight"));
                        layer.ff_norm_b   = get_tensor(string_format(TN_FFN_NORM,   prefix, il, "bias"));

                        layer.ff_norm_1_w = get_tensor(string_format(TN_FFN_NORM_1, prefix, il, "weight"));
                        layer.ff_norm_1_b = get_tensor(string_format(TN_FFN_NORM_1, prefix, il, "bias"));
                        layer.ff_up_1_w   = get_tensor(string_format(TN_FFN_UP_1,   prefix, il, "weight"));
                        layer.ff_up_1_b   = get_tensor(string_format(TN_FFN_UP_1,   prefix, il, "bias"));
                        layer.ff_down_1_w = get_tensor(string_format(TN_FFN_DOWN_1, prefix, il, "weight"));
                        layer.ff_down_1_b = get_tensor(string_format(TN_FFN_DOWN_1, prefix, il, "bias"));

                        layer.norm_conv_w = get_tensor(string_format(TN_NORM_CONV, prefix, il, "weight"));
                        layer.norm_conv_b = get_tensor(string_format(TN_NORM_CONV, prefix, il, "bias"));
                        layer.conv_norm_w = get_tensor(string_format(TN_CONV_NORM, prefix, il, "weight"));
                        layer.conv_norm_b = get_tensor(string_format(TN_CONV_NORM, prefix, il, "bias"));
                        layer.conv_dw_w   = get_tensor(string_format(TN_CONV_DW,   prefix, il, "weight"));
                        layer.conv_pw1_w  = get_tensor(string_format(TN_CONV_PW1,  prefix, il, "weight"));
                        layer.conv_pw1_b  = get_tensor(string_format(TN_CONV_PW1,  prefix, il, "bias"));
                        layer.conv_pw2_w  = get_tensor(string_format(TN_CONV_PW2,  prefix, il, "weight"));
                        layer.conv_pw2_b  = get_tensor(string_format(TN_CONV_PW2,  prefix, il, "bias"));
                    }

                    model.qf_proj_blocks.resize(1);
                    auto & qf = model.qf_proj_blocks[0];
                    qf.qf_proj_query    = get_tensor(string_format(TN_QF_PROJ_QUERY, prefix));
                    qf.qf_proj_norm_w   = get_tensor(string_format(TN_QF_PROJ_NORM, prefix, "weight"));
                    qf.qf_proj_norm_b   = get_tensor(string_format(TN_QF_PROJ_NORM, prefix, "bias"));
                    qf.qf_proj_linear_w = get_tensor(string_format(TN_QF_PROJ_LINEAR, prefix, "weight"));
                    qf.qf_proj_linear_b = get_tensor(string_format(TN_QF_PROJ_LINEAR, prefix, "bias"));

                    const int n_proj_layers = 2;
                    qf.qf_proj_layers.resize(n_proj_layers);
                    for (int il = 0; il < n_proj_layers; ++il) {
                        auto & pl = qf.qf_proj_layers[il];

                        pl.q_w    = get_tensor(string_format(TN_QF_SELF_ATTN_Q, prefix, il, "weight"));
                        pl.q_b    = get_tensor(string_format(TN_QF_SELF_ATTN_Q, prefix, il, "bias"));
                        pl.k_w    = get_tensor(string_format(TN_QF_SELF_ATTN_K, prefix, il, "weight"));
                        pl.k_b    = get_tensor(string_format(TN_QF_SELF_ATTN_K, prefix, il, "bias"));
                        pl.v_w    = get_tensor(string_format(TN_QF_SELF_ATTN_V, prefix, il, "weight"));
                        pl.v_b    = get_tensor(string_format(TN_QF_SELF_ATTN_V, prefix, il, "bias"));
                        pl.o_w    = get_tensor(string_format(TN_QF_SELF_ATTN_O, prefix, il, "weight"));
                        pl.o_b    = get_tensor(string_format(TN_QF_SELF_ATTN_O, prefix, il, "bias"));
                        pl.ln_1_w = get_tensor(string_format(TN_QF_SELF_ATTN_N, prefix, il, "weight"));
                        pl.ln_1_b = get_tensor(string_format(TN_QF_SELF_ATTN_N, prefix, il, "bias"));

                        pl.cross_attn_q_w    = get_tensor(string_format(TN_QF_CROSS_ATTN_Q, prefix, il, "weight"));
                        pl.cross_attn_q_b    = get_tensor(string_format(TN_QF_CROSS_ATTN_Q, prefix, il, "bias"));
                        pl.cross_attn_k_w    = get_tensor(string_format(TN_QF_CROSS_ATTN_K, prefix, il, "weight"));
                        pl.cross_attn_k_b    = get_tensor(string_format(TN_QF_CROSS_ATTN_K, prefix, il, "bias"));
                        pl.cross_attn_v_w    = get_tensor(string_format(TN_QF_CROSS_ATTN_V, prefix, il, "weight"));
                        pl.cross_attn_v_b    = get_tensor(string_format(TN_QF_CROSS_ATTN_V, prefix, il, "bias"));
                        pl.cross_attn_o_w    = get_tensor(string_format(TN_QF_CROSS_ATTN_O, prefix, il, "weight"));
                        pl.cross_attn_o_b    = get_tensor(string_format(TN_QF_CROSS_ATTN_O, prefix, il, "bias"));
                        pl.cross_attn_norm_w = get_tensor(string_format(TN_QF_CROSS_ATTN_N, prefix, il, "weight"));
                        pl.cross_attn_norm_b = get_tensor(string_format(TN_QF_CROSS_ATTN_N, prefix, il, "bias"));

                        pl.ff_up_w   = get_tensor(string_format(TN_QF_FFN_UP,   prefix, il, "weight"));
                        pl.ff_up_b   = get_tensor(string_format(TN_QF_FFN_UP,   prefix, il, "bias"));
                        pl.ff_down_w = get_tensor(string_format(TN_QF_FFN_DOWN, prefix, il, "weight"));
                        pl.ff_down_b = get_tensor(string_format(TN_QF_FFN_DOWN, prefix, il, "bias"));
                        pl.ln_2_w    = get_tensor(string_format(TN_QF_FFN_NORM, prefix, il, "weight"));
                        pl.ln_2_b    = get_tensor(string_format(TN_QF_FFN_NORM, prefix, il, "bias"));
                    }
                } break;
            case PROJECTOR_TYPE_GRANITE4_VISION:
                {
                    // image_newline lives at the top-level.
                    model.image_newline = get_tensor(TN_IMAGE_NEWLINE);

                    // Load separate layerwise and spatial projector tensors
                    const auto projector_count = hparams.feature_layers.size();
                    model.qf_proj_blocks.resize(projector_count);
                    for (size_t bid = 0; bid < projector_count; ++bid) {
                        auto & b = model.qf_proj_blocks[bid];

                        // non-layerwise tensors
                        b.qf_proj_img_pos     = get_tensor(string_format(TN_MULTI_PROJ_IMG_POS,           bid));
                        b.qf_proj_query       = get_tensor(string_format(TN_MULTI_PROJ_QUERY,     prefix, bid));
                        b.qf_proj_linear_w    = get_tensor(string_format(TN_MULTI_PROJ_LINEAR,    prefix, bid, "weight"));
                        b.qf_proj_linear_b    = get_tensor(string_format(TN_MULTI_PROJ_LINEAR,    prefix, bid, "bias"));
                        b.qf_proj_norm_w      = get_tensor(string_format(TN_MULTI_PROJ_NORM,      prefix, bid, "weight"));
                        b.qf_proj_norm_b      = get_tensor(string_format(TN_MULTI_PROJ_NORM,      prefix, bid, "bias"));
                        b.qf_proj_post_norm_w = get_tensor(string_format(TN_MULTI_PROJ_POST_NORM, prefix, bid, "weight"));
                        b.qf_proj_post_norm_b = get_tensor(string_format(TN_MULTI_PROJ_POST_NORM, prefix, bid, "bias"));

                        // laywerwise tensors
                        // NOTE: If any model uses multi-layer qformers, this will need to change
                        b.qf_proj_layers.resize(1);
                        auto & pl = b.qf_proj_layers[0];

                        pl.q_w    = get_tensor(string_format(TN_QF_SELF_ATTN_Q, prefix, bid, "weight"));
                        pl.q_b    = get_tensor(string_format(TN_QF_SELF_ATTN_Q, prefix, bid, "bias"));
                        pl.k_w    = get_tensor(string_format(TN_QF_SELF_ATTN_K, prefix, bid, "weight"));
                        pl.k_b    = get_tensor(string_format(TN_QF_SELF_ATTN_K, prefix, bid, "bias"));
                        pl.v_w    = get_tensor(string_format(TN_QF_SELF_ATTN_V, prefix, bid, "weight"));
                        pl.v_b    = get_tensor(string_format(TN_QF_SELF_ATTN_V, prefix, bid, "bias"));
                        pl.o_w    = get_tensor(string_format(TN_QF_SELF_ATTN_O, prefix, bid, "weight"));
                        pl.o_b    = get_tensor(string_format(TN_QF_SELF_ATTN_O, prefix, bid, "bias"));
                        pl.ln_1_w = get_tensor(string_format(TN_QF_SELF_ATTN_N, prefix, bid, "weight"));
                        pl.ln_1_b = get_tensor(string_format(TN_QF_SELF_ATTN_N, prefix, bid, "bias"));

                        pl.cross_attn_q_w    = get_tensor(string_format(TN_QF_CROSS_ATTN_Q, prefix, bid, "weight"));
                        pl.cross_attn_q_b    = get_tensor(string_format(TN_QF_CROSS_ATTN_Q, prefix, bid, "bias"));
                        pl.cross_attn_k_w    = get_tensor(string_format(TN_QF_CROSS_ATTN_K, prefix, bid, "weight"));
                        pl.cross_attn_k_b    = get_tensor(string_format(TN_QF_CROSS_ATTN_K, prefix, bid, "bias"));
                        pl.cross_attn_v_w    = get_tensor(string_format(TN_QF_CROSS_ATTN_V, prefix, bid, "weight"));
                        pl.cross_attn_v_b    = get_tensor(string_format(TN_QF_CROSS_ATTN_V, prefix, bid, "bias"));
                        pl.cross_attn_o_w    = get_tensor(string_format(TN_QF_CROSS_ATTN_O, prefix, bid, "weight"));
                        pl.cross_attn_o_b    = get_tensor(string_format(TN_QF_CROSS_ATTN_O, prefix, bid, "bias"));
                        pl.cross_attn_norm_w = get_tensor(string_format(TN_QF_CROSS_ATTN_N, prefix, bid, "weight"));
                        pl.cross_attn_norm_b = get_tensor(string_format(TN_QF_CROSS_ATTN_N, prefix, bid, "bias"));

                        pl.ff_up_w   = get_tensor(string_format(TN_QF_FFN_UP,   prefix, bid, "weight"));
                        pl.ff_up_b   = get_tensor(string_format(TN_QF_FFN_UP,   prefix, bid, "bias"));
                        pl.ff_down_w = get_tensor(string_format(TN_QF_FFN_DOWN, prefix, bid, "weight"));
                        pl.ff_down_b = get_tensor(string_format(TN_QF_FFN_DOWN, prefix, bid, "bias"));
                        pl.ln_2_w    = get_tensor(string_format(TN_QF_FFN_NORM, prefix, bid, "weight"));
                        pl.ln_2_b    = get_tensor(string_format(TN_QF_FFN_NORM, prefix, bid, "bias"));
                    }

                } break;
            case PROJECTOR_TYPE_SENSENOVA_U1:
                {
                    model.mm_0_w = get_tensor(string_format(TN_LLAVA_PROJ, 0, "weight"));
                    model.mm_0_b = get_tensor(string_format(TN_LLAVA_PROJ, 0, "bias"));
                } break;
            default:
                GGML_ASSERT(false && "unknown projector type");
        }

        // load data
        {
            std::vector<uint8_t> read_buf;

            // start loading event
            if (progress_callback){
                progress_callback(0.0, progress_callback_user_data);
            }

            // compute total tensor data size for progress reporting
            size_t total_data_size = 0;
            for (auto & t : tensors_to_load) {
                total_data_size += ggml_nbytes(t);
            }

            // alloc memory and offload data
            model.buf.reset(ggml_backend_alloc_ctx_tensors_from_buft(model.ctx_data.get(), buft));
            ggml_backend_buffer_set_usage(model.buf.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
            // read the weight from file
            if (!no_alloc) {
                size_t data_loaded = 0;
                for (auto & t : tensors_to_load) {
                    ggml_tensor * cur = ggml_get_tensor(model.ctx_data.get(), t->name);
                    GGML_ASSERT(cur && "tensor not found in ctx_data");
                    auto it_off = tensor_offset.find(t->name);
                    GGML_ASSERT(it_off != tensor_offset.end() && "no offset for tensor");
                    const size_t offset = it_off->second;
                    fin.seekg(offset, std::ios::beg);
                    if (!fin) {
                        throw std::runtime_error(string_format("%s: failed to seek for tensor %s\n", __func__, t->name));
                    }
                    size_t num_bytes = ggml_nbytes(cur);
                    if (ggml_backend_buft_is_host(buft)) {
                        // for the CPU and Metal backend, we can read directly into the tensor
                        fin.read(reinterpret_cast<char *>(cur->data), num_bytes);
                    } else {
                        // read into a temporary buffer first, then copy to device memory
                        read_buf.resize(num_bytes);
                        fin.read(reinterpret_cast<char *>(read_buf.data()), num_bytes);
                        ggml_backend_tensor_set(cur, read_buf.data(), 0, num_bytes);
                    }
                    data_loaded += num_bytes;
                    if (progress_callback && total_data_size > 0) {
                        const float progress = (float)data_loaded / (float)total_data_size;
                        if (!progress_callback(progress, progress_callback_user_data)) {
                            throw std::runtime_error(string_format("%s: model loading cancelled by progress_callback\n", __func__));
                        }
                    }
                }
                LOG_DBG("%s: loaded %zu tensors from %s\n", __func__, tensors_to_load.size(), fname.c_str());
            } else {
                LOG_DBG("%s: no_alloc is set, skipping tensor data loading (%zu tensors)\n", __func__, tensors_to_load.size());
            }
            fin.close();
        }

    }

    void get_bool(const std::string & key, bool & output, bool required = true) const {
        const int i = gguf_find_key(ctx_gguf.get(), key.c_str());
        if (i < 0) {
            if (required) {
                throw std::runtime_error("Key not found: " + key);
            }
            return;
        }
        output = gguf_get_val_bool(ctx_gguf.get(), i);
    }

    void get_i32(const std::string & key, int & output, bool required = true) const {
        const int i = gguf_find_key(ctx_gguf.get(), key.c_str());
        if (i < 0) {
            if (required) {
                throw std::runtime_error("Key not found: " + key);
            }
            return;
        }
        output = gguf_get_val_i32(ctx_gguf.get(), i);
    }

    void get_u32(const std::string & key, int & output, bool required = true) const {
        const int i = gguf_find_key(ctx_gguf.get(), key.c_str());
        if (i < 0) {
            if (required) {
                throw std::runtime_error("Key not found: " + key);
            }
            return;
        }
        const uint32_t val = gguf_get_val_u32(ctx_gguf.get(), i);
        // sanity check
        if (val > (uint32_t) INT32_MAX) {
            throw std::runtime_error(string_format("%s: value %u for key '%s' exceeds INT32_MAX\n",
                __func__, val, key.c_str()));
        }
        output = (int) val;
    }

    void get_f32(const std::string & key, float & output, bool required = true) const {
        const int i = gguf_find_key(ctx_gguf.get(), key.c_str());
        if (i < 0) {
            if (required) {
                throw std::runtime_error("Key not found: " + key);
            }
            return;
        }
        output = gguf_get_val_f32(ctx_gguf.get(), i);
    }

    void get_arr_f32(const std::string & key, std::vector<float> & output, bool required = true) const {
        const int i = gguf_find_key(ctx_gguf.get(), key.c_str());
        if (i < 0) {
            if (required) {
                throw std::runtime_error("Key not found: " + key);
            }
            return;
        }
        if (gguf_get_kv_type(ctx_gguf.get(), i) != GGUF_TYPE_ARRAY) {
            throw std::runtime_error(string_format("%s: key '%s' is not an array\n", __func__, key.c_str()));
        }
        const auto type = gguf_get_arr_type(ctx_gguf.get(), i);
        if (type != GGUF_TYPE_FLOAT32) {
            throw std::runtime_error(string_format("%s: array '%s' has type %d, expected %d (GGUF_TYPE_FLOAT32)\n", __func__, key.c_str(), type, GGUF_TYPE_FLOAT32));
        }
        const size_t n = gguf_get_arr_n(ctx_gguf.get(), i);
        if (n > (size_t) std::numeric_limits<int>::max()) {
            throw std::runtime_error(string_format("%s: array '%s' is too large (%zu elements)\n", __func__, key.c_str(), n));
        }
        output.resize(n);
        const float * values = (const float *)gguf_get_arr_data(ctx_gguf.get(), i);
        for (size_t j = 0; j < n; ++j) {
            output[j] = values[j];
        }
    }

    void get_string(const std::string & key, std::string & output, bool required = true) const {
        const int i = gguf_find_key(ctx_gguf.get(), key.c_str());
        if (i < 0) {
            if (required) {
                throw std::runtime_error("Key not found: " + key);
            }
            return;
        }
        output = std::string(gguf_get_val_str(ctx_gguf.get(), i));
    }

    void get_arr_int(const std::string & key, std::vector<int> & output, bool required = true) const {
        const int i = gguf_find_key(ctx_gguf.get(), key.c_str());
        if (i < 0) {
            if (required) {
                throw std::runtime_error("Key not found: " + key);
            }
            return;
        }
        if (gguf_get_kv_type(ctx_gguf.get(), i) != GGUF_TYPE_ARRAY) {
            throw std::runtime_error(string_format("%s: key '%s' is not an array\n", __func__, key.c_str()));
        }
        const auto type = gguf_get_arr_type(ctx_gguf.get(), i);
        if (type != GGUF_TYPE_INT32) {
            throw std::runtime_error(string_format("%s: array '%s' has type %d, expected %d (GGUF_TYPE_INT32)\n", __func__, key.c_str(), type, GGUF_TYPE_INT32));
        }
        const size_t n = gguf_get_arr_n(ctx_gguf.get(), i);
        if (n > (size_t) std::numeric_limits<int>::max()) {
            throw std::runtime_error(string_format("%s: array '%s' is too large (%zu elements)\n", __func__, key.c_str(), n));
        }
        output.resize(n);
        const int32_t * values = (const int32_t *)gguf_get_arr_data(ctx_gguf.get(), i);
        for (size_t j = 0; j < n; ++j) {
            output[j] = values[j];
        }
    }

    static void set_llava_uhd_res_candidates(mtmd_model & model, const int max_patches_per_side) {
        auto & hparams = model.hparams;
        for (int x = 1; x <= max_patches_per_side; x++) {
            for (int y = 1; y <= max_patches_per_side; y++) {
                if (x == 1 && y == 1) {
                    continue; // skip the first point
                }
                hparams.image_res_candidates.push_back(mtmd_image_size{
                    x*hparams.image_size,
                    y*hparams.image_size,
                });
            }
        }
    }

    static void set_internvl_dhr_res_candidates(mtmd_model & model) {
        auto & hparams = model.hparams;
        int min_num = hparams.preproc_min_tiles;
        int max_num = hparams.preproc_max_tiles;
        if (min_num < 1) {
           return; // avoid  divide by 0
        }
        for (int a = min_num; a <= max_num; ++a) {
            int b_lo = (min_num + a - 1) / a;
            int b_hi = max_num / a;
            b_lo = std::max(b_lo, min_num);
            b_hi = std::min(b_hi, max_num);
            for (int b = b_lo; b <= b_hi; ++b) {
                hparams.image_res_candidates.push_back(mtmd_image_size {
                    a*hparams.image_size,
                    b*hparams.image_size,
                });
            }
        }
    }
};

//
// loading
//

mtmd_models mtmd_model_load(const char * fname, const mtmd_model_params & params, ggml_backend_buffer_type_t buft) {
    mtmd_model_loader loader(fname,
        /* skip_tensors */ false,
        params.progress_callback,
        params.progress_callback_user_data);

    auto load = [&](mtmd_modality modality) {
        auto model = std::make_unique<mtmd_model>();
        model->hparams.custom_image_min_tokens = params.image_min_tokens;
        model->hparams.custom_image_max_tokens = params.image_max_tokens;
        loader.load_hparams(*model, modality);
        loader.load_tensors(*model, buft, params.no_alloc);
        return model;
    };

    mtmd_models res;
    if (loader.has_vision) {
        res.v = load(MTMD_MODALITY_VISION);
    }

    // TODO: we don't support audio for Gemma 3N, but GGUF contains audio tensors
    // we can remove this check when we implement audio support for Gemma 3N
    const bool skip_audio = res.v && res.v->proj_type == PROJECTOR_TYPE_GEMMA3NV;
    if (loader.has_audio && !skip_audio) {
        res.a = load(MTMD_MODALITY_AUDIO);
    }

    if (loader.has_gen_audio) {
        res.gen_a = load(MTMD_MODALITY_GEN_AUDIO);
    }

    return res;
}

mtmd_model_caps mtmd_model_get_caps(const char * fname) {
    mtmd_model_caps res;
    mtmd_model_loader loader(fname, /* skip_tensors= */ true);
    res.has_vision    = loader.has_vision;
    res.has_audio     = loader.has_audio;
    res.has_gen_audio = loader.has_gen_audio;
    return res;
}

//
// mtmd_model
//

int mtmd_model::image_max_tokens() const {
    if (proj_type == PROJECTOR_TYPE_DEEPSEEK4V || proj_type == PROJECTOR_TYPE_DEEPSEEK41V) {
        return hparams.dsv4_max_n_token;
    }
    if (hparams.image_max_pixels <= 0) {
        return -1;
    }
    const int patch_area = hparams.patch_size * hparams.patch_size * hparams.n_merge * hparams.n_merge;
    return hparams.image_max_pixels / patch_area;
}

int mtmd_model::n_output_tokens_x(const mtmd_image_f32 * img) const {
    const auto & params = hparams;
    const int n_total = n_output_tokens(img);
    const auto & proj = proj_type;
    switch (proj) {
        case PROJECTOR_TYPE_QWEN2VL:
        case PROJECTOR_TYPE_QWEN25VL:
        case PROJECTOR_TYPE_QWEN3VL:
        case PROJECTOR_TYPE_MAGEVL:
        case PROJECTOR_TYPE_SENSENOVA_U1:
        case PROJECTOR_TYPE_LING3VL:
        case PROJECTOR_TYPE_EXAONE4_5:
        case PROJECTOR_TYPE_MIMOVL:
        case PROJECTOR_TYPE_GLM4V:
        case PROJECTOR_TYPE_GLM5V:
        case PROJECTOR_TYPE_PADDLEOCR:
        case PROJECTOR_TYPE_HUNYUANVL:
        case PROJECTOR_TYPE_YOUTUVL:
        case PROJECTOR_TYPE_MUSE_GLIMMER:
        case PROJECTOR_TYPE_LFM3VL:
            return (img->nx() / params.patch_size) / 2;
        case PROJECTOR_TYPE_STEP3VL:
        case PROJECTOR_TYPE_MINICPMV4_6:
        case PROJECTOR_TYPE_MINICPMV4_7:
        case PROJECTOR_TYPE_MINICPM_OCR_1_0:
            return img->nx() / (params.patch_size * params.n_merge);
        case PROJECTOR_TYPE_DEEPSEEKOCR:
        case PROJECTOR_TYPE_DEEPSEEKOCR2:
            return (img->nx() / params.patch_size) / 4;
        default:
            break;
    }
    return n_total;
}

int mtmd_model::n_output_tokens_y(const mtmd_image_f32 * img) const {
    const auto & params = hparams;
    const auto & proj = proj_type;
    switch (proj) {
        case PROJECTOR_TYPE_QWEN2VL:
        case PROJECTOR_TYPE_QWEN25VL:
        case PROJECTOR_TYPE_QWEN3VL:
        case PROJECTOR_TYPE_MAGEVL:
        case PROJECTOR_TYPE_SENSENOVA_U1:
        case PROJECTOR_TYPE_LING3VL:
        case PROJECTOR_TYPE_EXAONE4_5:
        case PROJECTOR_TYPE_MIMOVL:
        case PROJECTOR_TYPE_GLM4V:
        case PROJECTOR_TYPE_GLM5V:
        case PROJECTOR_TYPE_PADDLEOCR:
        case PROJECTOR_TYPE_HUNYUANVL:
        case PROJECTOR_TYPE_YOUTUVL:
        case PROJECTOR_TYPE_MUSE_GLIMMER:
        case PROJECTOR_TYPE_LFM3VL:
            return (img->ny() / params.patch_size) / 2;
        case PROJECTOR_TYPE_STEP3VL:
        case PROJECTOR_TYPE_MINICPMV4_6:
        case PROJECTOR_TYPE_MINICPMV4_7:
        case PROJECTOR_TYPE_MINICPM_OCR_1_0:
            return img->ny() / (params.patch_size * params.n_merge);
        default:
            break;
    }
    return 1;
}

int mtmd_model::n_output_tokens(const mtmd_image_f32 * img) const {
    const auto & params = hparams;

    // for models with fixed size image, the input image is already pre-processed and resized to square
    int patch_size = params.patch_size;
    int n_patches = (img->nx() / patch_size) * (img->ny() / patch_size);

    projector_type proj = proj_type;

    switch (proj) {
        case PROJECTOR_TYPE_MLP:
        case PROJECTOR_TYPE_MLP_NORM:
        case PROJECTOR_TYPE_JANUS_PRO:
        case PROJECTOR_TYPE_PHI4:
            {
                // do nothing
            } break;
        case PROJECTOR_TYPE_YASA2:
            {
                n_patches = 64; // adaptive average pooling to 8x8 tokens
            } break;
        case PROJECTOR_TYPE_LDP:
        case PROJECTOR_TYPE_LDPV2:
        case PROJECTOR_TYPE_GLM_EDGE:
            {
                n_patches /= 4;
                if (mm_boi) {
                    n_patches += 2; // for BOI and EOI token embeddings
                }
            } break;
        case PROJECTOR_TYPE_MINICPMV:
            {
                // Use actual config value if available, otherwise fall back to hardcoded values
                if (params.minicpmv_query_num > 0) {
                    n_patches = params.minicpmv_query_num;
                } else {
                    // Fallback to hardcoded values for legacy models
                    if (params.minicpmv_version == 2) {
                        n_patches = 96;
                    } else if (params.minicpmv_version == 3) {
                        n_patches = 64;
                    } else if (params.minicpmv_version == 4) {
                        n_patches = 64;
                    } else if (params.minicpmv_version == 5) {
                        // MiniCPM-V 4.0
                        n_patches = 64;
                    } else if (params.minicpmv_version == 6) {
                        // MiniCPM-V 4.5
                        n_patches = 64;
                    } else if (params.minicpmv_version == 100045) {
                        // MiniCPM-o 4.5
                        n_patches = 64;
                    } else {
                        GGML_ABORT("Unknown minicpmv version");
                    }
                }
            } break;
        case PROJECTOR_TYPE_MINICPMV4_6:
        case PROJECTOR_TYPE_MINICPMV4_7:
        case PROJECTOR_TYPE_MINICPM_OCR_1_0:
            {
                n_patches /= params.n_merge * params.n_merge;
            } break;
        case PROJECTOR_TYPE_QWEN2VL:
        case PROJECTOR_TYPE_QWEN25VL:
        case PROJECTOR_TYPE_QWEN3VL:
        case PROJECTOR_TYPE_MAGEVL:
        case PROJECTOR_TYPE_SENSENOVA_U1:
        case PROJECTOR_TYPE_LING3VL:
        case PROJECTOR_TYPE_EXAONE4_5:
        case PROJECTOR_TYPE_MIMOVL:
        case PROJECTOR_TYPE_MINIMAX_M3:
        case PROJECTOR_TYPE_GLM4V:
        case PROJECTOR_TYPE_GLM5V:
        case PROJECTOR_TYPE_YOUTUVL:
        case PROJECTOR_TYPE_MUSE_GLIMMER:
        case PROJECTOR_TYPE_LFM3VL:
            {
                // dynamic size (2 conv, so double patch size)
                int x_patch = img->nx() / (params.patch_size * 2);
                int y_patch = img->ny() / (params.patch_size * 2);
                n_patches = x_patch * y_patch;
            } break;
        case PROJECTOR_TYPE_STEP3VL:
            {
                int x_patch = img->nx() / (params.patch_size * params.n_merge);
                int y_patch = img->ny() / (params.patch_size * params.n_merge);
                n_patches = x_patch * y_patch;
            } break;
        case PROJECTOR_TYPE_GEMMA3:
        case PROJECTOR_TYPE_GEMMA4V:
        case PROJECTOR_TYPE_GEMMA4UV:
        case PROJECTOR_TYPE_IDEFICS3:
        case PROJECTOR_TYPE_COHERE2V:
        case PROJECTOR_TYPE_INTERNVL:
        case PROJECTOR_TYPE_NEMOTRON_V2_VL:
        case PROJECTOR_TYPE_NEMOTRON_OMNI:
        case PROJECTOR_TYPE_LLAMA4:
            {
                // both X and Y are downscaled by the scale factor
                int scale_factor = hparams.n_merge;
                n_patches /= (scale_factor * scale_factor);
            } break;
        case PROJECTOR_TYPE_GEMMA3NV:
            {
                // MobileNetV5 MSFA adapter always outputs fixed 16x16 resolution
                // regardless of input size (see architecture description)
                n_patches = hparams.image_size / hparams.patch_size;
            } break;
        case PROJECTOR_TYPE_LFM2:
        case PROJECTOR_TYPE_D1OMNI_V:
        case PROJECTOR_TYPE_KIMIVL:
        case PROJECTOR_TYPE_KIMIK25:
            {
                // dynamic size
                int out_patch_size = params.patch_size * hparams.n_merge;
                int x_patch = MTMD_ALIGN(img->nx(), out_patch_size) / out_patch_size;
                int y_patch = MTMD_ALIGN(img->ny(), out_patch_size) / out_patch_size;
                n_patches = x_patch * y_patch;
            } break;
        case PROJECTOR_TYPE_DEEPSEEK4V:
            {
                const int out_patch_size = params.patch_size * params.n_merge;
                const int n_llm_w = MTMD_ALIGN(img->nx(), out_patch_size) / out_patch_size;
                const int n_llm_h = MTMD_ALIGN(img->ny(), out_patch_size) / out_patch_size;
                n_patches = dsv4_get_block_layout(n_llm_w, n_llm_h, img->lead_pad).n_out;
            } break;
        case PROJECTOR_TYPE_DEEPSEEK41V:
            {
                const int out_patch_size = params.patch_size * params.n_merge;
                const int n_llm_w = MTMD_ALIGN(img->nx(), out_patch_size) / out_patch_size;
                const int n_llm_h = MTMD_ALIGN(img->ny(), out_patch_size) / out_patch_size;
                n_patches = dsv41_n_block_tokens(n_llm_w, n_llm_h);
            } break;
        case PROJECTOR_TYPE_PADDLEOCR:
        case PROJECTOR_TYPE_DOTS_OCR:
        case PROJECTOR_TYPE_DOTS3NOTE_V:
            {
                // dynamic size
                int n_merge = hparams.n_merge;
                int stride = n_merge * n_merge;
                n_patches = MTMD_ALIGN(n_patches, stride) / stride;
            } break;
        case PROJECTOR_TYPE_DOTS3NOTE_A:
            {
                // 3x stride-2 conv2d over mel frames
                n_patches = (img->nx() + 7) / 8;
            } break;
        case PROJECTOR_TYPE_PIXTRAL:
        case PROJECTOR_TYPE_LIGHTONOCR:
            {
                // dynamic size
                int n_merge = hparams.n_merge;
                int n_patches_x = img->nx() / patch_size / n_merge;
                int n_patches_y = img->ny() / patch_size / n_merge;
                if (token_embd_img_break) {
                    n_patches = n_patches_y * n_patches_x + n_patches_y - 1; // + one [IMG_BREAK] per row, except the last row
                } else {
                    n_patches = n_patches_y * n_patches_x;
                }
            } break;
        case PROJECTOR_TYPE_VOXTRAL_RT:
            {
                // 2 context frames, conv stride 2, then stacked frames
                n_patches = (img->nx() - 2) / 2 / hparams.proj_stack_factor;
            } break;
        case PROJECTOR_TYPE_MUSE_SPARK_AUDIO:
            {
                const int64_t fold = hparams.audio_samples_per_token();
                n_patches = (int) ((img->nx() + fold - 1) / fold);
            } break;
        case PROJECTOR_TYPE_VOXTRAL:
        case PROJECTOR_TYPE_ULTRAVOX:
        case PROJECTOR_TYPE_QWEN2A:
        case PROJECTOR_TYPE_MERALION:
        case PROJECTOR_TYPE_MUSIC_FLAMINGO:
            {
                n_patches = img->nx();

                const int proj_stack_factor = hparams.proj_stack_factor;
                if (audio_has_stack_frames()) {
                    GGML_ASSERT(proj_stack_factor > 0);
                    const int n_len = MTMD_ALIGN(n_patches, proj_stack_factor);
                    n_patches = n_len / proj_stack_factor;
                }

                // whisper downscales input token by half after conv1d
                n_patches /= 2;

                if (audio_has_avgpool()) {
                    // divide by 2 because of nn.AvgPool1d(2, stride=2)
                    n_patches /= 2;
                }
            } break;
        case PROJECTOR_TYPE_QWEN3A:
            {
                // chunk_size=100 frames --> 3x stride-2 conv2d --> 13 tokens per chunk
                const int chunk_size       = 100;
                const int tokens_per_chunk = 13;
                n_patches = (img->nx() / chunk_size) * tokens_per_chunk;
            } break;
        case PROJECTOR_TYPE_GLMA:
            {
                n_patches = img->nx();
                // whisper downscales input token by half after conv1d
                n_patches /= 2;
                // reshape by merge_factor
                n_patches /= hparams.proj_stack_factor;
                // for BOI and EOI token embeddings
                n_patches += 2;
            } break;
        case PROJECTOR_TYPE_COGVLM:
            {
                n_patches += 2; // for BOI and EOI token embeddings
            } break;
        case PROJECTOR_TYPE_DEEPSEEKOCR:
        {
            // SAM encoder applies two stride-2 convolutions (net_2 and net_3)
            // that reduce spatial dimensions by 4x in each direction (16x total)
            // E.g., 64x64 -> 16x16 patches
            n_patches /= 16;

            if (img->add_viewsep) {
                // global view: one image-newline per token-row + trailing view separator
                const int h = static_cast<int>(std::sqrt(static_cast<float>(n_patches)));
                n_patches = h * (h + 1) + 1;
            } else if (img->ny() >= img->nx() && img->ny() % img->nx() == 0) {
                // tile row: one image-newline per token-row
                const int grid_w = img->ny() / img->nx();
                const int tile_patches = img->nx() / (patch_size * 4); // patches per tile side (SAM divides by 4)
                const int h = tile_patches;
                n_patches = (tile_patches * grid_w + 1) * h;
            }
        } break;
        case PROJECTOR_TYPE_HUNYUANVL:
            {
                int merge = hparams.n_merge;
                int ow = (img->nx() / patch_size) / merge;
                int oh = (img->ny() / patch_size) / merge;
                n_patches = (ow + 1) * oh + 2;
            } break;
        case PROJECTOR_TYPE_DEEPSEEKOCR2:
        {
            // 1024 global view -> 256 query tokens + 1 view separator = 257;
            // 768 local tile   -> 144 query tokens, no separator.
            n_patches /= 16;
            if (img->add_viewsep) {
                n_patches += 1; // view separator, appended only after the global view
            }
        } break;
        case PROJECTOR_TYPE_LFM2A:
        case PROJECTOR_TYPE_D1OMNI_A:
        case PROJECTOR_TYPE_LFM3A:
            {
                n_patches = ((((img->nx() + 1) / 2) + 1) / 2 + 1) / 2;
            } break;
        case PROJECTOR_TYPE_GEMMA4A:
            {
                // Two Conv2D stride-2: O = floor((I + 2p - k) / s) + 1, p=1, k=3, s=2
                // O = floor((I - 1) / 2) + 1
                int n = img->nx();
                for (int i = 0; i < 2; i++) {
                    n = (n - 1) / 2 + 1;
                }
                n_patches = n;
            } break;
        case PROJECTOR_TYPE_NEMOTRON_TRANSCRIBE:
        case PROJECTOR_TYPE_PARAKEET:
            {
                n_patches = (img->nx() + (params.subsampling_factor - 1)) / params.subsampling_factor;
            } break;
        case PROJECTOR_TYPE_GEMMA4UA:
            {
                n_patches = img->nx();  // no downsampling: one token per raw waveform frame
            } break;
        case PROJECTOR_TYPE_MIMO_AUDIO:
            {
                // conv1(s=1) + conv2(s=2) -> RVQ-encoder downsample conv(k=2,s=2)
                int n = img->nx();
                n = (n - 1) / 2 + 1;         // conv1 + conv2
                n = (n - 2) / 2 + 1;         // downsample conv
                const int group_size = params.audio_local_group_size;
                n_patches = (n + group_size - 1) / group_size;
            } break;
        case PROJECTOR_TYPE_GRANITE_SPEECH:
            {
                const int ws = hparams.audio_proj_window_size;
                const int ds = hparams.audio_proj_downsample_rate;
                n_patches = ((img->nx() + ws - 1) / ws) * (ws / ds);
            } break;
        case PROJECTOR_TYPE_QWEN3TTS_SPKENC:
            {
                // pooling gives one speaker embedding, whatever the clip length is
                n_patches = 1;
            } break;
        case PROJECTOR_TYPE_CHATTERBOX_SPKENC:
            {
                // statistics pooling gives one speaker embedding, whatever the clip length is
                n_patches = 1;
            } break;
        case PROJECTOR_TYPE_QWEN3TTS_GEN:
        case PROJECTOR_TYPE_LFM3A_GEN:
        case PROJECTOR_TYPE_DRAMA3_GEN:
            {
                // one hidden-state vector fed back to the talker per call
                n_patches = 1;
            } break;
        case PROJECTOR_TYPE_POCKETTTS_SPKENC:
            {
                // one conditioning row per 12.5Hz frame
                const int hop = hparams.mimi_downsample * 120;
                n_patches = img->nx() / hop;
            } break;
        case PROJECTOR_TYPE_POCKETTTS_GEN:
            {
                // one latent per call for GEN_CODE, GEN_WAV sizes its input from the caller
                n_patches = 1;
            } break;
        case PROJECTOR_TYPE_GRANITE4_VISION:
            {
                // Per-tile output token count: each projector block outputs
                // query_side^2 tokens per window x n^2 windows.
                // For 384x384 input: n = 24/8 = 3, query_side = 4 -> 144.
                const int window_side = hparams.downsample_window_side;
                const int query_side  = hparams.downsample_query_side;
                const int side        = img->nx() / params.patch_size;
                const int n           = side / window_side;
                const int out_side    = query_side * n;
                n_patches             = out_side * out_side;
                if (img->anyres.is_tiled()) {
                    // overview tile, then the unpadded tile grid with one newline per row
                    int off_x, off_y, w, h;
                    mtmd_anyres_unpad(img->anyres.grid_x * out_side, img->anyres.grid_y * out_side,
                                      img->anyres.orig_nx, img->anyres.orig_ny, off_x, off_y, w, h);
                    n_patches += h * (w + 1);
                }
            } break;
        default:
            GGML_ABORT("unsupported projector type");
    }

    n_patches += (int) mtmd_suffix_rows(img->suffix_type).size();

    return n_patches;
}

size_t mtmd_model::cbx_read_tensor(const char * name, float * out, size_t n_max) const {
    const auto & tensors = cbx_tensors;
    auto it = tensors.find(name);
    if (it == tensors.end() || !it->second) {
        return 0;
    }
    ggml_tensor * t = it->second;
    const size_t n = (size_t) ggml_nelements(t);
    if (!out) {
        return n;
    }
    if (n_max < n) {
        return 0;
    }
    if (t->type == GGML_TYPE_F32) {
        ggml_backend_tensor_get(t, out, 0, n * sizeof(float));
    } else if (t->type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> tmp(n);
        ggml_backend_tensor_get(t, tmp.data(), 0, n * sizeof(ggml_fp16_t));
        for (size_t i = 0; i < n; i++) out[i] = ggml_fp16_to_fp32(tmp[i]);
    } else {
        return 0;
    }
    return n;
}

int mtmd_model::n_mmproj_embd() const {
    switch (proj_type) {
        case PROJECTOR_TYPE_LDP:
            return mm_model_block_1_block_2_1_b->ne[0];
        case PROJECTOR_TYPE_LDPV2:
            return mm_model_peg_0_b->ne[0];
        case PROJECTOR_TYPE_MLP:
        case PROJECTOR_TYPE_PHI4:
        case PROJECTOR_TYPE_PIXTRAL:
        case PROJECTOR_TYPE_LIGHTONOCR:
        case PROJECTOR_TYPE_DOTS_OCR:
        case PROJECTOR_TYPE_DOTS3NOTE_V:
        case PROJECTOR_TYPE_DOTS3NOTE_A:
            return mm_2_w->ne[1];
        case PROJECTOR_TYPE_MLP_NORM:
            return mm_3_b->ne[0];
        case PROJECTOR_TYPE_MINICPMV:
            return mm_model_proj->ne[0];
        case PROJECTOR_TYPE_MINICPMV4_6:
        case PROJECTOR_TYPE_MINICPMV4_7:
        case PROJECTOR_TYPE_MINICPM_OCR_1_0:
            return mm_ffn_down_w->ne[1];
        case PROJECTOR_TYPE_GLM_EDGE:
            return mm_model_mlp_3_w->ne[1];
        case PROJECTOR_TYPE_MINIMAX_M3:
            return mm_merger_fc2_b->ne[0];
        case PROJECTOR_TYPE_MUSE_GLIMMER:
            return mm_2_w->ne[1];
        case PROJECTOR_TYPE_QWEN2VL:
        case PROJECTOR_TYPE_QWEN25VL:
        case PROJECTOR_TYPE_MAGEVL:
        case PROJECTOR_TYPE_LFM3VL:
        case PROJECTOR_TYPE_EXAONE4_5:
        case PROJECTOR_TYPE_JANUS_PRO:
        case PROJECTOR_TYPE_YOUTUVL:
            return mm_1_b->ne[0];
        case PROJECTOR_TYPE_SENSENOVA_U1:
            return hparams.projection_dim;
        case PROJECTOR_TYPE_QWEN3VL:
            // main path + deepstack paths
            return mm_1_b->ne[0] * (1 + n_deepstack_layers);
        case PROJECTOR_TYPE_LING3VL:
            return mm_1_b->ne[0];
        case PROJECTOR_TYPE_MIMOVL:
            return mm_1_w->ne[1];
        case PROJECTOR_TYPE_STEP3VL:
            return mm_model_proj->ne[1];
        case PROJECTOR_TYPE_GEMMA3:
        case PROJECTOR_TYPE_GEMMA3NV:
            return mm_input_proj_w->ne[0];
        case PROJECTOR_TYPE_GEMMA4V:
        case PROJECTOR_TYPE_GEMMA4UV:
        case PROJECTOR_TYPE_GEMMA4A:
        case PROJECTOR_TYPE_GEMMA4UA:
            return mm_input_proj_w->ne[1];
        case PROJECTOR_TYPE_IDEFICS3:
        case PROJECTOR_TYPE_MUSE_SPARK_AUDIO:
            return mm_fc_w->ne[1];
        case PROJECTOR_TYPE_ULTRAVOX:
        case PROJECTOR_TYPE_VOXTRAL:
        case PROJECTOR_TYPE_VOXTRAL_RT:
        case PROJECTOR_TYPE_MUSIC_FLAMINGO:
            return mm_2_w->ne[1];
        case PROJECTOR_TYPE_MERALION:
            return mm_3_w->ne[1]; // out_proj output dim
        case PROJECTOR_TYPE_INTERNVL:
        case PROJECTOR_TYPE_NEMOTRON_V2_VL:
        case PROJECTOR_TYPE_NEMOTRON_OMNI:
            return mm_3_w->ne[1];
        case PROJECTOR_TYPE_LLAMA4:
            return mm_model_proj->ne[1];
        case PROJECTOR_TYPE_QWEN2A:
        case PROJECTOR_TYPE_NEMOTRON_TRANSCRIBE:
            return mm_fc_w->ne[1];
        case PROJECTOR_TYPE_QWEN3A:
            return mm_2_w->ne[1];
        case PROJECTOR_TYPE_GLMA:
        case PROJECTOR_TYPE_LFM2:
        case PROJECTOR_TYPE_D1OMNI_V:
        case PROJECTOR_TYPE_KIMIVL:
        case PROJECTOR_TYPE_PADDLEOCR:
        case PROJECTOR_TYPE_KIMIK25:
        case PROJECTOR_TYPE_YASA2:
        case PROJECTOR_TYPE_DEEPSEEK4V:
        case PROJECTOR_TYPE_DEEPSEEK41V:
        case PROJECTOR_TYPE_COHERE2V:
            return mm_2_w->ne[1];
        case PROJECTOR_TYPE_HUNYUANVL:
            return mm_model_proj->ne[1];
        case PROJECTOR_TYPE_COGVLM:
            return mm_4h_to_h_w->ne[1];
        case PROJECTOR_TYPE_DEEPSEEKOCR:
        case PROJECTOR_TYPE_DEEPSEEKOCR2:
            return mm_fc_w->ne[1];
        case PROJECTOR_TYPE_LFM2A:
            return position_embeddings->ne[0];
        case PROJECTOR_TYPE_D1OMNI_A:
        case PROJECTOR_TYPE_LFM3A:
            return mm_3_w->ne[1];
        case PROJECTOR_TYPE_GRANITE_SPEECH:
            return qf_proj_blocks[0].qf_proj_linear_w->ne[1];
        case PROJECTOR_TYPE_GRANITE4_VISION:
            return qf_proj_blocks.size() * hparams.projection_dim;
        case PROJECTOR_TYPE_GLM4V:
        case PROJECTOR_TYPE_GLM5V:
            return mm_ffn_down_w->ne[1];
        case PROJECTOR_TYPE_MIMO_AUDIO:
            return mm_2_w->ne[1];
        case PROJECTOR_TYPE_QWEN3TTS_SPKENC:
            return mm_fc_w->ne[2];
        case PROJECTOR_TYPE_QWEN3TTS_GEN:
        case PROJECTOR_TYPE_LFM3A_GEN:
        case PROJECTOR_TYPE_DRAMA3_GEN:
            return gen_code_out_embd_w->ne[0];
        case PROJECTOR_TYPE_POCKETTTS_SPKENC:
            return spk_proj_w->ne[1];
        case PROJECTOR_TYPE_POCKETTTS_GEN:
            return gen_input_lin_w->ne[1];
        case PROJECTOR_TYPE_CHATTERBOX:
            // talker width, carried by the speech table
            return cbx_tensors.at("a.gen.code.out_embd.weight")->ne[0];
        case PROJECTOR_TYPE_CHATTERBOX_SPKENC:
            // talker conditioning rows, the speaker projection carries their width
            return cbx_tensors.at("a.cenc.spkr_enc.weight")->ne[1];
        case PROJECTOR_TYPE_SOPRANO:
            return hparams.projection_dim;
        case PROJECTOR_TYPE_PARAKEET:
            return mm_1_w->ne[1];
        default:
            GGML_ABORT("Unknown projector type");
    }
}

// TODO @ngxson : this is no longer correct with mtmd_batch API
// this was only meant to be used by qwen-vl-based models, to fuse 2 input images into one (qwen-vl video support)
// this logic should be refactored in near future to distinctly handle "merge frames" and "batching"
int mtmd_model::n_temporal_merge() const {
    switch (proj_type) {
        case PROJECTOR_TYPE_QWEN2VL:
        case PROJECTOR_TYPE_QWEN25VL:
        case PROJECTOR_TYPE_QWEN3VL:
        case PROJECTOR_TYPE_LFM3VL:
        case PROJECTOR_TYPE_NEMOTRON_OMNI:
        case PROJECTOR_TYPE_LING3VL:
            return 2;
        default:
            return 1;
    }
}
