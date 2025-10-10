from __future__ import annotations

from enum import Enum, IntEnum, auto
from typing import Any

#
# constants
#

GGUF_MAGIC             = 0x46554747  # "GGUF"
GGUF_VERSION           = 3
GGUF_DEFAULT_ALIGNMENT = 32
GGML_QUANT_VERSION     = 2  # GGML_QNT_VERSION from ggml.h

#
# metadata keys
#


class Keys:
    class General:
        TYPE                       = "general.type"
        ARCHITECTURE               = "general.architecture"
        QUANTIZATION_VERSION       = "general.quantization_version"
        ALIGNMENT                  = "general.alignment"
        FILE_TYPE                  = "general.file_type"

        # Authorship Metadata
        NAME                       = "general.name"
        AUTHOR                     = "general.author"
        VERSION                    = "general.version"
        ORGANIZATION               = "general.organization"

        FINETUNE                   = "general.finetune"
        BASENAME                   = "general.basename"

        DESCRIPTION                = "general.description"
        QUANTIZED_BY               = "general.quantized_by"

        SIZE_LABEL                 = "general.size_label"

        # Licensing details
        LICENSE                    = "general.license"
        LICENSE_NAME               = "general.license.name"
        LICENSE_LINK               = "general.license.link"

        # Typically represents the converted GGUF repo (Unless native)
        URL                        = "general.url" # Model Website/Paper
        DOI                        = "general.doi"
        UUID                       = "general.uuid"
        REPO_URL                   = "general.repo_url" # Model Source Repository (git/svn/etc...)

        # Model Source during conversion
        SOURCE_URL                 = "general.source.url" # Model Website/Paper
        SOURCE_DOI                 = "general.source.doi"
        SOURCE_UUID                = "general.source.uuid"
        SOURCE_REPO_URL            = "general.source.repo_url" # Model Source Repository (git/svn/etc...)

        # Base Model Source. There can be more than one source if it's a merged
        # model like with 'Mistral-7B-Merge-14-v0.1'. This will assist in
        # tracing linage of models as it is finetuned or merged over time.
        BASE_MODEL_COUNT           = "general.base_model.count"
        BASE_MODEL_NAME            = "general.base_model.{id}.name"
        BASE_MODEL_AUTHOR          = "general.base_model.{id}.author"
        BASE_MODEL_VERSION         = "general.base_model.{id}.version"
        BASE_MODEL_ORGANIZATION    = "general.base_model.{id}.organization"
        BASE_MODEL_DESCRIPTION     = "general.base_model.{id}.description"
        BASE_MODEL_URL             = "general.base_model.{id}.url" # Model Website/Paper
        BASE_MODEL_DOI             = "general.base_model.{id}.doi"
        BASE_MODEL_UUID            = "general.base_model.{id}.uuid"
        BASE_MODEL_REPO_URL        = "general.base_model.{id}.repo_url" # Model Source Repository (git/svn/etc...)

        # Dataset Source
        DATASET_COUNT           = "general.dataset.count"
        DATASET_NAME            = "general.dataset.{id}.name"
        DATASET_AUTHOR          = "general.dataset.{id}.author"
        DATASET_VERSION         = "general.dataset.{id}.version"
        DATASET_ORGANIZATION    = "general.dataset.{id}.organization"
        DATASET_DESCRIPTION     = "general.dataset.{id}.description"
        DATASET_URL             = "general.dataset.{id}.url" # Model Website/Paper
        DATASET_DOI             = "general.dataset.{id}.doi"
        DATASET_UUID            = "general.dataset.{id}.uuid"
        DATASET_REPO_URL        = "general.dataset.{id}.repo_url" # Model Source Repository (git/svn/etc...)

        # Array based KV stores
        TAGS                       = "general.tags"
        LANGUAGES                  = "general.languages"

    class LLM:
        VOCAB_SIZE                        = "{arch}.vocab_size"
        CONTEXT_LENGTH                    = "{arch}.context_length"
        EMBEDDING_LENGTH                  = "{arch}.embedding_length"
        FEATURES_LENGTH                   = "{arch}.features_length"
        BLOCK_COUNT                       = "{arch}.block_count"
        LEADING_DENSE_BLOCK_COUNT         = "{arch}.leading_dense_block_count"
        FEED_FORWARD_LENGTH               = "{arch}.feed_forward_length"
        EXPERT_FEED_FORWARD_LENGTH        = "{arch}.expert_feed_forward_length"
        EXPERT_SHARED_FEED_FORWARD_LENGTH = "{arch}.expert_shared_feed_forward_length"
        EXPERT_CHUNK_FEED_FORWARD_LENGTH  = "{arch}.expert_chunk_feed_forward_length"
        USE_PARALLEL_RESIDUAL             = "{arch}.use_parallel_residual"
        TENSOR_DATA_LAYOUT                = "{arch}.tensor_data_layout"
        EXPERT_COUNT                      = "{arch}.expert_count"
        EXPERT_USED_COUNT                 = "{arch}.expert_used_count"
        EXPERT_SHARED_COUNT               = "{arch}.expert_shared_count"
        EXPERT_WEIGHTS_SCALE              = "{arch}.expert_weights_scale"
        EXPERT_WEIGHTS_NORM               = "{arch}.expert_weights_norm"
        EXPERT_GATING_FUNC                = "{arch}.expert_gating_func"
        EXPERT_GROUP_SCALE                = "{arch}.expert_group_scale"
        EXPERTS_PER_GROUP                 = "{arch}.experts_per_group"
        MOE_EVERY_N_LAYERS                = "{arch}.moe_every_n_layers"
        NEXTN_PREDICT_LAYERS              = "{arch}.nextn_predict_layers"
        POOLING_TYPE                      = "{arch}.pooling_type"
        LOGIT_SCALE                       = "{arch}.logit_scale"
        DECODER_START_TOKEN_ID            = "{arch}.decoder_start_token_id"
        DECODER_BLOCK_COUNT               = "{arch}.decoder_block_count"
        ATTN_LOGIT_SOFTCAPPING            = "{arch}.attn_logit_softcapping"
        ROUTER_LOGIT_SOFTCAPPING          = "{arch}.router_logit_softcapping"
        FINAL_LOGIT_SOFTCAPPING           = "{arch}.final_logit_softcapping"
        SWIN_NORM                         = "{arch}.swin_norm"
        RESCALE_EVERY_N_LAYERS            = "{arch}.rescale_every_n_layers"
        TIME_MIX_EXTRA_DIM                = "{arch}.time_mix_extra_dim"
        TIME_DECAY_EXTRA_DIM              = "{arch}.time_decay_extra_dim"
        RESIDUAL_SCALE                    = "{arch}.residual_scale"
        EMBEDDING_SCALE                   = "{arch}.embedding_scale"
        TOKEN_SHIFT_COUNT                 = "{arch}.token_shift_count"
        INTERLEAVE_MOE_LAYER_STEP         = "{arch}.interleave_moe_layer_step"
        ACTIVATION_SPARSITY_SCALE         = "{arch}.activation_sparsity_scale"
        ALTUP_ACTIVE_IDX                  = "{arch}.altup.active_idx"
        ALTUP_NUM_INPUTS                  = "{arch}.altup.num_inputs"
        EMBD_LENGTH_PER_LAYER_INP         = "{arch}.embedding_length_per_layer_input"
        DENSE_FEAT_IN_SIZE                = "{arch}.{dense}_feat_in"
        DENSE_FEAT_OUT_SIZE               = "{arch}.{dense}_feat_out"

    class Attention:
        HEAD_COUNT                   = "{arch}.attention.head_count"
        HEAD_COUNT_KV                = "{arch}.attention.head_count_kv"
        MAX_ALIBI_BIAS               = "{arch}.attention.max_alibi_bias"
        CLAMP_KQV                    = "{arch}.attention.clamp_kqv"
        KEY_LENGTH                   = "{arch}.attention.key_length"
        VALUE_LENGTH                 = "{arch}.attention.value_length"
        LAYERNORM_EPS                = "{arch}.attention.layer_norm_epsilon"
        LAYERNORM_RMS_EPS            = "{arch}.attention.layer_norm_rms_epsilon"
        GROUPNORM_EPS                = "{arch}.attention.group_norm_epsilon"
        GROUPNORM_GROUPS             = "{arch}.attention.group_norm_groups"
        CAUSAL                       = "{arch}.attention.causal"
        Q_LORA_RANK                  = "{arch}.attention.q_lora_rank"
        KV_LORA_RANK                 = "{arch}.attention.kv_lora_rank"
        DECAY_LORA_RANK              = "{arch}.attention.decay_lora_rank"
        ICLR_LORA_RANK               = "{arch}.attention.iclr_lora_rank"
        VALUE_RESIDUAL_MIX_LORA_RANK = "{arch}.attention.value_residual_mix_lora_rank"
        GATE_LORA_RANK               = "{arch}.attention.gate_lora_rank"
        REL_BUCKETS_COUNT            = "{arch}.attention.relative_buckets_count"
        SLIDING_WINDOW               = "{arch}.attention.sliding_window"
        SCALE                        = "{arch}.attention.scale"
        OUTPUT_SCALE                 = "{arch}.attention.output_scale"
        TEMPERATURE_LENGTH           = "{arch}.attention.temperature_length"
        KEY_LENGTH_MLA               = "{arch}.attention.key_length_mla"
        VALUE_LENGTH_MLA             = "{arch}.attention.value_length_mla"
        SHARED_KV_LAYERS             = "{arch}.attention.shared_kv_layers"
        SLIDING_WINDOW_PATTERN       = "{arch}.attention.sliding_window_pattern"

    class Rope:
        DIMENSION_COUNT          = "{arch}.rope.dimension_count"
        DIMENSION_SECTIONS       = "{arch}.rope.dimension_sections"
        FREQ_BASE                = "{arch}.rope.freq_base"
        SCALING_TYPE             = "{arch}.rope.scaling.type"
        SCALING_FACTOR           = "{arch}.rope.scaling.factor"
        SCALING_ATTN_FACTOR      = "{arch}.rope.scaling.attn_factor"
        SCALING_ORIG_CTX_LEN     = "{arch}.rope.scaling.original_context_length"
        SCALING_FINETUNED        = "{arch}.rope.scaling.finetuned"
        SCALING_YARN_LOG_MUL     = "{arch}.rope.scaling.yarn_log_multiplier"
        SCALING_YARN_EXT_FACTOR  = "{arch}.rope.scaling.yarn_ext_factor"
        SCALING_YARN_ATTN_FACTOR = "{arch}.rope.scaling.yarn_attn_factor"
        SCALING_YARN_BETA_FAST   = "{arch}.rope.scaling.yarn_beta_fast"
        SCALING_YARN_BETA_SLOW   = "{arch}.rope.scaling.yarn_beta_slow"

    class Split:
        LLM_KV_SPLIT_NO            = "split.no"
        LLM_KV_SPLIT_COUNT         = "split.count"
        LLM_KV_SPLIT_TENSORS_COUNT = "split.tensors.count"

    class SSM:
        CONV_KERNEL    = "{arch}.ssm.conv_kernel"
        INNER_SIZE     = "{arch}.ssm.inner_size"
        STATE_SIZE     = "{arch}.ssm.state_size"
        TIME_STEP_RANK = "{arch}.ssm.time_step_rank"
        GROUP_COUNT    = "{arch}.ssm.group_count"
        DT_B_C_RMS     = "{arch}.ssm.dt_b_c_rms"

    class WKV:
        HEAD_SIZE = "{arch}.wkv.head_size"

    class PosNet:
        EMBEDDING_LENGTH = "{arch}.posnet.embedding_length"
        BLOCK_COUNT      = "{arch}.posnet.block_count"

    class ConvNext:
        EMBEDDING_LENGTH = "{arch}.convnext.embedding_length"
        BLOCK_COUNT      = "{arch}.convnext.block_count"

    class Classifier:
        OUTPUT_LABELS = "{arch}.classifier.output_labels"

    class ShortConv:
        L_CACHE = "{arch}.shortconv.l_cache"

    class Tokenizer:
        MODEL                = "tokenizer.ggml.model"
        PRE                  = "tokenizer.ggml.pre"
        LIST                 = "tokenizer.ggml.tokens"
        TOKEN_TYPE           = "tokenizer.ggml.token_type"
        TOKEN_TYPE_COUNT     = "tokenizer.ggml.token_type_count"  # for BERT-style token types
        SCORES               = "tokenizer.ggml.scores"
        MERGES               = "tokenizer.ggml.merges"
        BOS_ID               = "tokenizer.ggml.bos_token_id"
        EOS_ID               = "tokenizer.ggml.eos_token_id"
        EOT_ID               = "tokenizer.ggml.eot_token_id"
        EOM_ID               = "tokenizer.ggml.eom_token_id"
        UNK_ID               = "tokenizer.ggml.unknown_token_id"
        SEP_ID               = "tokenizer.ggml.seperator_token_id"
        PAD_ID               = "tokenizer.ggml.padding_token_id"
        MASK_ID              = "tokenizer.ggml.mask_token_id"
        ADD_BOS              = "tokenizer.ggml.add_bos_token"
        ADD_EOS              = "tokenizer.ggml.add_eos_token"
        ADD_SEP              = "tokenizer.ggml.add_sep_token"
        ADD_PREFIX           = "tokenizer.ggml.add_space_prefix"
        REMOVE_EXTRA_WS      = "tokenizer.ggml.remove_extra_whitespaces"
        PRECOMPILED_CHARSMAP = "tokenizer.ggml.precompiled_charsmap"
        HF_JSON              = "tokenizer.huggingface.json"
        RWKV                 = "tokenizer.rwkv.world"
        CHAT_TEMPLATE        = "tokenizer.chat_template"
        CHAT_TEMPLATE_N      = "tokenizer.chat_template.{name}"
        CHAT_TEMPLATES       = "tokenizer.chat_templates"
        # FIM/Infill special tokens constants
        FIM_PRE_ID           = "tokenizer.ggml.fim_pre_token_id"
        FIM_SUF_ID           = "tokenizer.ggml.fim_suf_token_id"
        FIM_MID_ID           = "tokenizer.ggml.fim_mid_token_id"
        FIM_PAD_ID           = "tokenizer.ggml.fim_pad_token_id"
        FIM_REP_ID           = "tokenizer.ggml.fim_rep_token_id"
        FIM_SEP_ID           = "tokenizer.ggml.fim_sep_token_id"
        # deprecated:
        PREFIX_ID            = "tokenizer.ggml.prefix_token_id"
        SUFFIX_ID            = "tokenizer.ggml.suffix_token_id"
        MIDDLE_ID            = "tokenizer.ggml.middle_token_id"

    class Adapter:
        TYPE                    = "adapter.type"
        LORA_ALPHA              = "adapter.lora.alpha"
        LORA_TASK_NAME          = "adapter.lora.task_name"
        LORA_PROMPT_PREFIX      = "adapter.lora.prompt_prefix"
        ALORA_INVOCATION_TOKENS = "adapter.alora.invocation_tokens"

    class IMatrix:
        CHUNK_COUNT = "imatrix.chunk_count"
        CHUNK_SIZE  = "imatrix.chunk_size"
        DATASETS    = "imatrix.datasets"

    class Clip:
        PROJECTOR_TYPE      = "clip.projector_type"
        HAS_VISION_ENCODER  = "clip.has_vision_encoder"
        HAS_AUDIO_ENCODER   = "clip.has_audio_encoder"
        HAS_LLAVA_PROJECTOR = "clip.has_llava_projector"

    class ClipVision:
        IMAGE_SIZE          = "clip.vision.image_size"
        PREPROC_IMAGE_SIZE  = "clip.vision.preproc_image_size"
        PATCH_SIZE          = "clip.vision.patch_size"
        EMBEDDING_LENGTH    = "clip.vision.embedding_length"
        FEED_FORWARD_LENGTH = "clip.vision.feed_forward_length"
        PROJECTION_DIM      = "clip.vision.projection_dim"
        BLOCK_COUNT         = "clip.vision.block_count"
        IMAGE_MEAN          = "clip.vision.image_mean"
        IMAGE_STD           = "clip.vision.image_std"
        SPATIAL_MERGE_SIZE  = "clip.vision.spatial_merge_size"
        USE_GELU            = "clip.use_gelu"
        USE_SILU            = "clip.use_silu"
        N_WA_PATTERN        = "clip.vision.n_wa_pattern" # used by qwen2.5vl

        class Attention:
            HEAD_COUNT      = "clip.vision.attention.head_count"
            LAYERNORM_EPS   = "clip.vision.attention.layer_norm_epsilon"

        class Projector:
            SCALE_FACTOR    = "clip.vision.projector.scale_factor"

    class ClipAudio:
        NUM_MEL_BINS        = "clip.audio.num_mel_bins"
        EMBEDDING_LENGTH    = "clip.audio.embedding_length"
        FEED_FORWARD_LENGTH = "clip.audio.feed_forward_length"
        PROJECTION_DIM      = "clip.audio.projection_dim"
        BLOCK_COUNT         = "clip.audio.block_count"

        class Attention:
            HEAD_COUNT      = "clip.audio.attention.head_count"
            LAYERNORM_EPS   = "clip.audio.attention.layer_norm_epsilon"

        class Projector:
            STACK_FACTOR    = "clip.audio.projector.stack_factor"

    class Diffusion:
        SHIFT_LOGITS        = "diffusion.shift_logits"

    class xIELU:
        ALPHA_P             = "xielu.alpha_p"
        ALPHA_N             = "xielu.alpha_n"
        BETA                = "xielu.beta"
        EPS                 = "xielu.eps"


#
# recommended mapping of model tensor names for storage in gguf
#


class GGUFType:
    MODEL   = "model"
    ADAPTER = "adapter"
    IMATRIX = "imatrix"
    MMPROJ  = "mmproj" # dummy, unused for now


class MODEL_ARCH(IntEnum):
    MMPROJ           = auto() # dummy arch for clip.cpp
    LLAMA            = auto()
    LLAMA4           = auto()
    DECI             = auto()
    FALCON           = auto()
    FALCON_H1        = auto()
    BAICHUAN         = auto()
    GROK             = auto()
    GPT2             = auto()
    GPTJ             = auto()
    GPTNEOX          = auto()
    MPT              = auto()
    STARCODER        = auto()
    REFACT           = auto()
    BERT             = auto()
    NOMIC_BERT       = auto()
    NOMIC_BERT_MOE   = auto()
    NEO_BERT         = auto()
    JINA_BERT_V2     = auto()
    JINA_BERT_V3     = auto()
    BLOOM            = auto()
    STABLELM         = auto()
    QWEN             = auto()
    QWEN2            = auto()
    QWEN2MOE         = auto()
    QWEN2VL          = auto()
    QWEN3            = auto()
    QWEN3MOE         = auto()
    PHI2             = auto()
    PHI3             = auto()
    PHIMOE           = auto()
    PLAMO            = auto()
    PLAMO2           = auto()
    CODESHELL        = auto()
    ORION            = auto()
    INTERNLM2        = auto()
    MINICPM          = auto()
    MINICPM3         = auto()
    GEMMA            = auto()
    GEMMA2           = auto()
    GEMMA3           = auto()
    GEMMA3N          = auto()
    GEMMA_EMBEDDING  = auto()
    STARCODER2       = auto()
    RWKV6            = auto()
    RWKV6QWEN2       = auto()
    RWKV7            = auto()
    ARWKV7           = auto()
    MAMBA            = auto()
    MAMBA2           = auto()
    JAMBA            = auto()
    XVERSE           = auto()
    COMMAND_R        = auto()
    COHERE2          = auto()
    DBRX             = auto()
    OLMO             = auto()
    OLMO2            = auto()
    OLMOE            = auto()
    OPENELM          = auto()
    ARCTIC           = auto()
    DEEPSEEK         = auto()
    DEEPSEEK2        = auto()
    CHATGLM          = auto()
    GLM4             = auto()
    GLM4_MOE         = auto()
    BITNET           = auto()
    T5               = auto()
    T5ENCODER        = auto()
    JAIS             = auto()
    NEMOTRON         = auto()
    NEMOTRON_H       = auto()
    EXAONE           = auto()
    EXAONE4          = auto()
    GRANITE          = auto()
    GRANITE_MOE      = auto()
    GRANITE_HYBRID   = auto()
    CHAMELEON        = auto()
    WAVTOKENIZER_DEC = auto()
    PLM              = auto()
    BAILINGMOE       = auto()
    DOTS1            = auto()
    ARCEE            = auto()
    ERNIE4_5         = auto()
    ERNIE4_5_MOE     = auto()
    HUNYUAN_MOE      = auto()
    HUNYUAN_DENSE    = auto()
    SMOLLM3          = auto()
    GPT_OSS          = auto()
    LFM2             = auto()
    LFM2MOE          = auto()
    DREAM            = auto()
    SMALLTHINKER     = auto()
    LLADA            = auto()
    LLADA_MOE        = auto()
    SEED_OSS         = auto()
    GROVEMOE         = auto()
    APERTUS          = auto()


class VISION_PROJECTOR_TYPE(IntEnum):
    MLP       = auto()
    LDP       = auto()
    LDPV2     = auto()
    RESAMPLER = auto()
    GLM_EDGE  = auto()
    MERGER    = auto()
    GEMMA3    = auto()


class MODEL_TENSOR(IntEnum):
    TOKEN_EMBD           = auto()
    TOKEN_EMBD_NORM      = auto()
    TOKEN_TYPES          = auto()
    POS_EMBD             = auto()
    OUTPUT               = auto()
    DENSE_2_OUT          = auto() # embeddinggemma 2_Dense
    DENSE_3_OUT          = auto() # embeddinggemma 3_Dense
    OUTPUT_NORM          = auto()
    ROPE_FREQS           = auto()
    ROPE_FACTORS_LONG    = auto()
    ROPE_FACTORS_SHORT   = auto()
    ATTN_Q               = auto()
    ATTN_K               = auto()
    ATTN_V               = auto()
    ATTN_QKV             = auto()
    ATTN_OUT             = auto()
    ATTN_NORM            = auto()
    ATTN_NORM_2          = auto()
    ATTN_OUT_NORM        = auto()
    ATTN_POST_NORM       = auto()
    ATTN_ROT_EMBD        = auto()
    ATTN_SINKS           = auto()
    FFN_GATE_INP         = auto()
    FFN_GATE_INP_SHEXP   = auto()
    FFN_NORM             = auto()
    FFN_PRE_NORM         = auto()
    FFN_POST_NORM        = auto()
    FFN_GATE             = auto()
    FFN_DOWN             = auto()
    FFN_UP               = auto()
    FFN_ACT              = auto()
    FFN_NORM_EXP         = auto()
    FFN_GATE_EXP         = auto()
    FFN_DOWN_EXP         = auto()
    FFN_UP_EXP           = auto()
    FFN_GATE_SHEXP       = auto()
    FFN_DOWN_SHEXP       = auto()
    FFN_UP_SHEXP         = auto()
    FFN_GATE_CHEXP       = auto()
    FFN_DOWN_CHEXP       = auto()
    FFN_UP_CHEXP         = auto()
    FFN_EXP_PROBS_B      = auto()
    ATTN_Q_NORM          = auto()
    ATTN_K_NORM          = auto()
    LAYER_OUT_NORM       = auto()
    PER_LAYER_TOKEN_EMBD = auto() # gemma3n
    PER_LAYER_MODEL_PROJ = auto() # gemma3n
    PER_LAYER_INP_GATE   = auto() # gemma3n
    PER_LAYER_PROJ       = auto() # gemma3n
    PER_LAYER_PROJ_NORM  = auto() # gemma3n
    PER_LAYER_POST_NORM  = auto() # gemma3n
    ALTUP_PROJ           = auto() # gemma3n
    ALTUP_UNEMBD_PROJ    = auto() # gemma3n
    ALTUP_CORRECT_COEF   = auto() # gemma3n
    ALTUP_CORRECT_SCALE  = auto() # gemma3n
    ALTUP_PREDICT_COEF   = auto() # gemma3n
    ALTUP_ROUTER         = auto() # gemma3n
    ALTUP_ROUTER_NORM    = auto() # gemma3n
    LAUREL_L             = auto() # gemma3n
    LAUREL_R             = auto() # gemma3n
    LAUREL_POST_NORM     = auto() # gemma3n
    SSM_IN               = auto()
    SSM_CONV1D           = auto()
    SSM_X                = auto()
    SSM_DT               = auto()
    SSM_DT_NORM          = auto()
    SSM_A                = auto()
    SSM_B_NORM           = auto()
    SSM_C_NORM           = auto()
    SSM_D                = auto()
    SSM_NORM             = auto()
    SSM_OUT              = auto()
    TIME_MIX_W0          = auto()
    TIME_MIX_W1          = auto()
    TIME_MIX_W2          = auto()
    TIME_MIX_A0          = auto()
    TIME_MIX_A1          = auto()
    TIME_MIX_A2          = auto()
    TIME_MIX_V0          = auto()
    TIME_MIX_V1          = auto()
    TIME_MIX_V2          = auto()
    TIME_MIX_G1          = auto()
    TIME_MIX_G2          = auto()
    TIME_MIX_K_K         = auto()
    TIME_MIX_K_A         = auto()
    TIME_MIX_R_K         = auto()
    TIME_MIX_LERP_X      = auto()
    TIME_MIX_LERP_K      = auto()
    TIME_MIX_LERP_V      = auto()
    TIME_MIX_LERP_R      = auto()
    TIME_MIX_LERP_G      = auto()
    TIME_MIX_LERP_FUSED  = auto()
    TIME_MIX_LERP_W      = auto()
    TIME_MIX_FIRST       = auto()
    TIME_MIX_DECAY       = auto()
    TIME_MIX_DECAY_W1    = auto()
    TIME_MIX_DECAY_W2    = auto()
    TIME_MIX_KEY         = auto()
    TIME_MIX_VALUE       = auto()
    TIME_MIX_RECEPTANCE  = auto()
    TIME_MIX_GATE        = auto()
    TIME_MIX_LN          = auto()
    TIME_MIX_OUTPUT      = auto()
    CHANNEL_MIX_LERP_K   = auto()
    CHANNEL_MIX_LERP_R   = auto()
    CHANNEL_MIX_KEY      = auto()
    CHANNEL_MIX_RECEPTANCE = auto()
    CHANNEL_MIX_VALUE    = auto()
    ATTN_Q_A             = auto()
    ATTN_Q_B             = auto()
    ATTN_KV_A_MQA        = auto()
    ATTN_KV_B            = auto()
    ATTN_K_B             = auto()
    ATTN_V_B             = auto()
    ATTN_Q_A_NORM        = auto()
    ATTN_KV_A_NORM       = auto()
    FFN_SUB_NORM         = auto()
    ATTN_SUB_NORM        = auto()
    DEC_ATTN_NORM        = auto()
    DEC_ATTN_Q           = auto()
    DEC_ATTN_K           = auto()
    DEC_ATTN_V           = auto()
    DEC_ATTN_OUT         = auto()
    DEC_ATTN_REL_B       = auto()
    DEC_CROSS_ATTN_NORM  = auto()
    DEC_CROSS_ATTN_Q     = auto()
    DEC_CROSS_ATTN_K     = auto()
    DEC_CROSS_ATTN_V     = auto()
    DEC_CROSS_ATTN_OUT   = auto()
    DEC_CROSS_ATTN_REL_B = auto()
    DEC_FFN_NORM         = auto()
    DEC_FFN_GATE         = auto()
    DEC_FFN_DOWN         = auto()
    DEC_FFN_UP           = auto()
    DEC_OUTPUT_NORM      = auto()
    ENC_ATTN_NORM        = auto()
    ENC_ATTN_Q           = auto()
    ENC_ATTN_K           = auto()
    ENC_ATTN_V           = auto()
    ENC_ATTN_OUT         = auto()
    ENC_ATTN_REL_B       = auto()
    ENC_FFN_NORM         = auto()
    ENC_FFN_GATE         = auto()
    ENC_FFN_DOWN         = auto()
    ENC_FFN_UP           = auto()
    ENC_OUTPUT_NORM      = auto()
    CLS                  = auto() # classifier
    CLS_OUT              = auto() # classifier output projection
    CONV1D               = auto()
    CONVNEXT_DW          = auto()
    CONVNEXT_NORM        = auto()
    CONVNEXT_PW1         = auto()
    CONVNEXT_PW2         = auto()
    CONVNEXT_GAMMA       = auto()
    POSNET_CONV1         = auto()
    POSNET_CONV2         = auto()
    POSNET_NORM          = auto()
    POSNET_NORM1         = auto()
    POSNET_NORM2         = auto()
    POSNET_ATTN_NORM     = auto()
    POSNET_ATTN_Q        = auto()
    POSNET_ATTN_K        = auto()
    POSNET_ATTN_V        = auto()
    POSNET_ATTN_OUT      = auto()
    SHORTCONV_CONV       = auto()
    SHORTCONV_INPROJ     = auto()
    SHORTCONV_OUTPROJ    = auto()
    # vision
    V_MMPROJ             = auto()
    V_MMPROJ_FC          = auto()
    V_MMPROJ_MLP         = auto()
    V_MMPROJ_PEG         = auto()
    V_ENC_EMBD_CLS       = auto()
    V_ENC_EMBD_PATCH     = auto()
    V_ENC_EMBD_POS       = auto()
    V_ENC_INPUT_NORM     = auto()
    V_ENC_ATTN_Q         = auto()
    V_ENC_ATTN_Q_NORM    = auto()
    V_ENC_ATTN_K         = auto()
    V_ENC_ATTN_K_NORM    = auto()
    V_ENC_ATTN_V         = auto()
    V_ENC_ATTN_O         = auto()
    V_ENC_ATTN_O_NORM    = auto()
    V_ENC_POST_ATTN_NORM = auto()
    V_ENC_FFN_UP         = auto()
    V_ENC_FFN_GATE       = auto()
    V_ENC_FFN_DOWN       = auto()
    V_LAYER_SCALE_1      = auto()
    V_LAYER_SCALE_2      = auto()
    V_PRE_NORM           = auto()
    V_POST_NORM          = auto()
    V_MM_INP_NORM        = auto()
    V_MM_INP_PROJ        = auto() # gemma3
    V_MM_SOFT_EMB_NORM   = auto() # gemma3
    V_RESMPL_POS_EMBD_K  = auto() # minicpmv
    V_RESMPL_ATTN_Q      = auto() # minicpmv
    V_RESMPL_ATTN_K      = auto() # minicpmv
    V_RESMPL_ATTN_V      = auto() # minicpmv
    V_RESMPL_ATTN_OUT    = auto() # minicpmv
    V_RESMPL_KV          = auto() # minicpmv
    V_RESMPL_KV_NORM     = auto() # minicpmv
    V_RESMPL_POST_NORM   = auto() # minicpmv
    V_RESMPL_Q_NORM      = auto() # minicpmv
    V_RESMPL_PROJ        = auto() # minicpmv
    V_RESMPL_QUERY       = auto() # minicpmv
    V_TOK_EMBD_IMG_BREAK = auto() # pixtral
    V_MM_PATCH_MERGER    = auto() # mistral small 3.1
    # audio (mtmd)
    A_ENC_EMBD_POS       = auto()
    A_ENC_CONV1D         = auto()
    A_PRE_NORM           = auto()
    A_POST_NORM          = auto()
    A_ENC_ATTN_Q         = auto()
    A_ENC_ATTN_K         = auto()
    A_ENC_ATTN_V         = auto()
    A_ENC_INPUT_NORM     = auto()
    A_ENC_OUTPUT         = auto()
    A_ENC_OUTPUT_NORM    = auto()
    A_ENC_FFN_UP         = auto()
    A_ENC_FFN_GATE       = auto()
    A_ENC_FFN_DOWN       = auto()
    A_MMPROJ             = auto()
    A_MMPROJ_FC          = auto()
    A_MM_NORM_PRE        = auto()
    A_MM_NORM_MID        = auto()
    # nextn/mtp
    NEXTN_EH_PROJ        = auto()
    NEXTN_EMBED_TOKENS   = auto()
    NEXTN_ENORM          = auto()
    NEXTN_HNORM          = auto()
    NEXTN_SHARED_HEAD_HEAD = auto()
    NEXTN_SHARED_HEAD_NORM = auto()


MODEL_ARCH_NAMES: dict[MODEL_ARCH, str] = {
    MODEL_ARCH.MMPROJ:           "clip", # dummy arch for clip.cpp
    MODEL_ARCH.LLAMA:            "llama",
    MODEL_ARCH.LLAMA4:           "llama4",
    MODEL_ARCH.DECI:             "deci",
    MODEL_ARCH.FALCON:           "falcon",
    MODEL_ARCH.BAICHUAN:         "baichuan",
    MODEL_ARCH.GROK:             "grok",
    MODEL_ARCH.GPT2:             "gpt2",
    MODEL_ARCH.GPTJ:             "gptj",
    MODEL_ARCH.GPTNEOX:          "gptneox",
    MODEL_ARCH.MPT:              "mpt",
    MODEL_ARCH.STARCODER:        "starcoder",
    MODEL_ARCH.REFACT:           "refact",
    MODEL_ARCH.BERT:             "bert",
    MODEL_ARCH.NOMIC_BERT:       "nomic-bert",
    MODEL_ARCH.NOMIC_BERT_MOE:   "nomic-bert-moe",
    MODEL_ARCH.NEO_BERT:         "neo-bert",
    MODEL_ARCH.JINA_BERT_V2:     "jina-bert-v2",
    MODEL_ARCH.JINA_BERT_V3:     "jina-bert-v3",
    MODEL_ARCH.BLOOM:            "bloom",
    MODEL_ARCH.STABLELM:         "stablelm",
    MODEL_ARCH.QWEN:             "qwen",
    MODEL_ARCH.QWEN2:            "qwen2",
    MODEL_ARCH.QWEN2MOE:         "qwen2moe",
    MODEL_ARCH.QWEN2VL:          "qwen2vl",
    MODEL_ARCH.QWEN3:            "qwen3",
    MODEL_ARCH.QWEN3MOE:         "qwen3moe",
    MODEL_ARCH.PHI2:             "phi2",
    MODEL_ARCH.PHI3:             "phi3",
    MODEL_ARCH.PHIMOE:           "phimoe",
    MODEL_ARCH.PLAMO:            "plamo",
    MODEL_ARCH.PLAMO2:           "plamo2",
    MODEL_ARCH.CODESHELL:        "codeshell",
    MODEL_ARCH.ORION:            "orion",
    MODEL_ARCH.INTERNLM2:        "internlm2",
    MODEL_ARCH.MINICPM:          "minicpm",
    MODEL_ARCH.MINICPM3:         "minicpm3",
    MODEL_ARCH.GEMMA:            "gemma",
    MODEL_ARCH.GEMMA2:           "gemma2",
    MODEL_ARCH.GEMMA3:           "gemma3",
    MODEL_ARCH.GEMMA3N:          "gemma3n",
    MODEL_ARCH.GEMMA_EMBEDDING:  "gemma-embedding",
    MODEL_ARCH.STARCODER2:       "starcoder2",
    MODEL_ARCH.RWKV6:            "rwkv6",
    MODEL_ARCH.RWKV6QWEN2:       "rwkv6qwen2",
    MODEL_ARCH.RWKV7:            "rwkv7",
    MODEL_ARCH.ARWKV7:           "arwkv7",
    MODEL_ARCH.MAMBA:            "mamba",
    MODEL_ARCH.MAMBA2:           "mamba2",
    MODEL_ARCH.JAMBA:            "jamba",
    MODEL_ARCH.XVERSE:           "xverse",
    MODEL_ARCH.COMMAND_R:        "command-r",
    MODEL_ARCH.COHERE2:          "cohere2",
    MODEL_ARCH.DBRX:             "dbrx",
    MODEL_ARCH.OLMO:             "olmo",
    MODEL_ARCH.OLMO2:            "olmo2",
    MODEL_ARCH.OLMOE:            "olmoe",
    MODEL_ARCH.OPENELM:          "openelm",
    MODEL_ARCH.ARCTIC:           "arctic",
    MODEL_ARCH.DEEPSEEK:         "deepseek",
    MODEL_ARCH.DEEPSEEK2:        "deepseek2",
    MODEL_ARCH.CHATGLM:          "chatglm",
    MODEL_ARCH.GLM4:             "glm4",
    MODEL_ARCH.GLM4_MOE:         "glm4moe",
    MODEL_ARCH.BITNET:           "bitnet",
    MODEL_ARCH.T5:               "t5",
    MODEL_ARCH.T5ENCODER:        "t5encoder",
    MODEL_ARCH.JAIS:             "jais",
    MODEL_ARCH.NEMOTRON:         "nemotron",
    MODEL_ARCH.NEMOTRON_H:       "nemotron_h",
    MODEL_ARCH.EXAONE:           "exaone",
    MODEL_ARCH.EXAONE4:          "exaone4",
    MODEL_ARCH.GRANITE:          "granite",
    MODEL_ARCH.GRANITE_MOE:      "granitemoe",
    MODEL_ARCH.GRANITE_HYBRID:   "granitehybrid",
    MODEL_ARCH.CHAMELEON:        "chameleon",
    MODEL_ARCH.WAVTOKENIZER_DEC: "wavtokenizer-dec",
    MODEL_ARCH.PLM:              "plm",
    MODEL_ARCH.BAILINGMOE:       "bailingmoe",
    MODEL_ARCH.DOTS1:            "dots1",
    MODEL_ARCH.ARCEE:            "arcee",
    MODEL_ARCH.ERNIE4_5:         "ernie4_5",
    MODEL_ARCH.ERNIE4_5_MOE:     "ernie4_5-moe",
    MODEL_ARCH.FALCON_H1:        "falcon-h1",
    MODEL_ARCH.HUNYUAN_MOE:      "hunyuan-moe",
    MODEL_ARCH.HUNYUAN_DENSE:    "hunyuan-dense",
    MODEL_ARCH.SMOLLM3:          "smollm3",
    MODEL_ARCH.GPT_OSS:          "gpt-oss",
    MODEL_ARCH.LFM2:             "lfm2",
    MODEL_ARCH.LFM2MOE:          "lfm2moe",
    MODEL_ARCH.DREAM:            "dream",
    MODEL_ARCH.SMALLTHINKER:     "smallthinker",
    MODEL_ARCH.LLADA:            "llada",
    MODEL_ARCH.LLADA_MOE:        "llada-moe",
    MODEL_ARCH.SEED_OSS:         "seed_oss",
    MODEL_ARCH.GROVEMOE:         "grovemoe",
    MODEL_ARCH.APERTUS:          "apertus",
}

VISION_PROJECTOR_TYPE_NAMES: dict[VISION_PROJECTOR_TYPE, str] = {
    VISION_PROJECTOR_TYPE.MLP:       "mlp",
    VISION_PROJECTOR_TYPE.LDP:       "ldp",
    VISION_PROJECTOR_TYPE.LDPV2:     "ldpv2",
    VISION_PROJECTOR_TYPE.RESAMPLER: "resampler",
    VISION_PROJECTOR_TYPE.GLM_EDGE:  "adapter",
    VISION_PROJECTOR_TYPE.MERGER:    "qwen2vl_merger",
    VISION_PROJECTOR_TYPE.GEMMA3:    "gemma3",
}

TENSOR_NAMES: dict[MODEL_TENSOR, str] = {
    MODEL_TENSOR.TOKEN_EMBD:                "token_embd",
    MODEL_TENSOR.TOKEN_EMBD_NORM:           "token_embd_norm",
    MODEL_TENSOR.TOKEN_TYPES:               "token_types",
    MODEL_TENSOR.POS_EMBD:                  "position_embd",
    MODEL_TENSOR.OUTPUT_NORM:               "output_norm",
    MODEL_TENSOR.OUTPUT:                    "output",
    MODEL_TENSOR.DENSE_2_OUT:                "dense_2", # embeddinggemma 2_Dense
    MODEL_TENSOR.DENSE_3_OUT:                "dense_3", # embeddinggemma 2_Dense
    MODEL_TENSOR.ROPE_FREQS:                "rope_freqs",
    MODEL_TENSOR.ROPE_FACTORS_LONG:         "rope_factors_long",
    MODEL_TENSOR.ROPE_FACTORS_SHORT:        "rope_factors_short",
    MODEL_TENSOR.ATTN_NORM:                 "blk.{bid}.attn_norm",
    MODEL_TENSOR.ATTN_NORM_2:               "blk.{bid}.attn_norm_2",
    MODEL_TENSOR.ATTN_QKV:                  "blk.{bid}.attn_qkv",
    MODEL_TENSOR.ATTN_Q:                    "blk.{bid}.attn_q",
    MODEL_TENSOR.ATTN_K:                    "blk.{bid}.attn_k",
    MODEL_TENSOR.ATTN_V:                    "blk.{bid}.attn_v",
    MODEL_TENSOR.ATTN_OUT:                  "blk.{bid}.attn_output",
    MODEL_TENSOR.ATTN_ROT_EMBD:             "blk.{bid}.attn_rot_embd",
    MODEL_TENSOR.ATTN_SINKS:                "blk.{bid}.attn_sinks",
    MODEL_TENSOR.ATTN_Q_NORM:               "blk.{bid}.attn_q_norm",
    MODEL_TENSOR.ATTN_K_NORM:               "blk.{bid}.attn_k_norm",
    MODEL_TENSOR.ATTN_OUT_NORM:             "blk.{bid}.attn_output_norm",
    MODEL_TENSOR.ATTN_POST_NORM:            "blk.{bid}.post_attention_norm",
    MODEL_TENSOR.FFN_GATE_INP:              "blk.{bid}.ffn_gate_inp",
    MODEL_TENSOR.FFN_GATE_INP_SHEXP:        "blk.{bid}.ffn_gate_inp_shexp",
    MODEL_TENSOR.FFN_NORM:                  "blk.{bid}.ffn_norm",
    MODEL_TENSOR.FFN_PRE_NORM:              "blk.{bid}.ffn_norm",
    MODEL_TENSOR.FFN_POST_NORM:             "blk.{bid}.post_ffw_norm",
    MODEL_TENSOR.FFN_GATE:                  "blk.{bid}.ffn_gate",
    MODEL_TENSOR.FFN_DOWN:                  "blk.{bid}.ffn_down",
    MODEL_TENSOR.FFN_UP:                    "blk.{bid}.ffn_up",
    MODEL_TENSOR.FFN_GATE_SHEXP:            "blk.{bid}.ffn_gate_shexp",
    MODEL_TENSOR.FFN_DOWN_SHEXP:            "blk.{bid}.ffn_down_shexp",
    MODEL_TENSOR.FFN_UP_SHEXP:              "blk.{bid}.ffn_up_shexp",
    MODEL_TENSOR.FFN_GATE_CHEXP:            "blk.{bid}.ffn_gate_chexps",
    MODEL_TENSOR.FFN_DOWN_CHEXP:            "blk.{bid}.ffn_down_chexps",
    MODEL_TENSOR.FFN_UP_CHEXP:              "blk.{bid}.ffn_up_chexps",
    MODEL_TENSOR.FFN_ACT:                   "blk.{bid}.ffn",
    MODEL_TENSOR.FFN_NORM_EXP:              "blk.{bid}.ffn_norm_exps",
    MODEL_TENSOR.FFN_GATE_EXP:              "blk.{bid}.ffn_gate_exps",
    MODEL_TENSOR.FFN_DOWN_EXP:              "blk.{bid}.ffn_down_exps",
    MODEL_TENSOR.FFN_UP_EXP:                "blk.{bid}.ffn_up_exps",
    MODEL_TENSOR.FFN_EXP_PROBS_B:           "blk.{bid}.exp_probs_b",
    MODEL_TENSOR.LAYER_OUT_NORM:            "blk.{bid}.layer_output_norm",
    MODEL_TENSOR.PER_LAYER_TOKEN_EMBD:      "per_layer_token_embd",           # gemma3n
    MODEL_TENSOR.PER_LAYER_MODEL_PROJ:      "per_layer_model_proj",           # gemma3n
    MODEL_TENSOR.PER_LAYER_PROJ_NORM:       "per_layer_proj_norm",            # gemma3n
    MODEL_TENSOR.ALTUP_UNEMBD_PROJ:         "altup_unembd_proj",              # gemma3n
    MODEL_TENSOR.ALTUP_PROJ:                "altup_proj",                     # gemma3n
    MODEL_TENSOR.PER_LAYER_INP_GATE:        "blk.{bid}.inp_gate",             # gemma3n
    MODEL_TENSOR.PER_LAYER_PROJ:            "blk.{bid}.proj",                 # gemma3n
    MODEL_TENSOR.PER_LAYER_POST_NORM:       "blk.{bid}.post_norm",            # gemma3n
    MODEL_TENSOR.ALTUP_CORRECT_COEF:        "blk.{bid}.altup_correct_coef",   # gemma3n
    MODEL_TENSOR.ALTUP_CORRECT_SCALE:       "blk.{bid}.altup_correct_scale",  # gemma3n
    MODEL_TENSOR.ALTUP_PREDICT_COEF:        "blk.{bid}.altup_predict_coef",   # gemma3n
    MODEL_TENSOR.ALTUP_ROUTER:              "blk.{bid}.altup_router",         # gemma3n
    MODEL_TENSOR.ALTUP_ROUTER_NORM:         "blk.{bid}.altup_router_norm",    # gemma3n
    MODEL_TENSOR.LAUREL_L:                  "blk.{bid}.laurel_l",             # gemma3n
    MODEL_TENSOR.LAUREL_R:                  "blk.{bid}.laurel_r",             # gemma3n
    MODEL_TENSOR.LAUREL_POST_NORM:          "blk.{bid}.laurel_post_norm",     # gemma3n
    MODEL_TENSOR.SSM_IN:                    "blk.{bid}.ssm_in",
    MODEL_TENSOR.SSM_CONV1D:                "blk.{bid}.ssm_conv1d",
    MODEL_TENSOR.SSM_X:                     "blk.{bid}.ssm_x",
    MODEL_TENSOR.SSM_DT:                    "blk.{bid}.ssm_dt",
    MODEL_TENSOR.SSM_DT_NORM:               "blk.{bid}.ssm_dt_norm",
    MODEL_TENSOR.SSM_A:                     "blk.{bid}.ssm_a",
    MODEL_TENSOR.SSM_B_NORM:                "blk.{bid}.ssm_b_norm",
    MODEL_TENSOR.SSM_C_NORM:                "blk.{bid}.ssm_c_norm",
    MODEL_TENSOR.SSM_D:                     "blk.{bid}.ssm_d",
    MODEL_TENSOR.SSM_NORM:                  "blk.{bid}.ssm_norm",
    MODEL_TENSOR.SSM_OUT:                   "blk.{bid}.ssm_out",
    MODEL_TENSOR.TIME_MIX_W0:               "blk.{bid}.time_mix_w0",
    MODEL_TENSOR.TIME_MIX_W1:               "blk.{bid}.time_mix_w1",
    MODEL_TENSOR.TIME_MIX_W2:               "blk.{bid}.time_mix_w2",
    MODEL_TENSOR.TIME_MIX_A0:               "blk.{bid}.time_mix_a0",
    MODEL_TENSOR.TIME_MIX_A1:               "blk.{bid}.time_mix_a1",
    MODEL_TENSOR.TIME_MIX_A2:               "blk.{bid}.time_mix_a2",
    MODEL_TENSOR.TIME_MIX_V0:               "blk.{bid}.time_mix_v0",
    MODEL_TENSOR.TIME_MIX_V1:               "blk.{bid}.time_mix_v1",
    MODEL_TENSOR.TIME_MIX_V2:               "blk.{bid}.time_mix_v2",
    MODEL_TENSOR.TIME_MIX_G1:               "blk.{bid}.time_mix_g1",
    MODEL_TENSOR.TIME_MIX_G2:               "blk.{bid}.time_mix_g2",
    MODEL_TENSOR.TIME_MIX_K_K:              "blk.{bid}.time_mix_k_k",
    MODEL_TENSOR.TIME_MIX_K_A:              "blk.{bid}.time_mix_k_a",
    MODEL_TENSOR.TIME_MIX_R_K:              "blk.{bid}.time_mix_r_k",
    MODEL_TENSOR.TIME_MIX_LERP_X:           "blk.{bid}.time_mix_lerp_x",
    MODEL_TENSOR.TIME_MIX_LERP_K:           "blk.{bid}.time_mix_lerp_k",
    MODEL_TENSOR.TIME_MIX_LERP_V:           "blk.{bid}.time_mix_lerp_v",
    MODEL_TENSOR.TIME_MIX_LERP_R:           "blk.{bid}.time_mix_lerp_r",
    MODEL_TENSOR.TIME_MIX_LERP_G:           "blk.{bid}.time_mix_lerp_g",
    MODEL_TENSOR.TIME_MIX_LERP_FUSED:       "blk.{bid}.time_mix_lerp_fused",
    MODEL_TENSOR.TIME_MIX_LERP_W:           "blk.{bid}.time_mix_lerp_w",
    MODEL_TENSOR.TIME_MIX_FIRST:            "blk.{bid}.time_mix_first",
    MODEL_TENSOR.TIME_MIX_DECAY:            "blk.{bid}.time_mix_decay",
    MODEL_TENSOR.TIME_MIX_DECAY_W1:         "blk.{bid}.time_mix_decay_w1",
    MODEL_TENSOR.TIME_MIX_DECAY_W2:         "blk.{bid}.time_mix_decay_w2",
    MODEL_TENSOR.TIME_MIX_KEY:              "blk.{bid}.time_mix_key",
    MODEL_TENSOR.TIME_MIX_VALUE:            "blk.{bid}.time_mix_value",
    MODEL_TENSOR.TIME_MIX_RECEPTANCE:       "blk.{bid}.time_mix_receptance",
    MODEL_TENSOR.TIME_MIX_GATE:             "blk.{bid}.time_mix_gate",
    MODEL_TENSOR.TIME_MIX_LN:               "blk.{bid}.time_mix_ln",
    MODEL_TENSOR.TIME_MIX_OUTPUT:           "blk.{bid}.time_mix_output",
    MODEL_TENSOR.CHANNEL_MIX_LERP_K:        "blk.{bid}.channel_mix_lerp_k",
    MODEL_TENSOR.CHANNEL_MIX_LERP_R:        "blk.{bid}.channel_mix_lerp_r",
    MODEL_TENSOR.CHANNEL_MIX_KEY:           "blk.{bid}.channel_mix_key",
    MODEL_TENSOR.CHANNEL_MIX_RECEPTANCE:    "blk.{bid}.channel_mix_receptance",
    MODEL_TENSOR.CHANNEL_MIX_VALUE:         "blk.{bid}.channel_mix_value",
    MODEL_TENSOR.ATTN_Q_A:                  "blk.{bid}.attn_q_a",
    MODEL_TENSOR.ATTN_Q_B:                  "blk.{bid}.attn_q_b",
    MODEL_TENSOR.ATTN_KV_A_MQA:             "blk.{bid}.attn_kv_a_mqa",
    MODEL_TENSOR.ATTN_KV_B:                 "blk.{bid}.attn_kv_b",
    MODEL_TENSOR.ATTN_K_B:                  "blk.{bid}.attn_k_b",
    MODEL_TENSOR.ATTN_V_B:                  "blk.{bid}.attn_v_b",
    MODEL_TENSOR.ATTN_Q_A_NORM:             "blk.{bid}.attn_q_a_norm",
    MODEL_TENSOR.ATTN_KV_A_NORM:            "blk.{bid}.attn_kv_a_norm",
    MODEL_TENSOR.ATTN_SUB_NORM:             "blk.{bid}.attn_sub_norm",
    MODEL_TENSOR.FFN_SUB_NORM:              "blk.{bid}.ffn_sub_norm",
    MODEL_TENSOR.DEC_ATTN_NORM:             "dec.blk.{bid}.attn_norm",
    MODEL_TENSOR.DEC_ATTN_Q:                "dec.blk.{bid}.attn_q",
    MODEL_TENSOR.DEC_ATTN_K:                "dec.blk.{bid}.attn_k",
    MODEL_TENSOR.DEC_ATTN_V:                "dec.blk.{bid}.attn_v",
    MODEL_TENSOR.DEC_ATTN_OUT:              "dec.blk.{bid}.attn_o",
    MODEL_TENSOR.DEC_ATTN_REL_B:            "dec.blk.{bid}.attn_rel_b",
    MODEL_TENSOR.DEC_CROSS_ATTN_NORM:       "dec.blk.{bid}.cross_attn_norm",
    MODEL_TENSOR.DEC_CROSS_ATTN_Q:          "dec.blk.{bid}.cross_attn_q",
    MODEL_TENSOR.DEC_CROSS_ATTN_K:          "dec.blk.{bid}.cross_attn_k",
    MODEL_TENSOR.DEC_CROSS_ATTN_V:          "dec.blk.{bid}.cross_attn_v",
    MODEL_TENSOR.DEC_CROSS_ATTN_OUT:        "dec.blk.{bid}.cross_attn_o",
    MODEL_TENSOR.DEC_CROSS_ATTN_REL_B:      "dec.blk.{bid}.cross_attn_rel_b",
    MODEL_TENSOR.DEC_FFN_NORM:              "dec.blk.{bid}.ffn_norm",
    MODEL_TENSOR.DEC_FFN_GATE:              "dec.blk.{bid}.ffn_gate",
    MODEL_TENSOR.DEC_FFN_DOWN:              "dec.blk.{bid}.ffn_down",
    MODEL_TENSOR.DEC_FFN_UP:                "dec.blk.{bid}.ffn_up",
    MODEL_TENSOR.DEC_OUTPUT_NORM:           "dec.output_norm",
    MODEL_TENSOR.ENC_ATTN_NORM:             "enc.blk.{bid}.attn_norm",
    MODEL_TENSOR.ENC_ATTN_Q:                "enc.blk.{bid}.attn_q",
    MODEL_TENSOR.ENC_ATTN_K:                "enc.blk.{bid}.attn_k",
    MODEL_TENSOR.ENC_ATTN_V:                "enc.blk.{bid}.attn_v",
    MODEL_TENSOR.ENC_ATTN_OUT:              "enc.blk.{bid}.attn_o",
    MODEL_TENSOR.ENC_ATTN_REL_B:            "enc.blk.{bid}.attn_rel_b",
    MODEL_TENSOR.ENC_FFN_NORM:              "enc.blk.{bid}.ffn_norm",
    MODEL_TENSOR.ENC_FFN_GATE:              "enc.blk.{bid}.ffn_gate",
    MODEL_TENSOR.ENC_FFN_DOWN:              "enc.blk.{bid}.ffn_down",
    MODEL_TENSOR.ENC_FFN_UP:                "enc.blk.{bid}.ffn_up",
    MODEL_TENSOR.ENC_OUTPUT_NORM:           "enc.output_norm",
    MODEL_TENSOR.CLS:                       "cls",
    MODEL_TENSOR.CLS_OUT:                   "cls.output",
    MODEL_TENSOR.CONV1D:                    "conv1d",
    MODEL_TENSOR.CONVNEXT_DW:               "convnext.{bid}.dw",
    MODEL_TENSOR.CONVNEXT_NORM:             "convnext.{bid}.norm",
    MODEL_TENSOR.CONVNEXT_PW1:              "convnext.{bid}.pw1",
    MODEL_TENSOR.CONVNEXT_PW2:              "convnext.{bid}.pw2",
    MODEL_TENSOR.CONVNEXT_GAMMA:            "convnext.{bid}.gamma",
    MODEL_TENSOR.POSNET_CONV1:              "posnet.{bid}.conv1",
    MODEL_TENSOR.POSNET_CONV2:              "posnet.{bid}.conv2",
    MODEL_TENSOR.POSNET_NORM:               "posnet.{bid}.norm",
    MODEL_TENSOR.POSNET_NORM1:              "posnet.{bid}.norm1",
    MODEL_TENSOR.POSNET_NORM2:              "posnet.{bid}.norm2",
    MODEL_TENSOR.POSNET_ATTN_NORM:          "posnet.{bid}.attn_norm",
    MODEL_TENSOR.POSNET_ATTN_Q:             "posnet.{bid}.attn_q",
    MODEL_TENSOR.POSNET_ATTN_K:             "posnet.{bid}.attn_k",
    MODEL_TENSOR.POSNET_ATTN_V:             "posnet.{bid}.attn_v",
    MODEL_TENSOR.POSNET_ATTN_OUT:           "posnet.{bid}.attn_output",
    MODEL_TENSOR.SHORTCONV_CONV:            "blk.{bid}.shortconv.conv",
    MODEL_TENSOR.SHORTCONV_INPROJ:          "blk.{bid}.shortconv.in_proj",
    MODEL_TENSOR.SHORTCONV_OUTPROJ:         "blk.{bid}.shortconv.out_proj",
    # vision
    MODEL_TENSOR.V_MMPROJ:                  "mm.{bid}",
    MODEL_TENSOR.V_MMPROJ_FC:               "mm.model.fc",
    MODEL_TENSOR.V_MMPROJ_MLP:              "mm.model.mlp.{bid}",
    MODEL_TENSOR.V_MMPROJ_PEG:              "mm.model.peg.{bid}",
    MODEL_TENSOR.V_ENC_EMBD_CLS:            "v.class_embd",
    MODEL_TENSOR.V_ENC_EMBD_PATCH:          "v.patch_embd",
    MODEL_TENSOR.V_ENC_EMBD_POS:            "v.position_embd",
    MODEL_TENSOR.V_ENC_ATTN_Q:              "v.blk.{bid}.attn_q",
    MODEL_TENSOR.V_ENC_ATTN_Q_NORM:         "v.blk.{bid}.attn_q_norm",
    MODEL_TENSOR.V_ENC_ATTN_K:              "v.blk.{bid}.attn_k",
    MODEL_TENSOR.V_ENC_ATTN_K_NORM:         "v.blk.{bid}.attn_k_norm",
    MODEL_TENSOR.V_ENC_ATTN_V:              "v.blk.{bid}.attn_v",
    MODEL_TENSOR.V_ENC_INPUT_NORM:          "v.blk.{bid}.ln1",
    MODEL_TENSOR.V_ENC_ATTN_O:              "v.blk.{bid}.attn_out",
    MODEL_TENSOR.V_ENC_ATTN_O_NORM:         "v.blk.{bid}.attn_out_norm",
    MODEL_TENSOR.V_ENC_POST_ATTN_NORM:      "v.blk.{bid}.ln2",
    MODEL_TENSOR.V_ENC_FFN_UP:              "v.blk.{bid}.ffn_up",
    MODEL_TENSOR.V_ENC_FFN_GATE:            "v.blk.{bid}.ffn_gate",
    MODEL_TENSOR.V_ENC_FFN_DOWN:            "v.blk.{bid}.ffn_down",
    MODEL_TENSOR.V_LAYER_SCALE_1:           "v.blk.{bid}.ls1",
    MODEL_TENSOR.V_LAYER_SCALE_2:           "v.blk.{bid}.ls2",
    MODEL_TENSOR.V_PRE_NORM:                "v.pre_ln",
    MODEL_TENSOR.V_POST_NORM:               "v.post_ln",
    MODEL_TENSOR.V_MM_INP_PROJ:             "mm.input_projection",
    MODEL_TENSOR.V_MM_INP_NORM:             "mm.input_norm",
    MODEL_TENSOR.V_MM_SOFT_EMB_NORM:        "mm.soft_emb_norm",
    MODEL_TENSOR.V_RESMPL_POS_EMBD_K:       "resampler.pos_embd_k",
    MODEL_TENSOR.V_RESMPL_ATTN_Q:           "resampler.attn.q",
    MODEL_TENSOR.V_RESMPL_ATTN_K:           "resampler.attn.k",
    MODEL_TENSOR.V_RESMPL_ATTN_V:           "resampler.attn.v",
    MODEL_TENSOR.V_RESMPL_ATTN_OUT:         "resampler.attn.out",
    MODEL_TENSOR.V_RESMPL_KV:               "resampler.kv",
    MODEL_TENSOR.V_RESMPL_KV_NORM:          "resampler.ln_kv",
    MODEL_TENSOR.V_RESMPL_POST_NORM:        "resampler.ln_post",
    MODEL_TENSOR.V_RESMPL_Q_NORM:           "resampler.ln_q",
    MODEL_TENSOR.V_RESMPL_PROJ:             "resampler.proj",
    MODEL_TENSOR.V_RESMPL_QUERY:            "resampler.query",
    MODEL_TENSOR.V_TOK_EMBD_IMG_BREAK:      "v.token_embd.img_break", # pixtral
    MODEL_TENSOR.V_MM_PATCH_MERGER:         "mm.patch_merger", # mistral small 3.1
    # audio (mtmd)
    MODEL_TENSOR.A_ENC_EMBD_POS:            "a.position_embd",
    MODEL_TENSOR.A_ENC_CONV1D:              "a.conv1d.{bid}",
    MODEL_TENSOR.A_PRE_NORM:                "a.pre_ln",
    MODEL_TENSOR.A_POST_NORM:               "a.post_ln",
    MODEL_TENSOR.A_ENC_ATTN_Q:              "a.blk.{bid}.attn_q",
    MODEL_TENSOR.A_ENC_ATTN_K:              "a.blk.{bid}.attn_k",
    MODEL_TENSOR.A_ENC_ATTN_V:              "a.blk.{bid}.attn_v",
    MODEL_TENSOR.A_ENC_INPUT_NORM:          "a.blk.{bid}.ln1",
    MODEL_TENSOR.A_ENC_OUTPUT:              "a.blk.{bid}.attn_out",
    MODEL_TENSOR.A_ENC_OUTPUT_NORM:         "a.blk.{bid}.ln2",
    MODEL_TENSOR.A_ENC_FFN_UP:              "a.blk.{bid}.ffn_up",
    MODEL_TENSOR.A_ENC_FFN_GATE:            "a.blk.{bid}.ffn_gate",
    MODEL_TENSOR.A_ENC_FFN_DOWN:            "a.blk.{bid}.ffn_down",
    MODEL_TENSOR.A_MMPROJ:                  "mm.a.mlp.{bid}",
    MODEL_TENSOR.A_MMPROJ_FC:               "mm.a.fc",
    MODEL_TENSOR.A_MM_NORM_PRE:             "mm.a.norm_pre",
    MODEL_TENSOR.A_MM_NORM_MID:             "mm.a.norm_mid",
    # NextN/MTP
    MODEL_TENSOR.NEXTN_EH_PROJ:             "blk.{bid}.nextn.eh_proj",
    MODEL_TENSOR.NEXTN_EMBED_TOKENS:        "blk.{bid}.nextn.embed_tokens",
    MODEL_TENSOR.NEXTN_ENORM:               "blk.{bid}.nextn.enorm",
    MODEL_TENSOR.NEXTN_HNORM:               "blk.{bid}.nextn.hnorm",
    MODEL_TENSOR.NEXTN_SHARED_HEAD_HEAD:    "blk.{bid}.nextn.shared_head_head",
    MODEL_TENSOR.NEXTN_SHARED_HEAD_NORM:    "blk.{bid}.nextn.shared_head_norm",
}

MODEL_TENSORS: dict[MODEL_ARCH, list[MODEL_TENSOR]] = {
    MODEL_ARCH.MMPROJ: [
        MODEL_TENSOR.V_MMPROJ,
        MODEL_TENSOR.V_MMPROJ_FC,
        MODEL_TENSOR.V_MMPROJ_MLP,
        MODEL_TENSOR.V_MMPROJ_PEG,
        MODEL_TENSOR.V_ENC_EMBD_CLS,
        MODEL_TENSOR.V_ENC_EMBD_PATCH,
        MODEL_TENSOR.V_ENC_EMBD_POS,
        MODEL_TENSOR.V_ENC_INPUT_NORM,
        MODEL_TENSOR.V_ENC_ATTN_Q,
        MODEL_TENSOR.V_ENC_ATTN_Q_NORM,
        MODEL_TENSOR.V_ENC_ATTN_K,
        MODEL_TENSOR.V_ENC_ATTN_K_NORM,
        MODEL_TENSOR.V_ENC_ATTN_V,
        MODEL_TENSOR.V_ENC_ATTN_O,
        MODEL_TENSOR.V_ENC_ATTN_O_NORM,
        MODEL_TENSOR.V_ENC_POST_ATTN_NORM,
        MODEL_TENSOR.V_ENC_FFN_UP,
        MODEL_TENSOR.V_ENC_FFN_GATE,
        MODEL_TENSOR.V_ENC_FFN_DOWN,
        MODEL_TENSOR.V_LAYER_SCALE_1,
        MODEL_TENSOR.V_LAYER_SCALE_2,
        MODEL_TENSOR.V_PRE_NORM,
        MODEL_TENSOR.V_POST_NORM,
        MODEL_TENSOR.V_MM_INP_PROJ,
        MODEL_TENSOR.V_MM_INP_NORM,
        MODEL_TENSOR.V_MM_SOFT_EMB_NORM,
        MODEL_TENSOR.V_RESMPL_POS_EMBD_K,
        MODEL_TENSOR.V_RESMPL_ATTN_Q,
        MODEL_TENSOR.V_RESMPL_ATTN_K,
        MODEL_TENSOR.V_RESMPL_ATTN_V,
        MODEL_TENSOR.V_RESMPL_ATTN_OUT,
        MODEL_TENSOR.V_RESMPL_KV,
        MODEL_TENSOR.V_RESMPL_KV_NORM,
        MODEL_TENSOR.V_RESMPL_POST_NORM,
        MODEL_TENSOR.V_RESMPL_Q_NORM,
        MODEL_TENSOR.V_RESMPL_PROJ,
        MODEL_TENSOR.V_RESMPL_QUERY,
        MODEL_TENSOR.V_TOK_EMBD_IMG_BREAK,
        MODEL_TENSOR.V_MM_PATCH_MERGER,
        # audio
        MODEL_TENSOR.A_ENC_EMBD_POS,
        MODEL_TENSOR.A_ENC_CONV1D,
        MODEL_TENSOR.A_PRE_NORM,
        MODEL_TENSOR.A_POST_NORM,
        MODEL_TENSOR.A_ENC_ATTN_Q,
        MODEL_TENSOR.A_ENC_ATTN_K,
        MODEL_TENSOR.A_ENC_ATTN_V,
        MODEL_TENSOR.A_ENC_INPUT_NORM,
        MODEL_TENSOR.A_ENC_OUTPUT,
        MODEL_TENSOR.A_ENC_OUTPUT_NORM,
        MODEL_TENSOR.A_ENC_FFN_UP,
        MODEL_TENSOR.A_ENC_FFN_GATE,
        MODEL_TENSOR.A_ENC_FFN_DOWN,
        MODEL_TENSOR.A_MMPROJ,
        MODEL_TENSOR.A_MMPROJ_FC,
        MODEL_TENSOR.A_MM_NORM_PRE,
        MODEL_TENSOR.A_MM_NORM_MID,
    ],
    MODEL_ARCH.LLAMA: [
        MODEL_TENSOR.TOKEN_EMBD,
        MODEL_TENSOR.OUTPUT_NORM,
        MODEL_TENSOR.OUTPUT,
        MODEL_TENSOR.ROPE_FREQS,
        MODEL_TENSOR.ATTN_NORM,
        MODEL_TENSOR.ATTN_Q,
        MODEL_TENSOR.ATTN_K,
        MODEL_TENSOR.ATTN_V,
        MODEL_TENSOR.ATTN_OUT,
        MODEL_TENSOR.ATTN_ROT_EMBD,
        MODEL_TENSOR.FFN_GATE_INP,
        MODEL_TENSOR.FFN_NORM,
        MODEL_TENSOR.FFN_GATE,
        MODEL_TENSOR.FFN_DOWN,
        MODEL_TENSOR.FFN_UP,
        MODEL_TENSOR.FFN_GATE_EXP,
        MODEL_TENSOR.FFN_DOWN_EXP,
        MODEL_TENSOR.FFN_UP_EXP,
    ],
    MODEL_ARCH.LLAMA4: [
        MODEL_TENSOR.TOKEN_EMBD,
        MODEL_TENSOR.OUTPUT_NORM,
        MODEL_TENSOR.OUTPUT,
        MODEL_TENSOR.ROPE_FREQS,
        MODEL_TENSOR.ATTN_NORM,
        MODEL_TENSOR.ATTN_Q,
        MODEL_TENSOR.ATTN_K,
        MODEL_TENSOR.ATTN_V,
        MODEL_TENSOR.ATTN_OUT,
        MODEL_TENSOR.ATTN_ROT_EMBD,
        MODEL_TENSOR.FFN_GATE_INP,
        MODEL_TENSOR.FFN_NORM,
        MODEL_TENSOR.FFN_GATE,
        MODEL_TENSOR.FFN_DOWN,
        MODEL_TENSOR.FFN_UP,
        MODEL_TENSOR.FFN_GATE_EXP,
        MODEL_TENSOR.FFN_DOWN_EXP,
        MODEL_TENSOR.FFN_UP_EXP,
        MODEL_TENSOR.FFN_GATE_SHEXP,
        MODEL_TENSOR.FFN_DOWN_SHEXP,
        MODEL_TENSOR.FFN_UP_SHEXP,
    ],
    MODEL_ARCH.DECI: [
        MODEL_TENSOR.TOKEN_EMBD,
        MODEL_TENSOR.OUTPUT_NORM,
        MODEL_TENSOR.OUTPUT,
        MODEL_TENSOR.ROPE_FREQS,
        MODEL_TENSOR.ATTN_NORM,
        MODEL_TENSOR.ATTN_Q,
        MODEL_TENSOR.ATTN_K,
        MODEL_TENSOR.ATTN_V,
        MODEL_TENSOR.ATTN_OUT,
        MODEL_TENSOR.ATTN_ROT_EMBD,
        MODEL_TENSOR.FFN_GATE_INP,
        MODEL_TENSOR.FFN_NORM,
        MODEL_TENSOR.FFN_GATE,
        MODEL_TENSOR.FFN_DOWN,
        MODEL_TENSOR.FFN_UP,
        MODEL_TENSOR.FFN_GATE_EXP,
        MODEL_TENSOR.FFN_DOWN_EXP,
        MODEL_TENSOR.FFN_UP_EXP,
    ],
    MODEL_ARCH.GROK: [
        MODEL_TENSOR.TOKEN_EMBD,
        MODEL_TENSOR.OUTPUT_NORM,
        MODEL_TENSOR.OUTPUT,
        MODEL_TENSOR.ROPE_FREQS,
        MODEL_TENSOR.ATTN_NORM,
        MODEL_TENSOR.ATTN_Q,
        MODEL_TENSOR.ATTN_K,
        MODEL_TENSOR.ATTN_V,
        MODEL_TENSOR.ATTN_OUT,
        MODEL_TENSOR.ATTN_ROT_EMBD,
        MODEL_TENSOR.ATTN_OUT_NORM,
        MODEL_TENSOR.FFN_GATE_INP,
        MODEL_TENSOR.FFN_NORM,
        MODEL_TENSOR.FFN_GATE,
        MODEL_TENSOR.FFN_DOWN,
        MODEL_TENSOR.FFN_UP,
        MODEL_TENSOR.FFN_GATE_EXP,
        MODEL_TENSOR.FFN_DOWN_EXP,
        MODEL_TENSOR.FFN_UP_EXP,
        MODEL_TENSOR.FFN_POST_NORM,
        MODEL_TENSOR.LAYER_OUT_NORM,
    ],
    MODEL_ARCH.GPTNEOX: [
        MODEL_TENSOR.TOKEN_EMBD,
        MODEL_TENSOR.OUTPUT_NORM,
        MODEL_TENSOR.OUTPUT,
        MODEL_TENSOR.ATTN_NORM,
        MODEL_TENSOR.ATTN_QKV,
        MODEL_TENSOR.ATTN_OUT,
        MODEL_TENSOR.FFN_NORM,
        MODEL_TENSOR.FFN_DOWN,
        MODEL_TENSOR.FFN_UP,
    ],
    MODEL_ARCH.FALCON: [
        MODEL_TENSOR.TOKEN_EMBD,
        MODEL_TENSOR.OUTPUT_NORM,
        MODEL_TENSOR.OUTPUT,
        MODEL_TENSOR.ATTN_NORM,
        MODEL_TENSOR.ATTN_NORM_2,
        MODEL_TENSOR.ATTN_QKV,
        MODEL_TENSOR.ATTN_OUT,
        MODEL_TENSOR.FFN_DOWN,
        MODEL_TENSOR.FFN_UP,
    ],
    MODEL_ARCH.BAICHUAN: [
        MODEL_TENSOR.TOKEN_EMBD,
        MODEL_TENSOR.OUTPUT_NORM,
        MODEL_TENSOR.OUTPUT,
        MODEL_TENSOR.ROPE_FREQS,
        MODEL_TENSOR.ATTN_NORM,
        MODEL_TENSOR.ATTN_Q,
        MODEL_TENSOR.ATTN_K,
        MODEL_TENSOR.ATTN_V,
        MODEL_TENSOR.ATTN_OUT,
        MODEL_TENSOR.ATTN_ROT_EMBD,
        MODEL_TENSOR.FFN_NORM,
        MODEL_TENSOR.FFN_GATE,
        MODEL_TENSOR.FFN_DOWN,
        MODEL_TENSOR.FFN_UP,
    ],
    MODEL_ARCH.STARCODER: [
        MODEL_TENSOR.TOKEN_EMBD,
        MODEL_TENSOR.POS_EMBD,
        MODEL_TENSOR.OUTPUT_NORM,
        MODEL_TENSOR.OUTPUT,
        MODEL_TENSOR.ATTN_NORM,
        MODEL_TENSOR.ATTN_QKV,
        MODEL_TENSOR.ATTN_OUT,
        MODEL_TENSOR.FFN_NORM,
        MODEL_TENSOR.FFN_DOWN,
        MODEL_TENSOR.FFN_UP,
    ],
    MODEL_ARCH.BERT: [
        MODEL_TENSOR.TOKEN_EMBD,
        MODEL_TENSOR.TOKEN_EMBD_NORM,
        MODEL_TENSOR.TOKEN_TYPES,
        MODEL_TENSOR.POS_EMBD,
        MODEL_TENSOR.OUTPUT_NORM,
        MODEL_TENSOR.ATTN_OUT_NORM,
        MODEL_TENSOR.ATTN_QKV,
        MODEL_TENSOR.ATTN_Q,
        MODEL_TENSOR.ATTN_K,
        MODEL_TENSOR.ATTN_V,
        MODEL_TENSOR.ATTN_OUT,
        MODEL_TENSOR.FFN_DOWN,
        MODEL_TENSOR.FFN_UP,
        MODEL_TENSOR.LAYER_OUT_NORM,
        MODEL_TENSOR.CLS,
        MODEL_TENSOR.CLS_OUT,
    ],
    MODEL_ARCH.NOMIC_BERT: [
        MODEL_TENSOR.TOKEN_EMBD,
        MODEL_TENSOR.TOKEN_EMBD_NORM,
        MODEL_TENSOR.TOKEN_TYPES,
        MODEL_TENSOR.POS_EMBD,
        MODEL_TENSOR.OUTPUT_NORM,
        MODEL_TENSOR.ATTN_OUT_NORM,
        MODEL_TENSOR.ATTN_QKV,
        MODEL_TENSOR.ATTN_OUT,
        MODEL_TENSOR.FFN_GATE,
        MODEL_TENSOR.FFN_DOWN,
        MODEL_TENSOR.FFN_UP,
        MODEL_TENSOR.LAYER_OUT_NORM,
    ],
    MODEL_ARCH.NOMIC_BERT_MOE: [
        MODEL_TENSOR.TOKEN_EMBD,
        MODEL_TENSOR.TOKEN_EMBD_NORM,
        MODEL_TENSOR.TOKEN_TYPES,
        MODEL_TENSOR.POS_EMBD,
        MODEL_TENSOR.OUTPUT_NORM,
        MODEL_TENSOR.ATTN_OUT_NORM,
        MODEL_TENSOR.ATTN_QKV,
        MODEL_TENSOR.ATTN_OUT,
        MODEL_TENSOR.FFN_DOWN,
        MODEL_TENSOR.FFN_UP,
        MODEL_TENSOR.FFN_GATE_INP,
        MODEL_TENSOR.FFN_DOWN_EXP,
        MODEL_TENSOR.FFN_UP_EXP,
        MODEL_TENSOR.LAYER_OUT_NORM,
    ],
    MODEL_ARCH.NEO_BERT: [
        MODEL_TENSOR.TOKEN_EMBD,
        MODEL_TENSOR.ATTN_NORM,
        MODEL_TENSOR.ATTN_QKV,
        MODEL_TENSOR.ATTN_OUT,
        MODEL_TENSOR.FFN_NORM,
        MODEL_TENSOR.FFN_DOWN,
        MODEL_TENSOR.FFN_UP,
        MODEL_TENSOR.ENC_OUTPUT_NORM,
        MODEL_TENSOR.CLS,
        MODEL_TENSOR.CLS_OUT,
    ],
    MODEL_ARCH.JINA_BERT_V2: [
        MODEL_TENSOR.TOKEN_EMBD,
        MODEL_TENSOR.TOKEN_EMBD_NORM,
        MODEL_TENSOR.TOKEN_TYPES,
        MODEL_TENSOR.ATTN_NORM_2,
        MODEL_TENSOR.ATTN_OUT_NORM,
        MODEL_TENSOR.ATTN_Q,
        MODEL_TENSOR.ATTN_Q_NORM,
        MODEL_TENSOR.ATTN_K,
        MODEL_TENSOR.ATTN_K_NORM,
        MODEL_TENSOR.ATTN_V,
        MODEL_TENSOR.ATTN_OUT,
        MODEL_TENSOR.FFN_UP,
        MODEL_TENSOR.FFN_GATE,
        MODEL_TENSOR.FFN_DOWN,
        MODEL_TENSOR.LAYER_OUT_NORM,
        MODEL_TENSOR.CLS,
    ],
    MODEL_ARCH.JINA_BERT_V3: [
        MODEL_TENSOR.TOKEN_EMBD,
        MODEL_TENSOR.TOKEN_EMBD_NORM,
        MODEL_TENSOR.TOKEN_TYPES,
        MODEL_TENSOR.OUTPUT_NORM,
        MODEL_TENSOR.ATTN_OUT_NORM,
        MODEL_TENSOR.ATTN_QKV,
        MODEL_TENSOR.ATTN_OUT,
        MODEL_TENSOR.FFN_DOWN,
        MODEL_TENSOR.FFN_UP,
        MODEL_TENSOR.LAYER_OUT_NORM,
    ],
    MODEL_ARCH.MPT: [
        MODEL_TENSOR.TOKEN_EMBD,
        MODEL_TENSOR.OUTPUT_NORM,
        MODEL_TENSOR.OUTPUT,
        MODEL_TENSOR.ATTN_NORM,
        MODEL_TENSOR.ATTN_QKV,
        MODEL_TENSOR.ATTN_OUT,
        MODEL_TENSOR.FFN_NORM,
        MODEL_TENSOR.FFN_DOWN,
        MODEL_TENSOR.FFN_UP,
        MODEL_TENSOR.FFN_ACT,
        MODEL_TENSOR.ATTN_Q_NORM,
        MODEL_TENSOR.ATTN_K_NORM,
        MODEL_TENSOR.POS_EMBD,
    ],
    MODEL_ARCH.GPTJ: [
        MODEL_TENSOR.TOKEN_EMBD,
        MODEL_TENSOR.OUTPUT_NORM,
        MODEL_TENSOR.OUTPUT,
        MODEL_TENSOR.ATTN_NORM,
        MODEL_TENSOR.ATTN_Q,
        MODEL_TENSOR.ATTN_K,
        MODEL_TENSOR.ATTN_V,
        MODEL_TENSOR.ATTN_OUT,
        MODEL_TENSOR.FFN_DOWN,
        MODEL_TENSOR.FFN_UP,
    ],
    MODEL_ARCH.REFACT: [
        MODEL_TENSOR.TOKEN_EMBD,
        MODEL_TENSOR.OUTPUT_NORM,
        MODEL_TENSOR.OUTPUT,
        MODEL_TENSOR.ATTN_NORM,
        MODEL_TENSOR.ATTN_Q,
        MODEL_TENSOR.ATTN_K,
        MODEL_TENSOR.ATTN_V,
        MODEL_TENSOR.ATTN_OUT,
        MODEL_TENSOR.FFN_NORM,
        MODEL_TENSOR.FFN_GATE,
        MODEL_TENSOR.FFN_DOWN,
        MODEL_TENSOR.FFN_UP,
    ],
    MODEL_ARCH.BLOOM: [
        MODEL_TENSOR.TOKEN_EMBD,
        MODEL_TENSOR.TOKEN_EMBD_NORM,
        MODEL_TENSOR.OUTPUT_NORM,
        MODEL_TENSOR.OUTPUT,
        MODEL_TENSOR.ATTN_NORM,
        MODEL_TENSOR.ATTN_QKV,
        MODEL_TENSOR.ATTN_OUT,
        MODEL_TENSOR.FFN_NORM,
        MODEL_TENSOR.FFN_DOWN,
        MODEL_TENSOR.FFN_UP,
    ],
    MODEL_ARCH.STABLELM: [
        MODEL_TENSOR.TOKEN_EMBD,
        MODEL_TENSOR.OUTPUT_NORM,
        MODEL_TENSOR.OUTPUT,
        MODEL_TENSOR.ROPE_FREQS,
        MODEL_TENSOR.ATTN_NORM,
        MODEL_TENSOR.ATTN_Q,
        MODEL_TENSOR.ATTN_K,
        MODEL_TENSOR.ATTN_V,
        MODEL_TENSOR.ATTN_OUT,
        MODEL_TENSOR.FFN_NORM,
        MODEL_TENSOR.FFN_GATE,
        MODEL_TENSOR.FFN_DOWN,
        MODEL_TENSOR.FFN_UP,
        MODEL_TENSOR.ATTN_Q_NORM,
        MODEL_TENSOR.ATTN_K_NORM,
    ],
    MODEL_ARCH.QWEN: [
        MODEL_TENSOR.TOKEN_EMBD,
        MODEL_TENSOR.OUTPUT_NORM,
        MODEL_TENSOR.OUTPUT,
        MODEL_TENSOR.ROPE_FREQS,
        MODEL_TENSOR.ATTN_NORM,
        MODEL_TENSOR.ATTN_QKV,
        MODEL_TENSOR.ATTN_OUT,
        MODEL_TENSOR.ATTN_ROT_EMBD,
        MODEL_TENSOR.FFN_NORM,
        MODEL_TENSOR.FFN_GATE,
        MODEL_TENSOR.FFN_DOWN,
        MODEL_TENSOR.FFN_UP,
    ],
    MODEL_ARCH.QWEN2: [
        MODEL_TENSOR.TOKEN_EMBD,
        MODEL_TENSOR.OUTPUT_NORM,
        MODEL_TENSOR.OUTPUT,
        MODEL_TENSOR.ROPE_FREQS,
        MODEL_TENSOR.ATTN_NORM,
        MODEL_TENSOR.ATTN_Q,
        MODEL_TENSOR.ATTN_K,
        MODEL_TENSOR.ATTN_V,
        MODEL_TENSOR.ATTN_OUT,
        MODEL_TENSOR.FFN_NORM,
        MODEL_TENSOR.FFN_GATE,
        MODEL_TENSOR.FFN_DOWN,
        MODEL_TENSOR.FFN_UP,
    ],
    MODEL_ARCH.DREAM: [
        MODEL_TENSOR.TOKEN_EMBD,
        MODEL_TENSOR.OUTPUT_NORM,
        MODEL_TENSOR.OUTPUT,
        MODEL_TENSOR.ROPE_FREQS,
        MODEL_TENSOR.ATTN_NORM,
        MODEL_TENSOR.ATTN_Q,
        MODEL_TENSOR.ATTN_K,
        MODEL_TENSOR.ATTN_V,
        MODEL_TENSOR.ATTN_OUT,
        MODEL_TENSOR.FFN_NORM,
        MODEL_TENSOR.FFN_GATE,
        MODEL_TENSOR.FFN_DOWN,
        MODEL_TENSOR.FFN_UP,
    ],
    MODEL_ARCH.LLADA: [
        MODEL_TENSOR.TOKEN_EMBD,
        MODEL_TENSOR.OUTPUT_NORM,
        MODEL_TENSOR.OUTPUT,
        MODEL_TENSOR.ROPE_FREQS,
        MODEL_TENSOR.ATTN_NORM,
        MODEL_TENSOR.ATTN_Q,
        MODEL_TENSOR.ATTN_K,
        MODEL_TENSOR.ATTN_V,
        MODEL_TENSOR.ATTN_OUT,
        MODEL_TENSOR.FFN_NORM,
        MODEL_TENSOR.FFN_GATE,
        MODEL_TENSOR.FFN_DOWN,
        MODEL_TENSOR.FFN_UP,
    ],
    MODEL_ARCH.QWEN2VL: [
        MODEL_TENSOR.TOKEN_EMBD,
        MODEL_TENSOR.OUTPUT_NORM,
        MODEL_TENSOR.OUTPUT,
        MODEL_TENSOR.ATTN_NORM,
        MODEL_TENSOR.ATTN_Q,
        MODEL_TENSOR.ATTN_K,
        MODEL_TENSOR.ATTN_V,
        MODEL_TENSOR.ATTN_OUT,
        MODEL_TENSOR.FFN_NORM,
        MODEL_TENSOR.FFN_GATE,
        MODEL_TENSOR.FFN_DOWN,
        MODEL_TENSOR.FFN_UP,
    ],
    MODEL_ARCH.QWEN2MOE: [
        MODEL_TENSOR.TOKEN_EMBD,
        MODEL_TENSOR.OUTPUT_NORM,
        MODEL_TENSOR.OUTPUT,
        MODEL_TENSOR.ATTN_NORM,
        MODEL_TENSOR.ATTN_Q,
        MODEL_TENSOR.ATTN_K,
        MODEL_TENSOR.ATTN_V,
        MODEL_TENSOR.ATTN_OUT,
        MODEL_TENSOR.FFN_NORM,
        MODEL_TENSOR.FFN_GATE_INP,
        MODEL_TENSOR.FFN_GATE_EXP,
        MODEL_TENSOR.FFN_DOWN_EXP,
        MODEL_TENSOR.FFN_UP_EXP,
        MODEL_TENSOR.FFN_GATE_INP_SHEXP,
        MODEL_TENSOR.FFN_GATE_SHEXP,
        MODEL_TENSOR.FFN_DOWN_SHEXP,
        MODEL_TENSOR.FFN_UP_SHEXP,
    ],
    MODEL_ARCH.QWEN3: [
        MODEL_TENSOR.TOKEN_EMBD,
        MODEL_TENSOR.OUTPUT_NORM,
        MODEL_TENSOR.OUTPUT,
        MODEL_TENSOR.ROPE_FREQS,
        MODEL_TENSOR.ATTN_NORM,
        MODEL_TENSOR.ATTN_Q,
        MODEL_TENSOR.ATTN_Q_NORM,
        MODEL_TENSOR.ATTN_K,
        MODEL_TENSOR.ATTN_K_NORM,
        MODEL_TENSOR.ATTN_V,
        MODEL_TENSOR.ATTN_OUT,
        MODEL_TENSOR.FFN_NORM,
        MODEL_TENSOR.FFN_GATE,
        MODEL_TENSOR.FFN_DOWN,
        MODEL_TENSOR.FFN_UP,
    ],
    MODEL_ARCH.QWEN3MOE: [
        MODEL_TENSOR.TOKEN_EMBD,
        MODEL_TENSOR.OUTPUT_NORM,
        MODEL_TENSOR.OUTPUT,
        MODEL_TENSOR.ATTN_NORM,
        MODEL_TENSOR.ATTN_Q,
        MODEL_TENSOR.ATTN_Q_NORM,
        MODEL_TENSOR.ATTN_K,
        MODEL_TENSOR.ATTN_K_NORM,
        MODEL_TENSOR.ATTN_V,
        MODEL_TENSOR.ATTN_OUT,
        MODEL_TENSOR.FFN_NORM,
        MODEL_TENSOR.FFN_GATE_INP,
        MODEL_TENSOR.FFN_GATE_EXP,
        MODEL_TENSOR.FFN_DOWN_EXP,
        MODEL_TENSOR.FFN_UP_EXP,
    ],
    MODEL_ARCH.PLAMO: [
        MODEL_TENSOR.TOKEN_EMBD,
        MODEL_TENSOR.OUTPUT_NORM,
        MODEL_TENSOR.OUTPUT,
        MODEL_TENSOR.ROPE_FREQS,
        MODEL_TENSOR.ATTN_NORM,
        MODEL_TENSOR.ATTN_Q,
        MODEL_TENSOR.ATTN_K,
        MODEL_TENSOR.ATTN_V,
        MODEL_TENSOR.ATTN_OUT,
        MODEL_TENSOR.ATTN_ROT_EMBD,
        MODEL_TENSOR.FFN_GATE,
        MODEL_TENSOR.FFN_DOWN,
        MODEL_TENSOR.FFN_UP,
    ],
    MODEL_ARCH.PLAMO2: [
        MODEL_TENSOR.TOKEN_EMBD,
        MODEL_TENSOR.OUTPUT_NORM,
        MODEL_TENSOR.OUTPUT,
        MODEL_TENSOR.ROPE_FREQS,
        MODEL_TENSOR.ATTN_NORM,
        MODEL_TENSOR.ATTN_QKV,
        MODEL_TENSOR.ATTN_Q,
        MODEL_TENSOR.ATTN_K,
        MODEL_TENSOR.ATTN_OUT,
        MODEL_TENSOR.ATTN_ROT_EMBD,
        MODEL_TENSOR.ATTN_Q_NORM,
        MODEL_TENSOR.ATTN_K_NORM,
        MODEL_TENSOR.ATTN_POST_NORM,
        MODEL_TENSOR.FFN_NORM,
        MODEL_TENSOR.FFN_GATE,
        MODEL_TENSOR.FFN_DOWN,
        MODEL_TENSOR.FFN_UP,
        MODEL_TENSOR.FFN_POST_NORM,
        MODEL_TENSOR.SSM_IN,
        MODEL_TENSOR.SSM_CONV1D,
        MODEL_TENSOR.SSM_X,
        MODEL_TENSOR.SSM_DT,
        MODEL_TENSOR.SSM_A,
        MODEL_TENSOR.SSM_D,
        MODEL_TENSOR.SSM_OUT,
        MODEL_TENSOR.SSM_DT_NORM,
        MODEL_TENSOR.SSM_B_NORM,
        MODEL_TENSOR.SSM_C_NORM,
    ],
    MODEL_ARCH.GPT2: [
        MODEL_TENSOR.TOKEN_EMBD,
        MODEL_TENSOR.POS_EMBD,
        MODEL_TENSOR.OUTPUT_NORM,
        MODEL_TENSOR.OUTPUT,
        MODEL_TENSOR.ATTN_NORM,
        MODEL_TENSOR.ATTN_QKV,
        MODEL_TENSOR.ATTN_OUT,
        MODEL_TENSOR.FFN_NORM,
        MODEL_TENSOR.FFN_DOWN,
        MODEL_TENSOR.FFN_UP,
    ],
    MODEL_ARCH.PHI2: [
        MODEL_TENSOR.TOKEN_EMBD,
        MODEL_TENSOR.OUTPUT_NORM,
        MODEL_TENSOR.OUTPUT,
        MODEL_TENSOR.ATTN_NORM,
        MODEL_TENSOR.ATTN_QKV,
        MODEL_TENSOR.ATTN_Q,
        MODEL_TENSOR.ATTN_K,
        MODEL_TENSOR.ATTN_V,
        MODEL_TENSOR.ATTN_OUT,
        MODEL_TENSOR.FFN_NORM,
        MODEL_TENSOR.FFN_DOWN,
        MODEL_TENSOR.FFN_UP,
    ],
    MODEL_ARCH.PHI3: [
        MODEL_TENSOR.TOKEN_EMBD,
        MODEL_TENSOR.OUTPUT_NORM,
        MODEL_TENSOR.OUTPUT,
        MODEL_TENSOR.ROPE_FACTORS_LONG,
        MODEL_TENSOR.ROPE_FACTORS_SHORT,
        MODEL_TENSOR.ATTN_NORM,
        MODEL_TENSOR.ATTN_QKV,
        MODEL_TENSOR.ATTN_Q,
        MODEL_TENSOR.ATTN_K,
        MODEL_TENSOR.ATTN_V,
        MODEL_TENSOR.ATTN_OUT,
        MODEL_TENSOR.FFN_NORM,
        MODEL_TENSOR.FFN_DOWN,
        MODEL_TENSOR.FFN_UP,
    ],
    MODEL_ARCH.PHIMOE: [
        MODEL_TENSOR.TOKEN_EMBD,
        MODEL_TENSOR.OUTPUT_NORM,
        MODEL_TENSOR.OUTPUT,
        MODEL_TENSOR.ROPE_FACTORS_LONG,
        MODEL_TENSOR.ROPE_FACTORS_SHORT,
        MODEL_TENSOR.ATTN_NORM,
        MODEL_TENSOR.ATTN_QKV,
        MODEL_TENSOR.ATTN_Q,
        MODEL_TENSOR.ATTN_K,
        MODEL_TENSOR.ATTN_V,
        MODEL_TENSOR.ATTN_OUT,
        MODEL_TENSOR.FFN_NORM,
        MODEL_TENSOR.FFN_GATE_INP,
        MODEL_TENSOR.FFN_GATE_EXP,
        MODEL_TENSOR.FFN_DOWN_EXP,
        MODEL_TENSOR.FFN_UP_EXP,
    ],
    MODEL_ARCH.CODESHELL: [
        MODEL_TENSOR.TOKEN_EMBD,
        MODEL_TENSOR.POS_EMBD,
        MODEL_TENSOR.OUTPUT_NORM,
        MODEL_TENSOR.OUTPUT,
        MODEL_TENSOR.ATTN_NORM,
        MODEL_TENSOR.ATTN_QKV,
        MODEL_TENSOR.ATTN_OUT,
        MODEL_TENSOR.ATTN_ROT_EMBD,
        MODEL_TENSOR.FFN_NORM,
        MODEL_TENSOR.FFN_DOWN,
        MODEL_TENSOR.FFN_UP,
    ],
    MODEL_ARCH.ORION: [
        MODEL_TENSOR.TOKEN_EMBD,
        MODEL_TENSOR.OUTPUT_NORM,
        MODEL_TENSOR.OUTPUT,
        MODEL_TENSOR.ROPE_FREQS,
        MODEL_TENSOR.ATTN_NORM,
        MODEL_TENSOR.ATTN_Q,
        MODEL_TENSOR.ATTN_K,
        MODEL_TENSOR.ATTN_V,
        MODEL_TENSOR.ATTN_OUT,
        MODEL_TENSOR.ATTN_ROT_EMBD,
        MODEL_TENSOR.FFN_NORM,
        MODEL_TENSOR.FFN_GATE,
        MODEL_TENSOR.FFN_DOWN,
        MODEL_TENSOR.FFN_UP,
    ],
    MODEL_ARCH.INTERNLM2: [
        MODEL_TENSOR.TOKEN_EMBD,
        MODEL_TENSOR.OUTPUT_NORM,
        MODEL_TENSOR.OUTPUT,
        MODEL_TENSOR.ATTN_NORM,
        MODEL_TENSOR.ATTN_Q,
        MODEL_TENSOR.ATTN_K,
        MODEL_TENSOR.ATTN_V,
        MODEL_TENSOR.ATTN_OUT,
        MODEL_TENSOR.ATTN_ROT_EMBD,
        MODEL_TENSOR.FFN_NORM,
        MODEL_TENSOR.FFN_GATE,
        MODEL_TENSOR.FFN_DOWN,
        MODEL_TENSOR.FFN_UP,
    ],
    MODEL_ARCH.MINICPM: [
        MODEL_TENSOR.TOKEN_EMBD,
        MODEL_TENSOR.OUTPUT,
        MODEL_TENSOR.OUTPUT_NORM,
        MODEL_TENSOR.ROPE_FREQS,
        MODEL_TENSOR.ROPE_FACTORS_LONG,
        MODEL_TENSOR.ROPE_FACTORS_SHORT,
        MODEL_TENSOR.ATTN_NORM,
        MODEL_TENSOR.ATTN_Q,
        MODEL_TENSOR.ATTN_K,
        MODEL_TENSOR.ATTN_V,
        MODEL_TENSOR.ATTN_OUT,
        MODEL_TENSOR.ATTN_ROT_EMBD,
        MODEL_TENSOR.FFN_GATE_INP,
        MODEL_TENSOR.FFN_NORM,
        MODEL_TENSOR.FFN_GATE,
        MODEL_TENSOR.FFN_DOWN,
        MODEL_TENSOR.FFN_UP,
        MODEL_TENSOR.FFN_GATE_EXP,
        MODEL_TENSOR.FFN_DOWN_EXP,
        MODEL_TENSOR.FFN_UP_EXP,
    ],
    MODEL_ARCH.MINICPM3: [
        MODEL_TENSOR.TOKEN_EMBD,
        MODEL_TENSOR.OUTPUT_NORM,
        MODEL_TENSOR.OUTPUT,
        MODEL_TENSOR.ROPE_FACTORS_LONG,
        MODEL_TENSOR.ROPE_FACTORS_SHORT,
        MODEL_TENSOR.ATTN_NORM,
        MODEL_TENSOR.ATTN_Q_A,
        MODEL_TENSOR.ATTN_Q_B,
        MODEL_TENSOR.ATTN_KV_A_MQA,
        MODEL_TENSOR.ATTN_KV_B,
        MODEL_TENSOR.ATTN_Q_A_NORM,
        MODEL_TENSOR.ATTN_KV_A_NORM,
        MODEL_TENSOR.ATTN_OUT,
        MODEL_TENSOR.FFN_NORM,
        MODEL_TENSOR.FFN_GATE,
        MODEL_TENSOR.FFN_DOWN,
        MODEL_TENSOR.FFN_UP,
    ],
    MODEL_ARCH.GEMMA: [
        MODEL_TENSOR.TOKEN_EMBD,
        MODEL_TENSOR.OUTPUT_NORM,
        MODEL_TENSOR.ATTN_NORM,
        MODEL_TENSOR.ATTN_Q,
        MODEL_TENSOR.ATTN_K,
        MODEL_TENSOR.ATTN_V,
        MODEL_TENSOR.ATTN_OUT,
        MODEL_TENSOR.FFN_GATE,
        MODEL_TENSOR.FFN_DOWN,
        MODEL_TENSOR.FFN_UP,
        MODEL_TENSOR.FFN_NORM,
    ],
    MODEL_ARCH.GEMMA2: [
        MODEL_TENSOR.TOKEN_EMBD,
        MODEL_TENSOR.OUTPUT_NORM,
        MODEL_TENSOR.ATTN_Q,
        MODEL_TENSOR.ATTN_K,
        MODEL_TENSOR.ATTN_V,
        MODEL_TENSOR.ATTN_OUT,
        MODEL_TENSOR.FFN_GATE,
        MODEL_TENSOR.FFN_DOWN,
        MODEL_TENSOR.FFN_UP,
        MODEL_TENSOR.ATTN_NORM,
        MODEL_TENSOR.ATTN_POST_NORM,
        MODEL_TENSOR.FFN_PRE_NORM,
        MODEL_TENSOR.FFN_POST_NORM,
    ],
    MODEL_ARCH.GEMMA3: [
        MODEL_TENSOR.TOKEN_EMBD,
        MODEL_TENSOR.OUTPUT,
        MODEL_TENSOR.OUTPUT_NORM,
        MODEL_TENSOR.ATTN_Q,
        MODEL_TENSOR.ATTN_Q_NORM,
        MODEL_TENSOR.ATTN_K,
        MODEL_TENSOR.ATTN_K_NORM,
        MODEL_TENSOR.ATTN_V,
        MODEL_TENSOR.ATTN_OUT,
        MODEL_TENSOR.FFN_GATE,
        MODEL_TENSOR.FFN_DOWN,
        MODEL_TENSOR.FFN_UP,
        MODEL_TENSOR.ATTN_NORM,
        MODEL_TENSOR.ATTN_POST_NORM,
        MODEL_TENSOR.FFN_PRE_NORM,
        MODEL_TENSOR.FFN_POST_NORM,
    ],
    MODEL_ARCH.GEMMA3N: [
        MODEL_TENSOR.TOKEN_EMBD,
        MODEL_TENSOR.OUTPUT,
        MODEL_TENSOR.OUTPUT_NORM,
        MODEL_TENSOR.ATTN_Q,
        MODEL_TENSOR.ATTN_Q_NORM,
        MODEL_TENSOR.ATTN_K,
        MODEL_TENSOR.ATTN_K_NORM,
        MODEL_TENSOR.ATTN_V,
        MODEL_TENSOR.ATTN_OUT,
        MODEL_TENSOR.FFN_GATE,
        MODEL_TENSOR.FFN_DOWN,
        MODEL_TENSOR.FFN_UP,
        MODEL_TENSOR.ATTN_NORM,
        MODEL_TENSOR.ATTN_POST_NORM,
        MODEL_TENSOR.FFN_PRE_NORM,
        MODEL_TENSOR.FFN_POST_NORM,
        # altup / laurel
        MODEL_TENSOR.PER_LAYER_TOKEN_EMBD,
        MODEL_TENSOR.PER_LAYER_MODEL_PROJ,
        MODEL_TENSOR.PER_LAYER_INP_GATE,
        MODEL_TENSOR.PER_LAYER_PROJ,
        MODEL_TENSOR.PER_LAYER_PROJ_NORM,
        MODEL_TENSOR.PER_LAYER_POST_NORM,
        MODEL_TENSOR.ALTUP_PROJ,
        MODEL_TENSOR.ALTUP_UNEMBD_PROJ,
        MODEL_TENSOR.ALTUP_CORRECT_COEF,
        MODEL_TENSOR.ALTUP_CORRECT_SCALE,
        MODEL_TENSOR.ALTUP_PREDICT_COEF,
        MODEL_TENSOR.ALTUP_ROUTER,
        MODEL_TENSOR.ALTUP_ROUTER_NORM,
        MODEL_TENSOR.LAUREL_L,
        MODEL_TENSOR.LAUREL_R,
        MODEL_TENSOR.LAUREL_POST_NORM,
    ],
    MODEL_ARCH.GEMMA_EMBEDDING: [
        MODEL_TENSOR.TOKEN_EMBD,
        MODEL_TENSOR.OUTPUT,
        MODEL_TENSOR.DENSE_2_OUT,
        MODEL_TENSOR.DENSE_3_OUT,
        MODEL_TENSOR.OUTPUT_NORM,
        MODEL_TENSOR.ATTN_Q,
        MODEL_TENSOR.ATTN_Q_NORM,
        MODEL_TENSOR.ATTN_K,
        MODEL_TENSOR.ATTN_K_NORM,
        MODEL_TENSOR.ATTN_V,
        MODEL_TENSOR.ATTN_OUT,
        MODEL_TENSOR.FFN_GATE,
        MODEL_TENSOR.FFN_DOWN,
        MODEL_TENSOR.FFN_UP,
        MODEL_TENSOR.ATTN_NORM,
        MODEL_TENSOR.ATTN_POST_NORM,
        MODEL_TENSOR.FFN_PRE_NORM,
        MODEL_TENSOR.FFN_POST_NORM,
    ],
    MODEL_ARCH.STARCODER2: [
        MODEL_TENSOR.TOKEN_EMBD,
        MODEL_TENSOR.OUTPUT_NORM,
        MODEL_TENSOR.OUTPUT,
        MODEL_TENSOR.ROPE_FREQS,
        MODEL_TENSOR.ATTN_NORM,
        MODEL_TENSOR.ATTN_Q,
        MODEL_TENSOR.ATTN_K,
        MODEL_TENSOR.ATTN_V,
        MODEL_TENSOR.ATTN_OUT,
        MODEL_TENSOR.ATTN_ROT_EMBD,
        MODEL_TENSOR.FFN_NORM,
        MODEL_TENSOR.FFN_DOWN,
        MODEL_TENSOR.FFN_UP,
    ],
    MODEL_ARCH.RWKV6: [
        MODEL_TENSOR.TOKEN_EMBD,
        MODEL_TENSOR.TOKEN_EMBD_NORM,
        MODEL_TENSOR.OUTPUT_NORM,
        MODEL_TENSOR.OUTPUT,
        MODEL_TENSOR.ATTN_NORM,
        MODEL_TENSOR.ATTN_NORM_2,
        MODEL_TENSOR.TIME_MIX_W1,
        MODEL_TENSOR.TIME_MIX_W2,
        MODEL_TENSOR.TIME_MIX_LERP_X,
        MODEL_TENSOR.TIME_MIX_LERP_K,
        MODEL_TENSOR.TIME_MIX_LERP_V,
        MODEL_TENSOR.TIME_MIX_LERP_R,
        MODEL_TENSOR.TIME_MIX_LERP_G,
        MODEL_TENSOR.TIME_MIX_LERP_W,
        MODEL_TENSOR.TIME_MIX_LERP_FUSED,
        MODEL_TENSOR.TIME_MIX_FIRST,
        MODEL_TENSOR.TIME_MIX_DECAY,
        MODEL_TENSOR.TIME_MIX_DECAY_W1,
        MODEL_TENSOR.TIME_MIX_DECAY_W2,
        MODEL_TENSOR.TIME_MIX_KEY,
        MODEL_TENSOR.TIME_MIX_VALUE,
        MODEL_TENSOR.TIME_MIX_RECEPTANCE,
        MODEL_TENSOR.TIME_MIX_GATE,
        MODEL_TENSOR.TIME_MIX_LN,
        MODEL_TENSOR.TIME_MIX_OUTPUT,
        MODEL_TENSOR.CHANNEL_MIX_LERP_K,
        MODEL_TENSOR.CHANNEL_MIX_LERP_R,
        MODEL_TENSOR.CHANNEL_MIX_KEY,
        MODEL_TENSOR.CHANNEL_MIX_RECEPTANCE,
        MODEL_TENSOR.CHANNEL_MIX_VALUE,
    ],
    MODEL_ARCH.RWKV6QWEN2: [
        MODEL_TENSOR.TOKEN_EMBD,
        MODEL_TENSOR.OUTPUT_NORM,
        MODEL_TENSOR.OUTPUT,
        MODEL_TENSOR.ATTN_NORM,
        MODEL_TENSOR.TIME_MIX_W1,
        MODEL_TENSOR.TIME_MIX_W2,
        MODEL_TENSOR.TIME_MIX_LERP_X,
        MODEL_TENSOR.TIME_MIX_LERP_K,
        MODEL_TENSOR.TIME_MIX_LERP_V,
        MODEL_TENSOR.TIME_MIX_LERP_R,
        MODEL_TENSOR.TIME_MIX_LERP_G,
        MODEL_TENSOR.TIME_MIX_LERP_W,
        MODEL_TENSOR.TIME_MIX_LERP_FUSED,
        MODEL_TENSOR.TIME_MIX_FIRST,
        MODEL_TENSOR.TIME_MIX_DECAY,
        MODEL_TENSOR.TIME_MIX_DECAY_W1,
        MODEL_TENSOR.TIME_MIX_DECAY_W2,
        MODEL_TENSOR.TIME_MIX_KEY,
        MODEL_TENSOR.TIME_MIX_VALUE,
        MODEL_TENSOR.TIME_MIX_RECEPTANCE,
        MODEL_TENSOR.TIME_MIX_GATE,
        MODEL_TENSOR.TIME_MIX_LN,
        MODEL_TENSOR.TIME_MIX_OUTPUT,
        MODEL_TENSOR.FFN_NORM,
        MODEL_TENSOR.FFN_GATE,
        MODEL_TENSOR.FFN_DOWN,
        MODEL_TENSOR.FFN_UP,
    ],
    MODEL_ARCH.RWKV7: [
        MODEL_TENSOR.TOKEN_EMBD,
        MODEL_TENSOR.TOKEN_EMBD_NORM,
        MODEL_TENSOR.OUTPUT_NORM,
        MODEL_TENSOR.OUTPUT,
        MODEL_TENSOR.ATTN_NORM,
        MODEL_TENSOR.ATTN_NORM_2,
        MODEL_TENSOR.TIME_MIX_LERP_FUSED,
        MODEL_TENSOR.TIME_MIX_W0,
        MODEL_TENSOR.TIME_MIX_W1,
        MODEL_TENSOR.TIME_MIX_W2,
        MODEL_TENSOR.TIME_MIX_A0,
        MODEL_TENSOR.TIME_MIX_A1,
        MODEL_TENSOR.TIME_MIX_A2,
        MODEL_TENSOR.TIME_MIX_V0,
        MODEL_TENSOR.TIME_MIX_V1,
        MODEL_TENSOR.TIME_MIX_V2,
        MODEL_TENSOR.TIME_MIX_G1,
        MODEL_TENSOR.TIME_MIX_G2,
        MODEL_TENSOR.TIME_MIX_K_K,
        MODEL_TENSOR.TIME_MIX_K_A,
        MODEL_TENSOR.TIME_MIX_R_K,
        MODEL_TENSOR.TIME_MIX_KEY,
        MODEL_TENSOR.TIME_MIX_VALUE,
        MODEL_TENSOR.TIME_MIX_RECEPTANCE,
        MODEL_TENSOR.TIME_MIX_LN,
        MODEL_TENSOR.TIME_MIX_OUTPUT,
        MODEL_TENSOR.CHANNEL_MIX_LERP_K,
        MODEL_TENSOR.CHANNEL_MIX_KEY,
        MODEL_TENSOR.CHANNEL_MIX_VALUE,
    ],
    MODEL_ARCH.ARWKV7: [
        MODEL_TENSOR.TOKEN_EMBD,
        MODEL_TENSOR.TOKEN_EMBD_NORM,
        MODEL_TENSOR.OUTPUT_NORM,
        MODEL_TENSOR.OUTPUT,
        MODEL_TENSOR.ATTN_NORM,
        MODEL_TENSOR.TIME_MIX_LERP_FUSED,
        MODEL_TENSOR.TIME_MIX_W0,
        MODEL_TENSOR.TIME_MIX_W1,
        MODEL_TENSOR.TIME_MIX_W2,
        MODEL_TENSOR.TIME_MIX_A0,
        MODEL_TENSOR.TIME_MIX_A1,
        MODEL_TENSOR.TIME_MIX_A2,
        MODEL_TENSOR.TIME_MIX_V0,
        MODEL_TENSOR.TIME_MIX_V1,
        MODEL_TENSOR.TIME_MIX_V2,
        MODEL_TENSOR.TIME_MIX_G1,
        MODEL_TENSOR.TIME_MIX_G2,
        MODEL_TENSOR.TIME_MIX_K_K,
        MODEL_TENSOR.TIME_MIX_K_A,
        MODEL_TENSOR.TIME_MIX_R_K,
        MODEL_TENSOR.TIME_MIX_KEY,
        MODEL_TENSOR.TIME_MIX_VALUE,
        MODEL_TENSOR.TIME_MIX_RECEPTANCE,
        MODEL_TENSOR.TIME_MIX_LN,
        MODEL_TENSOR.TIME_MIX_OUTPUT,
        MODEL_TENSOR.FFN_NORM,
        MODEL_TENSOR.FFN_GATE,
        MODEL_TENSOR.FFN_DOWN,
        MODEL_TENSOR.FFN_UP,
    ],
    MODEL_ARCH.MAMBA: [
        MODEL_TENSOR.TOKEN_EMBD,
        MODEL_TENSOR.OUTPUT_NORM,
        MODEL_TENSOR.OUTPUT,
        MODEL_TENSOR.ATTN_NORM,
        MODEL_TENSOR.SSM_IN,
        MODEL_TENSOR.SSM_CONV1D,
        MODEL_TENSOR.SSM_X,
        MODEL_TENSOR.SSM_DT,
        MODEL_TENSOR.SSM_A,
        MODEL_TENSOR.SSM_D,
        MODEL_TENSOR.SSM_OUT,
    ],
    MODEL_ARCH.MAMBA2: [
        MODEL_TENSOR.TOKEN_EMBD,
        MODEL_TENSOR.OUTPUT_NORM,
        MODEL_TENSOR.OUTPUT,
        MODEL_TENSOR.ATTN_NORM,
        MODEL_TENSOR.SSM_IN,
        MODEL_TENSOR.SSM_CONV1D,
        MODEL_TENSOR.SSM_DT,
        MODEL_TENSOR.SSM_A,
        MODEL_TENSOR.SSM_D,
        MODEL_TENSOR.SSM_NORM,
        MODEL_TENSOR.SSM_OUT,
    ],
    MODEL_ARCH.JAMBA: [
        MODEL_TENSOR.TOKEN_EMBD,
        MODEL_TENSOR.OUTPUT_NORM,
        MODEL_TENSOR.OUTPUT,
        MODEL_TENSOR.ATTN_NORM,
        MODEL_TENSOR.ATTN_Q,
        MODEL_TENSOR.ATTN_K,
        MODEL_TENSOR.ATTN_V,
        MODEL_TENSOR.ATTN_OUT,
        MODEL_TENSOR.SSM_IN,
        MODEL_TENSOR.SSM_CONV1D,
        MODEL_TENSOR.SSM_X,
        MODEL_TENSOR.SSM_DT,
        MODEL_TENSOR.SSM_DT_NORM,
        MODEL_TENSOR.SSM_A,
        MODEL_TENSOR.SSM_B_NORM,
        MODEL_TENSOR.SSM_C_NORM,
        MODEL_TENSOR.SSM_D,
        MODEL_TENSOR.SSM_OUT,
        MODEL_TENSOR.FFN_GATE_INP,
        MODEL_TENSOR.FFN_NORM,
        MODEL_TENSOR.FFN_GATE,
        MODEL_TENSOR.FFN_DOWN,
        MODEL_TENSOR.FFN_UP,
        MODEL_TENSOR.FFN_GATE_EXP,
        MODEL_TENSOR.FFN_DOWN_EXP,
        MODEL_TENSOR.FFN_UP_EXP,
    ],
    MODEL_ARCH.XVERSE: [
        MODEL_TENSOR.TOKEN_EMBD,
        MODEL_TENSOR.OUTPUT_NORM,
        MODEL_TENSOR.OUTPUT,
        MODEL_TENSOR.ROPE_FREQS,
        MODEL_TENSOR.ATTN_NORM,
        MODEL_TENSOR.ATTN_Q,
        MODEL_TENSOR.ATTN_K,
        MODEL_TENSOR.ATTN_V,
        MODEL_TENSOR.ATTN_OUT,
        MODEL_TENSOR.ATTN_ROT_EMBD,
        MODEL_TENSOR.FFN_NORM,
        MODEL_TENSOR.FFN_GATE,
        MODEL_TENSOR.FFN_DOWN,
        MODEL_TENSOR.FFN_UP,
    ],
    MODEL_ARCH.COMMAND_R: [
        MODEL_TENSOR.TOKEN_EMBD,
        MODEL_TENSOR.OUTPUT_NORM,
        MODEL_TENSOR.ATTN_NORM,
        MODEL_TENSOR.ATTN_Q,
        MODEL_TENSOR.ATTN_K,
        MODEL_TENSOR.ATTN_V,
        MODEL_TENSOR.ATTN_OUT,
        MODEL_TENSOR.FFN_GATE,
        MODEL_TENSOR.FFN_DOWN,
        MODEL_TENSOR.FFN_UP,
        MODEL_TENSOR.ATTN_K_NORM,
        MODEL_TENSOR.ATTN_Q_NORM,
    ],
    MODEL_ARCH.COHERE2: [
        MODEL_TENSOR.TOKEN_EMBD,
        MODEL_TENSOR.OUTPUT_NORM,
        MODEL_TENSOR.ATTN_NORM,
        MODEL_TENSOR.ATTN_Q,
        MODEL_TENSOR.ATTN_K,
        MODEL_TENSOR.ATTN_V,
        MODEL_TENSOR.ATTN_OUT,
        MODEL_TENSOR.FFN_GATE,
        MODEL_TENSOR.FFN_DOWN,
        MODEL_TENSOR.FFN_UP,
    ],
    MODEL_ARCH.DBRX: [
        MODEL_TENSOR.TOKEN_EMBD,
        MODEL_TENSOR.OUTPUT_NORM,
        MODEL_TENSOR.OUTPUT,
        MODEL_TENSOR.ATTN_NORM,
        MODEL_TENSOR.ATTN_QKV,
        MODEL_TENSOR.ATTN_OUT,
        MODEL_TENSOR.ATTN_OUT_NORM,
        MODEL_TENSOR.FFN_GATE_INP,
        MODEL_TENSOR.FFN_GATE_EXP,
        MODEL_TENSOR.FFN_DOWN_EXP,
        MODEL_TENSOR.FFN_UP_EXP,
    ],
    MODEL_ARCH.OLMO: [
        MODEL_TENSOR.TOKEN_EMBD,
        MODEL_TENSOR.OUTPUT,
        MODEL_TENSOR.ATTN_Q,
        MODEL_TENSOR.ATTN_K,
        MODEL_TENSOR.ATTN_V,
        MODEL_TENSOR.ATTN_OUT,
        MODEL_TENSOR.FFN_GATE,
        MODEL_TENSOR.FFN_DOWN,
        MODEL_TENSOR.FFN_UP,
    ],
    MODEL_ARCH.OLMO2: [
        MODEL_TENSOR.TOKEN_EMBD,
        MODEL_TENSOR.OUTPUT_NORM,
        MODEL_TENSOR.OUTPUT,
        MODEL_TENSOR.ATTN_Q,
        MODEL_TENSOR.ATTN_K,
        MODEL_TENSOR.ATTN_V,
        MODEL_TENSOR.ATTN_OUT,
        MODEL_TENSOR.ATTN_POST_NORM,
        MODEL_TENSOR.ATTN_Q_NORM,
        MODEL_TENSOR.ATTN_K_NORM,
        MODEL_TENSOR.FFN_POST_NORM,
        MODEL_TENSOR.FFN_GATE,
        MODEL_TENSOR.FFN_DOWN,
        MODEL_TENSOR.FFN_UP,
    ],
    MODEL_ARCH.SEED_OSS: [
        MODEL_TENSOR.TOKEN_EMBD,
        MODEL_TENSOR.ATTN_NORM,
        MODEL_TENSOR.ATTN_Q,
        MODEL_TENSOR.ATTN_K,
        MODEL_TENSOR.ATTN_V,
        MODEL_TENSOR.ATTN_OUT,
        MODEL_TENSOR.ATTN_POST_NORM,
        MODEL_TENSOR.FFN_GATE,
        MODEL_TENSOR.FFN_DOWN,
        MODEL_TENSOR.FFN_UP,
        MODEL_TENSOR.OUTPUT_NORM,
        MODEL_TENSOR.OUTPUT,
    ],
    MODEL_ARCH.OLMOE: [
        MODEL_TENSOR.TOKEN_EMBD,
        MODEL_TENSOR.OUTPUT_NORM,
        MODEL_TENSOR.OUTPUT,
        MODEL_TENSOR.ATTN_OUT,
        MODEL_TENSOR.ATTN_Q,
        MODEL_TENSOR.ATTN_K,
        MODEL_TENSOR.ATTN_V,
        MODEL_TENSOR.ATTN_NORM,
        MODEL_TENSOR.ATTN_Q_NORM,
        MODEL_TENSOR.ATTN_K_NORM,
        MODEL_TENSOR.FFN_NORM,
        MODEL_TENSOR.FFN_GATE_INP,
        MODEL_TENSOR.FFN_GATE_EXP,
        MODEL_TENSOR.FFN_UP_EXP,
        MODEL_TENSOR.FFN_DOWN_EXP,
    ],
    MODEL_ARCH.OPENELM: [
        MODEL_TENSOR.TOKEN_EMBD,
        MODEL_TENSOR.OUTPUT_NORM,
        MODEL_TENSOR.ATTN_NORM,
        MODEL_TENSOR.ATTN_QKV,
        MODEL_TENSOR.ATTN_Q_NORM,
        MODEL_TENSOR.ATTN_K_NORM,
        MODEL_TENSOR.ATTN_OUT,
        MODEL_TENSOR.FFN_NORM,
        MODEL_TENSOR.FFN_GATE,
        MODEL_TENSOR.FFN_DOWN,
        MODEL_TENSOR.FFN_UP,
    ],
    MODEL_ARCH.ARCTIC: [
        MODEL_TENSOR.TOKEN_EMBD,
        MODEL_TENSOR.OUTPUT_NORM,
        MODEL_TENSOR.OUTPUT,
        MODEL_TENSOR.ROPE_FREQS,
        MODEL_TENSOR.ATTN_NORM,
        MODEL_TENSOR.ATTN_Q,
        MODEL_TENSOR.ATTN_K,
        MODEL_TENSOR.ATTN_V,
        MODEL_TENSOR.ATTN_OUT,
        MODEL_TENSOR.ATTN_ROT_EMBD,
        MODEL_TENSOR.FFN_GATE_INP,
        MODEL_TENSOR.FFN_NORM,
        MODEL_TENSOR.FFN_GATE,
        MODEL_TENSOR.FFN_DOWN,
        MODEL_TENSOR.FFN_UP,
        MODEL_TENSOR.FFN_NORM_EXP,
        MODEL_TENSOR.FFN_GATE_EXP,
        MODEL_TENSOR.FFN_DOWN_EXP,
        MODEL_TENSOR.FFN_UP_EXP,
    ],
    MODEL_ARCH.DEEPSEEK: [
        MODEL_TENSOR.TOKEN_EMBD,
        MODEL_TENSOR.OUTPUT_NORM,
        MODEL_TENSOR.OUTPUT,
        MODEL_TENSOR.ROPE_FREQS,
        MODEL_TENSOR.ATTN_NORM,
        MODEL_TENSOR.ATTN_Q,
        MODEL_TENSOR.ATTN_K,
        MODEL_TENSOR.ATTN_V,
        MODEL_TENSOR.ATTN_OUT,
        MODEL_TENSOR.ATTN_ROT_EMBD,
        MODEL_TENSOR.FFN_GATE_INP,
        MODEL_TENSOR.FFN_NORM,
        MODEL_TENSOR.FFN_GATE,
        MODEL_TENSOR.FFN_DOWN,
        MODEL_TENSOR.FFN_UP,
        MODEL_TENSOR.FFN_GATE_EXP,
        MODEL_TENSOR.FFN_DOWN_EXP,
        MODEL_TENSOR.FFN_UP_EXP,
        MODEL_TENSOR.FFN_GATE_SHEXP,
        MODEL_TENSOR.FFN_DOWN_SHEXP,
        MODEL_TENSOR.FFN_UP_SHEXP,
    ],
    MODEL_ARCH.DEEPSEEK2: [
        MODEL_TENSOR.TOKEN_EMBD,
        MODEL_TENSOR.OUTPUT_NORM,
        MODEL_TENSOR.OUTPUT,
        MODEL_TENSOR.ROPE_FREQS,
        MODEL_TENSOR.ATTN_NORM,
        MODEL_TENSOR.ATTN_Q,
        MODEL_TENSOR.ATTN_Q_A,
        MODEL_TENSOR.ATTN_Q_B,
        MODEL_TENSOR.ATTN_KV_A_MQA,
        MODEL_TENSOR.ATTN_KV_B,
        MODEL_TENSOR.ATTN_K_B,
        MODEL_TENSOR.ATTN_V_B,
        MODEL_TENSOR.ATTN_Q_A_NORM,
        MODEL_TENSOR.ATTN_KV_A_NORM,
        MODEL_TENSOR.ATTN_OUT,
        MODEL_TENSOR.ATTN_ROT_EMBD,
        MODEL_TENSOR.FFN_GATE_INP,
        MODEL_TENSOR.FFN_NORM,
        MODEL_TENSOR.FFN_GATE,
        MODEL_TENSOR.FFN_DOWN,
        MODEL_TENSOR.FFN_UP,
        MODEL_TENSOR.FFN_GATE_EXP,
        MODEL_TENSOR.FFN_DOWN_EXP,
        MODEL_TENSOR.FFN_UP_EXP,
        MODEL_TENSOR.FFN_GATE_SHEXP,
        MODEL_TENSOR.FFN_DOWN_SHEXP,
        MODEL_TENSOR.FFN_UP_SHEXP,
        MODEL_TENSOR.FFN_EXP_PROBS_B,
    ],
    MODEL_ARCH.ERNIE4_5_MOE: [
        MODEL_TENSOR.TOKEN_EMBD,
        MODEL_TENSOR.OUTPUT_NORM,
        MODEL_TENSOR.OUTPUT,
        MODEL_TENSOR.ATTN_NORM,
        MODEL_TENSOR.ATTN_Q,
        MODEL_TENSOR.ATTN_K,
        MODEL_TENSOR.ATTN_V,
        MODEL_TENSOR.ATTN_OUT,
        MODEL_TENSOR.FFN_NORM,
        MODEL_TENSOR.FFN_GATE,
        MODEL_TENSOR.FFN_DOWN,
        MODEL_TENSOR.FFN_UP,
        MODEL_TENSOR.FFN_GATE_INP,
        MODEL_TENSOR.FFN_GATE_EXP,
        MODEL_TENSOR.FFN_DOWN_EXP,
        MODEL_TENSOR.FFN_UP_EXP,
        MODEL_TENSOR.FFN_GATE_SHEXP,
        MODEL_TENSOR.FFN_DOWN_SHEXP,
        MODEL_TENSOR.FFN_UP_SHEXP,
        MODEL_TENSOR.FFN_EXP_PROBS_B,
    ],
    MODEL_ARCH.PLM: [
        MODEL_TENSOR.TOKEN_EMBD,
        MODEL_TENSOR.OUTPUT,
        MODEL_TENSOR.OUTPUT_NORM,
        MODEL_TENSOR.ATTN_NORM,
        MODEL_TENSOR.ATTN_Q,
        MODEL_TENSOR.ATTN_KV_A_MQA,
        MODEL_TENSOR.ATTN_KV_A_NORM,
        MODEL_TENSOR.ATTN_KV_B,
        MODEL_TENSOR.ATTN_OUT,
        MODEL_TENSOR.FFN_NORM,
        MODEL_TENSOR.FFN_UP,
        MODEL_TENSOR.FFN_DOWN,
    ],
    MODEL_ARCH.CHATGLM : [
        MODEL_TENSOR.TOKEN_EMBD,
        MODEL_TENSOR.ROPE_FREQS,
        MODEL_TENSOR.OUTPUT_NORM,
        MODEL_TENSOR.OUTPUT,
        MODEL_TENSOR.ATTN_NORM,
        MODEL_TENSOR.ATTN_QKV,
        MODEL_TENSOR.ATTN_Q,
        MODEL_TENSOR.ATTN_K,
        MODEL_TENSOR.ATTN_V,
        MODEL_TENSOR.ATTN_OUT,
        MODEL_TENSOR.FFN_NORM,
        MODEL_TENSOR.FFN_DOWN,
        MODEL_TENSOR.FFN_UP,
    ],
    MODEL_ARCH.GLM4 : [
        MODEL_TENSOR.TOKEN_EMBD,
        MODEL_TENSOR.ROPE_FREQS,
        MODEL_TENSOR.OUTPUT_NORM,
        MODEL_TENSOR.OUTPUT,
        MODEL_TENSOR.ATTN_NORM,
        MODEL_TENSOR.ATTN_QKV,
        MODEL_TENSOR.ATTN_Q,
        MODEL_TENSOR.ATTN_K,
        MODEL_TENSOR.ATTN_V,
        MODEL_TENSOR.ATTN_OUT,
        MODEL_TENSOR.FFN_NORM,
        MODEL_TENSOR.FFN_DOWN,
        MODEL_TENSOR.FFN_UP,
        MODEL_TENSOR.ATTN_POST_NORM,
        MODEL_TENSOR.FFN_POST_NORM,
    ],
    MODEL_ARCH.GLM4_MOE: [
        MODEL_TENSOR.TOKEN_EMBD,
        MODEL_TENSOR.OUTPUT_NORM,
        MODEL_TENSOR.OUTPUT,
        MODEL_TENSOR.ATTN_NORM,
        MODEL_TENSOR.ATTN_POST_NORM,
        MODEL_TENSOR.ATTN_Q,
        MODEL_TENSOR.ATTN_K,
        MODEL_TENSOR.ATTN_V,
        MODEL_TENSOR.ATTN_OUT,
        MODEL_TENSOR.ATTN_Q_NORM,
        MODEL_TENSOR.ATTN_K_NORM,
        MODEL_TENSOR.FFN_GATE,
        MODEL_TENSOR.FFN_DOWN,
        MODEL_TENSOR.FFN_UP,
        MODEL_TENSOR.FFN_GATE_INP,
        MODEL_TENSOR.FFN_GATE_EXP,
        MODEL_TENSOR.FFN_DOWN_EXP,
        MODEL_TENSOR.FFN_UP_EXP,
        MODEL_TENSOR.FFN_GATE_SHEXP,
        MODEL_TENSOR.FFN_DOWN_SHEXP,
        MODEL_TENSOR.FFN_UP_SHEXP,
        MODEL_TENSOR.FFN_EXP_PROBS_B,
        # NextN/MTP tensors - preserved but unused
        MODEL_TENSOR.NEXTN_EH_PROJ,
        MODEL_TENSOR.NEXTN_EMBED_TOKENS,
        MODEL_TENSOR.NEXTN_ENORM,
        MODEL_TENSOR.NEXTN_HNORM,
        MODEL_TENSOR.NEXTN_SHARED_HEAD_HEAD,
        MODEL_TENSOR.NEXTN_SHARED_HEAD_NORM,
    ],
    MODEL_ARCH.BITNET: [
        MODEL_TENSOR.ATTN_Q,
        MODEL_TENSOR.ATTN_K,
        MODEL_TENSOR.ATTN_V,
        MODEL_TENSOR.TOKEN_EMBD,
        MODEL_TENSOR.OUTPUT_NORM,
        MODEL_TENSOR.ATTN_NORM,
        MODEL_TENSOR.ATTN_OUT,
        MODEL_TENSOR.FFN_NORM,
        MODEL_TENSOR.FFN_GATE,
        MODEL_TENSOR.FFN_DOWN,
        MODEL_TENSOR.FFN_UP,
        MODEL_TENSOR.ATTN_SUB_NORM,
        MODEL_TENSOR.FFN_SUB_NORM,
    ],
    MODEL_ARCH.T5: [
        MODEL_TENSOR.TOKEN_EMBD,
        MODEL_TENSOR.OUTPUT,
        MODEL_TENSOR.DEC_ATTN_NORM,
        MODEL_TENSOR.DEC_ATTN_Q,
        MODEL_TENSOR.DEC_ATTN_K,
        MODEL_TENSOR.DEC_ATTN_V,
        MODEL_TENSOR.DEC_ATTN_OUT,
        MODEL_TENSOR.DEC_ATTN_REL_B,
        MODEL_TENSOR.DEC_CROSS_ATTN_NORM,
        MODEL_TENSOR.DEC_CROSS_ATTN_Q,
        MODEL_TENSOR.DEC_CROSS_ATTN_K,
        MODEL_TENSOR.DEC_CROSS_ATTN_V,
        MODEL_TENSOR.DEC_CROSS_ATTN_OUT,
        MODEL_TENSOR.DEC_CROSS_ATTN_REL_B,
        MODEL_TENSOR.DEC_FFN_NORM,
        MODEL_TENSOR.DEC_FFN_GATE,
        MODEL_TENSOR.DEC_FFN_DOWN,
        MODEL_TENSOR.DEC_FFN_UP,
        MODEL_TENSOR.DEC_OUTPUT_NORM,
        MODEL_TENSOR.ENC_ATTN_NORM,
        MODEL_TENSOR.ENC_ATTN_Q,
        MODEL_TENSOR.ENC_ATTN_K,
        MODEL_TENSOR.ENC_ATTN_V,
        MODEL_TENSOR.ENC_ATTN_OUT,
        MODEL_TENSOR.ENC_ATTN_REL_B,
        MODEL_TENSOR.ENC_FFN_NORM,
        MODEL_TENSOR.ENC_FFN_GATE,
        MODEL_TENSOR.ENC_FFN_DOWN,
        MODEL_TENSOR.ENC_FFN_UP,
        MODEL_TENSOR.ENC_OUTPUT_NORM,
    ],
    MODEL_ARCH.T5ENCODER: [
        MODEL_TENSOR.TOKEN_EMBD,
        MODEL_TENSOR.OUTPUT,
        MODEL_TENSOR.ENC_ATTN_NORM,
        MODEL_TENSOR.ENC_ATTN_Q,
        MODEL_TENSOR.ENC_ATTN_K,
        MODEL_TENSOR.ENC_ATTN_V,
        MODEL_TENSOR.ENC_ATTN_OUT,
        MODEL_TENSOR.ENC_ATTN_REL_B,
        MODEL_TENSOR.ENC_FFN_NORM,
        MODEL_TENSOR.ENC_FFN_GATE,
        MODEL_TENSOR.ENC_FFN_DOWN,
        MODEL_TENSOR.ENC_FFN_UP,
        MODEL_TENSOR.ENC_OUTPUT_NORM,
    ],
    MODEL_ARCH.JAIS: [
        MODEL_TENSOR.TOKEN_EMBD,
        MODEL_TENSOR.OUTPUT_NORM,
        MODEL_TENSOR.OUTPUT,
        MODEL_TENSOR.ATTN_NORM,
        MODEL_TENSOR.ATTN_QKV,
        MODEL_TENSOR.ATTN_OUT,
        MODEL_TENSOR.FFN_NORM,
        MODEL_TENSOR.FFN_DOWN,
        MODEL_TENSOR.FFN_GATE,
        MODEL_TENSOR.FFN_UP,
    ],
    MODEL_ARCH.NEMOTRON: [
        MODEL_TENSOR.TOKEN_EMBD,
        MODEL_TENSOR.OUTPUT_NORM,
        MODEL_TENSOR.OUTPUT,
        MODEL_TENSOR.ROPE_FREQS,
        MODEL_TENSOR.ATTN_NORM,
        MODEL_TENSOR.ATTN_Q,
        MODEL_TENSOR.ATTN_K,
        MODEL_TENSOR.ATTN_V,
        MODEL_TENSOR.ATTN_OUT,
        MODEL_TENSOR.ATTN_ROT_EMBD,
        MODEL_TENSOR.FFN_NORM,
        MODEL_TENSOR.FFN_DOWN,
        MODEL_TENSOR.FFN_UP,
    ],
    MODEL_ARCH.NEMOTRON_H: [
        MODEL_TENSOR.TOKEN_EMBD,
        MODEL_TENSOR.OUTPUT_NORM,
        MODEL_TENSOR.OUTPUT,
        MODEL_TENSOR.ATTN_NORM,
        MODEL_TENSOR.SSM_IN,
        MODEL_TENSOR.SSM_CONV1D,
        MODEL_TENSOR.SSM_DT,
        MODEL_TENSOR.SSM_A,
        MODEL_TENSOR.SSM_D,
        MODEL_TENSOR.SSM_NORM,
        MODEL_TENSOR.SSM_OUT,
        MODEL_TENSOR.ATTN_Q,
        MODEL_TENSOR.ATTN_K,
        MODEL_TENSOR.ATTN_V,
        MODEL_TENSOR.ATTN_OUT,
        MODEL_TENSOR.FFN_DOWN,
        MODEL_TENSOR.FFN_UP,
    ],
    MODEL_ARCH.EXAONE: [
        MODEL_TENSOR.TOKEN_EMBD,
        MODEL_TENSOR.OUTPUT_NORM,
        MODEL_TENSOR.OUTPUT,
        MODEL_TENSOR.ROPE_FREQS,
        MODEL_TENSOR.ATTN_NORM,
        MODEL_TENSOR.ATTN_Q,
        MODEL_TENSOR.ATTN_K,
        MODEL_TENSOR.ATTN_V,
        MODEL_TENSOR.ATTN_OUT,
        MODEL_TENSOR.ATTN_ROT_EMBD,
        MODEL_TENSOR.FFN_NORM,
        MODEL_TENSOR.FFN_GATE,
        MODEL_TENSOR.FFN_DOWN,
        MODEL_TENSOR.FFN_UP,
    ],
    MODEL_ARCH.EXAONE4: [
        MODEL_TENSOR.TOKEN_EMBD,
        MODEL_TENSOR.OUTPUT_NORM,
        MODEL_TENSOR.OUTPUT,
        MODEL_TENSOR.ROPE_FREQS,
        MODEL_TENSOR.ATTN_Q,
        MODEL_TENSOR.ATTN_Q_NORM,
        MODEL_TENSOR.ATTN_K,
        MODEL_TENSOR.ATTN_K_NORM,
        MODEL_TENSOR.ATTN_V,
        MODEL_TENSOR.ATTN_OUT,
        MODEL_TENSOR.ATTN_POST_NORM,
        MODEL_TENSOR.FFN_GATE,
        MODEL_TENSOR.FFN_DOWN,
        MODEL_TENSOR.FFN_UP,
        MODEL_TENSOR.FFN_POST_NORM,
    ],
    MODEL_ARCH.GRANITE: [
        MODEL_TENSOR.TOKEN_EMBD,
        MODEL_TENSOR.OUTPUT_NORM,
        MODEL_TENSOR.OUTPUT,
        MODEL_TENSOR.ATTN_NORM,
        MODEL_TENSOR.ATTN_Q,
        MODEL_TENSOR.ATTN_K,
        MODEL_TENSOR.ATTN_V,
        MODEL_TENSOR.ATTN_OUT,
        MODEL_TENSOR.FFN_NORM,
        MODEL_TENSOR.FFN_GATE,
        MODEL_TENSOR.FFN_DOWN,
        MODEL_TENSOR.FFN_UP,
    ],
    MODEL_ARCH.GRANITE_MOE: [
        MODEL_TENSOR.TOKEN_EMBD,
        MODEL_TENSOR.OUTPUT_NORM,
        MODEL_TENSOR.OUTPUT,
        MODEL_TENSOR.ATTN_NORM,
        MODEL_TENSOR.ATTN_Q,
        MODEL_TENSOR.ATTN_K,
        MODEL_TENSOR.ATTN_V,
        MODEL_TENSOR.ATTN_OUT,
        MODEL_TENSOR.FFN_NORM,
        MODEL_TENSOR.FFN_GATE_INP,
        MODEL_TENSOR.FFN_GATE_EXP,
        MODEL_TENSOR.FFN_DOWN_EXP,
        MODEL_TENSOR.FFN_UP_EXP,
        MODEL_TENSOR.FFN_GATE_SHEXP,
        MODEL_TENSOR.FFN_UP_SHEXP,
        MODEL_TENSOR.FFN_DOWN_SHEXP,
    ],
    MODEL_ARCH.GRANITE_HYBRID: [
        MODEL_TENSOR.TOKEN_EMBD,
        MODEL_TENSOR.OUTPUT_NORM,
        MODEL_TENSOR.OUTPUT,
        MODEL_TENSOR.ATTN_NORM,
        MODEL_TENSOR.SSM_IN,
        MODEL_TENSOR.SSM_CONV1D,
        MODEL_TENSOR.SSM_DT,
        MODEL_TENSOR.SSM_A,
        MODEL_TENSOR.SSM_D,
        MODEL_TENSOR.SSM_NORM,
        MODEL_TENSOR.SSM_OUT,
        MODEL_TENSOR.ATTN_Q,
        MODEL_TENSOR.ATTN_K,
        MODEL_TENSOR.ATTN_V,
        MODEL_TENSOR.ATTN_OUT,
        MODEL_TENSOR.FFN_NORM,
        # MoE
        MODEL_TENSOR.FFN_GATE_INP,
        MODEL_TENSOR.FFN_GATE_EXP,
        MODEL_TENSOR.FFN_DOWN_EXP,
        MODEL_TENSOR.FFN_UP_EXP,
        MODEL_TENSOR.FFN_GATE_SHEXP,
        MODEL_TENSOR.FFN_UP_SHEXP,
        MODEL_TENSOR.FFN_DOWN_SHEXP,
        # Dense
        MODEL_TENSOR.FFN_GATE,
        MODEL_TENSOR.FFN_DOWN,
        MODEL_TENSOR.FFN_UP,
    ],
    MODEL_ARCH.CHAMELEON: [
        MODEL_TENSOR.TOKEN_EMBD,
        MODEL_TENSOR.OUTPUT_NORM,
        MODEL_TENSOR.OUTPUT,
        MODEL_TENSOR.ATTN_NORM,
        MODEL_TENSOR.ATTN_Q,
        MODEL_TENSOR.ATTN_Q_NORM,
        MODEL_TENSOR.ATTN_K,
        MODEL_TENSOR.ATTN_K_NORM,
        MODEL_TENSOR.ATTN_V,
        MODEL_TENSOR.ATTN_OUT,
        MODEL_TENSOR.FFN_NORM,
        MODEL_TENSOR.FFN_GATE,
        MODEL_TENSOR.FFN_DOWN,
        MODEL_TENSOR.FFN_UP,
    ],
    MODEL_ARCH.WAVTOKENIZER_DEC: [
        MODEL_TENSOR.TOKEN_EMBD,
        MODEL_TENSOR.TOKEN_EMBD_NORM,
        MODEL_TENSOR.CONV1D,
        MODEL_TENSOR.CONVNEXT_DW,
        MODEL_TENSOR.CONVNEXT_NORM,
        MODEL_TENSOR.CONVNEXT_PW1,
        MODEL_TENSOR.CONVNEXT_PW2,
        MODEL_TENSOR.CONVNEXT_GAMMA,
        MODEL_TENSOR.OUTPUT,
        MODEL_TENSOR.OUTPUT_NORM,
        MODEL_TENSOR.POSNET_CONV1,
        MODEL_TENSOR.POSNET_CONV2,
        MODEL_TENSOR.POSNET_NORM,
        MODEL_TENSOR.POSNET_NORM1,
        MODEL_TENSOR.POSNET_NORM2,
        MODEL_TENSOR.POSNET_ATTN_NORM,
        MODEL_TENSOR.POSNET_ATTN_Q,
        MODEL_TENSOR.POSNET_ATTN_K,
        MODEL_TENSOR.POSNET_ATTN_V,
        MODEL_TENSOR.POSNET_ATTN_OUT,
    ],
    MODEL_ARCH.BAILINGMOE: [
        MODEL_TENSOR.TOKEN_EMBD,
        MODEL_TENSOR.OUTPUT_NORM,
        MODEL_TENSOR.OUTPUT,
        MODEL_TENSOR.ROPE_FREQS,
        MODEL_TENSOR.ATTN_NORM,
        MODEL_TENSOR.ATTN_Q,
        MODEL_TENSOR.ATTN_K,
        MODEL_TENSOR.ATTN_V,
        MODEL_TENSOR.ATTN_OUT,
        MODEL_TENSOR.FFN_GATE_INP,
        MODEL_TENSOR.FFN_NORM,
        MODEL_TENSOR.FFN_GATE_EXP,
        MODEL_TENSOR.FFN_DOWN_EXP,
        MODEL_TENSOR.FFN_UP_EXP,
        MODEL_TENSOR.FFN_GATE_SHEXP,
        MODEL_TENSOR.FFN_DOWN_SHEXP,
        MODEL_TENSOR.FFN_UP_SHEXP,
    ],
    MODEL_ARCH.DOTS1: [
        MODEL_TENSOR.TOKEN_EMBD,
        MODEL_TENSOR.OUTPUT_NORM,
        MODEL_TENSOR.OUTPUT,
        MODEL_TENSOR.ATTN_NORM,
        MODEL_TENSOR.ATTN_Q,
        MODEL_TENSOR.ATTN_Q_NORM,
        MODEL_TENSOR.ATTN_K,
        MODEL_TENSOR.ATTN_K_NORM,
        MODEL_TENSOR.ATTN_V,
        MODEL_TENSOR.ATTN_OUT,
        MODEL_TENSOR.FFN_EXP_PROBS_B,
        MODEL_TENSOR.FFN_NORM,
        MODEL_TENSOR.FFN_GATE,
        MODEL_TENSOR.FFN_GATE_EXP,
        MODEL_TENSOR.FFN_GATE_INP,
        MODEL_TENSOR.FFN_GATE_SHEXP,
        MODEL_TENSOR.FFN_DOWN,
        MODEL_TENSOR.FFN_DOWN_EXP,
        MODEL_TENSOR.FFN_DOWN_SHEXP,
        MODEL_TENSOR.FFN_UP,
        MODEL_TENSOR.FFN_UP_EXP,
        MODEL_TENSOR.FFN_UP_SHEXP,
    ],
    MODEL_ARCH.ARCEE: [
        MODEL_TENSOR.TOKEN_EMBD,
        MODEL_TENSOR.OUTPUT_NORM,
        MODEL_TENSOR.OUTPUT,
        MODEL_TENSOR.ROPE_FREQS,
        MODEL_TENSOR.ATTN_NORM,
        MODEL_TENSOR.ATTN_Q,
        MODEL_TENSOR.ATTN_K,
        MODEL_TENSOR.ATTN_V,
        MODEL_TENSOR.ATTN_OUT,
        MODEL_TENSOR.ATTN_ROT_EMBD,
        MODEL_TENSOR.FFN_NORM,
        MODEL_TENSOR.FFN_DOWN,
        MODEL_TENSOR.FFN_UP,
    ],
    MODEL_ARCH.ERNIE4_5: [
        MODEL_TENSOR.TOKEN_EMBD,
        MODEL_TENSOR.OUTPUT_NORM,
        MODEL_TENSOR.OUTPUT,
        MODEL_TENSOR.ATTN_NORM,
        MODEL_TENSOR.ATTN_Q,
        MODEL_TENSOR.ATTN_K,
        MODEL_TENSOR.ATTN_V,
        MODEL_TENSOR.ATTN_OUT,
        MODEL_TENSOR.FFN_NORM,
        MODEL_TENSOR.FFN_GATE,
        MODEL_TENSOR.FFN_DOWN,
        MODEL_TENSOR.FFN_UP,
    ],
    MODEL_ARCH.FALCON_H1: [
        # Token embedding
        MODEL_TENSOR.TOKEN_EMBD,

        # Input layernorm
        MODEL_TENSOR.ATTN_NORM,

        # Attention components
        MODEL_TENSOR.ATTN_Q,         # Query projection
        MODEL_TENSOR.ATTN_K,         # Key projection
        MODEL_TENSOR.ATTN_V,         # Value projection
        MODEL_TENSOR.ATTN_OUT,       # Output projection

        # SSM components (Mamba2 specific)
        MODEL_TENSOR.SSM_IN,         # Input projection for SSM
        MODEL_TENSOR.SSM_CONV1D,     # Convolution layer
        MODEL_TENSOR.SSM_DT,         # Delta time projection
        MODEL_TENSOR.SSM_A,          # A parameter (log form)
        MODEL_TENSOR.SSM_D,          # D parameter
        MODEL_TENSOR.SSM_NORM,       # Normalization in SSM
        MODEL_TENSOR.SSM_OUT,        # Output projection

        # Pre-feedforward layernorm
        MODEL_TENSOR.FFN_PRE_NORM,

        # Feed-forward network components
        MODEL_TENSOR.FFN_GATE,       # Gate projection (SwiGLU)
        MODEL_TENSOR.FFN_DOWN,       # Down projection
        MODEL_TENSOR.FFN_UP,         # Up projection

        # Post-feedforward layernorm
        MODEL_TENSOR.OUTPUT_NORM,    # Final layer norm
        MODEL_TENSOR.OUTPUT,         # Output projection (lm_head)
    ],
    MODEL_ARCH.HUNYUAN_MOE: [
        MODEL_TENSOR.TOKEN_EMBD,
        MODEL_TENSOR.OUTPUT_NORM,
        MODEL_TENSOR.OUTPUT,
        MODEL_TENSOR.ROPE_FREQS,
        MODEL_TENSOR.ATTN_NORM,
        MODEL_TENSOR.ATTN_Q,
        MODEL_TENSOR.ATTN_Q_NORM,
        MODEL_TENSOR.ATTN_K,
        MODEL_TENSOR.ATTN_K_NORM,
        MODEL_TENSOR.ATTN_V,
        MODEL_TENSOR.ATTN_OUT,
        MODEL_TENSOR.FFN_GATE_INP,
        MODEL_TENSOR.FFN_NORM,
        MODEL_TENSOR.FFN_GATE_EXP,
        MODEL_TENSOR.FFN_DOWN_EXP,
        MODEL_TENSOR.FFN_UP_EXP,
        MODEL_TENSOR.FFN_GATE_SHEXP,
        MODEL_TENSOR.FFN_DOWN_SHEXP,
        MODEL_TENSOR.FFN_UP_SHEXP,
    ],
    MODEL_ARCH.HUNYUAN_DENSE: [
        MODEL_TENSOR.TOKEN_EMBD,
        MODEL_TENSOR.OUTPUT_NORM,
        MODEL_TENSOR.OUTPUT,
        MODEL_TENSOR.ATTN_NORM,
        MODEL_TENSOR.ATTN_Q,
        MODEL_TENSOR.ATTN_Q_NORM,
        MODEL_TENSOR.ATTN_K,
        MODEL_TENSOR.ATTN_K_NORM,
        MODEL_TENSOR.ATTN_V,
        MODEL_TENSOR.ATTN_OUT,
        MODEL_TENSOR.FFN_NORM,
        MODEL_TENSOR.FFN_GATE,
        MODEL_TENSOR.FFN_DOWN,
        MODEL_TENSOR.FFN_UP,
    ],
    MODEL_ARCH.SMOLLM3: [
        MODEL_TENSOR.TOKEN_EMBD,
        MODEL_TENSOR.OUTPUT_NORM,
        MODEL_TENSOR.OUTPUT,
        MODEL_TENSOR.ROPE_FREQS,
        MODEL_TENSOR.ATTN_NORM,
        MODEL_TENSOR.ATTN_Q,
        MODEL_TENSOR.ATTN_K,
        MODEL_TENSOR.ATTN_V,
        MODEL_TENSOR.ATTN_OUT,
        MODEL_TENSOR.ATTN_ROT_EMBD,
        MODEL_TENSOR.FFN_NORM,
        MODEL_TENSOR.FFN_GATE,
        MODEL_TENSOR.FFN_DOWN,
        MODEL_TENSOR.FFN_UP,
    ],
    MODEL_ARCH.GPT_OSS: [
        MODEL_TENSOR.TOKEN_EMBD,
        MODEL_TENSOR.OUTPUT_NORM,
        MODEL_TENSOR.OUTPUT,
        MODEL_TENSOR.ATTN_NORM,
        MODEL_TENSOR.ATTN_POST_NORM,
        MODEL_TENSOR.ATTN_Q,
        MODEL_TENSOR.ATTN_K,
        MODEL_TENSOR.ATTN_V,
        MODEL_TENSOR.ATTN_OUT,
        MODEL_TENSOR.ATTN_SINKS,
        MODEL_TENSOR.FFN_GATE_INP,
        MODEL_TENSOR.FFN_GATE_EXP,
        MODEL_TENSOR.FFN_DOWN_EXP,
        MODEL_TENSOR.FFN_UP_EXP,
    ],
    MODEL_ARCH.LFM2: [
        MODEL_TENSOR.TOKEN_EMBD,
        MODEL_TENSOR.TOKEN_EMBD_NORM,
        MODEL_TENSOR.SHORTCONV_CONV,
        MODEL_TENSOR.SHORTCONV_INPROJ,
        MODEL_TENSOR.SHORTCONV_OUTPROJ,
        MODEL_TENSOR.FFN_GATE,
        MODEL_TENSOR.FFN_DOWN,
        MODEL_TENSOR.FFN_UP,
        MODEL_TENSOR.FFN_NORM,
        MODEL_TENSOR.ATTN_NORM, # operator_norm
        MODEL_TENSOR.ATTN_Q_NORM,
        MODEL_TENSOR.ATTN_K_NORM,
        MODEL_TENSOR.ATTN_Q,
        MODEL_TENSOR.ATTN_K,
        MODEL_TENSOR.ATTN_V,
        MODEL_TENSOR.ATTN_OUT,
        MODEL_TENSOR.OUTPUT,
    ],
    MODEL_ARCH.LFM2MOE: [
        MODEL_TENSOR.TOKEN_EMBD,
        MODEL_TENSOR.TOKEN_EMBD_NORM,
        MODEL_TENSOR.SHORTCONV_CONV,
        MODEL_TENSOR.SHORTCONV_INPROJ,
        MODEL_TENSOR.SHORTCONV_OUTPROJ,
        MODEL_TENSOR.FFN_GATE,
        MODEL_TENSOR.FFN_DOWN,
        MODEL_TENSOR.FFN_UP,
        MODEL_TENSOR.FFN_NORM,
        MODEL_TENSOR.ATTN_NORM, # operator_norm
        MODEL_TENSOR.ATTN_Q_NORM,
        MODEL_TENSOR.ATTN_K_NORM,
        MODEL_TENSOR.ATTN_Q,
        MODEL_TENSOR.ATTN_K,
        MODEL_TENSOR.ATTN_V,
        MODEL_TENSOR.ATTN_OUT,
        MODEL_TENSOR.FFN_GATE_INP,
        MODEL_TENSOR.FFN_GATE_EXP,
        MODEL_TENSOR.FFN_DOWN_EXP,
        MODEL_TENSOR.FFN_UP_EXP,
        MODEL_TENSOR.FFN_EXP_PROBS_B,
    ],
    MODEL_ARCH.SMALLTHINKER: [
        MODEL_TENSOR.TOKEN_EMBD,
        MODEL_TENSOR.OUTPUT_NORM,
        MODEL_TENSOR.OUTPUT,
        MODEL_TENSOR.ATTN_NORM,
        MODEL_TENSOR.ATTN_Q,
        MODEL_TENSOR.ATTN_K,
        MODEL_TENSOR.ATTN_V,
        MODEL_TENSOR.ATTN_OUT,
        MODEL_TENSOR.FFN_NORM,
        MODEL_TENSOR.FFN_GATE,
        MODEL_TENSOR.FFN_DOWN,
        MODEL_TENSOR.FFN_UP,
        MODEL_TENSOR.FFN_GATE_INP,
        MODEL_TENSOR.FFN_GATE_EXP,
        MODEL_TENSOR.FFN_DOWN_EXP,
        MODEL_TENSOR.FFN_UP_EXP,
    ],
    MODEL_ARCH.APERTUS: [
        MODEL_TENSOR.TOKEN_EMBD,
        MODEL_TENSOR.OUTPUT_NORM,
        MODEL_TENSOR.OUTPUT,
        MODEL_TENSOR.ROPE_FREQS,
        MODEL_TENSOR.ATTN_NORM,
        MODEL_TENSOR.ATTN_Q,
        MODEL_TENSOR.ATTN_K,
        MODEL_TENSOR.ATTN_V,
        MODEL_TENSOR.ATTN_OUT,
        MODEL_TENSOR.ATTN_ROT_EMBD,
        MODEL_TENSOR.ATTN_Q_NORM,
        MODEL_TENSOR.ATTN_K_NORM,
        MODEL_TENSOR.FFN_NORM,
        MODEL_TENSOR.FFN_GATE,
        MODEL_TENSOR.FFN_DOWN,
        MODEL_TENSOR.FFN_UP,
    ],
    MODEL_ARCH.LLADA_MOE: [
        MODEL_TENSOR.TOKEN_EMBD,
        MODEL_TENSOR.OUTPUT_NORM,
        MODEL_TENSOR.OUTPUT,
        MODEL_TENSOR.ATTN_OUT,
        MODEL_TENSOR.ATTN_Q,
        MODEL_TENSOR.ATTN_K,
        MODEL_TENSOR.ATTN_V,
        MODEL_TENSOR.ATTN_NORM,
        MODEL_TENSOR.ATTN_Q_NORM,
        MODEL_TENSOR.ATTN_K_NORM,
        MODEL_TENSOR.FFN_NORM,
        MODEL_TENSOR.FFN_GATE_INP,
        MODEL_TENSOR.FFN_GATE_EXP,
        MODEL_TENSOR.FFN_UP_EXP,
        MODEL_TENSOR.FFN_DOWN_EXP,
    ],
    MODEL_ARCH.GROVEMOE: [
        MODEL_TENSOR.TOKEN_EMBD,
        MODEL_TENSOR.OUTPUT_NORM,
        MODEL_TENSOR.OUTPUT,
        MODEL_TENSOR.ATTN_NORM,
        MODEL_TENSOR.ATTN_Q,
        MODEL_TENSOR.ATTN_Q_NORM,
        MODEL_TENSOR.ATTN_K,
        MODEL_TENSOR.ATTN_K_NORM,
        MODEL_TENSOR.ATTN_V,
        MODEL_TENSOR.ATTN_OUT,
        MODEL_TENSOR.FFN_NORM,
        MODEL_TENSOR.FFN_GATE_INP,
        MODEL_TENSOR.FFN_GATE_EXP,
        MODEL_TENSOR.FFN_DOWN_EXP,
        MODEL_TENSOR.FFN_UP_EXP,
        MODEL_TENSOR.FFN_GATE_CHEXP,
        MODEL_TENSOR.FFN_DOWN_CHEXP,
        MODEL_TENSOR.FFN_UP_CHEXP,
    ],
    # TODO
}

# tensors that will not be serialized
MODEL_TENSOR_SKIP: dict[MODEL_ARCH, list[MODEL_TENSOR]] = {
    MODEL_ARCH.LLAMA: [
        MODEL_TENSOR.ROPE_FREQS,
        MODEL_TENSOR.ATTN_ROT_EMBD,
    ],
    MODEL_ARCH.DECI: [
        MODEL_TENSOR.ROPE_FREQS,
        MODEL_TENSOR.ATTN_ROT_EMBD,
    ],
    MODEL_ARCH.BAICHUAN: [
        MODEL_TENSOR.ROPE_FREQS,
        MODEL_TENSOR.ATTN_ROT_EMBD,
    ],
    MODEL_ARCH.QWEN: [
        MODEL_TENSOR.ROPE_FREQS,
        MODEL_TENSOR.ATTN_ROT_EMBD,
    ],
    MODEL_ARCH.CODESHELL: [
        MODEL_TENSOR.ROPE_FREQS,
        MODEL_TENSOR.ATTN_ROT_EMBD,
    ],
    MODEL_ARCH.ORION: [
        MODEL_TENSOR.ROPE_FREQS,
        MODEL_TENSOR.ATTN_ROT_EMBD,
    ],
    MODEL_ARCH.STARCODER2: [
        MODEL_TENSOR.ROPE_FREQS,
        MODEL_TENSOR.ATTN_ROT_EMBD,
    ],
    MODEL_ARCH.XVERSE: [
        MODEL_TENSOR.ROPE_FREQS,
        MODEL_TENSOR.ATTN_ROT_EMBD,
    ],
    MODEL_ARCH.DEEPSEEK: [
        MODEL_TENSOR.ROPE_FREQS,
        MODEL_TENSOR.ATTN_ROT_EMBD,
    ],
    MODEL_ARCH.DEEPSEEK2: [
        MODEL_TENSOR.ROPE_FREQS,
        MODEL_TENSOR.ATTN_ROT_EMBD,
    ],
    MODEL_ARCH.CHATGLM: [
        MODEL_TENSOR.ROPE_FREQS,
    ],
    MODEL_ARCH.NEMOTRON: [
        MODEL_TENSOR.ROPE_FREQS,
        MODEL_TENSOR.ATTN_ROT_EMBD,
    ],
    MODEL_ARCH.BAILINGMOE: [
        MODEL_TENSOR.ROPE_FREQS,
    ],
}

#
# types
#


class TokenType(IntEnum):
    NORMAL       = 1
    UNKNOWN      = 2
    CONTROL      = 3
    USER_DEFINED = 4
    UNUSED       = 5
    BYTE         = 6


class RopeScalingType(Enum):
    NONE     = 'none'
    LINEAR   = 'linear'
    YARN     = 'yarn'
    LONGROPE = 'longrope'


class PoolingType(IntEnum):
    NONE = 0
    MEAN = 1
    CLS  = 2
    LAST = 3
    RANK = 4


class GGMLQuantizationType(IntEnum):
    F32     = 0
    F16     = 1
    Q4_0    = 2
    Q4_1    = 3
    Q5_0    = 6
    Q5_1    = 7
    Q8_0    = 8
    Q8_1    = 9
    Q2_K    = 10
    Q3_K    = 11
    Q4_K    = 12
    Q5_K    = 13
    Q6_K    = 14
    Q8_K    = 15
    IQ2_XXS = 16
    IQ2_XS  = 17
    IQ3_XXS = 18
    IQ1_S   = 19
    IQ4_NL  = 20
    IQ3_S   = 21
    IQ2_S   = 22
    IQ4_XS  = 23
    I8      = 24
    I16     = 25
    I32     = 26
    I64     = 27
    F64     = 28
    IQ1_M   = 29
    BF16    = 30
    TQ1_0   = 34
    TQ2_0   = 35
    MXFP4   = 39


class ExpertGatingFuncType(IntEnum):
    SOFTMAX  = 1
    SIGMOID  = 2


# TODO: add GGMLFileType from ggml_ftype in ggml.h


# from llama_ftype in llama.h
# ALL VALUES SHOULD BE THE SAME HERE AS THEY ARE OVER THERE.
class LlamaFileType(IntEnum):
    ALL_F32              = 0
    MOSTLY_F16           = 1   # except 1d tensors
    MOSTLY_Q4_0          = 2   # except 1d tensors
    MOSTLY_Q4_1          = 3   # except 1d tensors
    # MOSTLY_Q4_1_SOME_F16 = 4   # tok_embeddings.weight and output.weight are F16
    # MOSTLY_Q4_2        = 5   # support has been removed
    # MOSTLY_Q4_3        = 6   # support has been removed
    MOSTLY_Q8_0          = 7   # except 1d tensors
    MOSTLY_Q5_0          = 8   # except 1d tensors
    MOSTLY_Q5_1          = 9   # except 1d tensors
    MOSTLY_Q2_K          = 10  # except 1d tensors
    MOSTLY_Q3_K_S        = 11  # except 1d tensors
    MOSTLY_Q3_K_M        = 12  # except 1d tensors
    MOSTLY_Q3_K_L        = 13  # except 1d tensors
    MOSTLY_Q4_K_S        = 14  # except 1d tensors
    MOSTLY_Q4_K_M        = 15  # except 1d tensors
    MOSTLY_Q5_K_S        = 16  # except 1d tensors
    MOSTLY_Q5_K_M        = 17  # except 1d tensors
    MOSTLY_Q6_K          = 18  # except 1d tensors
    MOSTLY_IQ2_XXS       = 19  # except 1d tensors
    MOSTLY_IQ2_XS        = 20  # except 1d tensors
    MOSTLY_Q2_K_S        = 21  # except 1d tensors
    MOSTLY_IQ3_XS        = 22  # except 1d tensors
    MOSTLY_IQ3_XXS       = 23  # except 1d tensors
    MOSTLY_IQ1_S         = 24  # except 1d tensors
    MOSTLY_IQ4_NL        = 25  # except 1d tensors
    MOSTLY_IQ3_S         = 26  # except 1d tensors
    MOSTLY_IQ3_M         = 27  # except 1d tensors
    MOSTLY_IQ2_S         = 28  # except 1d tensors
    MOSTLY_IQ2_M         = 29  # except 1d tensors
    MOSTLY_IQ4_XS        = 30  # except 1d tensors
    MOSTLY_IQ1_M         = 31  # except 1d tensors
    MOSTLY_BF16          = 32  # except 1d tensors
    # MOSTLY_Q4_0_4_4      = 33  # removed from gguf files, use Q4_0 and runtime repack
    # MOSTLY_Q4_0_4_8      = 34  # removed from gguf files, use Q4_0 and runtime repack
    # MOSTLY_Q4_0_8_8      = 35  # removed from gguf files, use Q4_0 and runtime repack
    MOSTLY_TQ1_0         = 36  # except 1d tensors
    MOSTLY_TQ2_0         = 37  # except 1d tensors

    GUESSED              = 1024  # not specified in the model file


class GGUFEndian(IntEnum):
    LITTLE = 0
    BIG = 1


class GGUFValueType(IntEnum):
    UINT8   = 0
    INT8    = 1
    UINT16  = 2
    INT16   = 3
    UINT32  = 4
    INT32   = 5
    FLOAT32 = 6
    BOOL    = 7
    STRING  = 8
    ARRAY   = 9
    UINT64  = 10
    INT64   = 11
    FLOAT64 = 12

    @staticmethod
    def get_type(val: Any) -> GGUFValueType:
        if isinstance(val, (str, bytes, bytearray)):
            return GGUFValueType.STRING
        elif isinstance(val, list):
            return GGUFValueType.ARRAY
        elif isinstance(val, float):
            return GGUFValueType.FLOAT32
        elif isinstance(val, bool):
            return GGUFValueType.BOOL
        elif isinstance(val, int):
            return GGUFValueType.INT32
        # TODO: need help with 64-bit types in Python
        else:
            raise ValueError(f"Unknown type: {type(val)}")


class VisionProjectorType:
    GEMMA3 = "gemma3"
    IDEFICS3 = "idefics3"
    PIXTRAL = "pixtral"
    LLAMA4 = "llama4"
    QWEN2VL = "qwen2vl_merger"
    QWEN25VL = "qwen2.5vl_merger"
    ULTRAVOX = "ultravox"
    INTERNVL = "internvl"
    QWEN2A = "qwen2a" # audio
    QWEN25O = "qwen2.5o" # omni
    VOXTRAL = "voxtral"
    LFM2 = "lfm2"
    KIMIVL = "kimivl"


# Items here are (block size, type size)
QK_K = 256
GGML_QUANT_SIZES: dict[GGMLQuantizationType, tuple[int, int]] = {
    GGMLQuantizationType.F32:     (1, 4),
    GGMLQuantizationType.F16:     (1, 2),
    GGMLQuantizationType.Q4_0:    (32, 2 + 16),
    GGMLQuantizationType.Q4_1:    (32, 2 + 2 + 16),
    GGMLQuantizationType.Q5_0:    (32, 2 + 4 + 16),
    GGMLQuantizationType.Q5_1:    (32, 2 + 2 + 4 + 16),
    GGMLQuantizationType.Q8_0:    (32, 2 + 32),
    GGMLQuantizationType.Q8_1:    (32, 4 + 4 + 32),
    GGMLQuantizationType.Q2_K:    (256, 2 + 2 + QK_K // 16 + QK_K // 4),
    GGMLQuantizationType.Q3_K:    (256, 2 + QK_K // 4 + QK_K // 8 + 12),
    GGMLQuantizationType.Q4_K:    (256, 2 + 2 + QK_K // 2 + 12),
    GGMLQuantizationType.Q5_K:    (256, 2 + 2 + QK_K // 2 + QK_K // 8 + 12),
    GGMLQuantizationType.Q6_K:    (256, 2 + QK_K // 2 + QK_K // 4 + QK_K // 16),
    GGMLQuantizationType.Q8_K:    (256, 4 + QK_K + QK_K // 8),
    GGMLQuantizationType.IQ2_XXS: (256, 2 + QK_K // 4),
    GGMLQuantizationType.IQ2_XS:  (256, 2 + QK_K // 4 + QK_K // 32),
    GGMLQuantizationType.IQ3_XXS: (256, 2 + QK_K // 4 + QK_K // 8),
    GGMLQuantizationType.IQ1_S:   (256, 2 + QK_K // 8 + QK_K // 16),
    GGMLQuantizationType.IQ4_NL:  (32, 2 + 16),
    GGMLQuantizationType.IQ3_S:   (256, 2 + QK_K // 4 + QK_K // 8 + QK_K // 32 + 4),
    GGMLQuantizationType.IQ2_S:   (256, 2 + QK_K // 4 + QK_K // 16),
    GGMLQuantizationType.IQ4_XS:  (256, 2 + 2 + QK_K // 2 + QK_K // 64),
    GGMLQuantizationType.I8:      (1, 1),
    GGMLQuantizationType.I16:     (1, 2),
    GGMLQuantizationType.I32:     (1, 4),
    GGMLQuantizationType.I64:     (1, 8),
    GGMLQuantizationType.F64:     (1, 8),
    GGMLQuantizationType.IQ1_M:   (256, QK_K // 8 + QK_K // 16  + QK_K // 32),
    GGMLQuantizationType.BF16:    (1, 2),
    GGMLQuantizationType.TQ1_0:   (256, 2 + 4 * 13),
    GGMLQuantizationType.TQ2_0:   (256, 2 + 64),
    GGMLQuantizationType.MXFP4:   (32, 1 + 16),
}


# Aliases for backward compatibility.

# general
KEY_GENERAL_ARCHITECTURE         = Keys.General.ARCHITECTURE
KEY_GENERAL_QUANTIZATION_VERSION = Keys.General.QUANTIZATION_VERSION
KEY_GENERAL_ALIGNMENT            = Keys.General.ALIGNMENT
KEY_GENERAL_NAME                 = Keys.General.NAME
KEY_GENERAL_AUTHOR               = Keys.General.AUTHOR
KEY_GENERAL_URL                  = Keys.General.URL
KEY_GENERAL_DESCRIPTION          = Keys.General.DESCRIPTION
KEY_GENERAL_LICENSE              = Keys.General.LICENSE
KEY_GENERAL_SOURCE_URL           = Keys.General.SOURCE_URL
KEY_GENERAL_FILE_TYPE            = Keys.General.FILE_TYPE

# LLM
KEY_VOCAB_SIZE            = Keys.LLM.VOCAB_SIZE
KEY_CONTEXT_LENGTH        = Keys.LLM.CONTEXT_LENGTH
KEY_EMBEDDING_LENGTH      = Keys.LLM.EMBEDDING_LENGTH
KEY_BLOCK_COUNT           = Keys.LLM.BLOCK_COUNT
KEY_FEED_FORWARD_LENGTH   = Keys.LLM.FEED_FORWARD_LENGTH
KEY_USE_PARALLEL_RESIDUAL = Keys.LLM.USE_PARALLEL_RESIDUAL
KEY_TENSOR_DATA_LAYOUT    = Keys.LLM.TENSOR_DATA_LAYOUT

# attention
KEY_ATTENTION_HEAD_COUNT        = Keys.Attention.HEAD_COUNT
KEY_ATTENTION_HEAD_COUNT_KV     = Keys.Attention.HEAD_COUNT_KV
KEY_ATTENTION_MAX_ALIBI_BIAS    = Keys.Attention.MAX_ALIBI_BIAS
KEY_ATTENTION_CLAMP_KQV         = Keys.Attention.CLAMP_KQV
KEY_ATTENTION_LAYERNORM_EPS     = Keys.Attention.LAYERNORM_EPS
KEY_ATTENTION_LAYERNORM_RMS_EPS = Keys.Attention.LAYERNORM_RMS_EPS

# RoPE
KEY_ROPE_DIMENSION_COUNT      = Keys.Rope.DIMENSION_COUNT
KEY_ROPE_FREQ_BASE            = Keys.Rope.FREQ_BASE
KEY_ROPE_SCALING_TYPE         = Keys.Rope.SCALING_TYPE
KEY_ROPE_SCALING_FACTOR       = Keys.Rope.SCALING_FACTOR
KEY_ROPE_SCALING_ORIG_CTX_LEN = Keys.Rope.SCALING_ORIG_CTX_LEN
KEY_ROPE_SCALING_FINETUNED    = Keys.Rope.SCALING_FINETUNED

# SSM
KEY_SSM_CONV_KERNEL    = Keys.SSM.CONV_KERNEL
KEY_SSM_INNER_SIZE     = Keys.SSM.INNER_SIZE
KEY_SSM_STATE_SIZE     = Keys.SSM.STATE_SIZE
KEY_SSM_TIME_STEP_RANK = Keys.SSM.TIME_STEP_RANK
KEY_SSM_GROUP_COUNT    = Keys.SSM.GROUP_COUNT
KEY_SSM_DT_B_C_RMS     = Keys.SSM.DT_B_C_RMS

# tokenization
KEY_TOKENIZER_MODEL      = Keys.Tokenizer.MODEL
KEY_TOKENIZER_PRE        = Keys.Tokenizer.PRE
KEY_TOKENIZER_LIST       = Keys.Tokenizer.LIST
KEY_TOKENIZER_TOKEN_TYPE = Keys.Tokenizer.TOKEN_TYPE
KEY_TOKENIZER_SCORES     = Keys.Tokenizer.SCORES
KEY_TOKENIZER_MERGES     = Keys.Tokenizer.MERGES
KEY_TOKENIZER_BOS_ID     = Keys.Tokenizer.BOS_ID
KEY_TOKENIZER_EOS_ID     = Keys.Tokenizer.EOS_ID
KEY_TOKENIZER_EOT_ID     = Keys.Tokenizer.EOT_ID
KEY_TOKENIZER_EOM_ID     = Keys.Tokenizer.EOM_ID
KEY_TOKENIZER_UNK_ID     = Keys.Tokenizer.UNK_ID
KEY_TOKENIZER_SEP_ID     = Keys.Tokenizer.SEP_ID
KEY_TOKENIZER_PAD_ID     = Keys.Tokenizer.PAD_ID
KEY_TOKENIZER_MASK_ID    = Keys.Tokenizer.MASK_ID
KEY_TOKENIZER_HF_JSON    = Keys.Tokenizer.HF_JSON
KEY_TOKENIZER_RWKV       = Keys.Tokenizer.RWKV

KEY_TOKENIZER_FIM_PRE_ID = Keys.Tokenizer.FIM_PRE_ID
KEY_TOKENIZER_FIM_SUF_ID = Keys.Tokenizer.FIM_SUF_ID
KEY_TOKENIZER_FIM_MID_ID = Keys.Tokenizer.FIM_MID_ID
KEY_TOKENIZER_FIM_PAD_ID = Keys.Tokenizer.FIM_PAD_ID
KEY_TOKENIZER_FIM_REP_ID = Keys.Tokenizer.FIM_REP_ID
KEY_TOKENIZER_FIM_SEP_ID = Keys.Tokenizer.FIM_SEP_ID

# deprecated
KEY_TOKENIZER_PREFIX_ID  = Keys.Tokenizer.PREFIX_ID
KEY_TOKENIZER_SUFFIX_ID  = Keys.Tokenizer.SUFFIX_ID
KEY_TOKENIZER_MIDDLE_ID  = Keys.Tokenizer.MIDDLE_ID
