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
        USE_PARALLEL_RESIDUAL             = "{arch}.use_parallel_residual"
        TENSOR_DATA_LAYOUT                = "{arch}.tensor_data_layout"
        EXPERT_COUNT                      = "{arch}.expert_count"
        EXPERT_USED_COUNT                 = "{arch}.expert_used_count"
        EXPERT_SHARED_COUNT               = "{arch}.expert_shared_count"
        EXPERT_WEIGHTS_SCALE              = "{arch}.expert_weights_scale"
        EXPERT_WEIGHTS_NORM               = "{arch}.expert_weights_norm"
        EXPERT_GATING_FUNC                = "{arch}.expert_gating_func"
        POOLING_TYPE                      = "{arch}.pooling_type"
        LOGIT_SCALE                       = "{arch}.logit_scale"
        DECODER_START_TOKEN_ID            = "{arch}.decoder_start_token_id"
        ATTN_LOGIT_SOFTCAPPING            = "{arch}.attn_logit_softcapping"
        FINAL_LOGIT_SOFTCAPPING           = "{arch}.final_logit_softcapping"
        SWIN_NORM                         = "{arch}.swin_norm"
        RESCALE_EVERY_N_LAYERS            = "{arch}.rescale_every_n_layers"
        TIME_MIX_EXTRA_DIM                = "{arch}.time_mix_extra_dim"
        TIME_DECAY_EXTRA_DIM              = "{arch}.time_decay_extra_dim"
        RESIDUAL_SCALE                    = "{arch}.residual_scale"
        EMBEDDING_SCALE                   = "{arch}.embedding_scale"
        TOKEN_SHIFT_COUNT                 = "{arch}.token_shift_count"

    class Attention:
        HEAD_COUNT        = "{arch}.attention.head_count"
        HEAD_COUNT_KV     = "{arch}.attention.head_count_kv"
        MAX_ALIBI_BIAS    = "{arch}.attention.max_alibi_bias"
        CLAMP_KQV         = "{arch}.attention.clamp_kqv"
        KEY_LENGTH        = "{arch}.attention.key_length"
        VALUE_LENGTH      = "{arch}.attention.value_length"
        LAYERNORM_EPS     = "{arch}.attention.layer_norm_epsilon"
        LAYERNORM_RMS_EPS = "{arch}.attention.layer_norm_rms_epsilon"
        GROUPNORM_EPS     = "{arch}.attention.group_norm_epsilon"
        GROUPNORM_GROUPS  = "{arch}.attention.group_norm_groups"
        CAUSAL            = "{arch}.attention.causal"
        Q_LORA_RANK       = "{arch}.attention.q_lora_rank"
        KV_LORA_RANK      = "{arch}.attention.kv_lora_rank"
        REL_BUCKETS_COUNT = "{arch}.attention.relative_buckets_count"
        SLIDING_WINDOW    = "{arch}.attention.sliding_window"
        SCALE             = "{arch}.attention.scale"

    class Rope:
        DIMENSION_COUNT         = "{arch}.rope.dimension_count"
        DIMENSION_SECTIONS      = "{arch}.rope.dimension_sections"
        FREQ_BASE               = "{arch}.rope.freq_base"
        SCALING_TYPE            = "{arch}.rope.scaling.type"
        SCALING_FACTOR          = "{arch}.rope.scaling.factor"
        SCALING_ATTN_FACTOR     = "{arch}.rope.scaling.attn_factor"
        SCALING_ORIG_CTX_LEN    = "{arch}.rope.scaling.original_context_length"
        SCALING_FINETUNED       = "{arch}.rope.scaling.finetuned"
        SCALING_YARN_LOG_MUL    = "{arch}.rope.scaling.yarn_log_multiplier"

    class Split:
        LLM_KV_SPLIT_NO            = "split.no"
        LLM_KV_SPLIT_COUNT         = "split.count"
        LLM_KV_SPLIT_TENSORS_COUNT = "split.tensors.count"

    class SSM:
        CONV_KERNEL    = "{arch}.ssm.conv_kernel"
        INNER_SIZE     = "{arch}.ssm.inner_size"
        STATE_SIZE     = "{arch}.ssm.state_size"
        TIME_STEP_RANK = "{arch}.ssm.time_step_rank"
        DT_B_C_RMS     = "{arch}.ssm.dt_b_c_rms"

    class WKV:
        HEAD_SIZE = "{arch}.wkv.head_size"

    class PosNet:
        EMBEDDING_LENGTH = "{arch}.posnet.embedding_length"
        BLOCK_COUNT      = "{arch}.posnet.block_count"

    class ConvNext:
        EMBEDDING_LENGTH = "{arch}.convnext.embedding_length"
        BLOCK_COUNT      = "{arch}.convnext.block_count"

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
        TYPE       = "adapter.type"
        LORA_ALPHA = "adapter.lora.alpha"

#
# recommended mapping of model tensor names for storage in gguf
#


class GGUFType:
    MODEL   = "model"
    ADAPTER = "adapter"


class MODEL_ARCH(IntEnum):
    LLAMA            = auto()
    DECI             = auto()
    FALCON           = auto()
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
    JINA_BERT_V2     = auto()
    BLOOM            = auto()
    STABLELM         = auto()
    QWEN             = auto()
    QWEN2            = auto()
    QWEN2MOE         = auto()
    QWEN2VL          = auto()
    PHI2             = auto()
    PHI3             = auto()
    PHIMOE           = auto()
    PLAMO            = auto()
    CODESHELL        = auto()
    ORION            = auto()
    INTERNLM2        = auto()
    MINICPM          = auto()
    MINICPM3         = auto()
    GEMMA            = auto()
    GEMMA2           = auto()
    STARCODER2       = auto()
    RWKV6            = auto()
    RWKV6QWEN2       = auto()
    MAMBA            = auto()
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
    BITNET           = auto()
    T5               = auto()
    T5ENCODER        = auto()
    JAIS             = auto()
    NEMOTRON         = auto()
    EXAONE           = auto()
    GRANITE          = auto()
    GRANITE_MOE      = auto()
    CHAMELEON        = auto()
    WAVTOKENIZER_DEC = auto()


class MODEL_TENSOR(IntEnum):
    TOKEN_EMBD           = auto()
    TOKEN_EMBD_NORM      = auto()
    TOKEN_TYPES          = auto()
    POS_EMBD             = auto()
    OUTPUT               = auto()
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
    FFN_EXP_PROBS_B      = auto()
    ATTN_Q_NORM          = auto()
    ATTN_K_NORM          = auto()
    LAYER_OUT_NORM       = auto()
    SSM_IN               = auto()
    SSM_CONV1D           = auto()
    SSM_X                = auto()
    SSM_DT               = auto()
    SSM_A                = auto()
    SSM_D                = auto()
    SSM_OUT              = auto()
    TIME_MIX_W1          = auto()
    TIME_MIX_W2          = auto()
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


MODEL_ARCH_NAMES: dict[MODEL_ARCH, str] = {
    MODEL_ARCH.LLAMA:            "llama",
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
    MODEL_ARCH.JINA_BERT_V2:     "jina-bert-v2",
    MODEL_ARCH.BLOOM:            "bloom",
    MODEL_ARCH.STABLELM:         "stablelm",
    MODEL_ARCH.QWEN:             "qwen",
    MODEL_ARCH.QWEN2:            "qwen2",
    MODEL_ARCH.QWEN2MOE:         "qwen2moe",
    MODEL_ARCH.QWEN2VL:          "qwen2vl",
    MODEL_ARCH.PHI2:             "phi2",
    MODEL_ARCH.PHI3:             "phi3",
    MODEL_ARCH.PHIMOE:           "phimoe",
    MODEL_ARCH.PLAMO:            "plamo",
    MODEL_ARCH.CODESHELL:        "codeshell",
    MODEL_ARCH.ORION:            "orion",
    MODEL_ARCH.INTERNLM2:        "internlm2",
    MODEL_ARCH.MINICPM:          "minicpm",
    MODEL_ARCH.MINICPM3:         "minicpm3",
    MODEL_ARCH.GEMMA:            "gemma",
    MODEL_ARCH.GEMMA2:           "gemma2",
    MODEL_ARCH.STARCODER2:       "starcoder2",
    MODEL_ARCH.RWKV6:            "rwkv6",
    MODEL_ARCH.RWKV6QWEN2:       "rwkv6qwen2",
    MODEL_ARCH.MAMBA:            "mamba",
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
    MODEL_ARCH.BITNET:           "bitnet",
    MODEL_ARCH.T5:               "t5",
    MODEL_ARCH.T5ENCODER:        "t5encoder",
    MODEL_ARCH.JAIS:             "jais",
    MODEL_ARCH.NEMOTRON:         "nemotron",
    MODEL_ARCH.EXAONE:           "exaone",
    MODEL_ARCH.GRANITE:          "granite",
    MODEL_ARCH.GRANITE_MOE:      "granitemoe",
    MODEL_ARCH.CHAMELEON:        "chameleon",
    MODEL_ARCH.WAVTOKENIZER_DEC: "wavtokenizer-dec",
}

TENSOR_NAMES: dict[MODEL_TENSOR, str] = {
    MODEL_TENSOR.TOKEN_EMBD:                "token_embd",
    MODEL_TENSOR.TOKEN_EMBD_NORM:           "token_embd_norm",
    MODEL_TENSOR.TOKEN_TYPES:               "token_types",
    MODEL_TENSOR.POS_EMBD:                  "position_embd",
    MODEL_TENSOR.OUTPUT_NORM:               "output_norm",
    MODEL_TENSOR.OUTPUT:                    "output",
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
    MODEL_TENSOR.FFN_ACT:                   "blk.{bid}.ffn",
    MODEL_TENSOR.FFN_NORM_EXP:              "blk.{bid}.ffn_norm_exps",
    MODEL_TENSOR.FFN_GATE_EXP:              "blk.{bid}.ffn_gate_exps",
    MODEL_TENSOR.FFN_DOWN_EXP:              "blk.{bid}.ffn_down_exps",
    MODEL_TENSOR.FFN_UP_EXP:                "blk.{bid}.ffn_up_exps",
    MODEL_TENSOR.FFN_EXP_PROBS_B:           "blk.{bid}.exp_probs_b",
    MODEL_TENSOR.LAYER_OUT_NORM:            "blk.{bid}.layer_output_norm",
    MODEL_TENSOR.SSM_IN:                    "blk.{bid}.ssm_in",
    MODEL_TENSOR.SSM_CONV1D:                "blk.{bid}.ssm_conv1d",
    MODEL_TENSOR.SSM_X:                     "blk.{bid}.ssm_x",
    MODEL_TENSOR.SSM_DT:                    "blk.{bid}.ssm_dt",
    MODEL_TENSOR.SSM_A:                     "blk.{bid}.ssm_a",
    MODEL_TENSOR.SSM_D:                     "blk.{bid}.ssm_d",
    MODEL_TENSOR.SSM_OUT:                   "blk.{bid}.ssm_out",
    MODEL_TENSOR.TIME_MIX_W1:               "blk.{bid}.time_mix_w1",
    MODEL_TENSOR.TIME_MIX_W2:               "blk.{bid}.time_mix_w2",
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
}

MODEL_TENSORS: dict[MODEL_ARCH, list[MODEL_TENSOR]] = {
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
