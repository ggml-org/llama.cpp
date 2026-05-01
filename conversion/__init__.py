from __future__ import annotations

from .base import (
    ModelBase, TextModel, MmprojModel, ModelType, SentencePieceTokenTypes,
    logger, _mistral_common_installed, _mistral_import_error_msg,
    get_model_architecture, LazyTorchTensor,
)
from typing import Type


__all__ = [
    "ModelBase", "TextModel", "MmprojModel", "ModelType", "SentencePieceTokenTypes",
    "get_model_architecture", "LazyTorchTensor", "logger",
    "_mistral_common_installed", "_mistral_import_error_msg",
    "get_model_class", "print_registered_models", "load_all_models",
]


TEXT_MODEL_MAP: dict[str, str] = {
    "AfmoeForCausalLM": "afmoe",
    "ApertusForCausalLM": "llama",
    "ArceeForCausalLM": "llama",
    "ArcticForCausalLM": "arctic",
    "AudioFlamingo3ForConditionalGeneration": "qwen",
    "BaiChuanForCausalLM": "baichuan",
    "BaichuanForCausalLM": "baichuan",
    "BailingMoeForCausalLM": "bailing",
    "BailingMoeV2ForCausalLM": "bailing",
    "BambaForCausalLM": "granite",
    "BertForMaskedLM": "bert",
    "BertForSequenceClassification": "bert",
    "BertModel": "bert",
    "BitnetForCausalLM": "bitnet",
    "BloomForCausalLM": "bloom",
    "BloomModel": "bloom",
    "CamembertModel": "bert",
    "ChameleonForCausalLM": "chameleon",
    "ChameleonForConditionalGeneration": "chameleon",
    "ChatGLMForConditionalGeneration": "chatglm",
    "ChatGLMModel": "chatglm",
    "CodeShellForCausalLM": "codeshell",
    "CogVLMForCausalLM": "cogvlm",
    "Cohere2ForCausalLM": "command_r",
    "CohereForCausalLM": "command_r",
    "DbrxForCausalLM": "dbrx",
    "DeciLMForCausalLM": "deci",
    "DeepseekForCausalLM": "deepseek",
    "DeepseekV2ForCausalLM": "deepseek",
    "DeepseekV3ForCausalLM": "deepseek",
    "DistilBertForMaskedLM": "bert",
    "DistilBertForSequenceClassification": "bert",
    "DistilBertModel": "bert",
    "Dots1ForCausalLM": "dots1",
    "DotsOCRForCausalLM": "qwen",
    "DreamModel": "dream",
    "Ernie4_5ForCausalLM": "ernie",
    "Ernie4_5_ForCausalLM": "ernie",
    "Ernie4_5_MoeForCausalLM": "ernie",
    "EuroBertModel": "bert",
    "Exaone4ForCausalLM": "exaone",
    "ExaoneForCausalLM": "exaone",
    "ExaoneMoEForCausalLM": "exaone",
    "FalconForCausalLM": "falcon",
    "FalconH1ForCausalLM": "falcon_h1",
    "FalconMambaForCausalLM": "mamba",
    "GPT2LMHeadModel": "gpt2",
    "GPTBigCodeForCausalLM": "starcoder",
    "GPTNeoXForCausalLM": "gpt_neox",
    "GPTRefactForCausalLM": "refact",
    "Gemma2ForCausalLM": "gemma",
    "Gemma3ForCausalLM": "gemma",
    "Gemma3ForConditionalGeneration": "gemma",
    "Gemma3TextModel": "gemma",
    "Gemma3nForCausalLM": "gemma",
    "Gemma3nForConditionalGeneration": "gemma",
    "Gemma4ForConditionalGeneration": "gemma",
    "GemmaForCausalLM": "gemma",
    "Glm4ForCausalLM": "glm",
    "Glm4MoeForCausalLM": "glm",
    "Glm4MoeLiteForCausalLM": "glm",
    "Glm4vForConditionalGeneration": "glm",
    "Glm4vMoeForConditionalGeneration": "glm",
    "GlmForCausalLM": "chatglm",
    "GlmMoeDsaForCausalLM": "glm",
    "GlmOcrForConditionalGeneration": "glm",
    "GptOssForCausalLM": "gpt_oss",
    "GraniteForCausalLM": "granite",
    "GraniteMoeForCausalLM": "granite",
    "GraniteMoeHybridForCausalLM": "granite",
    "GraniteMoeSharedForCausalLM": "granite",
    "Grok1ForCausalLM": "grok",
    "GrokForCausalLM": "grok",
    "GroveMoeForCausalLM": "grove",
    "HunYuanDenseV1ForCausalLM": "hunyuan",
    "HunYuanMoEV1ForCausalLM": "hunyuan",
    "HunYuanVLForConditionalGeneration": "hunyuan",
    "IQuestCoderForCausalLM": "llama",
    "InternLM2ForCausalLM": "internlm",
    "InternLM3ForCausalLM": "internlm",
    "JAISLMHeadModel": "jais",
    "Jais2ForCausalLM": "jais",
    "JambaForCausalLM": "jamba",
    "JanusForConditionalGeneration": "janus_pro",
    "JinaBertForMaskedLM": "bert",
    "JinaBertModel": "bert",
    "JinaEmbeddingsV5Model": "bert",
    "KORMoForCausalLM": "qwen",
    "KimiK25ForConditionalGeneration": "deepseek",
    "KimiLinearForCausalLM": "kimi_linear",
    "KimiLinearModel": "kimi_linear",
    "KimiVLForConditionalGeneration": "deepseek",
    "LFM2ForCausalLM": "lfm2",
    "LLaDAMoEModel": "llada",
    "LLaDAMoEModelLM": "llada",
    "LLaDAModelLM": "llada",
    "LLaMAForCausalLM": "llama",
    "Lfm25AudioTokenizer": "lfm2",
    "Lfm2ForCausalLM": "lfm2",
    "Lfm2Model": "lfm2",
    "Lfm2MoeForCausalLM": "lfm2",
    "Llama4ForCausalLM": "llama",
    "Llama4ForConditionalGeneration": "llama",
    "LlamaBidirectionalModel": "llama",
    "LlamaForCausalLM": "llama",
    "LlamaModel": "llama",
    "LlavaForConditionalGeneration": "llama",
    "LlavaStableLMEpochForCausalLM": "stable_lm",
    "MPTForCausalLM": "mpt",
    "MT5ForConditionalGeneration": "t5",
    "MaincoderForCausalLM": "maincoder",
    "Mamba2ForCausalLM": "mamba",
    "MambaForCausalLM": "mamba",
    "MambaLMHeadModel": "mamba",
    "MiMoV2FlashForCausalLM": "mimo",
    "MiniCPM3ForCausalLM": "minicpm",
    "MiniCPMForCausalLM": "minicpm",
    "MiniMaxM2ForCausalLM": "minimax",
    "Ministral3ForCausalLM": "mistral3",
    "Mistral3ForConditionalGeneration": "mistral3",
    "MistralForCausalLM": "llama",
    "MixtralForCausalLM": "llama",
    "ModernBertForMaskedLM": "bert",
    "ModernBertForSequenceClassification": "bert",
    "ModernBertModel": "bert",
    "NemotronForCausalLM": "nemotron",
    "NemotronHForCausalLM": "nemotron",
    "NeoBERT": "bert",
    "NeoBERTForSequenceClassification": "bert",
    "NeoBERTLMHead": "bert",
    "NomicBertModel": "bert",
    "OLMoForCausalLM": "olmo",
    "Olmo2ForCausalLM": "olmo",
    "Olmo3ForCausalLM": "olmo",
    "OlmoForCausalLM": "olmo",
    "OlmoeForCausalLM": "olmo",
    "OpenELMForCausalLM": "openelm",
    "OrionForCausalLM": "orion",
    "PLMForCausalLM": "plm",
    "PLaMo2ForCausalLM": "plamo",
    "PLaMo3ForCausalLM": "plamo",
    "PaddleOCRVLForConditionalGeneration": "ernie",
    "PanguEmbeddedForCausalLM": "pangu",
    "Phi3ForCausalLM": "phi",
    "Phi4ForCausalLMV": "phi",
    "PhiForCausalLM": "phi",
    "PhiMoEForCausalLM": "phi",
    "Plamo2ForCausalLM": "plamo",
    "Plamo3ForCausalLM": "plamo",
    "PlamoForCausalLM": "plamo",
    "QWenLMHeadModel": "qwen",
    "Qwen2AudioForConditionalGeneration": "qwen",
    "Qwen2ForCausalLM": "qwen",
    "Qwen2Model": "qwen",
    "Qwen2MoeForCausalLM": "qwen",
    "Qwen2VLForConditionalGeneration": "qwen_vl",
    "Qwen2VLModel": "qwen_vl",
    "Qwen2_5OmniModel": "qwen_vl",
    "Qwen2_5_VLForConditionalGeneration": "qwen_vl",
    "Qwen3ASRForConditionalGeneration": "qwen3_vl",
    "Qwen3ForCausalLM": "qwen",
    "Qwen3Model": "qwen",
    "Qwen3MoeForCausalLM": "qwen",
    "Qwen3NextForCausalLM": "qwen",
    "Qwen3OmniMoeForConditionalGeneration": "qwen3_vl",
    "Qwen3VLForConditionalGeneration": "qwen3_vl",
    "Qwen3VLMoeForConditionalGeneration": "qwen3_vl",
    "Qwen3_5ForCausalLM": "qwen",
    "Qwen3_5ForConditionalGeneration": "qwen",
    "Qwen3_5MoeForCausalLM": "qwen",
    "Qwen3_5MoeForConditionalGeneration": "qwen",
    "RND1": "qwen",
    "RWForCausalLM": "falcon",
    "RWKV6Qwen2ForCausalLM": "rwkv",
    "RWKV7ForCausalLM": "rwkv",
    "RobertaForSequenceClassification": "bert",
    "RobertaModel": "bert",
    "RuGPT3XLForCausalLM": "gpt2",
    "Rwkv6ForCausalLM": "rwkv",
    "Rwkv7ForCausalLM": "rwkv",
    "RwkvHybridForCausalLM": "rwkv",
    "SeedOssForCausalLM": "olmo",
    "SmallThinkerForCausalLM": "small_thinker",
    "SmolLM3ForCausalLM": "llama",
    "SolarOpenForCausalLM": "glm",
    "StableLMEpochForCausalLM": "stable_lm",
    "StableLmForCausalLM": "stable_lm",
    "Starcoder2ForCausalLM": "starcoder",
    "Step3p5ForCausalLM": "step3",
    "StepVLForConditionalGeneration": "step3",
    "T5EncoderModel": "t5",
    "T5ForConditionalGeneration": "t5",
    "T5WithLMHeadModel": "t5",
    "UMT5ForConditionalGeneration": "t5",
    "UMT5Model": "t5",
    "UltravoxModel": "ultravox",
    "VLlama3ForCausalLM": "llama",
    "VoxtralForConditionalGeneration": "llama",
    "WavTokenizerDec": "wav_tokenizer",
    "XLMRobertaForSequenceClassification": "bert",
    "XLMRobertaModel": "bert",
    "XverseForCausalLM": "xverse",
    "YoutuForCausalLM": "deepseek",
    "YoutuVLForConditionalGeneration": "deepseek",
    "modeling_grove_moe.GroveMoeForCausalLM": "grove",
}


MMPROJ_MODEL_MAP: dict[str, str] = {
    "AudioFlamingo3ForConditionalGeneration": "ultravox",
    "CogVLMForCausalLM": "cogvlm",
    "DeepseekOCRForCausalLM": "deepseek",
    "DotsOCRForCausalLM": "dots_ocr",
    "Gemma3ForConditionalGeneration": "gemma",
    "Gemma3nForConditionalGeneration": "gemma",
    "Gemma4ForConditionalGeneration": "gemma",
    "Glm4vForConditionalGeneration": "qwen3_vl",
    "Glm4vMoeForConditionalGeneration": "qwen3_vl",
    "GlmOcrForConditionalGeneration": "qwen3_vl",
    "GlmasrModel": "ultravox",
    "HunYuanVLForConditionalGeneration": "hunyuan",
    "Idefics3ForConditionalGeneration": "smolvlm",
    "InternVisionModel": "intern_vision",
    "JanusForConditionalGeneration": "janus_pro",
    "KimiK25ForConditionalGeneration": "kimi_vl",
    "KimiVLForConditionalGeneration": "kimi_vl",
    "Lfm2AudioForConditionalGeneration": "lfm2",
    "Lfm2VlForConditionalGeneration": "lfm2",
    "LightOnOCRForConditionalGeneration": "lighton_ocr",
    "Llama4ForConditionalGeneration": "llama4",
    "LlavaForConditionalGeneration": "llava",
    "MERaLiON2ForConditionalGeneration": "ultravox",
    "Mistral3ForConditionalGeneration": "llava",
    "NemotronH_Nano_VL_V2": "nemotron",
    "PaddleOCRVisionModel": "ernie",
    "Phi4ForCausalLMV": "phi",
    "Qwen2AudioForConditionalGeneration": "ultravox",
    "Qwen2VLForConditionalGeneration": "qwen_vl",
    "Qwen2VLModel": "qwen_vl",
    "Qwen2_5OmniModel": "qwen_vl",
    "Qwen2_5_VLForConditionalGeneration": "qwen_vl",
    "Qwen3ASRForConditionalGeneration": "qwen3_vl",
    "Qwen3OmniMoeForConditionalGeneration": "qwen3_vl",
    "Qwen3VLForConditionalGeneration": "qwen3_vl",
    "Qwen3VLMoeForConditionalGeneration": "qwen3_vl",
    "Qwen3_5ForConditionalGeneration": "qwen3_vl",
    "Qwen3_5MoeForConditionalGeneration": "qwen3_vl",
    "RADIOModel": "nemotron",
    "SmolVLMForConditionalGeneration": "smolvlm",
    "StepVLForConditionalGeneration": "step3",
    "UltravoxModel": "ultravox",
    "VoxtralForConditionalGeneration": "ultravox",
    "YoutuVLForConditionalGeneration": "youtu_vl",
}


_TEXT_MODEL_MODULES = sorted(set(TEXT_MODEL_MAP.values()))
_MMPROJ_MODEL_MODULES = sorted(set(MMPROJ_MODEL_MAP.values()))


_loaded_text_modules: set[str] = set()
_loaded_mmproj_modules: set[str] = set()


def load_all_models() -> None:
    """Import all model modules to trigger @ModelBase.register() decorators."""
    if len(_loaded_text_modules) != len(_TEXT_MODEL_MODULES):
        for module_name in _TEXT_MODEL_MODULES:
            if module_name not in _loaded_text_modules:
                try:
                    __import__(f"conversion.{module_name}")
                    _loaded_text_modules.add(module_name)
                except Exception as e:
                    logger.warning(f"Failed to load model module {module_name}: {e}")

    if len(_loaded_mmproj_modules) != len(_MMPROJ_MODEL_MODULES):
        for module_name in _MMPROJ_MODEL_MODULES:
            if module_name not in _loaded_mmproj_modules:
                try:
                    __import__(f"conversion.{module_name}")
                    _loaded_mmproj_modules.add(module_name)
                except Exception as e:
                    logger.warning(f"Failed to load model module {module_name}: {e}")


def get_model_class(name: str, mmproj: bool = False) -> Type[ModelBase]:
    """Dynamically import and return a model class by its HuggingFace architecture name."""
    relevant_map = MMPROJ_MODEL_MAP if mmproj else TEXT_MODEL_MAP
    if name not in relevant_map:
        raise NotImplementedError(f"Architecture {name!r} not supported!")
    module_name = relevant_map[name]
    __import__(f"conversion.{module_name}")
    model_type = ModelType.MMPROJ if mmproj else ModelType.TEXT
    return ModelBase._model_classes[model_type][name]


def print_registered_models() -> None:
    load_all_models()
    logger.error("TEXT models:")
    for name in sorted(TEXT_MODEL_MAP.keys()):
        logger.error(f"  - {name}")
    logger.error("MMPROJ models:")
    for name in sorted(MMPROJ_MODEL_MAP.keys()):
        logger.error(f"  - {name}")
