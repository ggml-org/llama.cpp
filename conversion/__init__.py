from __future__ import annotations
from .base import (
    ModelBase, TextModel, MmprojModel, ModelType, SentencePieceTokenTypes,
    logger, _mistral_common_installed, _mistral_import_error_msg,
    get_model_architecture, LazyTorchTensor
)
from typing import TYPE_CHECKING, Type
if TYPE_CHECKING:
    from torch import Tensor # type: ignore  # noqa: F401

__all__ = [
    "ModelBase", "TextModel", "MmprojModel", "ModelType", "SentencePieceTokenTypes",
    "get_model_architecture", "LazyTorchTensor", "logger",
    "_mistral_common_installed", "_mistral_import_error_msg", "get_model_class", "print_registered_models"
]

TEXT_MODEL_MAP = {
    # Text models - keys are the names from @ModelBase.register(...) annotations
    "AfmoeForCausalLM": ("afmoe", "AfmoeModel"),
    "ApertusForCausalLM": ("apertus", "ApertusModel"),
    "ArceeForCausalLM": ("arcee", "ArceeModel"),
    "ArcticForCausalLM": ("arctic", "ArcticModel"),
    "BaiChuanForCausalLM": ("baichuan", "BaichuanModel"),
    "BaichuanForCausalLM": ("baichuan", "BaichuanModel"),
    "BailingMoeForCausalLM": ("bailing", "BailingMoeModel"),
    "BailingMoeV2ForCausalLM": ("bailing", "BailingMoeV2Model"),
    "BambaForCausalLM": ("granite", "GraniteHybridModel"),
    "BertForMaskedLM": ("bert", "BertModel"),
    "BertForSequenceClassification": ("bert", "BertModel"),
    "BertModel": ("bert", "BertModel"),
    "BitnetForCausalLM": ("bitnet", "BitnetModel"),
    "BloomForCausalLM": ("bloom", "BloomModel"),
    "BloomModel": ("bloom", "BloomModel"),
    "CamembertModel": ("bert", "BertModel"),
    "ChameleonForCausalLM": ("chameleon", "ChameleonModel"),
    "ChameleonForConditionalGeneration": ("chameleon", "ChameleonModel"),
    "ChatGLMForConditionalGeneration": ("chatglm", "ChatGLMModel"),
    "ChatGLMModel": ("chatglm", "ChatGLMModel"),
    "CodeShellForCausalLM": ("codeshell", "CodeShellModel"),
    "CogVLMForCausalLM": ("cogvlm", "CogVLMModel"),
    "Cohere2ForCausalLM": ("command_r", "Cohere2Model"),
    "CohereForCausalLM": ("command_r", "CommandR2Model"),
    "DbrxForCausalLM": ("dbrx", "DbrxModel"),
    "DeciLMForCausalLM": ("deci", "DeciModel"),
    "DeepseekForCausalLM": ("deepseek", "DeepseekModel"),
    "DeepseekV2ForCausalLM": ("deepseek", "DeepseekV2Model"),
    "DeepseekV3ForCausalLM": ("deepseek", "DeepseekV2Model"),
    "DistilBertForMaskedLM": ("bert", "DistilBertModel"),
    "DistilBertForSequenceClassification": ("bert", "DistilBertModel"),
    "DistilBertModel": ("bert", "DistilBertModel"),
    "Dots1ForCausalLM": ("dots1", "Dots1Model"),
    "DreamModel": ("dream", "DreamModel"),
    "Ernie4_5ForCausalLM": ("ernie", "Ernie4_5Model"),
    "Ernie4_5_ForCausalLM": ("ernie", "Ernie4_5Model"),
    "Ernie4_5_MoeForCausalLM": ("ernie", "Ernie4_5MoeModel"),
    "Exaone4ForCausalLM": ("exaone", "Exaone4Model"),
    "ExaoneForCausalLM": ("exaone", "ExaoneModel"),
    "FalconForCausalLM": ("falcon", "FalconModel"),
    "FalconH1ForCausalLM": ("falcon_h1", "FalconH1Model"),
    "FalconMambaForCausalLM": ("mamba", "MambaModel"),
    "GPT2LMHeadModel": ("gpt2", "GPT2Model"),
    "GPTBigCodeForCausalLM": ("starcoder", "StarCoderModel"),
    "GPTNeoXForCausalLM": ("gpt_neox", "GPTNeoXModel"),
    "GPTRefactForCausalLM": ("refact", "RefactModel"),
    "Gemma2ForCausalLM": ("gemma", "Gemma2Model"),
    "Gemma3ForCausalLM": ("gemma", "Gemma3Model"),
    "Gemma3ForConditionalGeneration": ("gemma", "Gemma3Model"),
    "Gemma3TextModel": ("gemma", "EmbeddingGemma"),
    "Gemma3nForConditionalGeneration": ("gemma", "Gemma3NModel"),
    "GemmaForCausalLM": ("gemma", "GemmaModel"),
    "Glm4ForCausalLM": ("glm", "Glm4Model"),
    "Glm4MoeForCausalLM": ("glm", "Glm4MoeModel"),
    "Glm4vForConditionalGeneration": ("glm", "Glm4Model"),
    "GlmForCausalLM": ("chatglm", "ChatGLMModel"),
    "GptOssForCausalLM": ("gpt_oss", "GptOssModel"),
    "GraniteForCausalLM": ("granite", "GraniteModel"),
    "GraniteMoeForCausalLM": ("granite", "GraniteMoeModel"),
    "GraniteMoeHybridForCausalLM": ("granite", "GraniteHybridModel"),
    "GraniteMoeSharedForCausalLM": ("granite", "GraniteMoeModel"),
    "Grok1ForCausalLM": ("grok", "GrokModel"),
    "GrokForCausalLM": ("grok", "GrokModel"),
    "GroveMoeForCausalLM": ("grove", "GroveMoeModel"),
    "HunYuanDenseV1ForCausalLM": ("hunyuan", "HunYuanModel"),
    "HunYuanMoEV1ForCausalLM": ("hunyuan", "HunYuanMoEModel"),
    "InternLM2ForCausalLM": ("internlm", "InternLM2Model"),
    "InternLM3ForCausalLM": ("internlm", "InternLM3Model"),
    "JAISLMHeadModel": ("jais", "JaisModel"),
    "JambaForCausalLM": ("jamba", "JambaModel"),
    "JanusForConditionalGeneration": ("janus_pro", "JanusProModel"),
    "JinaBertForMaskedLM": ("bert", "JinaBertV2Model"),
    "JinaBertModel": ("bert", "JinaBertV2Model"),
    "KimiVLForConditionalGeneration": ("deepseek", "DeepseekV2Model"),
    "LFM2ForCausalLM": ("lfm2", "LFM2Model"),
    "LLaDAMoEModel": ("llada", "LLaDAMoEModel"),
    "LLaDAMoEModelLM": ("llada", "LLaDAMoEModel"),
    "LLaDAModelLM": ("llada", "LLaDAModel"),
    "LLaMAForCausalLM": ("llama", "LlamaModel"),
    "Lfm2ForCausalLM": ("lfm2", "LFM2Model"),
    "Lfm2MoeForCausalLM": ("lfm2", "LFM2MoeModel"),
    "LlamaForCausalLM": ("llama", "LlamaModel"),
    "Llama4ForCausalLM": ("llama", "Llama4Model"),
    "Llama4ForConditionalGeneration": ("llama", "Llama4Model"),
    "LlamaModel": ("llama", "LlamaModel"),
    "LlavaForConditionalGeneration": ("llama", "LlamaModel"),
    "LlavaStableLMEpochForCausalLM": ("stable_lm", "StableLMModel"),
    "MPTForCausalLM": ("mpt", "MPTModel"),
    "MT5ForConditionalGeneration": ("t5", "T5Model"),
    "Mamba2ForCausalLM": ("mamba", "Mamba2Model"),
    "MambaForCausalLM": ("mamba", "MambaModel"),
    "MambaLMHeadModel": ("mamba", "MambaModel"),
    "MiniCPM3ForCausalLM": ("minicpm", "MiniCPM3Model"),
    "MiniCPMForCausalLM": ("minicpm", "MiniCPMModel"),
    "MiniMaxM2ForCausalLM": ("minimax", "MiniMaxM2Model"),
    "MistralForCausalLM": ("llama", "LlamaModel"),
    "Mistral3ForConditionalGeneration": ("llava", "LlavaVisionModel"),
    "MixtralForCausalLM": ("llama", "LlamaModel"),
    "NemotronForCausalLM": ("nemotron", "NemotronModel"),
    "NemotronHForCausalLM": ("nemotron", "NemotronHModel"),
    "NeoBERT": ("bert", "NeoBert"),
    "NeoBERTForSequenceClassification": ("bert", "NeoBert"),
    "NeoBERTLMHead": ("bert", "NeoBert"),
    "NomicBertModel": ("bert", "NomicBertModel"),
    "OLMoForCausalLM": ("olmo", "OlmoModel"),
    "Olmo2ForCausalLM": ("olmo", "Olmo2Model"),
    "Olmo3ForCausalLM": ("olmo", "Olmo2Model"),
    "OlmoForCausalLM": ("olmo", "OlmoModel"),
    "OlmoeForCausalLM": ("olmo", "OlmoeModel"),
    "OpenELMForCausalLM": ("openelm", "OpenELMModel"),
    "OrionForCausalLM": ("orion", "OrionModel"),
    "PLMForCausalLM": ("plm", "PLMModel"),
    "PLaMo2ForCausalLM": ("plamo", "Plamo2Model"),
    "PanguEmbeddedForCausalLM": ("pangu", "PanguEmbeddedModel"),
    "Phi3ForCausalLM": ("phi", "Phi3MiniModel"),
    "PhiForCausalLM": ("phi", "Phi2Model"),
    "PhiMoEForCausalLM": ("phi", "PhiMoeModel"),
    "Plamo2ForCausalLM": ("plamo", "Plamo2Model"),
    "PlamoForCausalLM": ("plamo", "PlamoModel"),
    "QWenLMHeadModel": ("qwen", "QwenModel"),
    "Qwen2AudioForConditionalGeneration": ("qwen", "Qwen2Model"),
    "Qwen2ForCausalLM": ("qwen", "Qwen2Model"),
    "Qwen2Model": ("qwen", "Qwen2Model"),
    "Qwen2MoeForCausalLM": ("qwen", "Qwen2MoeModel"),
    "Qwen2VLForConditionalGeneration": ("qwen_vl", "Qwen2VLModel"),
    "Qwen2VLModel": ("qwen_vl", "Qwen2VLModel"),
    "Qwen2_5OmniModel": ("qwen_vl", "Qwen2VLModel"),
    "Qwen2_5_VLForConditionalGeneration": ("qwen_vl", "Qwen2VLModel"),
    "Qwen3ForCausalLM": ("qwen", "Qwen3Model"),
    "Qwen3MoeForCausalLM": ("qwen", "Qwen3MoeModel"),
    "Qwen3VLForConditionalGeneration": ("qwen3_vl", "Qwen3VLVisionModel"),
    "Qwen3VLMoeForConditionalGeneration": ("qwen3_vl", "Qwen3VLVisionModel"),
    "RWForCausalLM": ("falcon", "FalconModel"),
    "RWKV6Qwen2ForCausalLM": ("rwkv", "RWKV6Qwen2Model"),
    "RWKV7ForCausalLM": ("rwkv", "Rwkv7Model"),
    "RobertaForSequenceClassification": ("bert", "RobertaModel"),
    "RobertaModel": ("bert", "RobertaModel"),
    "Rwkv6ForCausalLM": ("rwkv", "Rwkv6Model"),
    "Rwkv7ForCausalLM": ("rwkv", "Rwkv7Model"),
    "RwkvHybridForCausalLM": ("rwkv", "ARwkv7Model"),
    "SeedOssForCausalLM": ("olmo", "SeedOssModel"),
    "SmallThinkerForCausalLM": ("small_thinker", "SmallThinkerModel"),
    "StableLMEpochForCausalLM": ("stable_lm", "StableLMModel"),
    "StableLmForCausalLM": ("stable_lm", "StableLMModel"),
    "Starcoder2ForCausalLM": ("starcoder", "StarCoder2Model"),
    "SmolLM3ForCausalLM": ("smollm", "SmolLM3Model"),
    "T5EncoderModel": ("t5", "T5EncoderModel"),
    "T5ForConditionalGeneration": ("t5", "T5Model"),
    "T5WithLMHeadModel": ("t5", "T5Model"),
    "UMT5ForConditionalGeneration": ("t5", "T5Model"),
    "UMT5Model": ("t5", "T5Model"),
    "UltravoxModel": ("ultravox", "UltravoxModel"),
    "VLlama3ForCausalLM": ("llama", "LlamaModel"),
    "VoxtralForConditionalGeneration": ("llama", "LlamaModel"),
    "WavTokenizerDec": ("wav_tokenizer", "WavTokenizerDecModel"),
    "XverseForCausalLM": ("xverse", "XverseModel"),
    "XLMRobertaForSequenceClassification": ("bert", "XLMRobertaModel"),
    "XLMRobertaModel": ("bert", "XLMRobertaModel"),
    "modeling_grove_moe.GroveMoeForCausalLM": ("grove", "GroveMoeModel"),
}

MMPROJ_MODEL_MAP = {
    # Multimodal models - keys are the names from @ModelBase.register(...) annotations
    "CogVLMForCausalLM": ("llava", "LlavaVisionModel"),
    "Gemma3ForConditionalGeneration": ("gemma", "Gemma3VisionModel"),
    "Idefics3ForConditionalGeneration": ("smolvlm", "SmolVLMModel"),
    "InternVisionModel": ("intern_vision", "InternVisionModel"),
    "JanusForConditionalGeneration": ("llava", "LlavaVisionModel"),
    "KimiVLForConditionalGeneration": ("kimi_vl", "KimiVLModel"),
    "Lfm2VlForConditionalGeneration": ("lfm2", "LFM2VLModel"),
    "LightOnOCRForConditionalGeneration": ("llava", "LlavaVisionModel"),
    "Llama4ForConditionalGeneration": ("llama4", "Llama4VisionModel"),
    "LlavaForConditionalGeneration": ("llava", "LlavaVisionModel"),
    "Mistral3ForConditionalGeneration": ("llava", "LlavaVisionModel"),
    "Qwen2AudioForConditionalGeneration": ("ultravox", "WhisperEncoderModel"),
    "Qwen2VLForConditionalGeneration": ("qwen_vl", "Qwen2VLVisionModel"),
    "Qwen2VLModel": ("qwen_vl", "Qwen2VLVisionModel"),
    "Qwen2_5OmniModel": ("qwen_vl", "Qwen25OmniModel"),
    "Qwen2_5_VLForConditionalGeneration": ("qwen_vl", "Qwen2VLVisionModel"),
    "Qwen3VLForConditionalGeneration": ("qwen3_vl", "Qwen3VLVisionModel"),
    "Qwen3VLMoeForConditionalGeneration": ("qwen3_vl", "Qwen3VLVisionModel"),
    "SmolVLMForConditionalGeneration": ("smolvlm", "SmolVLMModel"),
    "UltravoxModel": ("ultravox", "UltravoxWhisperEncoderModel"),
    "VoxtralForConditionalGeneration": ("ultravox", "VoxtralWhisperEncoderModel"),
}

# List of all model module names (used for lazy loading)
_TEXT_MODEL_MODULES = sorted(list(set(y[0] for (_, y) in TEXT_MODEL_MAP.items())))
_MMPROJ_MODEL_MODULES = sorted(list(set(y[0] for (_, y) in MMPROJ_MODEL_MAP.items())))

# Track which modules have been loaded
_loaded_text_modules = set()
_loaded_mmproj_modules = set()


# Function to load all model modules
def _load_all_models():
    """Import all model modules to trigger registration."""
    if not len(_loaded_text_modules) == len(_TEXT_MODEL_MODULES):
        for module_name in _TEXT_MODEL_MODULES:
            if module_name not in _loaded_text_modules:
                try:
                    __import__(f"conversion.{module_name}")
                    _loaded_text_modules.add(module_name)
                except Exception as e:
                    # Log but don't fail - some models might have issues
                    logger.warning(f"Failed to load model module {module_name}: {e}")

    if not len(_loaded_mmproj_modules) == len(_MMPROJ_MODEL_MODULES):
        for module_name in _MMPROJ_MODEL_MODULES:
            if module_name not in _loaded_mmproj_modules:
                try:
                    __import__(f"conversion.{module_name}")
                    _loaded_mmproj_modules.add(module_name)
                except Exception as e:
                    # Log but don't fail - some models might have issues
                    logger.warning(f"Failed to load model module {module_name}: {e}")


# Function to get a model class by name
def get_model_class(name: str, mmproj: bool = False) -> Type[ModelBase]:
    """
    Dynamically import and return a model class by name.
    This avoids circular dependencies by only importing when needed.
    """
    # Map model names to their module and class name
    relevant_map = None
    if mmproj:
        relevant_map = MMPROJ_MODEL_MAP
    else:
        relevant_map = TEXT_MODEL_MAP

    if name not in relevant_map:
        raise ValueError(f"Unknown model class: {name}, valid classes are: {relevant_map}")
    module_name, class_name = relevant_map[name]
    module = __import__(f"conversion.{module_name}", fromlist=[class_name])
    return getattr(module, class_name)


def print_registered_models():
    logger.error("TEXT models:")
    for name in sorted(TEXT_MODEL_MAP.keys()):
        logger.error(f"  - {name}")

    logger.error("\nMMPROJ models:")
    for name in sorted(MMPROJ_MODEL_MAP.keys()):
        logger.error(f"  - {name}")
