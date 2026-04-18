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


TEXT_MODEL_MAP: dict[str, tuple[str, str]] = {
    "AfmoeForCausalLM": ("afmoe", "AfmoeModel"),
    "ApertusForCausalLM": ("llama", "ApertusModel"),
    "ArceeForCausalLM": ("llama", "ArceeModel"),
    "ArcticForCausalLM": ("arctic", "ArcticModel"),
    "AudioFlamingo3ForConditionalGeneration": ("qwen", "Qwen2Model"),
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
    "DotsOCRForCausalLM": ("qwen", "Qwen2Model"),
    "DreamModel": ("dream", "DreamModel"),
    "Ernie4_5ForCausalLM": ("ernie", "Ernie4_5Model"),
    "Ernie4_5_ForCausalLM": ("ernie", "Ernie4_5Model"),
    "Ernie4_5_MoeForCausalLM": ("ernie", "Ernie4_5MoeModel"),
    "EuroBertModel": ("bert", "EuroBertModel"),
    "Exaone4ForCausalLM": ("exaone", "Exaone4Model"),
    "ExaoneForCausalLM": ("exaone", "ExaoneModel"),
    "ExaoneMoEForCausalLM": ("exaone", "ExaoneMoEModel"),
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
    "Gemma3nForCausalLM": ("gemma", "Gemma3NModel"),
    "Gemma3nForConditionalGeneration": ("gemma", "Gemma3NModel"),
    "Gemma4ForConditionalGeneration": ("gemma", "Gemma4Model"),
    "GemmaForCausalLM": ("gemma", "GemmaModel"),
    "Glm4ForCausalLM": ("glm", "Glm4Model"),
    "Glm4MoeForCausalLM": ("glm", "Glm4MoeModel"),
    "Glm4MoeLiteForCausalLM": ("glm", "Glm4MoeLiteModel"),
    "Glm4vForConditionalGeneration": ("glm", "Glm4Model"),
    "Glm4vMoeForConditionalGeneration": ("glm", "Glm4MoeModel"),
    "GlmForCausalLM": ("chatglm", "ChatGLMModel"),
    "GlmMoeDsaForCausalLM": ("glm", "GlmMoeDsaModel"),
    "GlmOcrForConditionalGeneration": ("glm", "GlmOCRModel"),
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
    "HunYuanVLForConditionalGeneration": ("hunyuan", "HunyuanVLTextModel"),
    "IQuestCoderForCausalLM": ("llama", "LlamaModel"),
    "InternLM2ForCausalLM": ("internlm", "InternLM2Model"),
    "InternLM3ForCausalLM": ("internlm", "InternLM3Model"),
    "JAISLMHeadModel": ("jais", "JaisModel"),
    "Jais2ForCausalLM": ("jais", "Jais2Model"),
    "JambaForCausalLM": ("jamba", "JambaModel"),
    "JanusForConditionalGeneration": ("janus_pro", "JanusProModel"),
    "JinaBertForMaskedLM": ("bert", "JinaBertV2Model"),
    "JinaBertModel": ("bert", "JinaBertV2Model"),
    "JinaEmbeddingsV5Model": ("bert", "EuroBertModel"),
    "KORMoForCausalLM": ("qwen", "Qwen2Model"),
    "KimiK25ForConditionalGeneration": ("deepseek", "DeepseekV2Model"),
    "KimiLinearForCausalLM": ("kimi_linear", "KimiLinearModel"),
    "KimiLinearModel": ("kimi_linear", "KimiLinearModel"),
    "KimiVLForConditionalGeneration": ("deepseek", "DeepseekV2Model"),
    "LFM2ForCausalLM": ("lfm2", "LFM2Model"),
    "LLaDAMoEModel": ("llada", "LLaDAMoEModel"),
    "LLaDAMoEModelLM": ("llada", "LLaDAMoEModel"),
    "LLaDAModelLM": ("llada", "LLaDAModel"),
    "LLaMAForCausalLM": ("llama", "LlamaModel"),
    "Lfm25AudioTokenizer": ("lfm2", "LFM25AudioTokenizer"),
    "Lfm2ForCausalLM": ("lfm2", "LFM2Model"),
    "Lfm2Model": ("lfm2", "LFM2ColBertModel"),
    "Lfm2MoeForCausalLM": ("lfm2", "LFM2MoeModel"),
    "Llama4ForCausalLM": ("llama", "Llama4Model"),
    "Llama4ForConditionalGeneration": ("llama", "Llama4Model"),
    "LlamaBidirectionalModel": ("llama", "LlamaEmbedNemotronModel"),
    "LlamaForCausalLM": ("llama", "LlamaModel"),
    "LlamaModel": ("llama", "LlamaModel"),
    "LlavaForConditionalGeneration": ("llama", "LlamaModel"),
    "LlavaStableLMEpochForCausalLM": ("stable_lm", "StableLMModel"),
    "MPTForCausalLM": ("mpt", "MPTModel"),
    "MT5ForConditionalGeneration": ("t5", "T5Model"),
    "MaincoderForCausalLM": ("maincoder", "MaincoderModel"),
    "Mamba2ForCausalLM": ("mamba", "Mamba2Model"),
    "MambaForCausalLM": ("mamba", "MambaModel"),
    "MambaLMHeadModel": ("mamba", "MambaModel"),
    "MiMoV2FlashForCausalLM": ("mimo", "MimoV2Model"),
    "MiniCPM3ForCausalLM": ("minicpm", "MiniCPM3Model"),
    "MiniCPMForCausalLM": ("minicpm", "MiniCPMModel"),
    "MiniMaxM2ForCausalLM": ("minimax", "MiniMaxM2Model"),
    "Ministral3ForCausalLM": ("mistral3", "Mistral3Model"),
    "Mistral3ForConditionalGeneration": ("mistral3", "Mistral3Model"),
    "MistralForCausalLM": ("llama", "LlamaModel"),
    "MixtralForCausalLM": ("llama", "LlamaModel"),
    "ModernBertForMaskedLM": ("bert", "ModernBertModel"),
    "ModernBertForSequenceClassification": ("bert", "ModernBertModel"),
    "ModernBertModel": ("bert", "ModernBertModel"),
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
    "PLaMo3ForCausalLM": ("plamo", "Plamo3Model"),
    "PaddleOCRVLForConditionalGeneration": ("ernie", "PaddleOCRModel"),
    "PanguEmbeddedForCausalLM": ("pangu", "PanguEmbeddedModel"),
    "Phi3ForCausalLM": ("phi", "Phi3MiniModel"),
    "Phi4ForCausalLMV": ("phi", "Phi3MiniModel"),
    "PhiForCausalLM": ("phi", "Phi2Model"),
    "PhiMoEForCausalLM": ("phi", "PhiMoeModel"),
    "Plamo2ForCausalLM": ("plamo", "Plamo2Model"),
    "Plamo3ForCausalLM": ("plamo", "Plamo3Model"),
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
    "Qwen3ASRForConditionalGeneration": ("qwen3_vl", "Qwen3ASRTextModel"),
    "Qwen3ForCausalLM": ("qwen", "Qwen3Model"),
    "Qwen3Model": ("qwen", "Qwen3Model"),
    "Qwen3MoeForCausalLM": ("qwen", "Qwen3MoeModel"),
    "Qwen3NextForCausalLM": ("qwen", "Qwen3NextModel"),
    "Qwen3OmniMoeForConditionalGeneration": ("qwen3_vl", "Qwen3OmniMoeTextModel"),
    "Qwen3VLForConditionalGeneration": ("qwen3_vl", "Qwen3VLTextModel"),
    "Qwen3VLMoeForConditionalGeneration": ("qwen3_vl", "Qwen3VLMoeTextModel"),
    "Qwen3_5ForCausalLM": ("qwen", "Qwen3_5TextModel"),
    "Qwen3_5ForConditionalGeneration": ("qwen", "Qwen3_5TextModel"),
    "Qwen3_5MoeForCausalLM": ("qwen", "Qwen3_5MoeTextModel"),
    "Qwen3_5MoeForConditionalGeneration": ("qwen", "Qwen3_5MoeTextModel"),
    "RND1": ("qwen", "RND1Model"),
    "RWForCausalLM": ("falcon", "FalconModel"),
    "RWKV6Qwen2ForCausalLM": ("rwkv", "RWKV6Qwen2Model"),
    "RWKV7ForCausalLM": ("rwkv", "Rwkv7Model"),
    "RobertaForSequenceClassification": ("bert", "RobertaModel"),
    "RobertaModel": ("bert", "RobertaModel"),
    "RuGPT3XLForCausalLM": ("gpt2", "RuGPT3XLModel"),
    "Rwkv6ForCausalLM": ("rwkv", "Rwkv6Model"),
    "Rwkv7ForCausalLM": ("rwkv", "Rwkv7Model"),
    "RwkvHybridForCausalLM": ("rwkv", "ARwkv7Model"),
    "SeedOssForCausalLM": ("olmo", "SeedOssModel"),
    "SmallThinkerForCausalLM": ("small_thinker", "SmallThinkerModel"),
    "SmolLM3ForCausalLM": ("llama", "SmolLM3Model"),
    "SolarOpenForCausalLM": ("glm", "SolarOpenModel"),
    "StableLMEpochForCausalLM": ("stable_lm", "StableLMModel"),
    "StableLmForCausalLM": ("stable_lm", "StableLMModel"),
    "Starcoder2ForCausalLM": ("starcoder", "StarCoder2Model"),
    "Step3p5ForCausalLM": ("step3", "Step35Model"),
    "StepVLForConditionalGeneration": ("step3", "Step3VLTextModel"),
    "T5EncoderModel": ("t5", "T5EncoderModel"),
    "T5ForConditionalGeneration": ("t5", "T5Model"),
    "T5WithLMHeadModel": ("t5", "T5Model"),
    "UMT5ForConditionalGeneration": ("t5", "T5Model"),
    "UMT5Model": ("t5", "T5Model"),
    "UltravoxModel": ("ultravox", "UltravoxModel"),
    "VLlama3ForCausalLM": ("llama", "LlamaModel"),
    "VoxtralForConditionalGeneration": ("llama", "LlamaModel"),
    "WavTokenizerDec": ("wav_tokenizer", "WavTokenizerDecModel"),
    "XLMRobertaForSequenceClassification": ("bert", "XLMRobertaModel"),
    "XLMRobertaModel": ("bert", "XLMRobertaModel"),
    "XverseForCausalLM": ("xverse", "XverseModel"),
    "YoutuForCausalLM": ("deepseek", "DeepseekV2Model"),
    "YoutuVLForConditionalGeneration": ("deepseek", "DeepseekV2Model"),
    "modeling_grove_moe.GroveMoeForCausalLM": ("grove", "GroveMoeModel"),
}


MMPROJ_MODEL_MAP: dict[str, tuple[str, str]] = {
    "AudioFlamingo3ForConditionalGeneration": ("ultravox", "AudioFlamingo3WhisperEncoderModel"),
    "CogVLMForCausalLM": ("cogvlm", "CogVLMVisionModel"),
    "DeepseekOCRForCausalLM": ("deepseek", "DeepseekOCRVisionModel"),
    "DotsOCRForCausalLM": ("dots_ocr", "DotsOCRVisionModel"),
    "Gemma3ForConditionalGeneration": ("gemma", "Gemma3VisionModel"),
    "Gemma3nForConditionalGeneration": ("gemma", "Gemma3nVisionAudioModel"),
    "Gemma4ForConditionalGeneration": ("gemma", "Gemma4VisionAudioModel"),
    "Glm4vForConditionalGeneration": ("qwen3_vl", "Glm4VVisionModel"),
    "Glm4vMoeForConditionalGeneration": ("qwen3_vl", "Glm4VVisionModel"),
    "GlmOcrForConditionalGeneration": ("qwen3_vl", "Glm4VVisionModel"),
    "GlmasrModel": ("ultravox", "GlmASRWhisperEncoderModel"),
    "HunYuanVLForConditionalGeneration": ("hunyuan", "HunyuanVLVisionModel"),
    "Idefics3ForConditionalGeneration": ("smolvlm", "SmolVLMModel"),
    "InternVisionModel": ("intern_vision", "InternVisionModel"),
    "JanusForConditionalGeneration": ("janus_pro", "JanusProVisionModel"),
    "KimiK25ForConditionalGeneration": ("kimi_vl", "KimiK25Model"),
    "KimiVLForConditionalGeneration": ("kimi_vl", "KimiVLModel"),
    "Lfm2AudioForConditionalGeneration": ("lfm2", "LFM2AudioModel"),
    "Lfm2VlForConditionalGeneration": ("lfm2", "LFM2VLModel"),
    "LightOnOCRForConditionalGeneration": ("lighton_ocr", "LightOnOCRVisionModel"),
    "Llama4ForConditionalGeneration": ("llama4", "Llama4VisionModel"),
    "LlavaForConditionalGeneration": ("llava", "LlavaVisionModel"),
    "MERaLiON2ForConditionalGeneration": ("ultravox", "MERaLiONWhisperEncoderModel"),
    "Mistral3ForConditionalGeneration": ("llava", "LlavaVisionModel"),
    "NemotronH_Nano_VL_V2": ("nemotron", "NemotronNanoV2VLModel"),
    "PaddleOCRVisionModel": ("ernie", "PaddleOCRVisionModel"),
    "Phi4ForCausalLMV": ("phi", "Phi4VisionMmprojModel"),
    "Qwen2AudioForConditionalGeneration": ("ultravox", "WhisperEncoderModel"),
    "Qwen2VLForConditionalGeneration": ("qwen_vl", "Qwen2VLVisionModel"),
    "Qwen2VLModel": ("qwen_vl", "Qwen2VLVisionModel"),
    "Qwen2_5OmniModel": ("qwen_vl", "Qwen25OmniModel"),
    "Qwen2_5_VLForConditionalGeneration": ("qwen_vl", "Qwen2VLVisionModel"),
    "Qwen3ASRForConditionalGeneration": ("qwen3_vl", "Qwen3ASRMmprojModel"),
    "Qwen3OmniMoeForConditionalGeneration": ("qwen3_vl", "Qwen3OmniMmprojModel"),
    "Qwen3VLForConditionalGeneration": ("qwen3_vl", "Qwen3VLVisionModel"),
    "Qwen3VLMoeForConditionalGeneration": ("qwen3_vl", "Qwen3VLVisionModel"),
    "Qwen3_5ForConditionalGeneration": ("qwen3_vl", "Qwen3VLVisionModel"),
    "Qwen3_5MoeForConditionalGeneration": ("qwen3_vl", "Qwen3VLVisionModel"),
    "RADIOModel": ("nemotron", "NemotronNanoV2VLModel"),
    "SmolVLMForConditionalGeneration": ("smolvlm", "SmolVLMModel"),
    "StepVLForConditionalGeneration": ("step3", "Step3VLVisionModel"),
    "UltravoxModel": ("ultravox", "UltravoxWhisperEncoderModel"),
    "VoxtralForConditionalGeneration": ("ultravox", "VoxtralWhisperEncoderModel"),
    "YoutuVLForConditionalGeneration": ("youtu_vl", "YoutuVLVisionModel"),
}


_TEXT_MODEL_MODULES = sorted({v[0] for v in TEXT_MODEL_MAP.values()})
_MMPROJ_MODEL_MODULES = sorted({v[0] for v in MMPROJ_MODEL_MAP.values()})


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
    module_name, class_name = relevant_map[name]
    module = __import__(f"conversion.{module_name}", fromlist=[class_name])
    return getattr(module, class_name)


def print_registered_models() -> None:
    load_all_models()
    logger.error("TEXT models:")
    for name in sorted(TEXT_MODEL_MAP.keys()):
        logger.error(f"  - {name}")
    logger.error("MMPROJ models:")
    for name in sorted(MMPROJ_MODEL_MAP.keys()):
        logger.error(f"  - {name}")
