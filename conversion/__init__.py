from __future__ import annotations
from .base import (
    ModelBase, TextModel, MmprojModel, ModelType, SentencePieceTokenTypes,
    logger, _mistral_common_installed, _mistral_import_error_msg,
    get_model_architecture, LazyTorchTensor
)
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from torch import Tensor # type: ignore  # noqa: F401

__all__ = [
    "ModelBase", "TextModel", "MmprojModel", "ModelType", "SentencePieceTokenTypes",
    "get_model_architecture", "LazyTorchTensor", "logger",
    "_mistral_common_installed", "_mistral_import_error_msg"
]
# List of all model module names (used for lazy loading)
_MODEL_MODULES = [
    'gpt_neox', 'bloom', 'mpt', 'orion', 'baichuan', 'xverse', 'falcon', 'starcoder',
    'refact', 'stable_lm', 'llama', 'arcee', 'mistral', 'mistral3', 'deci', 'bitnet', 'grok',
    'dbrx', 'minicpm', 'qwen', 'qwen_vl', 'qwen3_vl', 'dream', 'llada', 'ernie',
    'intern_vision', 'wav_tokenizer', 'gpt2', 'phi', 'plamo', 'codeshell', 'internlm',
    'bert', 'gemma', 'rwkv', 'mamba', 'jamba', 'command_r', 'olmo', 'openelm',
    'arctic', 'deepseek', 'minimax', 'pangu', 'dots1', 'plm', 't5', 'jais', 'glm',
    'chatglm', 'nemotron', 'exaone', 'granite', 'bailing', 'grove', 'chameleon',
    'ultravox', 'falcon_h1', 'hunyuan', 'smollm', 'gpt_oss', 'lfm2', 'small_thinker',
    'apertus', 'pixtral', 'lighton_ocr', 'kimi_vl', 'cogvlm', 'janus_pro', 'llama4',
    'smolvlm'
]
# Track which modules have been loaded
_loaded_modules = set()


# Function to load all model modules
def _load_all_models():
    """Import all model modules to trigger registration."""
    if len(_loaded_modules) == len(_MODEL_MODULES):
        return  # Already loaded
    for module_name in _MODEL_MODULES:
        if module_name not in _loaded_modules:
            try:
                __import__(f"conversion.{module_name}")
                _loaded_modules.add(module_name)
            except Exception as e:
                # Log but don't fail - some models might have issues
                logger.warning(f"Failed to load model module {module_name}: {e}")


# Function to get a model class by name
def get_model_class(name: str, mmproj: bool = False):
    """
    Dynamically import and return a model class by name.
    This avoids circular dependencies by only importing when needed.
    """
    # Map model names to their module and class name
    model_map = {
        # Text models
        "LlamaModel": ("llama", "LlamaModel"),
        "MistralModel": ("mistral", "MistralModel"),
        "GPTNeoXModel": ("gpt_neox", "GPTNeoXModel"),
        "BloomModel": ("bloom", "BloomModel"),
        "MPTModel": ("mpt", "MPTModel"),
        "OrionModel": ("orion", "OrionModel"),
        "BaichuanModel": ("baichuan", "BaichuanModel"),
        "XverseModel": ("xverse", "XverseModel"),
        "FalconModel": ("falcon", "FalconModel"),
        "StarCoderModel": ("starcoder", "StarCoderModel"),
        "StarCoder2Model": ("starcoder", "StarCoder2Model"),
        "RefactModel": ("refact", "RefactModel"),
        "StableLMModel": ("stable_lm", "StableLMModel"),
        "ArceeModel": ("arcee", "ArceeModel"),
        "Mistral3Model": ("mistral3", "Mistral3Model"),
        "DeciModel": ("deci", "DeciModel"),
        "BitnetModel": ("bitnet", "BitnetModel"),
        "GrokModel": ("grok", "GrokModel"),
        "DbrxModel": ("dbrx", "DbrxModel"),
        "MiniCPMModel": ("minicpm", "MiniCPMModel"),
        "MiniCPM3Model": ("minicpm", "MiniCPM3Model"),
        "QwenModel": ("qwen", "QwenModel"),
        "Qwen2Model": ("qwen", "Qwen2Model"),
        "Qwen2MoeModel": ("qwen", "Qwen2MoeModel"),
        "Qwen3Model": ("qwen", "Qwen3Model"),
        "Qwen3MoeModel": ("qwen", "Qwen3MoeModel"),
        "Qwen25OmniModel": ("qwen_vl", "Qwen25OmniModel"),
        "Qwen3VLTextModel": ("qwen3_vl", "Qwen3VLTextModel"),
        "Qwen3VLMoeTextModel": ("qwen3_vl", "Qwen3VLMoeTextModel"),
        "DreamModel": ("dream", "DreamModel"),
        "LLaDAModel": ("llada", "LLaDAModel"),
        "LLaDAMoEModel": ("llada", "LLaDAMoEModel"),
        "Ernie4_5Model": ("ernie", "Ernie4_5Model"),
        "Ernie4_5MoeModel": ("ernie", "Ernie4_5MoeModel"),
        "InternVisionModel": ("intern_vision", "InternVisionModel"),
        "WavTokenizerDecModel": ("wav_tokenizer", "WavTokenizerDecModel"),
        "GPT2Model": ("gpt2", "GPT2Model"),
        "Phi2Model": ("phi", "Phi2Model"),
        "Phi3MiniModel": ("phi", "Phi3MiniModel"),
        "PhiMoeModel": ("phi", "PhiMoeModel"),
        "PlamoModel": ("plamo", "PlamoModel"),
        "Plamo2Model": ("plamo", "Plamo2Model"),
        "CodeShellModel": ("codeshell", "CodeShellModel"),
        "InternLM2Model": ("internlm", "InternLM2Model"),
        "InternLM3Model": ("internlm", "InternLM3Model"),
        "BertModel": ("bert", "BertModel"),
        "DistilBertModel": ("bert", "DistilBertModel"),
        "RobertaModel": ("bert", "RobertaModel"),
        "NomicBertModel": ("bert", "NomicBertModel"),
        "NeoBert": ("bert", "NeoBert"),
        "XLMRobertaModel": ("bert", "XLMRobertaModel"),
        "JinaBertV2Model": ("bert", "JinaBertV2Model"),
        "GemmaModel": ("gemma", "GemmaModel"),
        "Gemma2Model": ("gemma", "Gemma2Model"),
        "Gemma3Model": ("gemma", "Gemma3Model"),
        "EmbeddingGemma": ("gemma", "EmbeddingGemma"),
        "Gemma3NModel": ("gemma", "Gemma3NModel"),
        "Rwkv6Model": ("rwkv", "Rwkv6Model"),
        "RWKV6Qwen2Model": ("rwkv", "RWKV6Qwen2Model"),
        "Rwkv7Model": ("rwkv", "Rwkv7Model"),
        "ARwkv7Model": ("rwkv", "ARwkv7Model"),
        "MambaModel": ("mamba", "MambaModel"),
        "Mamba2Model": ("mamba", "Mamba2Model"),
        "JambaModel": ("jamba", "JambaModel"),
        "CommandR2Model": ("command_r", "CommandR2Model"),
        "Cohere2Model": ("command_r", "Cohere2Model"),
        "OlmoModel": ("olmo", "OlmoModel"),
        "OlmoForCausalLM": ("olmo", "OlmoModel"),
        "SeedOssModel": ("olmo", "SeedOssModel"),
        "Olmo2Model": ("olmo", "Olmo2Model"),
        "OlmoeModel": ("olmo", "OlmoeModel"),
        "OpenELMModel": ("openelm", "OpenELMModel"),
        "ArcticModel": ("arctic", "ArcticModel"),
        "DeepseekModel": ("deepseek", "DeepseekModel"),
        "DeepseekV2Model": ("deepseek", "DeepseekV2Model"),
        "MiniMaxM2Model": ("minimax", "MiniMaxM2Model"),
        "PanguEmbeddedModel": ("pangu", "PanguEmbeddedModel"),
        "Dots1Model": ("dots1", "Dots1Model"),
        "PLMModel": ("plm", "PLMModel"),
        "T5Model": ("t5", "T5Model"),
        "T5ForConditionalGeneration": ("t5", "T5Model"),
        "T5WithLMHeadModel": ("t5", "T5Model"),
        "T5EncoderModel": ("t5", "T5EncoderModel"),
        "JaisModel": ("jais", "JaisModel"),
        "Glm4Model": ("glm", "Glm4Model"),
        "Glm4MoeModel": ("glm", "Glm4MoeModel"),
        "ChatGLMModel": ("chatglm", "ChatGLMModel"),
        "NemotronModel": ("nemotron", "NemotronModel"),
        "NemotronHModel": ("nemotron", "NemotronHModel"),
        "ExaoneModel": ("exaone", "ExaoneModel"),
        "Exaone4Model": ("exaone", "Exaone4Model"),
        "GraniteModel": ("granite", "GraniteModel"),
        "GraniteMoeModel": ("granite", "GraniteMoeModel"),
        "GraniteHybridModel": ("granite", "GraniteHybridModel"),
        "BailingMoeModel": ("bailing", "BailingMoeModel"),
        "BailingMoeV2Model": ("bailing", "BailingMoeV2Model"),
        "GroveMoeModel": ("grove", "GroveMoeModel"),
        "ChameleonModel": ("chameleon", "ChameleonModel"),
        "HunYuanMoEModel": ("hunyuan", "HunYuanMoEModel"),
        "HunYuanModel": ("hunyuan", "HunYuanModel"),
        "SmolLM3Model": ("smollm", "SmolLM3Model"),
        "GptOssModel": ("gpt_oss", "GptOssModel"),
        "LFM2Model": ("lfm2", "LFM2Model"),
        "LFM2MoeModel": ("lfm2", "LFM2MoeModel"),
        "SmallThinkerModel": ("small_thinker", "SmallThinkerModel"),
        "ApertusModel": ("apertus", "ApertusModel"),
        "PixtralModel": ("pixtral", "PixtralModel"),
        "LightOnOCRVisionModel": ("lighton_ocr", "LightOnOCRVisionModel"),
        "KimiVLModel": ("kimi_vl", "KimiVLModel"),
        "CogVLMModel": ("cogvlm", "CogVLMModel"),
        "JanusProModel": ("janus_pro", "JanusProModel"),
        # Multimodal models
        "LlavaVisionModel": ("llava", "LlavaVisionModel"),
        "SmolVLMModel": ("smolvlm", "SmolVLMModel"),
        "Llama4Model": ("llama4", "Llama4Model"),
        "Llama4VisionModel": ("llama4", "Llama4VisionModel"),
        "Qwen2VLModel": ("qwen_vl", "Qwen2VLVisionModel"),
        "Qwen2VLVisionModel": ("qwen_vl", "Qwen2VLVisionModel"),
        "Qwen2_5_VLForConditionalGeneration": ("qwen_vl", "Qwen2VLVisionModel"),
        "Qwen3VLVisionModel": ("qwen3_vl", "Qwen3VLVisionModel"),
        "Gemma3VisionModel": ("gemma", "Gemma3VisionModel"),
        "LFM2VLModel": ("lfm2", "LFM2VLModel"),
        "UltravoxModel": ("ultravox", "UltravoxModel"),
        "WhisperEncoderModel": ("ultravox", "WhisperEncoderModel"),
        "UltravoxWhisperEncoderModel": ("ultravox", "UltravoxWhisperEncoderModel"),
        "VoxtralWhisperEncoderModel": ("ultravox", "VoxtralWhisperEncoderModel"),
        "FalconH1Model": ("falcon_h1", "FalconH1Model"),
        "CogVLMVisionModel": ("cogvlm", "CogVLMVisionModel"),
        "JanusProVisionModel": ("janus_pro", "JanusProVisionModel"),
    }
    if name not in model_map:
        raise ValueError(f"Unknown model class: {name}")
    module_name, class_name = model_map[name]
    module = __import__(f"conversion.{module_name}", fromlist=[class_name])
    return getattr(module, class_name)
