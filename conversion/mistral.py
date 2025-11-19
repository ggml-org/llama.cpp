from __future__ import annotations
from .base import (
    gguf, MistralVocab, MistralTokenizerType,
    _mistral_common_installed, _mistral_import_error_msg
)
from pathlib import Path
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    pass
from .llama import LlamaModel

if _mistral_common_installed:
    from mistral_common.tokens.tokenizers.base import Tokenizer as TokenizerBase
    from mistral_common.tokens.tokenizers.tekken import Tekkenizer
    from mistral_common.tokens.tokenizers.sentencepiece import SentencePieceTokenizer
    from mistral_common.tokens.tokenizers.base import TokenizerVersion
else:
    TokenizerBase = None
    Tekkenizer = None
    SentencePieceTokenizer = None
    TokenizerVersion = None


class MistralModel(LlamaModel):
    model_arch = gguf.MODEL_ARCH.LLAMA
    model_name = "Mistral"
    hf_arch = ""
    is_mistral_format = True
    undo_permute = False

    @staticmethod
    def get_community_chat_template(vocab: MistralVocab, templates_dir: Path, is_mistral_format: bool):
        assert TokenizerVersion is not None and Tekkenizer is not None and SentencePieceTokenizer is not None, _mistral_import_error_msg
        assert isinstance(vocab.tokenizer, (Tekkenizer, SentencePieceTokenizer)), (
            f"Expected Tekkenizer or SentencePieceTokenizer, got {type(vocab.tokenizer)}"
        )

        if vocab.tokenizer.version == TokenizerVersion.v1:
            return "mistral-v1"
        elif vocab.tokenizer.version == TokenizerVersion.v3 and vocab.tokenizer_type == MistralTokenizerType.spm:
            return "mistral-v3"
        elif vocab.tokenizer.version == TokenizerVersion.v3 and vocab.tokenizer_type == MistralTokenizerType.tekken:
            return "mistral-v3-tekken"
        elif vocab.tokenizer.version == TokenizerVersion.v7 and vocab.tokenizer_type == MistralTokenizerType.spm:
            return "mistral-v7"
        elif vocab.tokenizer.version == TokenizerVersion.v7 and vocab.tokenizer_type == MistralTokenizerType.tekken:
            return "mistral-v7-tekken"
        elif vocab.tokenizer.version == TokenizerVersion.v11:
            template_file = "Mistral-Small-3.2-24B-Instruct-2506.jinja"
        elif vocab.tokenizer.version == TokenizerVersion.v13:
            template_file = "unsloth-mistral-Devstral-Small-2507.jinja"
        else:
            err_message = f"Unknown tokenizer type: {vocab.tokenizer_type} and version {vocab.tokenizer.version}"
            if is_mistral_format:
                err_message += (
                    " . Please pass --disable-mistral-community-chat-template argument to the CLI "
                    "if you want to skip this error and use the Mistral official `mistral-common` pre-processing library."
                )
            raise ValueError(err_message)

        template_path = templates_dir / template_file
        if not template_path.exists():
            raise FileNotFoundError(f"Template file not found: {template_path}")

        with open(template_path, "r", encoding="utf-8") as f:
            template = f.read()

        return template
