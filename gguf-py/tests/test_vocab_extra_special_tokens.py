#!/usr/bin/env python3

import unittest
from pathlib import Path
import os
import sys
import json
import tempfile

# Necessary to load the local gguf package
if "NO_LOCAL_GGUF" not in os.environ and (Path(__file__).parent.parent.parent / 'gguf-py').exists():
    sys.path.insert(0, str(Path(__file__).parent.parent))

import gguf


class TestVocabExtraSpecialTokens(unittest.TestCase):
    # https://github.com/ggml-org/llama.cpp/issues/24460
    def test_extra_special_tokens_as_list(self):
        try:
            from tokenizers import Tokenizer, models, decoders
            from transformers import AutoTokenizer
        except ImportError:
            self.skipTest("tokenizers / transformers not installed")

        tok = Tokenizer(models.BPE(vocab={}, merges=[], byte_fallback=True))
        tok.decoder = decoders.Sequence([decoders.ByteFallback()])

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            tok.save(str(tmp_path / "tokenizer.json"))
            with open(tmp_path / "tokenizer_config.json", "w", encoding="utf-8") as f:
                json.dump({
                    "tokenizer_class": "PreTrainedTokenizerFast",
                    "extra_special_tokens": ["<|foo|>", "<|bar|>"],
                }, f)

            vocab = gguf.LlamaHfVocab(tmp_path)
            self.assertGreaterEqual(vocab.vocab_size_base, 0)

            # same config with a proper dict must keep working
            with open(tmp_path / "tokenizer_config.json", "w", encoding="utf-8") as f:
                json.dump({
                    "tokenizer_class": "PreTrainedTokenizerFast",
                    "extra_special_tokens": {"<|foo|>": "<|foo|>"},
                }, f)

            vocab = gguf.LlamaHfVocab(tmp_path)
            self.assertGreaterEqual(vocab.vocab_size_base, 0)


if __name__ == '__main__':
    unittest.main()
