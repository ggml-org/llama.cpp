import json
from pathlib import Path
from typing import Any

import pytest

from gguf import GGUFReader, GGUFWriter, SpecialVocab, TokenType


def _write_model_dir(model_dir: Path, model_type: str, prepend_scheme: str) -> None:
    model_dir.mkdir()
    with open(model_dir / 'tokenizer.json', 'w', encoding = 'utf-8') as f:
        json.dump({
            'model': {'type': model_type, 'vocab': {'<s>': 0}, 'merges': []},
            'pre_tokenizer': {
                'type': 'Sequence',
                'pretokenizers': [
                    {'type': 'Metaspace', 'replacement': '\u2581', 'prepend_scheme': prepend_scheme, 'split': True},
                    {'type': 'ByteLevel', 'add_prefix_space': False, 'trim_offsets': False, 'use_regex': False},
                ],
            },
            'added_tokens': [],
        }, f)
    with open(model_dir / 'tokenizer_config.json', 'w', encoding = 'utf-8') as f:
        json.dump({'bos_token': '<s>', 'eos_token': '</s>'}, f)


def _write_gguf(tmp_path: Path, tokenizer_model: str, model_type: str, prepend_scheme: str) -> dict[str, Any]:
    model_dir = tmp_path / 'model'
    _write_model_dir(model_dir, model_type, prepend_scheme)

    gguf_path = tmp_path / 'model.gguf'
    writer = GGUFWriter(gguf_path, 'modern-bert')
    writer.add_tokenizer_model(tokenizer_model)
    writer.add_token_list(['<s>', '</s>'])
    writer.add_token_types([TokenType.CONTROL, TokenType.CONTROL])
    SpecialVocab(model_dir, load_merges = False).add_to_gguf(writer)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    reader = GGUFReader(gguf_path)
    try:
        return {key: field.contents() for key, field in reader.fields.items()}
    finally:
        # release the mmap so that pytest can delete the temporary directory on Windows
        reader.data._mmap.close()


# ref: https://github.com/ggml-org/llama.cpp/issues/32882
# llama.cpp aborts with "metaspace prepend_scheme 'always' is not supported for BPE vocab"
# when a BPE vocab declares prepend_scheme "always", so it must not be written
def test_bpe_vocab_skips_metaspace_prepend_scheme_always(tmp_path):
    fields = _write_gguf(tmp_path, 'gpt2', 'BPE', 'always')
    assert 'tokenizer.ggml.metaspace.prepend_scheme' not in fields


@pytest.mark.parametrize('prepend_scheme', ['first', 'never'])
def test_bpe_vocab_keeps_supported_metaspace_prepend_scheme(tmp_path, prepend_scheme):
    fields = _write_gguf(tmp_path, 'gpt2', 'BPE', prepend_scheme)
    assert fields['tokenizer.ggml.metaspace.prepend_scheme'] == prepend_scheme


def test_spm_vocab_keeps_metaspace_prepend_scheme_always(tmp_path):
    fields = _write_gguf(tmp_path, 'llama', 'Unigram', 'always')
    assert fields['tokenizer.ggml.metaspace.prepend_scheme'] == 'always'
