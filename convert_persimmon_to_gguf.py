#!/usr/bin/env python3
import argparse
import os
import sys
from pathlib import Path
from pprint import pprint

import torch
from sentencepiece import SentencePieceProcessor

if 'NO_LOCAL_GGUF' not in os.environ:
    sys.path.insert(1, str(Path(__file__).parent / 'gguf-py'))
import gguf


def _flatten_dict(dct, tensors, prefix=None):
    assert isinstance(dct, dict)
    for key in dct.keys():
        new_prefix = prefix + '.' + key if prefix is not None else key
        if isinstance(dct[key], torch.Tensor):
            tensors[new_prefix] = dct[key]
        elif isinstance(dct[key], dict):
            _flatten_dict(dct[key], tensors, new_prefix)
        else:
            raise ValueError(type(dct[key]))
    return None


def _get_sentencepiece_tokenizer_info(dir_model: Path):
    tokenizer_path = dir_model / 'adept_vocab.model'
    print('gguf: getting sentencepiece tokenizer from', tokenizer_path)
    tokenizer = SentencePieceProcessor(str(tokenizer_path))
    print('gguf: adding tokens')
    tokens: list[bytes] = []
    scores: list[float] = []
    toktypes: list[int] = []

    for i in range(tokenizer.vocab_size()):
        text: bytes
        score: float

        piece = tokenizer.id_to_piece(i)
        text = piece.encode("utf-8")
        score = tokenizer.get_score(i)

        toktype = 1
        if tokenizer.is_unknown(i):
            toktype = 2
        if tokenizer.is_control(i):
            toktype = 3
        if tokenizer.is_unused(i):
            toktype = 5
        if tokenizer.is_byte(i):
            toktype = 6

        tokens.append(text)
        scores.append(score)
        toktypes.append(toktype)
        pass
    return tokens, scores, toktypes


def main():
    parser = argparse.ArgumentParser(description="Convert a Persimmon model from Adept (e.g. Persimmon 8b chat) to a GGML compatible file")
    parser.add_argument("--outfile",             type=Path, help="path to write to; default: based on input")
    parser.add_argument("--ckpt-path",           type=Path, help="path to persimmon checkpoint .pt file")
    parser.add_argument("--model-dir",           type=Path, help="directory containing model e.g. 8b_chat_model_release")
    parser.add_argument("--adept-inference-dir", type=str, help="path to adept-inference code directory")
    args = parser.parse_args()
    sys.path.append(str(args.adept_inference_dir))
    persimmon_model = torch.load(args.ckpt_path)
    hparams = persimmon_model['args']
    pprint(hparams)
    tensors: dict[str, torch.Tensor] = {}
    _flatten_dict(persimmon_model['model'], tensors, None)

    arch = gguf.MODEL_ARCH.PERSIMMON
    gguf_writer = gguf.GGUFWriter(args.outfile, gguf.MODEL_ARCH_NAMES[arch])

    block_count = hparams.num_layers
    head_count = hparams.num_attention_heads
    head_count_kv = head_count
    ctx_length = hparams.seq_length
    hidden_size = hparams.hidden_size

    gguf_writer.add_name('persimmon-8b-chat')
    gguf_writer.add_context_length(ctx_length)
    gguf_writer.add_embedding_length(hidden_size)
    gguf_writer.add_block_count(block_count)
    gguf_writer.add_feed_forward_length(hparams.ffn_hidden_size)
    # ref: https://github.com/ggerganov/llama.cpp/pull/4889/commits/eea19039fc52ea2dbd1aab45b59ab4e3e29a3443
    gguf_writer.add_rope_dimension_count(hidden_size // head_count // 2)
    gguf_writer.add_head_count(head_count)
    gguf_writer.add_head_count_kv(head_count_kv)
    gguf_writer.add_rope_freq_base(hparams.rotary_emb_base)
    gguf_writer.add_layer_norm_eps(hparams.layernorm_epsilon)

    tokens, scores, toktypes = _get_sentencepiece_tokenizer_info(args.model_dir)
    gguf_writer.add_tokenizer_model('llama')
    gguf_writer.add_token_list(tokens)
    gguf_writer.add_token_scores(scores)
    gguf_writer.add_token_types(toktypes)
    gguf_writer.add_bos_token_id(71013)
    gguf_writer.add_eos_token_id(71013)

    tensor_map = gguf.get_tensor_name_map(arch, block_count)
    print(tensor_map)
    for name in tensors.keys():
        data = tensors[name]
        if name.endswith(".self_attention.rotary_emb.inv_freq"):
            continue
        old_dtype = data.dtype
        # TODO: FP16 conversion produces garbage outputs. (Q8_0 does not, so..?)
        data = data.to(torch.float32).squeeze().numpy()
        new_name = tensor_map.get_name(name, try_suffixes = (".weight", ".bias"))
        if new_name is None:
            print("Can not map tensor '" + name + "'")
            sys.exit()
        n_dims = len(data.shape)
        print(new_name + ", n_dims = " + str(n_dims) + ", " + str(old_dtype) + " --> " + str(data.dtype))
        gguf_writer.add_tensor(new_name, data)
    print("gguf: write header")
    gguf_writer.write_header_to_file()
    print("gguf: write metadata")
    gguf_writer.write_kv_data_to_file()
    print("gguf: write tensors")
    gguf_writer.write_tensors_to_file()

    gguf_writer.close()

    print(f"gguf: model successfully exported to '{args.outfile}'")
    print("")


if __name__ == '__main__':
    main()

