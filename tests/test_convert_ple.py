import sys

import numpy as np
import pytest
import torch
from safetensors.torch import save_file

import convert_hf_to_gguf as converter
import gguf
from conversion.qwen4exp import Qwen4ExpTextModel


@pytest.fixture
def checkpoint(tmp_path, monkeypatch):
    tensors = {
        "model.embed_tokens.weight": torch.arange(16 * 128, dtype=torch.float32).reshape(16, 128),
        "model.ple_embedding.ngram_embedding.shard_0.weight": torch.arange(12 * 32, dtype=torch.float32).reshape(12, 32),
        "model.ple_embedding.ngram_embedding.shard_1.weight": torch.arange(12 * 32, 24 * 32, dtype=torch.float32).reshape(12, 32),
        "model.ple_embedding.layer_multipliers": torch.tensor([23703573157769, 20109073645365, 8052911324071]),
        "model.ple_embedding.ngram_heads_offsets": torch.tensor([0, 6, 12, 18]),
        "model.ple_embedding.ngram_heads_vocab_sizes": torch.tensor([6, 6, 6, 6]),
    }
    save_file(tensors, tmp_path / "model.safetensors")
    hparams = {
        "architectures": ["Qwen4ExpForCausalLM"], "num_hidden_layers": 1,
        "hidden_size": 128, "num_attention_heads": 4, "num_key_value_heads": 2,
        "intermediate_size": 256, "vocab_size": 16, "rms_norm_eps": 1e-6,
        "linear_conv_kernel_dim": 4, "linear_key_head_dim": 16,
        "linear_num_key_heads": 2, "linear_num_value_heads": 4, "linear_value_head_dim": 16,
        "hc_count": 4, "hc_lowrank": 8, "indexer_n_heads": 4,
        "indexer_head_dim": 32, "indexer_budget": 16, "indexer_compress_ratio": 4,
        "layer_types": ["linear_attention"], "ple_layer_ids": [1], "ngram_size": 3,
        "heads_per_ngram": 2, "ple_conv_kernel_size": 4, "split_ngram_parts": 2,
        "eos_token_id": 0,
    }

    def set_vocab(model):
        model.gguf_writer.add_tokenizer_model("none")
        model.gguf_writer.add_vocab_size(16)

    monkeypatch.setattr(Qwen4ExpTextModel, "no_mtp", True)
    monkeypatch.setattr(Qwen4ExpTextModel, "set_vocab", set_vocab)
    return tmp_path, hparams


@pytest.mark.parametrize("ftype", [gguf.LlamaFileType.ALL_F32, gguf.LlamaFileType.MOSTLY_F16, gguf.LlamaFileType.MOSTLY_Q8_0])
@pytest.mark.parametrize("use_temp_file", [False, True])
def test_qwen_ple_exports(checkpoint, ftype, use_temp_file):
    directory, hparams = checkpoint
    readers = {}
    for mode in ["full", "ple", "backbone"]:
        path = directory / f"{mode}.gguf"
        model = Qwen4ExpTextModel(directory, ftype, path, hparams=hparams,
                                  ple_only=mode == "ple", no_ple=mode == "backbone", use_temp_file=use_temp_file)
        model.write()
        readers[mode] = gguf.GGUFReader(path)

    tensors = {mode: {t.name: t for t in reader.tensors} for mode, reader in readers.items()}
    table = "per_layer_token_embd.weight"
    assert set(tensors["ple"]) == {table}
    assert set(tensors["backbone"]) == {"token_embd.weight"}
    assert set(tensors["full"]) == set(tensors["ple"]) | set(tensors["backbone"])
    for mode in ["ple", "backbone"]:
        for name, tensor in tensors[mode].items():
            assert tensor.tensor_type == tensors["full"][name].tensor_type
            np.testing.assert_array_equal(tensor.data, tensors["full"][name].data)
        for key, field in readers["full"].fields.items():
            if key.startswith("qwen4exp.ple.") or key == "qwen4exp.embedding_length_per_layer_input":
                assert readers[mode].fields[key].contents() == field.contents()
    assert readers["backbone"].fields["tokenizer.ggml.model"].contents() == "none"
    assert "tokenizer.ggml.model" not in readers["ple"].fields
    assert readers["backbone"].fields["qwen4exp.ple.layer_multipliers"].contents()[0] == 23703573157769


def test_no_ple_skips_table_data(checkpoint, monkeypatch):
    directory, hparams = checkpoint

    def load_shard(*args):
        raise AssertionError("excluded PLE table was read")

    monkeypatch.setattr(Qwen4ExpTextModel, "_load_ple_shard", lambda *args: load_shard)
    model = Qwen4ExpTextModel(directory, gguf.LlamaFileType.ALL_F32, directory / "backbone.gguf",
                              hparams=hparams, no_ple=True)
    model.write()


@pytest.mark.parametrize("mode", ["ple_only", "no_ple"])
def test_default_names_and_missing_table(checkpoint, mode):
    directory, hparams = checkpoint
    model = Qwen4ExpTextModel(directory, gguf.LlamaFileType.ALL_F32, directory, hparams=hparams, **{mode: True})
    model.write()
    assert model.fname_out.name.startswith("ple-" if mode == "ple_only" else "no-ple-")
    hparams = dict(hparams, ple_layer_ids=[])
    save_file({"model.embed_tokens.weight": torch.zeros(16, 128)}, directory / "model.safetensors")
    path = directory / "missing.gguf"
    model = Qwen4ExpTextModel(directory, gguf.LlamaFileType.ALL_F32, path, hparams=hparams, **{mode: True})
    with pytest.raises(ValueError, match="no per-layer embedding table"):
        model.write()
    assert not path.exists()


@pytest.mark.parametrize("flags", [
    ["--ple-only", "--no-ple"], ["--ple-only", "--vocab-only"], ["--no-ple", "--vocab-only"],
    ["--ple-only", "--mtp"], ["--no-ple", "--mtp"], ["--ple-only", "--mmproj"],
    ["--no-ple", "--dspark"], ["--ple-only", "--no-tensor-first-split"],
])
def test_invalid_ple_arguments(monkeypatch, flags):
    monkeypatch.setattr(sys, "argv", ["convert_hf_to_gguf.py", "model", *flags])
    with pytest.raises(SystemExit) as error:
        converter.parse_args()
    assert error.value.code == 2


@pytest.mark.parametrize("flag, expected", [("--ple-only", "per_layer_token_embd.weight"), ("--no-ple", "token_embd.weight")])
def test_converter_cli(checkpoint, monkeypatch, flag, expected):
    directory, hparams = checkpoint
    path = directory / "cli.gguf"
    monkeypatch.setattr(converter.ModelBase, "load_hparams", lambda *args, **kwargs: hparams)
    monkeypatch.setattr(sys, "argv", ["convert_hf_to_gguf.py", str(directory), flag, "--no-mtp",
                                      "--outtype", "f32", "--outfile", str(path)])
    converter.main()
    assert [tensor.name for tensor in gguf.GGUFReader(path).tensors] == [expected]


def test_split_backbone_counts_exclude_ple(checkpoint):
    directory, hparams = checkpoint
    from safetensors.torch import load_file
    path = directory / "model.safetensors"
    tensors = load_file(path)
    tensors["model.layers.0.ple.key_proj.weight"] = torch.zeros(512, 128)
    save_file(tensors, path)
    model = Qwen4ExpTextModel(directory, gguf.LlamaFileType.ALL_F32, directory / "split.gguf",
                              hparams=hparams, no_ple=True, split_max_tensors=1)
    model.write()
    shards = [gguf.GGUFReader(path) for path in sorted(directory.glob("split-*.gguf"))]
    assert len(shards) == 2
    assert shards[0].fields["split.tensors.count"].contents() == 2
    assert {tensor.name for shard in shards for tensor in shard.tensors} == {"token_embd.weight", "blk.0.ple_key.weight"}


@pytest.mark.parametrize("ple_only", [False, True])
def test_gemma_ple_exports(tmp_path, monkeypatch, ple_only):
    from conversion.gemma import Gemma4Model
    save_file({"model.embed_tokens_per_layer.weight": torch.zeros(16, 32),
               "model.per_layer_model_projection.weight": torch.zeros(32, 128)}, tmp_path / "model.safetensors")
    hparams = {
        "architectures": ["Gemma4ForCausalLM"], "num_hidden_layers": 1, "hidden_size": 128,
        "num_attention_heads": 4, "num_key_value_heads": 2, "intermediate_size": 256,
        "num_kv_shared_layers": 0, "hidden_size_per_layer_input": 32,
        "layer_types": ["sliding_attention"], "global_head_dim": 32, "head_dim": 32,
        "sliding_window": 32, "rope_parameters": {
            "full_attention": {"rope_type": "proportional", "partial_rotary_factor": 0.25},
        },
    }
    monkeypatch.setattr(Gemma4Model, "set_vocab", lambda model: model.gguf_writer.add_tokenizer_model("none"))
    path = tmp_path / "gemma.gguf"
    model = Gemma4Model(tmp_path, gguf.LlamaFileType.ALL_F32, path, hparams=hparams,
                        ple_only=ple_only, no_ple=not ple_only)
    model.write()
    reader = gguf.GGUFReader(path)
    names = {tensor.name for tensor in reader.tensors}
    assert ("per_layer_token_embd.weight" in names) == ple_only
    assert ("per_layer_model_proj.weight" in names) != ple_only
    assert reader.fields["gemma4.embedding_length_per_layer_input"].contents() == 32
