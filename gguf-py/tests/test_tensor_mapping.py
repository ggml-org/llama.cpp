import pytest

from gguf import MODEL_ARCH, get_tensor_name_map


@pytest.mark.parametrize("arch, prefix, tensors", [
    (MODEL_ARCH.QWEN3NEXT, "model.layers.0.linear_attn", ("conv1d", "norm", "out_proj")),
    (MODEL_ARCH.MAMBA, "backbone.layers.0.mixer", ("conv1d", "out_proj")),
])
def test_gigachat4_mapping_isolation(arch, prefix, tensors):
    expected = {"conv1d": "ssm_conv1d", "norm": "ssm_norm", "out_proj": "ssm_out"}
    before = get_tensor_name_map(arch, 1)
    gigachat = get_tensor_name_map(MODEL_ARCH.GIGACHAT4, 1)
    after = get_tensor_name_map(arch, 1)

    for name in tensors:
        key = f"{prefix}.{name}.weight"
        target = f"blk.0.{expected[name]}.weight"
        assert before.get_name(key, (".weight",)) == target
        assert after.get_name(key, (".weight",)) == target

    for name, target in expected.items():
        key = f"model.layers.0.self_attn.{name}.weight"
        assert gigachat.get_name(key, (".weight",)) == f"blk.0.{target}.weight"

    assert after.mapping == before.mapping
    assert get_tensor_name_map(MODEL_ARCH.GIGACHAT4, 1).mapping == gigachat.mapping


@pytest.mark.parametrize("source, target", [
    ("k_norm", "k_norm"),
    ("weights_proj", "proj"),
    ("wk", "attn_k"),
    ("wq_b", "attn_q_b"),
])
def test_gigachat4_indexer_mapping(source, target):
    name = f"model.layers.0.self_attn.indexer.{source}.weight"
    expected = f"blk.0.indexer.{target}.weight"
    for arch in (MODEL_ARCH.GIGACHAT4, MODEL_ARCH.DEEPSEEK32):
        assert get_tensor_name_map(arch, 1).get_name(name, (".weight",)) == expected
