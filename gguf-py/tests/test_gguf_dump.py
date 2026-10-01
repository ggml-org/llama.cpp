from types import SimpleNamespace

import pytest

from gguf import GGUFValueType
from gguf.scripts.gguf_dump import dump_markdown_metadata, sanitize_name


def test_sanitize_name_preserves_printable_text():
    assert sanitize_name("blk.0/attention_é.weight") == "blk.0/attention_é.weight"


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("field\nname", "field\\nname"),
        ("field\tname", "field\\tname"),
        ("field\x1b]52;c;payload\x07", "field\\x1b]52;c;payload\\x07"),
        ("field\x7fname", "field\\x7fname"),
        ("field\x9b31mred", "field\\x9b31mred"),
    ],
)
def test_sanitize_name_escapes_non_printable_characters(name, expected):
    assert sanitize_name(name) == expected

def test_dump_markdown_metadata_sanitizes_names(capsys):
    payload = "\x1b]52;c;payload\x07"
    field = SimpleNamespace(
        name=f"field{payload}",
        types=[GGUFValueType.UINT8],
        data=[1],
        parts=[[1]],
    )
    tensor = SimpleNamespace(
        name=f"blk.{payload}.attn_q.weight",
        n_elements=1,
        n_bytes=4,
        data_offset=0,
        shape=[1],
        tensor_type=SimpleNamespace(name="F32"),
    )
    reader = SimpleNamespace(
        endianess=SimpleNamespace(name="LITTLE"),
        byte_order="N",
        fields={"field": field},
        tensors=[tensor],
        gguf_scalar_to_np={GGUFValueType.UINT8: object()},
    )
    args = SimpleNamespace(model="model.gguf", no_tensors=False)

    dump_markdown_metadata(reader, args)

    output = capsys.readouterr().out
    assert payload not in output
    assert "\\x1b]52;c;payload\\x07" in output

