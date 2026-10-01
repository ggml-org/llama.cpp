import pytest

from gguf.scripts.gguf_dump import sanitize_name


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
