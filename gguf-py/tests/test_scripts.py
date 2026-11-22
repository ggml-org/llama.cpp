import argparse

import pytest

from gguf import GGUFReader, GGUFWriter
from gguf.scripts import gguf_set_metadata


def test_parse_bool():
    for value in ("1", "true", "TRUE", "yes", "on"):
        assert gguf_set_metadata.parse_bool(value) is True
    for value in ("0", "false", "False", "no", "off"):
        assert gguf_set_metadata.parse_bool(value) is False
    with pytest.raises(ValueError):
        gguf_set_metadata.parse_bool("maybe")


def _write_bool(path, value):
    writer = GGUFWriter(path, "llama")
    writer.add_bool("test.bool", value)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.close()


def test_set_metadata_boolean_false(tmp_path):
    path = tmp_path / "bool.gguf"
    _write_bool(path, True)

    reader = GGUFReader(path, "r+")
    gguf_set_metadata.set_metadata(
        reader,
        argparse.Namespace(key="test.bool", value="false", dry_run=False, force=True),
    )
    reader.data.flush()
    del reader

    assert GGUFReader(path).fields["test.bool"].contents() is False


def test_set_metadata_rejects_invalid_boolean(tmp_path):
    path = tmp_path / "bool.gguf"
    _write_bool(path, True)

    reader = GGUFReader(path, "r+")
    with pytest.raises(SystemExit):
        gguf_set_metadata.set_metadata(
            reader,
            argparse.Namespace(key="test.bool", value="maybe", dry_run=False, force=True),
        )
