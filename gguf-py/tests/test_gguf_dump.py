import argparse
import struct

import pytest

from gguf.gguf_reader import GGUFReader
from gguf.scripts.gguf_dump import dump_markdown_metadata, dump_metadata, sanitize_control_chars


def _write_hostile_gguf(path, key, tensor_name):
    buf = b'GGUF' + struct.pack('<IQQ', 3, 1, 1)

    value = b'innocent'
    buf += struct.pack('<Q', len(key)) + key
    buf += struct.pack('<I', 8)  # GGUFValueType.STRING
    buf += struct.pack('<Q', len(value)) + value

    buf += struct.pack('<Q', len(tensor_name)) + tensor_name
    buf += struct.pack('<I', 1)
    buf += struct.pack('<Q', 1)
    buf += struct.pack('<I', 0)  # F32
    buf += struct.pack('<Q', 0)
    buf += b'\x00' * 64

    path.write_bytes(buf)


def _make_args(path):
    return argparse.Namespace(
        model=str(path), no_tensors=False, json=False, json_array=False,
        markdown=False, data_offset=False, data_alignment=False, verbose=False,
    )


def test_sanitize_control_chars_passthrough():
    assert sanitize_control_chars('blk.1.attn_q.weight') == 'blk.1.attn_q.weight'
    assert sanitize_control_chars('') == ''


def test_sanitize_control_chars_escapes_c0_c1():
    assert sanitize_control_chars('a\x1bb') == 'a\\x1bb'
    assert sanitize_control_chars('a\x07b') == 'a\\x07b'
    assert sanitize_control_chars('a\nb') == 'a\\x0ab'
    assert sanitize_control_chars('a\x7fb') == 'a\\x7fb'
    assert sanitize_control_chars('a\x9bb') == 'a\\x9bb'
    assert sanitize_control_chars('\x1b]52;c;aGVsbG8=\x07') == '\\x1b]52;c;aGVsbG8=\\x07'


def test_console_dump_escapes_hostile_names(tmp_path, capsys):
    p = tmp_path / 'evil.gguf'
    _write_hostile_gguf(p, b'evil\x1b]52;c;aGVsbG8=\x07key', b'blk.1.attn\x1b]8;;https://evil.example\x1b\\q.weight')
    reader = GGUFReader(p)

    dump_metadata(reader, _make_args(p))
    out = capsys.readouterr().out

    assert '\x1b' not in out
    assert '\x07' not in out
    assert 'evil\\x1b]52;c;aGVsbG8=\\x07key' in out
    assert 'blk.1.attn\\x1b]8;;https://evil.example\\x1b\\q.weight' in out


def test_console_dump_leaves_clean_names_untouched(tmp_path, capsys):
    p = tmp_path / 'clean.gguf'
    _write_hostile_gguf(p, b'general.name', b'blk.1.attn_q.weight')
    reader = GGUFReader(p)

    dump_metadata(reader, _make_args(p))
    out = capsys.readouterr().out

    assert 'general.name' in out
    assert 'blk.1.attn_q.weight' in out
    assert '\\x' not in out


def test_markdown_dump_escapes_hostile_names(tmp_path, capsys):
    p = tmp_path / 'evil.gguf'
    _write_hostile_gguf(p, b'evil\x1b]52;c;aGVsbG8=\x07key', b'blk.1.attn\x1b]8;;https://evil.example\x1b\\q.weight')
    reader = GGUFReader(p)

    dump_markdown_metadata(reader, _make_args(p))
    out = capsys.readouterr().out

    assert '\x1b' not in out
    assert '\x07' not in out
    assert 'evil\\x1b]52;c;aGVsbG8=\\x07key' in out
    assert 'blk.1.attn\\x1b]8;;https://evil.example\\x1b\\q.weight' in out
