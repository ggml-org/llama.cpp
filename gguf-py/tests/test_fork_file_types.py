import re
from pathlib import Path

import pytest
from gguf import GGUFReader, GGUFWriter, LlamaFileType


@pytest.mark.parametrize('name,value', [('Q8_CR', 512), ('Q5_CR', 513), ('Q6_CR', 514)])
def test_cr_file_type_serialization(tmp_path, name, value):
    header = (Path(__file__).resolve().parents[2] / 'include/llama.h').read_text()
    match = re.search(rf'LLAMA_FTYPE_MOSTLY_{name}\s*=\s*(\d+)', header)
    assert match is not None and int(match[1]) == value
    assert LlamaFileType[f'MOSTLY_{name}'] == value
    assert value < LlamaFileType.GUESSED

    path = tmp_path / 'file-type.gguf'
    writer = GGUFWriter(path, 'llama')
    writer.add_file_type(value)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.close()
    reader = GGUFReader(path)
    assert reader.get_field('general.file_type').contents() == value
