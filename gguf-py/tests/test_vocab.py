import json

import pytest

from gguf.vocab import SpecialVocab


def _write_tokenizer(tmp_path, merges):
    (tmp_path / 'tokenizer.json').write_text(json.dumps({
        'model': {'merges': merges},
        'added_tokens': [],
    }))


def _load(tmp_path, merges):
    _write_tokenizer(tmp_path, merges)
    return SpecialVocab(tmp_path, load_merges = True)


def test_merges_str_list(tmp_path):
    v = _load(tmp_path, ['a b', 'c d'])
    assert v.merges == ['a b', 'c d']


def test_merges_pair_list(tmp_path):
    v = _load(tmp_path, [['a', 'b'], ['c', 'd']])
    assert v.merges == ['a b', 'c d']


def test_merges_pair_list_with_space(tmp_path):
    v = _load(tmp_path, [['a', ' b'], ['x', 'y']])
    assert v.merges == ['a ' + chr(ord(' ') + 256) + 'b', 'x y']


@pytest.mark.parametrize('merges', [
    [['a', 'b'], 42],            # non-pair element after a valid pair
    [['a', 'b', 'c']],           # 3-element pair silently mangled before
    [['a', 5]],                  # non-str element inside a pair
    [['a', 'b'], 'x'],           # bare string inside a pair list
    [[['a'], 'b']],              # nested list inside a pair
    [['a', 'b'], None],          # null element
])
def test_merges_pair_list_malformed(tmp_path, merges):
    # the first element used to decide the format; the rest were converted
    # unchecked, so these crashed with TypeError or produced wrong merges
    with pytest.raises(ValueError, match='Unknown tokenizer merges format'):
        _load(tmp_path, merges)


def test_merges_empty(tmp_path):
    v = _load(tmp_path, [])
    assert v.merges is None or v.merges == []
