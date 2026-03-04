from __future__ import annotations

from enum import Enum
import re
import logging
import json
import os
from pathlib import Path
from typing import Any, Callable, Sequence, Mapping, Iterable, Protocol, ClassVar, runtime_checkable

try:
    from sentencepiece import SentencePieceProcessor
except ImportError:
    SentencePieceProcessor = None

try:
    from mistral_common.tokens.tokenizers.mistral import MistralTokenizer # pyright: ignore[reportMissingImports]
    from mistral_common.tokens.tokenizers.tekken import Tekkenizer # pyright: ignore[reportMissingImports]
    from mistral_common.tokens.tokenizers.utils import ( # pyright: ignore[reportMissingImports]
        _filter_valid_tokenizer_files,
    )
    from mistral_common.tokens.tokenizers.sentencepiece import ( # pyright: ignore[reportMissingImports]
        SentencePieceTokenizer,
    )
except ImportError:
    _mistral_common_installed = False
    MistralTokenizer = None
    Tekkenizer = None
    SentencePieceTokenizer = None
    _filter_valid_tokenizer_files = None
else:
    _mistral_common_installed = True

try:
    from mistral_common.tokens.tokenizers.utils import ( # pyright: ignore[reportMissingImports]
        get_one_valid_tokenizer_file,
    )
except ImportError:
    # We still want the conversion to work with older mistral-common versions.
    get_one_valid_tokenizer_file = None


import gguf

from .gguf_writer import GGUFWriter

logger = logging.getLogger(__name__)


class SpecialVocab:
    merges: list[str]
    add_special_token: dict[str, bool]
    special_token_ids: dict[str, int]
    chat_template: str | Sequence[Mapping[str, str]] | None

    def __init__(
        self, path: str | os.PathLike[str], load_merges: bool = False,
        special_token_types: Iterable[str] | None = None,
        n_vocab: int | None = None,
    ):
        self.special_token_ids = {}
        self.add_special_token = {}
        self.n_vocab = n_vocab
        self.load_merges = load_merges
        self.merges = []
        self.chat_template = None
        if special_token_types is not None:
            self.special_token_types = special_token_types
        else:
            self.special_token_types = ('bos', 'eos', 'unk', 'sep', 'pad', 'cls', 'mask')
        self._load(Path(path))

    def __repr__(self) -> str:
        return '<SpecialVocab with {} merges, special tokens {}, add special tokens {}>'.format(
            len(self.merges), self.special_token_ids or "unset", self.add_special_token or "unset",
        )

    def add_to_gguf(self, gw: GGUFWriter, quiet: bool = False) -> None:
        if self.merges:
            if not quiet:
                logger.info(f'Adding {len(self.merges)} merge(s).')
            gw.add_token_merges(self.merges)
        elif self.load_merges:
            logger.warning('Adding merges requested but no merges found, output may be non-functional.')
        for typ, tokid in self.special_token_ids.items():
            id_handler: Callable[[int], None] | None = getattr(gw, f'add_{typ}_token_id', None)
            if id_handler is None:
                logger.warning(f'No handler for special token type {typ} with id {tokid} - skipping')
                continue
            if not quiet:
                logger.info(f'Setting special token type {typ} to {tokid}')
            id_handler(tokid)
        for typ, value in self.add_special_token.items():
            add_handler: Callable[[bool], None] | None = getattr(gw, f'add_add_{typ}_token', None)
            if add_handler is None:
                logger.warning(f'No handler for add_{typ}_token with value {value} - skipping')
                continue
            if not quiet:
                logger.info(f'Setting add_{typ}_token to {value}')
            add_handler(value)
        if self.chat_template is not None:
            if not quiet:
                logger.info(f'Setting chat_template to {self.chat_template}')
            gw.add_chat_template(self.chat_template)

    def _load(self, path: Path) -> None:
        self._try_load_from_tokenizer_json(path)
        self._try_load_from_config_json(path)
        if self.load_merges and not self.merges:
            self._try_load_merges_txt(path)

    def _try_load_merges_txt(self, path: Path) -> bool:
        merges_file = path / 'merges.txt'
        if not merges_file.is_file():
            return False
        with open(merges_file, 'r', encoding = 'utf-8') as fp:
            first_line = next(fp, '').strip()
            if not first_line.startswith('#'):
                fp.seek(0)
                line_num = 0
            else:
                line_num = 1
            merges = []
            for line in fp:
                line_num += 1
                line = line.strip()
                if not line:
                    continue
                parts = line.split(None, 3)
                if len(parts) != 2:
                    logger.warning(f'{merges_file.name}: Line {line_num}: Entry malformed, ignoring')
                    continue
                merges.append(f'{parts[0]} {parts[1]}')
        self.merges = merges
        return True

    def _set_special_token(self, typ: str, tid: Any) -> None:
        if not isinstance(tid, int):
            return
        if tid < 0:
            raise ValueError(f'invalid value for special token type {typ}: {tid}')
        if self.n_vocab is None or tid < self.n_vocab:
            if typ in self.special_token_ids:
                return
            self.special_token_ids[typ] = tid
            return
        logger.warning(f'Special token type {typ}, id {tid} out of range, must be under {self.n_vocab} - skipping')

    def _try_load_from_tokenizer_json(self, path: Path) -> bool:
        tokenizer = None
        tokenizer_file = path / 'tokenizer.json'
        if tokenizer_file.is_file():
            with open(tokenizer_file, encoding = 'utf-8') as f:
                tokenizer = json.load(f)
            if self.load_merges:
                merges = tokenizer.get('model', {}).get('merges')
                if isinstance(merges, list) and merges:
                    if isinstance(merges[0], str):
                        self.merges = merges
                    elif isinstance(merges[0], list) and len(merges[0]) == 2 and isinstance(merges[0][0], str):
                        # New format since transformers 4.45 to support spaces in merges
                        # ref: https://github.com/ggml-org/llama.cpp/issues/9692
                        # TODO: internally store as the new format instead of converting to old
                        if any(' ' in s for pair in merges for s in pair):
                            logger.warning(f'Spaces in merges detected, encoding as {chr(ord(" ") + 256)!r}')
                        self.merges = [
                            ' '.join(
                                [
                                    # ensure the spaces are properly encoded
                                    ''.join(
                                        chr(ord(c) + 256) if c == ' ' else c
                                        for c in part
                                    )
                                    for part in pair
                                ]
                            )
                            for pair in merges
                        ]
                    else:
                        raise ValueError("Unknown tokenizer merges format")
            added_tokens = tokenizer.get('added_tokens', {})
        else:
            added_tokens = {}
        tokenizer_config = None
        tokenizer_config_file = path / 'tokenizer_config.json'
        if tokenizer_config_file.is_file():
            with open(tokenizer_config_file, encoding = 'utf-8') as f:
                tokenizer_config = json.load(f)
        if tokenizer:
            special_bos = (tokenizer_config or {}).get('bos_token')
            special_cls = (tokenizer_config or {}).get('cls_token')
            special_eos = (tokenizer_config or {}).get('eos_token')
            special_sep = (tokenizer_config or {}).get('sep_token')
            if not special_bos and special_cls and tokenizer_config:
                tokenizer_config['bos_token'] = special_bos = special_cls
            if not special_eos and special_sep and tokenizer_config:
                tokenizer_config['eos_token'] = special_eos = special_sep
            if post_processor := tokenizer.get('post_processor'):
                for processor in post_processor.get('processors', [post_processor]):
                    if processor.get('type') == 'RobertaProcessing':
                        self.add_special_token['bos'] = True
                        self.add_special_token['eos'] = True
                        self.add_special_token['sep'] = True
                        if not special_cls and tokenizer_config:
                            special_cls = processor.get('cls', [special_bos])[0]
                            tokenizer_config['cls_token'] = special_cls
                        if not special_sep and tokenizer_config:
                            special_sep = processor.get('sep', [special_eos])[0]
                            tokenizer_config['sep_token'] = special_sep
                        continue
                    # Crude parsing of TemplateProcessing to determine if BOS/SEP/EOS should be added
                    # Only works with simple templates, **will** get it wrong on unusual sequences
                    if processor.get('type') == 'TemplateProcessing':
                        tmpl_single = processor.get('single', [])
                        tmpl_pair = processor.get('pair', [])
                        special_first = None
                        special_last = None
                        if len(tmpl_single) > 1:
                            if special_first := tmpl_single[0].get('SpecialToken', {}).get('id'):
                                if not tokenizer_config:
                                    special_bos = special_first
                                self.add_special_token['bos'] = True if special_first in (special_bos, special_cls) else False
                                if special_first not in (special_bos, special_cls):
                                    logger.warning(f'Unknown leading special token {special_first!r} in TemplateProcessing<single>')
                            if special_last := tmpl_single[-1].get('SpecialToken', {}).get('id'):
                                if not tokenizer_config:
                                    special_eos = special_last
                                elif special_last != special_eos:
                                    if 'eot' not in self.special_token_types:
                                        self.special_token_types = tuple(self.special_token_types) + ('eot', )
                                        tokenizer_config['eot_token'] = special_eos
                                    elif 'eom' not in self.special_token_types:
                                        self.special_token_types = tuple(self.special_token_types) + ('eom', )
                                        tokenizer_config['eom_token'] = special_eos
                                    else:
                                        logger.warning(f'Overriding EOS token {special_eos!r} with {special_last!r} without EOT/EOM fallback!')
                                    tokenizer_config['eos_token'] = special_eos = special_last
                                self.add_special_token['eos'] = True if special_last == special_eos else False
                                if special_last != special_eos:
                                    logger.warning(f'Unknown trailing special token {special_last!r} in TemplateProcessing<single>')
                        if tmpl_pair:
                            seq_start = 1 if special_first and tmpl_pair[0].get('SpecialToken', {}).get('id') == special_first else 0
                            seq_stop = -1 if special_last and tmpl_pair[-1].get('SpecialToken', {}).get('id') == special_last else None
                            if (special_first and seq_start == 0) or (special_last and seq_stop is None):
                                logger.warning('TemplateProcessing<single> leading/trailing special tokens do not match TemplateProcessing<pair>')
                            if tmpl_pair := tmpl_pair[slice(seq_start, seq_stop)]:
                                tmpl_a = tmpl_pair[0].get('Sequence', {}).get('id')
                                tmpl_b = tmpl_pair[-1].get('Sequence', {}).get('id')
                                if tmpl_a != 'A' or tmpl_b != 'B':
                                    logger.warning(f'Unknown sequence {tmpl_a}...{tmpl_b} in TemplateProcessing<pair>')
                                # A [sep] [eos] B
                                if tmpl_a == 'A' and tmpl_b == 'B' and (tmpl_pair := tmpl_pair[1:-1]):
                                    add_sep = False
                                    if special_entry := tmpl_pair[0].get('SpecialToken', {}).get('id'):
                                        if special_entry in (special_sep, special_eos) and not special_last:
                                            add_sep = True
                                        if special_entry not in (special_sep, special_eos):
                                            logger.warning(f'Unknown separator token {special_entry!r} in TemplateProcessing<pair>')
                                    else:
                                        logger.warning(f'Unknown middle sequence {tmpl_pair[0]!r} in TemplateProcessing<pair>')
                                    if len(tmpl_pair) == 2:
                                        if special_entry := tmpl_pair[1].get('SpecialToken', {}).get('id'):
                                            if special_entry in (special_sep, special_eos):
                                                add_sep = True
                                            if special_entry not in (special_sep, special_eos):
                                                logger.warning(f'Unknown second separator token {special_entry!r} in TemplateProcessing<pair>')
                                        else:
                                            logger.warning(f'Unknown second middle sequence {tmpl_pair[1]!r} in TemplateProcessing<pair>')
                                    self.add_special_token['sep'] = add_sep
                                    if add_sep and not special_sep and tokenizer_config:
                                        tokenizer_config['sep_token'] = special_eos
                        continue
        if not tokenizer_config:
            return True
        chat_template_alt = None
        chat_template_json = path / 'chat_template.json'
        chat_template_jinja = path / 'chat_template.jinja'
        if chat_template_jinja.is_file():
            with open(chat_template_jinja, encoding = 'utf-8') as f:
                chat_template_alt = f.read()
            if additional_templates := list((path / 'additional_chat_templates').glob('*.jinja')):
                chat_template_alt = [{'name': 'default', 'template': chat_template_alt}]
                for template_path in additional_templates:
                    with open(template_path, encoding = 'utf-8') as fp:
                        chat_template_alt.append({'name': template_path.stem, 'template': fp.read()})
        elif chat_template_json.is_file():
            with open(chat_template_json, encoding = 'utf-8') as f:
                chat_template_alt = json.load(f).get('chat_template')
        chat_template = tokenizer_config.get('chat_template', chat_template_alt)
        if chat_template is None or isinstance(chat_template, (str, list)):
            self.chat_template = chat_template
        else:
            logger.warning(f'Bad type for chat_template field in {tokenizer_config_file!r} - ignoring')
        for typ in self.special_token_types:
            add_entry = tokenizer_config.get(f'add_{typ}_token')
            if isinstance(add_entry, bool):
                self.add_special_token[typ] = add_entry
            entry = tokenizer_config.get(f'{typ}_token')
            if isinstance(entry, str):
                tc_content = entry
            elif isinstance(entry, dict):
                entry_content = entry.get('content')
                if not isinstance(entry_content, str):
                    continue
                tc_content = entry_content
            else:
                continue
            # We only need the first match here.
            maybe_token_id = next(
                (atok.get('id') for atok in added_tokens if atok.get('content') == tc_content),
                None,
            )
            self._set_special_token(typ, maybe_token_id)
        return True

    def _try_load_from_config_json(self, path: Path) -> bool:
        config_file = path / 'config.json'
        if not config_file.is_file():
            return False
        with open(config_file, encoding = 'utf-8') as f:
            config = json.load(f)
        for typ in self.special_token_types:
            token_id = config.get(f'{typ}_token_id')
            # If not found at root, check in text_config (for multimodal models like Kimi-VL)
            if token_id is None and 'text_config' in config:
                token_id = config['text_config'].get(f'{typ}_token_id')
            self._set_special_token(typ, token_id)
        return True


@runtime_checkable
class BaseVocab(Protocol):
    tokenizer_model: ClassVar[str]
    name: ClassVar[str]


@runtime_checkable
class Vocab(BaseVocab, Protocol):
    vocab_size: int
    added_tokens_dict: dict[str, int]
    added_tokens_list: list[str]
    fname_tokenizer: Path

    def __init__(self, base_path: Path): ...
    def all_tokens(self) -> Iterable[tuple[bytes, float, gguf.TokenType]]: ...


class NoVocab(BaseVocab):
    tokenizer_model = "no_vocab"
    name = "no_vocab"

    def __repr__(self) -> str:
        return "<NoVocab for a model without integrated vocabulary>"


class BpeVocab(Vocab):
    tokenizer_model = "gpt2"
    name = "bpe"

    def __init__(self, base_path: Path):
        added_tokens: dict[str, int] = {}

        if (fname_tokenizer := base_path / 'vocab.json').exists():
            # "slow" tokenizer
            with open(fname_tokenizer, encoding="utf-8") as f:
                self.vocab = json.load(f)

            try:
                # FIXME: Verify that added tokens here _cannot_ overlap with the main vocab.
                with open(base_path / 'added_tokens.json', encoding="utf-8") as f:
                    added_tokens = json.load(f)
            except FileNotFoundError:
                pass
        else:
            # "fast" tokenizer
            fname_tokenizer = base_path / 'tokenizer.json'

            # if this fails, FileNotFoundError propagates to caller
            with open(fname_tokenizer, encoding="utf-8") as f:
                tokenizer_json = json.load(f)

            tokenizer_model: dict[str, Any] = tokenizer_json['model']
            if (
                tokenizer_model['type'] != 'BPE' or tokenizer_model.get('byte_fallback', False)
                or tokenizer_json['decoder']['type'] != 'ByteLevel'
            ):
                raise FileNotFoundError('Cannot find GPT-2 BPE tokenizer')

            self.vocab = tokenizer_model["vocab"]

            if (added := tokenizer_json.get('added_tokens')) is not None:
                # Added tokens here can be duplicates of the main vocabulary.
                added_tokens = {item['content']: item['id']
                                for item in added
                                if item['content'] not in self.vocab}

        vocab_size   = len(self.vocab)
        expected_ids = list(range(vocab_size, vocab_size + len(added_tokens)))
        actual_ids   = sorted(added_tokens.values())
        if expected_ids != actual_ids:
            expected_end_id = vocab_size + len(actual_ids) - 1
            raise ValueError(f"Expected the {len(actual_ids)} added token ID(s) to be sequential in the range "
                             f"{vocab_size} - {expected_end_id}; got {actual_ids}")

        items = sorted(added_tokens.items(), key=lambda text_idx: text_idx[1])
        self.added_tokens_dict    = added_tokens
        self.added_tokens_list    = [text for (text, idx) in items]
        self.vocab_size_base      = vocab_size
        self.vocab_size           = self.vocab_size_base + len(self.added_tokens_list)
        self.fname_tokenizer      = fname_tokenizer

    def bpe_tokens(self) -> Iterable[tuple[bytes, float, gguf.TokenType]]:
        reverse_vocab = {id: encoded_tok for encoded_tok, id in self.vocab.items()}

        for i, _ in enumerate(self.vocab):
            yield reverse_vocab[i], 0.0, gguf.TokenType.NORMAL

    def added_tokens(self) -> Iterable[tuple[bytes, float, gguf.TokenType]]:
        for text in self.added_tokens_list:
            score = -1000.0
            yield text.encode("utf-8"), score, gguf.TokenType.CONTROL

    def all_tokens(self) -> Iterable[tuple[bytes, float, gguf.TokenType]]:
        yield from self.bpe_tokens()
        yield from self.added_tokens()

    def __repr__(self) -> str:
        return f"<BpeVocab with {self.vocab_size_base} base tokens and {len(self.added_tokens_list)} added tokens>"


class SentencePieceVocab(Vocab):
    tokenizer_model = "llama"
    name = "spm"

    def __init__(self, base_path: Path):
        if SentencePieceProcessor is None:
            raise RuntimeError("sentencepiece is not installed")

        added_tokens: dict[str, int] = {}
        if (fname_tokenizer := base_path / 'tokenizer.model').exists():
            # normal location
            try:
                with open(base_path / 'added_tokens.json', encoding="utf-8") as f:
                    added_tokens = json.load(f)
            except FileNotFoundError:
                pass
        elif not (fname_tokenizer := base_path.parent / 'tokenizer.model').exists():
            # not found in alternate location either
            raise FileNotFoundError('Cannot find tokenizer.model')

        self.sentencepiece_tokenizer = SentencePieceProcessor()
        self.sentencepiece_tokenizer.LoadFromFile(str(fname_tokenizer))
        vocab_size = self.sentencepiece_tokenizer.vocab_size()

        new_tokens       = {id: piece for piece, id in added_tokens.items() if id >= vocab_size}
        expected_new_ids = list(range(vocab_size, vocab_size + len(new_tokens)))
        actual_new_ids   = sorted(new_tokens.keys())

        if expected_new_ids != actual_new_ids:
            raise ValueError(f"Expected new token IDs {expected_new_ids} to be sequential; got {actual_new_ids}")

        # Token pieces that were added to the base vocabulary.
        self.added_tokens_dict  = added_tokens
        self.added_tokens_list  = [new_tokens[id] for id in actual_new_ids]
        self.vocab_size_base    = vocab_size
        self.vocab_size         = self.vocab_size_base + len(self.added_tokens_list)
        self.fname_tokenizer    = fname_tokenizer

    def sentencepiece_tokens(self) -> Iterable[tuple[bytes, float, gguf.TokenType]]:
        tokenizer = self.sentencepiece_tokenizer
        for i in range(tokenizer.vocab_size()):
            piece = tokenizer.IdToPiece(i)
            text         = piece.encode("utf-8")
            score: float = tokenizer.GetScore(i)

            toktype = gguf.TokenType.NORMAL
            if tokenizer.IsUnknown(i):
                toktype = gguf.TokenType.UNKNOWN
            if tokenizer.IsControl(i):
                toktype = gguf.TokenType.CONTROL

            # NOTE: I think added_tokens are user defined.
            # ref: https://github.com/google/sentencepiece/blob/master/src/sentencepiece_model.proto
            # if tokenizer.is_user_defined(i): toktype = gguf.TokenType.USER_DEFINED

            if tokenizer.IsUnused(i):
                toktype = gguf.TokenType.UNUSED
            if tokenizer.IsByte(i):
                toktype = gguf.TokenType.BYTE

            yield text, score, toktype

    def added_tokens(self) -> Iterable[tuple[bytes, float, gguf.TokenType]]:
        for text in self.added_tokens_list:
            score = -1000.0
            yield text.encode("utf-8"), score, gguf.TokenType.USER_DEFINED

    def all_tokens(self) -> Iterable[tuple[bytes, float, gguf.TokenType]]:
        yield from self.sentencepiece_tokens()
        yield from self.added_tokens()

    def __repr__(self) -> str:
        return f"<SentencePieceVocab with {self.vocab_size_base} base tokens and {len(self.added_tokens_list)} added tokens>"


class LlamaHfVocab(Vocab):
    tokenizer_model = "llama"
    name = "hfft"

    def __init__(self, base_path: Path):
        fname_tokenizer = base_path / 'tokenizer.json'
        # if this fails, FileNotFoundError propagates to caller
        with open(fname_tokenizer, encoding='utf-8') as f:
            tokenizer_json = json.load(f)

        # pre-check so we know if we need transformers
        tokenizer_model: dict[str, Any] = tokenizer_json['model']
        is_llama3 = (
            tokenizer_model['type'] == 'BPE' and tokenizer_model.get('ignore_merges', False)
            and not tokenizer_model.get('byte_fallback', True)
        )
        if is_llama3:
            raise TypeError('Llama 3 must be converted with BpeVocab')

        if not is_llama3 and (
            tokenizer_model['type'] != 'BPE' or not tokenizer_model.get('byte_fallback', False)
            or tokenizer_json['decoder']['type'] != 'Sequence'
        ):
            raise FileNotFoundError('Cannot find Llama BPE tokenizer')

        try:
            from transformers import AutoTokenizer
        except ImportError as e:
            raise ImportError(
                "To use LlamaHfVocab, please install the `transformers` package. "
                "You can install it with `pip install transformers`."
            ) from e

        # Allow the tokenizer to default to slow or fast versions.
        # Explicitly set tokenizer to use local paths.
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_path,
            cache_dir=base_path,
            local_files_only=True,
        )
        assert self.tokenizer.is_fast  # assume tokenizer.json is used

        # Initialize lists and dictionaries for added tokens
        self.added_tokens_list = []
        self.added_tokens_dict = dict()
        self.added_tokens_ids  = set()

        # Process added tokens
        for tok, tokidx in sorted(
            self.tokenizer.get_added_vocab().items(), key=lambda x: x[1]
        ):
            # Only consider added tokens that are not in the base vocabulary
            if tokidx >= self.tokenizer.vocab_size:
                self.added_tokens_list.append(tok)
                self.added_tokens_dict[tok] = tokidx
                self.added_tokens_ids.add(tokidx)

        # Store special tokens and their IDs
        self.specials = {
            tok: self.tokenizer.get_vocab()[tok]
            for tok in self.tokenizer.all_special_tokens
        }
        self.special_ids = set(self.tokenizer.all_special_ids)

        # Set vocabulary sizes
        self.vocab_size_base = self.tokenizer.vocab_size
        self.vocab_size      = self.vocab_size_base + len(self.added_tokens_list)

        self.fname_tokenizer = fname_tokenizer

    def hf_tokens(self) -> Iterable[tuple[bytes, float, gguf.TokenType]]:
        reverse_vocab = {
            id: encoded_tok for encoded_tok, id in self.tokenizer.get_vocab().items()
        }

        for token_id in range(self.vocab_size_base):
            # Skip processing added tokens here
            if token_id in self.added_tokens_ids:
                continue

            # Convert token text to bytes
            token_text = reverse_vocab[token_id].encode("utf-8")

            # Yield token text, score, and type
            yield token_text, self.get_token_score(token_id), self.get_token_type(
                token_id, token_text, self.special_ids  # Reuse already stored special IDs
            )

    def get_token_type(self, token_id: int, token_text: bytes, special_ids: set[int]) -> gguf.TokenType:
        # Special case for byte tokens
        if re.fullmatch(br"<0x[0-9A-Fa-f]{2}>", token_text):
            return gguf.TokenType.BYTE

        # Determine token type based on whether it's a special token
        return gguf.TokenType.CONTROL if token_id in special_ids else gguf.TokenType.NORMAL

    def get_token_score(self, token_id: int) -> float:
        # Placeholder for actual logic to determine the token's score
        # This needs to be implemented based on specific requirements
        return -1000.0  # Default score

    def added_tokens(self) -> Iterable[tuple[bytes, float, gguf.TokenType]]:
        for text in self.added_tokens_list:
            if text in self.specials:
                toktype = self.get_token_type(self.specials[text], b'', self.special_ids)
                score = self.get_token_score(self.specials[text])
            else:
                toktype = gguf.TokenType.USER_DEFINED
                score = -1000.0

            yield text.encode("utf-8"), score, toktype

    def has_newline_token(self):
        return "<0x0A>" in self.tokenizer.vocab or "\n" in self.tokenizer.vocab

    def all_tokens(self) -> Iterable[tuple[bytes, float, gguf.TokenType]]:
        yield from self.hf_tokens()
        yield from self.added_tokens()

    def __repr__(self) -> str:
        return f"<LlamaHfVocab with {self.vocab_size_base} base tokens and {len(self.added_tokens_list)} added tokens>"


class MistralTokenizerType(str, Enum):
    spm = "spm"
    tekken = "tekken"


# Copied from Transformers (Apache 2.0)
# https://github.com/huggingface/transformers/blob/main/src/transformers/convert_slow_tokenizer.py#L1544

def bytes_to_unicode() -> dict[int, str]:
    """
    Returns list of utf-8 byte and a mapping to unicode strings. We specifically avoids mapping to whitespace/control
    characters the bpe code barfs on.

    The reversible bpe codes work on unicode strings. This means you need a large # of unicode characters in your vocab
    if you want to avoid UNKs. When you're at something like a 10B token dataset you end up needing around 5K for
    decent coverage. This is a significant percentage of your normal, say, 32K bpe vocab. To avoid that, we want lookup
    tables between utf-8 bytes and unicode strings.
    """
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    cs_str = [chr(n) for n in cs]
    return dict(zip(bs, cs_str))


class MistralVocab(Vocab):
    tokenizer_model = "mistral"
    name = "mistral"

    added_tokens_dict: dict[str, int] = {}
    added_tokens_list: list[str] = []

    def __init__(self, base_path: Path):
        if not _mistral_common_installed:
            raise ImportError(
                "To use MistralVocab, please install the `mistral-common` package. "
                "You can install it with `pip install mistral-common`."
            )
        assert _filter_valid_tokenizer_files is not None, "mistral_common is not installed"
        assert MistralTokenizer is not None, "mistral_common is not installed"
        assert Tekkenizer is not None, "mistral_common is not installed"

        logger.info(f"Loading Mistral tokenizer from {base_path}")

        # Find the tokenizer files
        all_files = [f.as_posix() for f in base_path.glob("**/*") if f.is_file()]

        if get_one_valid_tokenizer_file is not None:
            tokenizer_file_path = get_one_valid_tokenizer_file(all_files)
        else:
            valid_tokenizer_files = _filter_valid_tokenizer_files(all_files)

            if len(valid_tokenizer_files) == 0:
                raise ValueError(f"No tokenizer file found in the directory: {base_path}")
            # If there are multiple tokenizer files, we use tekken.json if it exists, otherwise the versioned one.
            if len(valid_tokenizer_files) > 1:
                if "tekken.json" in valid_tokenizer_files:
                    tokenizer_file = "tekken.json"
                else:
                    tokenizer_file = sorted(valid_tokenizer_files)[-1]
                logger.warning(
                    f"Multiple tokenizer files found in {base_path}. Using {tokenizer_file}"
                )
            else:
                tokenizer_file = valid_tokenizer_files[0]

            tokenizer_file_path = base_path / tokenizer_file

        self.tokenizer = MistralTokenizer.from_file(
            tokenizer_file_path
        ).instruct_tokenizer.tokenizer
        self.tokenizer_type = (
            MistralTokenizerType.tekken
            if isinstance(self.tokenizer, Tekkenizer)
            else MistralTokenizerType.spm
        )
        self.vocab_size = self.tokenizer.n_words
        self.fname_tokenizer = tokenizer_file_path
        self._name = (
            "mistral-" + self.tokenizer_type.value + "-" + self.tokenizer.version
        )

    @property
    def tokenizer_name(self) -> str:
        return self._name

    @property
    def gguf_tokenizer_model(self) -> str:
        return "llama" if self.tokenizer_type == MistralTokenizerType.spm else "gpt2"

    def _sentencepiece_tokens(self) -> Iterable[tuple[bytes, float, gguf.TokenType]]:
        assert SentencePieceTokenizer is not None, "mistral_common is not installed"
        assert isinstance(self.tokenizer, SentencePieceTokenizer), (
            f"Expected SentencePieceTokenizer, got {type(self.tokenizer)}"
        )

        for i in range(self.tokenizer._model.vocab_size()):
            piece = self.tokenizer._model.IdToPiece(i)
            text = piece.encode("utf-8")
            score: float = self.tokenizer._model.GetScore(i)

            toktype = gguf.TokenType.NORMAL
            if self.tokenizer._model.IsUnknown(i):
                toktype = gguf.TokenType.UNKNOWN
            if self.tokenizer._model.IsControl(i):
                toktype = gguf.TokenType.CONTROL

            if self.tokenizer._model.IsUnused(i):
                toktype = gguf.TokenType.UNUSED
            if self.tokenizer._model.IsByte(i):
                toktype = gguf.TokenType.BYTE

            yield text, score, toktype

    def _tekken_tokens(self) -> Iterable[tuple[bytes, float, gguf.TokenType]]:
        assert Tekkenizer is not None, "mistral_common is not installed"
        assert isinstance(self.tokenizer, Tekkenizer), (
            f"Expected Tekkenizer, got {type(self.tokenizer)}"
        )

        byte_encoder = bytes_to_unicode()
        for token_id in range(self.tokenizer.num_special_tokens):
            yield (
                self.tokenizer.id_to_piece(token_id).encode("utf-8"),
                0,
                gguf.TokenType.CONTROL
            )
        for token in self.tokenizer._tekken_token2id_nospecial:
            yield (
                self.token_bytes_to_string(token, byte_encoder).encode("utf-8"),
                0,
                gguf.TokenType.NORMAL,
            )

    def get_token_id(self, token: str) -> int:
        assert SentencePieceTokenizer is not None and Tekkenizer is not None, "mistral_common is not installed"
        if self.tokenizer_type == MistralTokenizerType.spm:
            assert isinstance(self.tokenizer, SentencePieceTokenizer)
            return self.tokenizer._vocab.index(token)
        elif self.tokenizer_type == MistralTokenizerType.tekken:
            assert isinstance(self.tokenizer, Tekkenizer)
            return (
                self.tokenizer._vocab.index(token) + self.tokenizer.num_special_tokens
            )
        else:
            raise ValueError(f"Unknown tokenizer type: {self.tokenizer_type}")

    @property
    def bos_id(self) -> int:
        return self.tokenizer.bos_id

    @property
    def eos_id(self) -> int:
        return self.tokenizer.eos_id

    @property
    def pad_id(self) -> int:
        if self.tokenizer.pad_id == -1:
            return self.eos_id
        return self.tokenizer.pad_id

    @property
    def unk_id(self) -> int:
        return self.tokenizer.unk_id

    @property
    def bos_token(self) -> str:
        return self.tokenizer.id_to_piece(self.tokenizer.bos_id)

    @property
    def eos_token(self) -> str:
        return self.tokenizer.id_to_piece(self.tokenizer.eos_id)

    @property
    def pad_token(self) -> str:
        return self.tokenizer.id_to_piece(self.tokenizer.pad_id)

    @property
    def unk_token(self) -> str:
        return self.tokenizer.id_to_piece(self.tokenizer.unk_id)

    def all_tokens(self) -> Iterable[tuple[bytes, float, gguf.TokenType]]:
        if self.tokenizer_type == MistralTokenizerType.spm:
            yield from self._sentencepiece_tokens()

        elif self.tokenizer_type == MistralTokenizerType.tekken:
            yield from self._tekken_tokens()

        else:
            raise ValueError(f"Unknown tokenizer type: {self.tokenizer_type}")

    @staticmethod
    def token_bytes_to_string(b, byte_encoder):
        return "".join([byte_encoder[ord(char)] for char in b.decode("latin-1")])

    def extract_vocab_merges_from_model(self):
        # Adapted from Transformers (Apache 2.0)
        # https://github.com/huggingface/transformers/blob/main/src/transformers/convert_slow_tokenizer.py
        assert Tekkenizer is not None and isinstance(self.tokenizer, Tekkenizer), (
            f"Expected Tekkenizer, got {type(self.tokenizer)}"
        )
        mergeable_ranks = self.tokenizer._model._mergeable_ranks
        token_bytes_map = {
            rank: token_bytes for token_bytes, rank in mergeable_ranks.items()
        }
        merge_pairs = []

        # Sort vocab by rank to ensure correct merge order
        for i in range(256, self.vocab_size - self.tokenizer.num_special_tokens):
            merged_token = token_bytes_map[i]
            local = []
            for j in range(1, len(merged_token)):
                left = merged_token[:j]
                right = merged_token[j:]
                if (
                    left in mergeable_ranks
                    and right in mergeable_ranks
                    and (left + right) in mergeable_ranks
                ):
                    local.append((left, right, i))
            if not local:
                raise ValueError(
                    f"Could not find valid merge for token at rank {i}: {merged_token.decode('latin-1')}"
                )
            local = sorted(
                local,
                key=lambda x: (mergeable_ranks[x[0]], mergeable_ranks[x[1]]),
                reverse=False,
            )
            merge_pairs.extend(local)
        merge_pairs = sorted(merge_pairs, key=lambda val: val[2], reverse=False)

        byte_encoder = bytes_to_unicode()

        decoded_merge_pairs = [
            [
                self.token_bytes_to_string(val[0], byte_encoder),
                self.token_bytes_to_string(val[1], byte_encoder),
            ]
            for val in merge_pairs
        ]

        merges = [
            " ".join(
                [
                    # ensure the spaces are properly encoded
                    "".join(chr(ord(c) + 256) if c == " " else c for c in part)
                    for part in pair
                ]
            )
            for pair in decoded_merge_pairs
        ]

        return merges
