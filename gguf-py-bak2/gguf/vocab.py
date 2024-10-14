from __future__ import annotations

import logging
import json
import os
from pathlib import Path
from typing import Any, Callable

from .gguf_writer import GGUFWriter

logger = logging.getLogger(__name__)


class SpecialVocab:
    merges: list[str]
    add_special_token: dict[str, bool]
    special_token_ids: dict[str, int]
    chat_template: str | None

    def __init__(
        self, path: str | os.PathLike[str], load_merges: bool = False,
        special_token_types: tuple[str, ...] | None = None,
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
            self.special_token_ids[typ] = tid
            return
        logger.warning(f'Special token type {typ}, id {tid} out of range, must be under {self.n_vocab} - skipping')

    def _try_load_from_tokenizer_json(self, path: Path) -> bool:
        tokenizer_file = path / 'tokenizer.json'
        if tokenizer_file.is_file():
            with open(tokenizer_file, encoding = 'utf-8') as f:
                tokenizer = json.load(f)
            if self.load_merges:
                merges = tokenizer.get('model', {}).get('merges')
                if isinstance(merges, list) and merges and isinstance(merges[0], str):
                    self.merges = merges
            added_tokens = tokenizer.get('added_tokens', {})
        else:
            added_tokens = {}
        tokenizer_config_file = path / 'tokenizer_config.json'
        if not tokenizer_config_file.is_file():
            return True
        with open(tokenizer_config_file, encoding = 'utf-8') as f:
            tokenizer_config = json.load(f)
        chat_template = tokenizer_config.get('chat_template')
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
            self._set_special_token(typ, config.get(f'{typ}_token_id'))
        return True
