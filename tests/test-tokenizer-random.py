# Test libllama tokenizer == AutoTokenizer.
# Brute force random words/text generation.
#
# Sample usage:
#
#   python3 tests/test-tokenizer-random.py ./models/ggml-vocab-llama-bpe.gguf ./models/tokenizers/llama-bpe
#

from __future__ import annotations

import time
import logging
import argparse
import subprocess
import random
import unicodedata

from pathlib import Path
from typing import Any, Iterator, cast
from typing_extensions import Buffer

import cffi
from transformers import AutoTokenizer, PreTrainedTokenizer


logger = logging.getLogger("test-tokenizer-random")


class LibLlama:

    DEFAULT_PATH_LLAMA_H = "./include/llama.h"
    DEFAULT_PATH_INCLUDES = ["./ggml/include/", "./include/"]
    DEFAULT_PATH_LIBLLAMA = "./build/src/libllama.so"  # CMakeLists.txt: BUILD_SHARED_LIBS ON

    def __init__(self, path_llama_h: str | None = None, path_includes: list[str] = [], path_libllama: str | None = None):
        path_llama_h = path_llama_h or self.DEFAULT_PATH_LLAMA_H
        path_includes = path_includes or self.DEFAULT_PATH_INCLUDES
        path_libllama = path_libllama or self.DEFAULT_PATH_LIBLLAMA
        (self.ffi, self.lib) = self._load_libllama_cffi(path_llama_h, path_includes, path_libllama)
        self.lib.llama_backend_init()

    def _load_libllama_cffi(self, path_llama_h: str, path_includes: list[str], path_libllama: str) -> tuple[cffi.FFI, Any]:
        cmd = ["gcc", "-O0", "-E", "-P", "-D__restrict=", "-D__attribute__(x)=", "-D__asm__(x)="]
        cmd += ["-I" + path for path in path_includes] + [path_llama_h]
        res = subprocess.run(cmd, stdout=subprocess.PIPE)
        assert (res.returncode == 0)
        source = res.stdout.decode()
        ffi = cffi.FFI()
        if True:  # workarounds for pycparser
            source = "typedef struct { } __builtin_va_list;" + "\n" + source
            source = source.replace("sizeof (int)",    str(ffi.sizeof("int")))
            source = source.replace("sizeof (void *)", str(ffi.sizeof("void*")))
            source = source.replace("sizeof (size_t)", str(ffi.sizeof("size_t")))
            source = source.replace("sizeof(int32_t)", str(ffi.sizeof("int32_t")))
        ffi.cdef(source, override=True)
        lib = ffi.dlopen(path_libllama)
        return (ffi, lib)

    def model_default_params(self, **kwargs):
        mparams = self.lib.llama_model_default_params()
        for k, v in kwargs.items():
            setattr(mparams, k, v)
        return mparams

    def context_default_params(self, **kwargs):
        cparams = self.lib.llama_context_default_params()
        for k, v in kwargs.items():
            setattr(cparams, k, v)
        return cparams


class LibLlamaModel:

    def __init__(self, libllama: LibLlama, path_model: str, mparams={}, cparams={}):
        self.lib: Any = libllama.lib
        self.ffi = libllama.ffi
        if isinstance(mparams, dict):
            mparams = libllama.model_default_params(**mparams)
        self.model = self.lib.llama_load_model_from_file(path_model.encode(), mparams)
        if not self.model:
            raise RuntimeError("error: failed to load model '%s'" % path_model)
        if isinstance(cparams, dict):
            cparams = libllama.context_default_params(**cparams)
        self.ctx = self.lib.llama_new_context_with_model(self.model, cparams)
        if not self.ctx:
            raise RuntimeError("error: failed to create context for model '%s'" % path_model)
        n_tokens_max = self.lib.llama_n_ctx(self.ctx)
        self.token_ids = self.ffi.new("llama_token[]", n_tokens_max)
        self.text_buff = self.ffi.new("uint8_t[]", 1024)

    def free(self):
        if self.ctx:
            self.lib.llama_free(self.ctx)
        if self.model:
            self.lib.llama_free_model(self.model)
        self.ctx = None
        self.model = None
        self.lib = None

    def tokenize(self, text: str, add_special: bool = False, parse_special: bool = False) -> list[int]:
        encoded_text: bytes = text.encode("utf-8")
        num = self.lib.llama_tokenize(self.model, encoded_text, len(encoded_text), self.token_ids, len(self.token_ids), add_special, parse_special)
        while num < 0 and len(self.token_ids) < (16 << 20):
            self.token_ids = self.ffi.new("llama_token[]", -2 * num)
            num = self.lib.llama_tokenize(self.model, encoded_text, len(encoded_text), self.token_ids, len(self.token_ids), add_special, parse_special)
        return list(self.token_ids[0:num])

    def detokenize(self, ids: list[int], remove_special: bool = False, unparse_special: bool = False) -> str:
        if len(self.token_ids) < len(ids):
            self.token_ids = self.ffi.new("llama_token[]", 2 * len(ids))
        for i, id in enumerate(ids):
            self.token_ids[i] = id
        num = self.lib.llama_detokenize(self.model, self.token_ids, len(ids), self.text_buff, len(self.text_buff), remove_special, unparse_special)
        while num < 0 and len(self.text_buff) < (16 << 20):
            self.text_buff = self.ffi.new("uint8_t[]", -2 * num)
            num = self.lib.llama_detokenize(self.model, self.token_ids, len(ids), self.text_buff, len(self.text_buff), remove_special, unparse_special)
        return str(cast(Buffer, self.ffi.buffer(self.text_buff, num)), encoding="utf-8", errors="replace")  # replace errors with '\uFFFD'


class Tokenizer:

    def encode(self, text: str) -> list[int]:
        raise NotImplementedError

    def decode(self, ids: list[int]) -> str:
        raise NotImplementedError


class TokenizerGroundtruth (Tokenizer):

    def __init__(self, dir_tokenizer: str):
        self.model: PreTrainedTokenizer = AutoTokenizer.from_pretrained(dir_tokenizer)
        # guess BOS and EOS
        ids = self.encode("a")
        assert 1 <= len(ids) <= 3
        add_bos_token = len(ids) > 1 and self.model.bos_token_id == ids[0]
        add_eos_token = len(ids) > 1 and self.model.eos_token_id == ids[-1]
        self.add_bos_token = getattr(self.model, "add_bos_token", add_bos_token)
        self.add_eos_token = getattr(self.model, "add_eos_token", add_eos_token)
        # build vocab
        tokens = list(self.model.get_vocab().values())
        self.vocab = self.model.batch_decode(tokens, skip_special_tokens=True)
        self.vocab = list(sorted(self.vocab))
        # tokens and lists
        self.special_tokens = list(self.model.all_special_tokens)
        self.added_tokens   = self.model.batch_decode(self.model.added_tokens_encoder.values(), skip_special_tokens=False)
        self.bos_token = self.model.bos_token
        self.eos_token = self.model.eos_token

    def encode(self, text: str) -> list[int]:
        return self.model.encode(text, add_special_tokens=True)

    def decode(self, ids: list[int]) -> str:
        return self.model.decode(ids, skip_special_tokens=False)


class TokenizerLlamaCpp (Tokenizer):

    libllama: LibLlama | None = None

    def __init__(self, vocab_file: str):
        if not self.libllama:
            self.libllama = LibLlama()
        self.model = LibLlamaModel(self.libllama, vocab_file, mparams=dict(vocab_only=True), cparams=dict(n_ctx=4096))

    def encode(self, text: str) -> list[int]:
        return self.model.tokenize(text, add_special=True, parse_special=True)

    def decode(self, ids: list[int]) -> str:
        return self.model.detokenize(ids, remove_special=False, unparse_special=True)


def generator_custom_text() -> Iterator[str]:
    """General tests"""
    yield from [
        "",
        " ",
        "  ",
        "   ",
        "\t",
        "\n",
        "\n\n",
        "\n\n\n",
        "\t\n",
        "Hello world",
        " Hello world",
        "Hello World",
        " Hello World",
        " Hello World!",
        "Hello, world!",
        " Hello, world!",
        " this is 🦙.cpp",
        "w048 7tuijk dsdfhu",
        "нещо на Български",
        "កាន់តែពិសេសអាចខលចេញ",
        "🚀 (normal) 😶‍🌫️ (multiple emojis concatenated) ✅ (only emoji that has its own token)",
        "Hello",
        " Hello",
        "  Hello",
        "   Hello",
        "    Hello",
        "    Hello\n    Hello",
        " (",
        "\n =",
        "' era",
        "Hello, y'all! How are you 😁 ?我想在apple工作1314151天～",
        "3",
        "33",
        "333",
        "3333",
        "33333",
        "333333",
        "3333333",
        "33333333",
        "333333333",
    ]


def generator_custom_text_edge_cases() -> Iterator[str]:
    """Edge cases found while debugging"""
    yield from [
        '\x1f-a',     # unicode_ranges_control, {0x00001C, 0x00001F}
        '¼-a',        # unicode_ranges_digit, 0x00BC
        '½-a',        # unicode_ranges_digit, 0x00BD
        '¾-a',        # unicode_ranges_digit, 0x00BE
        'a 〇b',      # unicode_ranges_digit, 0x3007
        'Ⅵ-a',       # unicode_ranges_digit, {0x00002150, 0x0000218F} // Number Forms
        '\uFEFF//',   # unicode_ranges_control, 0xFEFF (BOM)
        'Cửa Việt',   # llama-3, ignore_merges = true
        '<s>a',       # Phi-3 fail
        '<unk><|endoftext|><s>',  # Phi-3 fail
        'a\na',            # bert fail
        '"`',              # falcon
        ' \u2e4e',         # falcon
        '\n\x0b  ',        # falcon
        'a\xa0\xa0\x00b',  # jina-v2-es
        'one <mask>',      # jina-v2-es  <mask> lstrip=true
        'a </s> b',        # rstrip phi-3
        'a <mask> b',      # lstrip jina-v2
        '\xa0aC',          # deepseek
        '\u2029 \uA3E4',   # deepseek-llm
        "a ?",
        'å',               # mpt
        '\U000ac517',      # utf-8 encode error, falcon
        '\U000522f4',      # utf-8 encode error, starcoder
        "<s><s><unk><s>a<s>b<s>c<unk>d<unk></s>",
        "<s> <s> <unk><s>a<s>b<s>c<unk>d<unk></s>",
    ]


def generator_vocab_words(tokenizer: TokenizerGroundtruth) -> Iterator[str]:
    """Brute force check all vocab words"""
    yield from tokenizer.vocab


def generator_ascii_lr_strip() -> Iterator[str]:
    WHITESPACES = ["", " ", "  "]
    CHARACTERS = list(chr(i) for i in range(1, 0x80)) + [""]
    for char1 in CHARACTERS:
        for char2 in CHARACTERS:
            for lstrip in WHITESPACES:
                for rstrip in WHITESPACES:
                    yield lstrip + char1 + char2 + rstrip
                    yield lstrip + char1 + rstrip + char2
                    yield char1 + lstrip + char2 + rstrip


def generator_apostrophe() -> Iterator[str]:
    WHITESPACES = ["", " ", "  "]
    CHARACTERS = list(chr(i) for i in range(1, 0x80)) + [""]
    for char1 in CHARACTERS:
        for char2 in CHARACTERS:
            for lstrip in WHITESPACES:
                for rstrip in WHITESPACES:
                    yield char1 + lstrip + "'" + rstrip + char2
                    yield char1 + char2 + lstrip + "'" + rstrip + "z"
                    yield "a" + lstrip + "'" + rstrip + char1 + char2


def generator_added_lr_strip(tokenizer: TokenizerGroundtruth) -> Iterator[str]:
    WHITESPACES = ["", " ", "  ", "\n", "\r\n", "\n\n", "\t", "\t\t"]
    all_tokens = list(sorted(set(tokenizer.special_tokens + tokenizer.added_tokens)))
    for token in all_tokens:
        for lstrip in WHITESPACES:
            for rstrip in WHITESPACES:
                yield lstrip + token + rstrip
                yield "a" + lstrip + token + rstrip
                yield lstrip + token + rstrip + "z"
                yield "a" + lstrip + token + rstrip + "z"


def generator_random_added_tokens(tokenizer: TokenizerGroundtruth, iterations=100) -> Iterator[str]:
    separations = [" ", "\n", "\t", "-", "!", "one", "1", "<s>", "</s>"]
    all_tokens  = list(sorted(set(tokenizer.special_tokens + tokenizer.added_tokens + separations)))
    rand = random.Random()
    for m in range(iterations):
        rand.seed(m)
        words = rand.choices(all_tokens, k=500)
        if words and words[0] == tokenizer.bos_token:  # skip spam warning of double BOS
            while len(words) > 1 and words[1] == tokenizer.bos_token:  # leave one starting BOS
                words.pop(0)
            if tokenizer.add_bos_token:  # drop all starting BOS
                words.pop(0)
        if words and words[-1] == tokenizer.eos_token:  # skip spam warning of double EOS
            while len(words) > 1 and words[-2] == tokenizer.eos_token:  # leave one trailing EOS
                words.pop(-1)
            if tokenizer.add_bos_token:  # drop all trailing EOS
                words.pop(-1)
        yield "".join(words)


def generator_random_chars(iterations=100) -> Iterator[str]:
    """Brute force random text with simple characters"""

    NUM_WORDS = 400
    WHITESPACES = list(" " * 20 + "\n" * 5 + "\r\n" * 5 + "\t" * 5)
    CHARS = list(sorted(set("""
        ABCDEFGHIJKLMNOPQRSTUVWXYZ
        abcdefghijklmnopqrstuvwxyz
        ÁÉÍÓÚÀÈÌÒÙÂÊÎÔÛÄËÏÖÜ
        áéíóúàèìòùâêîôûäëïöü
        .-,*/-+ª!"·$%&/()=?¿[]{}<>\\|@#~½¬~;:_
    """)))

    rand = random.Random()
    for m in range(iterations):
        rand.seed(m)
        text = []
        for _ in range(NUM_WORDS):
            k = rand.randint(1, 7)
            word = rand.choices(CHARS, k=k)
            word.append(rand.choice(WHITESPACES))
            text.append("".join(word))
        yield "".join(text)


def generator_unicodes() -> Iterator[str]:
    """Iterate unicode characters"""

    MAX_CODEPOINTS = 0x30000  # 0x110000

    def _valid(cpt):
        if cpt >= 0x30000:  # unassigned and supplement­ary
            return False
        # if cpt == 0x2029:  # deepseek-llm
        #    return False
        if unicodedata.category(chr(cpt)) in ("Cn", "Cs", "Co"):  # undefined, surrogates, private
            return False
        return True

    characters = [chr(cpt) for cpt in range(0, MAX_CODEPOINTS) if _valid(cpt)]

    yield from characters


def generator_random_unicodes(iterations=100) -> Iterator[str]:
    """Brute force random text with unicode characters"""

    NUM_WORDS = 200
    WHITESPACES = list(" " * 20 + "\n" * 5 + "\r\n" * 5 + "\t" * 5)

    characters = list(generator_unicodes())

    rand = random.Random()
    for m in range(iterations):
        rand.seed(m)
        text = []
        for _ in range(NUM_WORDS):
            k = rand.randint(1, 7)
            word = rand.choices(characters, k=k)
            word.append(rand.choice(WHITESPACES))
            text.append("".join(word))
        yield "".join(text)


def generator_random_vocab_chars(tokenizer: TokenizerGroundtruth, iterations=100) -> Iterator[str]:
    """Brute force random text with vocab characters"""

    vocab_chars = set()
    for word in tokenizer.vocab:
        vocab_chars.update(word)
    vocab_chars = list(sorted(vocab_chars))

    rand = random.Random()
    for m in range(iterations):
        rand.seed(m)
        text = rand.choices(vocab_chars, k=1024)
        yield "".join(text)


def generator_random_vocab_words(tokenizer: TokenizerGroundtruth, iterations=100) -> Iterator[str]:
    """Brute force random text from vocab words"""

    vocab = [w.strip() for w in tokenizer.vocab]
    yield from vocab

    rand = random.Random()
    for m in range(iterations):
        rand.seed(m)
        text = []
        num_words = rand.randint(300, 400)
        for i in range(num_words):
            k = rand.randint(1, 3)
            words = rand.choices(vocab, k=k)
            sep = rand.choice("     \n\r\t")
            text.append("".join(words) + sep)
        yield "".join(text)


def compare_tokenizers(tokenizer1: TokenizerGroundtruth, tokenizer2: TokenizerLlamaCpp, generator: Iterator[str]):

    def find_first_mismatch(ids1: list[int] | str, ids2: list[int] | str):
        for i, (a, b) in enumerate(zip(ids1, ids2)):
            if a != b:
                return i
        if len(ids1) == len(ids2):
            return -1
        return min(len(ids1), len(ids2))

    def check_detokenizer(text: str, text1: str, text2: str) -> bool:
        if text1 == text2:  # equal to TokenizerGroundtruth?
            return True
        # equal to source text?
        if tokenizer1.add_bos_token:  # remove BOS
            if text2.startswith(tokenizer1.bos_token):
                text2 = text2[len(tokenizer1.bos_token):]
        if tokenizer1.add_eos_token:  # remove EOS
            if text2.endswith(tokenizer1.eos_token):
                text2 = text2[:-len(tokenizer1.eos_token)]
        return text == text2

    t_encode1 = 0
    t_encode2 = 0
    t_decode1 = 0
    t_decode2 = 0
    t_start = time.perf_counter()
    encode_errors = 0
    decode_errors = 0
    MAX_ERRORS = 10

    logger.info("%s: %s" % (generator.__qualname__, "ini"))
    for text in generator:
        # print(repr(text), text.encode())
        # print(repr(text), hex(ord(text[0])), text.encode())
        t0 = time.perf_counter()
        ids1 = tokenizer1.encode(text)
        t1 = time.perf_counter()
        ids2 = tokenizer2.encode(text)
        t2 = time.perf_counter()
        text1 = tokenizer1.decode(ids1)
        t3 = time.perf_counter()
        text2 = tokenizer2.decode(ids1)
        t4 = time.perf_counter()
        t_encode1 += t1 - t0
        t_encode2 += t2 - t1
        t_decode1 += t3 - t2
        t_decode2 += t4 - t3
        if encode_errors < MAX_ERRORS and ids1 != ids2:
            i = find_first_mismatch(ids1, ids2)
            ids1 = list(ids1)[max(0, i - 2) : i + 5 + 1]
            ids2 = list(ids2)[max(0, i - 2) : i + 5 + 1]
            logger.error(" Expected: " + str(ids1))
            logger.error("   Result: " + str(ids2))
            encode_errors += 1
            logger.error(f" {encode_errors=}")
        if decode_errors < MAX_ERRORS and not check_detokenizer(text, text1, text2):
            i = find_first_mismatch(text1, text2)
            text1 = list(text1[max(0, i - 2) : i + 5 + 1])
            text2 = list(text2[max(0, i - 2) : i + 5 + 1])
            logger.error(" Expected: " + " ".join(hex(ord(x)) for x in text1))
            logger.error("   Result: " + " ".join(hex(ord(x)) for x in text2))
            decode_errors += 1
            logger.error(f" {decode_errors=}")
        if encode_errors >= MAX_ERRORS and decode_errors >= MAX_ERRORS:
            logger.error(f" EXIT: {encode_errors=} {decode_errors=}")
            # raise Exception()
            break

    t_total = time.perf_counter() - t_start
    logger.info(f"{generator.__qualname__}: end,  {t_encode1=:.3f} {t_encode2=:.3f}  {t_decode1=:.3f} {t_decode2=:.3f}  {t_total=:.3f}")


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser()
    parser.add_argument("vocab_file", type=str, help="path to vocab 'gguf' file")
    parser.add_argument("dir_tokenizer", type=str, help="directory containing 'tokenizer.model' file")
    parser.add_argument("--verbose", action="store_true", help="increase output verbosity")
    args = parser.parse_args(argv)

    logging.basicConfig(level = logging.DEBUG if args.verbose else logging.INFO)
    logger.info(f"VOCABFILE: '{args.vocab_file}'")

    tokenizer1 = TokenizerGroundtruth(args.dir_tokenizer)
    tokenizer2 = TokenizerLlamaCpp(args.vocab_file)

    # compare_tokenizers(tokenizer1, tokenizer2, generator_custom_text())
    # compare_tokenizers(tokenizer1, tokenizer2, generator_custom_text_edge_cases())
    compare_tokenizers(tokenizer1, tokenizer2, generator_ascii_lr_strip())
    compare_tokenizers(tokenizer1, tokenizer2, generator_apostrophe())
    compare_tokenizers(tokenizer1, tokenizer2, generator_unicodes())
    compare_tokenizers(tokenizer1, tokenizer2, generator_vocab_words(tokenizer1))
    compare_tokenizers(tokenizer1, tokenizer2, generator_added_lr_strip(tokenizer1))
    # compare_tokenizers(tokenizer1, tokenizer2, generator_random_added_tokens(tokenizer1, 10_000))
    # compare_tokenizers(tokenizer1, tokenizer2, generator_random_chars(10_000))
    # compare_tokenizers(tokenizer1, tokenizer2, generator_random_unicodes(10_000))
    # compare_tokenizers(tokenizer1, tokenizer2, generator_random_vocab_chars(tokenizer1, 10_000))
    # compare_tokenizers(tokenizer1, tokenizer2, generator_random_vocab_words(tokenizer1, 5_000))

    tokenizer2.model.free()


if __name__ == "__main__":
    # main()

    if True:
        logging.basicConfig(
            level    = logging.DEBUG,
            format   = "%(asctime)s.%(msecs)03d %(name)s %(levelname)s %(message)s",
            datefmt  = "%Y-%m-%d %H:%M:%S",
            filename = logger.name + ".log",
            filemode = "a"
        )
    logging.basicConfig(
        level    = logging.DEBUG,
        format   = "%(levelname)s %(message)s",
    )

    path_tokenizers   = Path("./models/tokenizers/")
    path_vocab_format = "./models/ggml-vocab-%s.gguf"

    tokenizers = [
        "llama-spm",      # SPM
        "phi-3",          # SPM
        "gemma",          # SPM
        "gemma-2",        # SPM
        "baichuan",       # SPM
        "bert-bge",       # WPM
        "jina-v2-en",     # WPM
        "llama-bpe",      # BPE
        "phi-2",          # BPE
        "deepseek-llm",   # BPE
        "deepseek-coder", # BPE
        "falcon",         # BPE
        "mpt",            # BPE
        "starcoder",      # BPE
        "gpt-2",          # BPE
        "stablelm2",      # BPE
        "refact",         # BPE
        "qwen2",          # BPE
        "olmo",           # BPE
        "jina-v2-es",     # BPE
        "jina-v2-de",     # BPE
        "smaug-bpe",      # BPE
        "poro-chat",      # BPE
        "jina-v2-code",   # BPE
        "viking",         # BPE
        "jais",           # BPE
    ]

    logger.info("=" * 50)
    for tokenizer in tokenizers:
        logger.info("-" * 50)
        logger.info(f"TOKENIZER: '{tokenizer}'")
        vocab_file = Path(path_vocab_format % tokenizer)
        dir_tokenizer = path_tokenizers / tokenizer
        main([str(vocab_file), str(dir_tokenizer), "--verbose"])
