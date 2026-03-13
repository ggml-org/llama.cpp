import time
import argparse
import sys
import threading
import os
from pathlib import Path
from typing import Dict, List, Tuple
from transformers import AutoTokenizer


def setup_console():
    """Setup console for proper Unicode handling on Windows."""
    if sys.platform == 'win32':
        try:
            import ctypes
            kernel32 = ctypes.windll.kernel32
            kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
        except Exception as e:
            print(f"Warning: Failed to setup Windows console: {e}")


def read_tests(fname_inp: str, fname_out: str) -> Dict[str, List[int]]:
    """Read test cases from input and output files."""
    tests = {}

    try:
        if not os.path.isfile(fname_inp):
            print(f"{__name__} : error: could not open file '{fname_inp}'")
            return {}

        if not os.path.isfile(fname_out):
            print(f"{__name__} : error: could not open file '{fname_out}'")
            return {}

        with open(fname_inp, 'r', encoding='utf-8') as f:
            raw_input = f.read()

        with open(fname_out, 'r', encoding='utf-8') as f:
            outputs = [line.strip() for line in f]

        separator = "\n__ggml_vocab_test__\n"
        inputs = raw_input.split(separator)

        if len(inputs) != len(outputs):
            print(f"{__name__} : error: input and output files have different number of tests")
            return {}

        for inp, out in zip(inputs, outputs):
            tokens = [int(tok) for tok in out.split()]
            tests[inp.strip()] = tokens

        return tests

    except Exception as e:
        print(f"{__name__} : error: {e}")
        return {}


def run_tests(tokenizer, tests: Dict[str, List[int]], thread_id: int) -> bool:
    """Run tokenization tests and verify results."""
    success = True

    for test_input, expected_tokens in tests.items():
        should_print = (thread_id == 0)

        try:
            result_tokens = tokenizer.encode(test_input, add_special_tokens=False)

            if should_print:
                print(f"\nsrc: '{test_input}'")
                print(f"res: '{tokenizer.decode(result_tokens)}'")
                print(f"tok:", " ".join(str(t) for t in result_tokens))

            correct = (len(result_tokens) == len(expected_tokens))
            if correct:
                for res, exp in zip(result_tokens, expected_tokens):
                    if res != exp:
                        correct = False
                        break

            if not correct and should_print:
                print(f"{__name__} : failed test:    '{test_input}'")
                print(
                    f"{__name__} : detokenized to: '{tokenizer.decode(result_tokens)}' instead of \
                        '{tokenizer.decode(expected_tokens)}'")
                print(f"{__name__} : expected tokens: ", end='')
                for t in expected_tokens:
                    print(f"{t:6d} '{tokenizer.decode([t])}', ", end='')
                print()
                print(f"{__name__} : got tokens:      ", end='')
                for t in result_tokens:
                    print(f"{t:6d} '{tokenizer.decode([t])}', ", end='')
                print()
                success = False

        except Exception as e:
            print(f"{__name__} : error processing test '{test_input}': {e}")
            success = False

    return success


def process_text_file(tokenizer, fname: str) -> Tuple[List[int], float, List[str]]:
    """Process a single text file and return tokens, processing time, and lines."""
    if not os.path.isfile(fname):
        print(f"{__name__} : error: could not open file '{fname}'")
        return [], 0.0, []

    try:
        print(f"tokenizing file: {fname}")
        with open(fname, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            text = ''.join(lines)

        print(f"{__name__} : text size: {len(text)}")

        start_time = time.time()
        tokens = tokenizer.encode(text, add_special_tokens=False)
        processing_time = (time.time() - start_time) * 1000

        return tokens, processing_time, lines

    except Exception as e:
        print(f"{__name__} : error: {e}")
        return [], 0.0, []


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dir_tokenizer", help="directory containing 'tokenizer.model' file")
    parser.add_argument("--fname-tok", help="path to a text file to tokenize", required=True)
    args = parser.parse_args()

    setup_console()

    try:
        tokenizer = AutoTokenizer.from_pretrained(args.dir_tokenizer)
        # LLaMA v3 for some reason strips the space for these tokens (and others)
        # if x == 662:
        #     f.write(str(x) + ' \' ' + tokenizer.decode(x) + '\'\n')
        # elif x == 1174:
        #     f.write(str(x) + ' \' ' + tokenizer.decode(x) + '\'\n')
        # elif x == 2564:
        #     f.write(str(x) + ' \' ' + tokenizer.decode(x) + '\'\n')
        # elif x == 758:
        #     f.write(str(x) + ' \' ' + tokenizer.decode(x) + '\'\n')
        # elif x == 949:
        #     f.write(str(x) + ' \' ' + tokenizer.decode(x) + '\'\n')
        # elif x == 5354:
        #     f.write(str(x) + ' \' ' + tokenizer.decode(x) + '\'\n')
        # else:
        #     f.write(str(x) + ' \'' + tokenizer.decode(x) + '\'\n')
        # f.write(str(x) + ' \'' + tokenizer.decode(x).strip() + '\'\n')
        # Process file mode
        tokens, processing_time, lines = process_text_file(tokenizer, args.fname_tok)
        if not tokens:
            return 1

        fname_out = args.fname_tok + '.tok'
        with open(fname_out, 'w', encoding='utf-8') as f:
            for token in tokens:
                f.write(f"{token}\n")

        print(f"\nmain : tokenized in {processing_time:.3f} ms (py)")
        print(f"len(res): {len(tokens)}")
        print(f"len(lines): {len(lines)}")
        print(f"results written to: {fname_out}")
        return 0

    except Exception as e:
        print(f"{__name__} : error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
