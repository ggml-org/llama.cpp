#!/usr/bin/env python3
"""File-based integration tests for deterministic draft plugins.

Reads test files from a directory, feeds code character-by-character to the
plugin via the bitmask API, and verifies accept/reject behavior matches
expectations.

Test file format:
    Line 1:   // TEST: <language>        (or # TEST: for Python)
    Line 2:   // DESC: <description>
    Line 3:   // EXPECT: accept_all | reject_any
    Line 4+:  code content

Usage:
    python3 test_file_based.py <test_dir> <plugin.so>

Example:
    python3 test_file_based.py tests/ build/deterministic-draft.so
"""

import ctypes
import os
import re
import sys
import glob


PASS = 0
FAIL = 0
SKIP = 0

VOCAB_SIZE = 256


def test(name, condition, detail=""):
    global PASS, FAIL
    if condition:
        print(f"  PASS: {name}")
        PASS += 1
    else:
        msg = f"  FAIL: {name}"
        if detail:
            msg += f" -- {detail}"
        print(msg)
        FAIL += 1


def skip(name, reason=""):
    global SKIP
    print(f"  SKIP: {name}" + (f" ({reason})" if reason else ""))
    SKIP += 1


class Plugin:
    def __init__(self, path):
        mode = getattr(os, 'RTLD_LAZY', 0)
        self.lib = ctypes.CDLL(path, mode=mode)

        self.lib.deterministic_draft_create.restype = ctypes.c_void_p
        self.lib.deterministic_draft_create.argtypes = []
        self.lib.deterministic_draft_destroy.restype = None
        self.lib.deterministic_draft_destroy.argtypes = [ctypes.c_void_p]
        self.lib.deterministic_draft_set_vocab.restype = ctypes.c_bool
        self.lib.deterministic_draft_set_vocab.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_char_p),
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_int32),
            ctypes.c_int,
        ]
        self.lib.deterministic_draft_set_language.restype = ctypes.c_bool
        self.lib.deterministic_draft_set_language.argtypes = [
            ctypes.c_void_p, ctypes.c_int, ctypes.c_char_p,
        ]
        self.lib.deterministic_draft_fill_bitmask.restype = ctypes.c_bool
        self.lib.deterministic_draft_fill_bitmask.argtypes = [
            ctypes.c_void_p, ctypes.c_int,
            ctypes.POINTER(ctypes.c_uint32), ctypes.c_int,
        ]
        self.lib.deterministic_draft_commit.restype = None
        self.lib.deterministic_draft_commit.argtypes = [
            ctypes.c_void_p, ctypes.c_int,
            ctypes.c_int32, ctypes.c_char_p, ctypes.c_int,
        ]
        self.lib.deterministic_draft_reset.restype = None
        self.lib.deterministic_draft_reset.argtypes = [
            ctypes.c_void_p, ctypes.c_int,
        ]

        self.state = self.lib.deterministic_draft_create()
        if not self.state:
            raise RuntimeError("Failed to create plugin state")

        vocab = (ctypes.c_char_p * VOCAB_SIZE)()
        for i in range(VOCAB_SIZE):
            vocab[i] = bytes([i])
        if not self.lib.deterministic_draft_set_vocab(
            self.state, vocab, VOCAB_SIZE, None, 0
        ):
            raise RuntimeError("set_vocab failed")

    def set_language(self, lang):
        return self.lib.deterministic_draft_set_language(
            self.state, -1, lang.encode("utf-8")
        )

    def reset(self):
        self.lib.deterministic_draft_reset(self.state, -1)

    def accept_char(self, ch):
        n_words = (VOCAB_SIZE + 31) // 32
        bitmask = (ctypes.c_uint32 * n_words)()
        has_mask = self.lib.deterministic_draft_fill_bitmask(
            self.state, -1, bitmask, VOCAB_SIZE
        )
        if not has_mask:
            return False
        byte_val = ord(ch) if isinstance(ch, str) else ch
        word_idx = byte_val // 32
        bit_idx = byte_val % 32
        return bool(bitmask[word_idx] & (1 << bit_idx))

    def commit_char(self, ch):
        byte_val = ord(ch) if isinstance(ch, str) else ch
        tok = bytes([byte_val])
        self.lib.deterministic_draft_commit(self.state, -1, byte_val, tok, 1)

    def close(self):
        if self.state:
            self.lib.deterministic_draft_destroy(self.state)
            self.state = None


def parse_test_file(filepath):
    with open(filepath, "r") as f:
        lines = f.readlines()

    language = None
    description = None
    expect_type = None
    expect_token = None
    code_start = 0

    for i, line in enumerate(lines):
        stripped = line.strip()
        m = re.match(r'^(?://|#)\s*TEST:\s*(\w+)', stripped)
        if m:
            language = m.group(1)
            code_start = i + 1
            continue
        m = re.match(r'^(?://|#)\s*DESC:\s*(.*)', stripped)
        if m:
            description = m.group(1)
            code_start = i + 1
            continue
        m = re.match(r'^(?://|#)\s*EXPECT:\s*(\w+)(?:\s+"([^"]*)")?', stripped)
        if m:
            expect_type = m.group(1)
            expect_token = m.group(2)
            code_start = i + 1
            continue
        if stripped and not stripped.startswith("//") and not stripped.startswith("#"):
            code_start = i
            break

    if not language:
        return None
    code = "".join(lines[code_start:])
    return language, description, expect_type, expect_token, code


def run_test_file(plugin, filepath):
    global PASS, FAIL, SKIP

    parsed = parse_test_file(filepath)
    if not parsed:
        skip(f"{os.path.basename(filepath)}", "could not parse")
        return

    language, description, expect_type, expect_token, code = parsed
    basename = os.path.basename(filepath)

    print(f"\n  [{basename}] {language}: {description}")

    if not plugin.set_language(language):
        skip(f"{basename}", f"set_language({language}) failed")
        return

    plugin.reset()

    if not code:
        skip(f"{basename}", "no code")
        return

    if expect_type == "accept_all":
        rejected_at = None
        rejected_char = None
        for i, ch in enumerate(code):
            if not plugin.accept_char(ch):
                rejected_at = i
                rejected_char = ch
                break
            plugin.commit_char(ch)

        if rejected_at is not None:
            context_start = max(0, rejected_at - 20)
            context = code[context_start:rejected_at + 20]
            test(
                f"{basename}: all chars accepted",
                False,
                f"rejected '{rejected_char}' (0x{ord(rejected_char):02x}) at offset {rejected_at}, context: {repr(context)}"
            )
        else:
            test(f"{basename}: all chars accepted", True)

    elif expect_type == "reject_any":
        found_rejection = False
        for ch in code:
            if not plugin.accept_char(ch):
                found_rejection = True
                break
            plugin.commit_char(ch)

        test(f"{basename}: rejected at some point", found_rejection)

    elif expect_type == "reject_at":
        found_rejection = False
        rejection_char = None
        for ch in code:
            if not plugin.accept_char(ch):
                found_rejection = True
                rejection_char = ch
                break
            plugin.commit_char(ch)

        if found_rejection:
            test(
                f"{basename}: rejected at '{expect_token}'",
                rejection_char == expect_token,
                f"rejected at '{rejection_char}' instead of '{expect_token}'"
            )
        else:
            test(
                f"{basename}: rejected at '{expect_token}'",
                False,
                "no rejection occurred - all chars accepted"
            )

    else:
        skip(f"{basename}", f"unknown expect type: {expect_type}")


def main():
    global PASS, FAIL, SKIP

    if len(sys.argv) < 3:
        print("Usage: python3 test_file_based.py <test_dir> <plugin.so>")
        print("Example: python3 test_file_based.py tests/ build/deterministic-draft.so")
        sys.exit(1)

    test_dir = sys.argv[1]
    plugin_path = sys.argv[2]

    if not os.path.isdir(test_dir):
        print(f"ERROR: test directory not found: {test_dir}")
        sys.exit(1)

    if not os.path.isfile(plugin_path):
        print(f"ERROR: plugin not found: {plugin_path}")
        sys.exit(1)

    test_files = sorted(glob.glob(os.path.join(test_dir, "**", "*.test"), recursive=True))

    if not test_files:
        print(f"ERROR: no .test files found in {test_dir}")
        sys.exit(1)

    print("=" * 60)
    print("File-Based Integration Tests")
    print("=" * 60)
    print(f"Test directory: {test_dir}")
    print(f"Plugin: {plugin_path}")
    print(f"Test files: {len(test_files)}")

    try:
        plugin = Plugin(plugin_path)
    except Exception as e:
        print(f"ERROR: Failed to load plugin: {e}")
        sys.exit(1)

    for filepath in test_files:
        run_test_file(plugin, filepath)

    plugin.close()

    print(f"\n{'=' * 60}")
    print(f"Results: {PASS} passed, {FAIL} failed, {SKIP} skipped")
    print(f"{'=' * 60}")

    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
