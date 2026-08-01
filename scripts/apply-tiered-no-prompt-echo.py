#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

path = Path("examples/tiered-memory/tiered.cpp")
text = path.read_text(encoding="utf-8")

marker = "Do not echo the tokenized prompt"
if marker in text:
    print(f"already patched {path}")
    raise SystemExit(0)

old = '''    for (const llama_token token : prompt_tokens) {
        char piece[256];
        const int length = llama_token_to_piece(
                vocab, token, piece, sizeof(piece), 0, true);
        if (length > 0) {
            std::fwrite(piece, 1, static_cast<size_t>(length), stdout);
        }
    }
    std::fflush(stdout);

    llama_batch batch = llama_batch_get_one(
'''
new = '''    // Do not echo the tokenized prompt. Besides producing duplicate CLI text,
    // token-to-piece conversion can expose model-specific BOS/EOS markers.
    // The caller already owns the prompt and only generated tokens belong on
    // standard output.
    llama_batch batch = llama_batch_get_one(
'''

if old not in text:
    raise SystemExit("prompt echo block did not match expected source")

path.write_text(text.replace(old, new, 1), encoding="utf-8")
print(f"patched {path}")
