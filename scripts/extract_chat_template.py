#!/usr/bin/env python3
import sys

def main():
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} /path/to/model.gguf", file=sys.stderr)
        sys.exit(2)

    path = sys.argv[1]

    # gguf-py API (в llama.cpp) обычно предоставляет GGUFReader
    from gguf import GGUFReader

    r = GGUFReader(path)
    # метаданные лежат как key/value
    key = "tokenizer.chat_template"
    if key not in r.fields:
        # иногда ключи могут быть в другом месте, но обычно именно так
        print(f"ERROR: key '{key}' not found in GGUF metadata", file=sys.stderr)
        sys.exit(1)

    field = r.fields[key]
    # field.parts[0] обычно bytes/str; приведём аккуратно
    val = field.parts[0]
    if isinstance(val, (bytes, bytearray)):
        val = val.decode("utf-8", errors="replace")
    print(val)

if __name__ == "__main__":
    main()
