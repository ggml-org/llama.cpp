#!/usr/bin/env python3
import sys

def main():
    if len(sys.argv) != 3:
        print(f"usage: {sys.argv[0]} /path/to/model.gguf /path/to/out.jinja", file=sys.stderr)
        sys.exit(2)

    gguf_path = sys.argv[1]
    out_path  = sys.argv[2]

    import gguf

    reader = gguf.GGUFReader(gguf_path)

    key = "tokenizer.chat_template"
    if key not in reader.fields:
        print(f"ERROR: key not found in GGUF: {key}", file=sys.stderr)
        keys = [k for k in reader.fields.keys() if "chat" in k or "tokenizer" in k]
        for k in sorted(keys):
            print(f"found: {k}", file=sys.stderr)
        sys.exit(1)

    field = reader.fields[key]
    val = field.parts[0]

    # gguf-py can return str, bytes, memoryview, numpy.memmap, etc.
    if isinstance(val, str):
        text = val
    else:
        try:
            # memoryview/memmap/ndarray-like -> bytes
            b = val.tobytes() if hasattr(val, "tobytes") else bytes(val)
        except TypeError:
            # last resort: convert via memoryview
            b = memoryview(val).tobytes()
        text = b.decode("utf-8", errors="replace")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)

    print(out_path)

if __name__ == "__main__":
    main()
