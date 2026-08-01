#!/usr/bin/env python3
#
# embed_metal_source.py
#
# Generate a C header that embeds a Metal source file as a string literal,
# used as the runtime-compile fallback when tessera-metal.metallib is not on
# disk. Run via CMake whenever tessera-metal.metal changes.
#
#   embed_metal_source.py <input.metal> <output.h>

import sys

def main():
    if len(sys.argv) != 3:
        sys.stderr.write("usage: embed_metal_source.py <input.metal> <output.h>\n")
        return 1
    src_path = sys.argv[1]
    out_path = sys.argv[2]
    with open(src_path, "r") as f:
        src = f.read()

    lines = []
    lines.append("// AUTO-GENERATED. Do not edit. Edit tessera-metal.metal and rebuild.")
    lines.append("#pragma once")
    lines.append("// Embedded Metal kernel source. Used as the runtime-compile fallback")
    lines.append("// when tessera-metal.metallib is not on disk.")
    lines.append("")
    lines.append("inline const char * ts_metal_kernel_source(void) {")
    lines.append("    static const char src[] =")
    for ln in src.split("\n"):
        # escape backslash and double-quote for a C string literal
        esc = ln.replace("\\", "\\\\").replace('"', '\\"')
        lines.append('        "' + esc + '\\n"')
    lines.append("        ;")
    lines.append("    return src;")
    lines.append("}")
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return 0

if __name__ == "__main__":
    sys.exit(main())
