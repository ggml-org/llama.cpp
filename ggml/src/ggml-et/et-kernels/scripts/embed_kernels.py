#!/usr/bin/env python3
"""
Embed ET kernel ELF files as C byte arrays.
"""

import os
import argparse
from pathlib import Path


def embed_elf_as_bytes(elf_path: Path, var_name: str) -> tuple[str, str]:
    """
    Read an ELF file and convert it to C byte array declarations.

    Returns:
        (header_decl, source_def) - declarations for .hpp and definitions for .cpp
    """
    with open(elf_path, 'rb') as f:
        data = f.read()

    size = len(data)

    # Generate header declaration
    header = f"extern unsigned char {var_name}_data[{size}];\n"
    header += f"const uint64_t {var_name}_len = {size};\n"

    # Generate source definition
    source = f"unsigned char {var_name}_data[{size}] = {{\n"

    # Write bytes in rows of 12
    for i, byte in enumerate(data):
        source += f"0x{byte:02x},"
        if (i + 1) % 12 == 0:
            source += "\n"

    source += "\n};\n"

    return header, source


def main():
    parser = argparse.ArgumentParser(description='Embed ET kernel ELF files as C arrays')
    parser.add_argument('--input-dir', required=True, help='Directory containing .elf files')
    parser.add_argument('--output-hpp', required=True, help='Output header file path')
    parser.add_argument('--output-cpp', required=True, help='Output source file path')
    parser.add_argument('--kernels', nargs='+', required=True, help='List of kernel names')

    args = parser.parse_args()

    input_dir = Path(args.input_dir)

    # Collect all kernels
    kernels = []
    for kernel_name in args.kernels:
        elf_path = input_dir / f"{kernel_name}.elf"
        if not elf_path.exists():
            print(f"Warning: Kernel ELF not found: {elf_path}")
            continue
        kernels.append((kernel_name, elf_path))

    # Sort for deterministic output
    kernels.sort()

    # Generate header file
    with open(args.output_hpp, 'w') as hpp:
        hpp.write("// Auto-generated kernel embeddings\n")
        hpp.write("// Do not edit manually\n\n")
        hpp.write("#pragma once\n\n")
        hpp.write("#include <cstdint>\n")
        hpp.write("#include <unordered_map>\n")
        hpp.write("#include <string>\n")
        hpp.write("#include <utility>\n\n")

        for kernel_name, _ in kernels:
            header_decl, _ = embed_elf_as_bytes(Path('/dev/null'), kernel_name)  # Just for header
            # Re-read to get actual header
            elf_path = input_dir / f"{kernel_name}.elf"
            with open(elf_path, 'rb') as f:
                size = len(f.read())
            hpp.write(f"extern unsigned char {kernel_name}_data[{size}];\n")
            hpp.write(f"extern const uint64_t {kernel_name}_len;\n\n")

        # Add kernel lookup map
        hpp.write("// Kernel name -> (data, length) lookup map\n")
        hpp.write("extern const std::unordered_map<std::string, std::pair<const unsigned char*, uint64_t>> ggml_et_embedded_kernels;\n")

    # Generate source file
    with open(args.output_cpp, 'w') as cpp:
        cpp.write("// Auto-generated kernel embeddings\n")
        cpp.write("// Do not edit manually\n\n")
        cpp.write(f'#include "{Path(args.output_hpp).name}"\n\n')

        for kernel_name, elf_path in kernels:
            _, source_def = embed_elf_as_bytes(elf_path, kernel_name)
            cpp.write(source_def)
            cpp.write(f"\nconst uint64_t {kernel_name}_len = {len(open(elf_path, 'rb').read())};\n\n")

        # Generate kernel lookup map
        cpp.write("// Kernel name -> (data, length) lookup map\n")
        cpp.write("const std::unordered_map<std::string, std::pair<const unsigned char*, uint64_t>> ggml_et_embedded_kernels = {\n")
        for kernel_name, _ in kernels:
            cpp.write(f'    {{"{kernel_name}", {{{kernel_name}_data, {kernel_name}_len}}}},\n')
        cpp.write("};\n")

    print(f"Generated {len(kernels)} embedded kernels")
    print(f"  Header: {args.output_hpp}")
    print(f"  Source: {args.output_cpp}")


if __name__ == '__main__':
    main()
