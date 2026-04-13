#!/usr/bin/env python3
"""
Generate uberkernel kernel-id mapping shared by host and device code.
"""

import argparse
from pathlib import Path


def enum_name(kernel_name: str) -> str:
    return f"GGML_ET_UBERKERNEL_KERNEL_{kernel_name.upper()}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate ET uberkernel mapping")
    parser.add_argument("--output-h", required=True, help="Output header path")
    parser.add_argument("--output-cpp", required=True, help="Output C++ source path")
    parser.add_argument("--kernels", nargs="+", required=True, help="Supported uberkernel kernel names")
    args = parser.parse_args()

    kernels = sorted(args.kernels)
    out_h = Path(args.output_h)
    out_cpp = Path(args.output_cpp)

    with out_h.open("w") as h:
        h.write("// Auto-generated uberkernel kernel-id mapping\n")
        h.write("// Do not edit manually\n\n")
        h.write("#pragma once\n\n")
        h.write("#include <stdint.h>\n")
        h.write("\n")
        h.write("enum ggml_et_uberkernel_kernel_id {\n")
        h.write("    GGML_ET_UBERKERNEL_KERNEL_INVALID = 0,\n")
        for idx, kernel in enumerate(kernels, start=1):
            h.write(f"    {enum_name(kernel)} = {idx},\n")
        h.write("};\n\n")
        h.write("#ifdef GGML_ET_UBERKERNEL_HOST_LOOKUP\n")
        h.write("uint16_t ggml_et_uberkernel_kernel_id_from_name(const char * kernel_name);\n")
        h.write("#endif\n")

    with out_cpp.open("w") as cpp:
        h_name = out_h.name
        cpp.write("// Auto-generated uberkernel kernel-id mapping\n")
        cpp.write("// Do not edit manually\n\n")
        cpp.write(f'#include "{h_name}"\n\n')
        cpp.write("#ifdef GGML_ET_UBERKERNEL_HOST_LOOKUP\n")
        cpp.write("#include <string>\n")
        cpp.write("#include <unordered_map>\n\n")
        cpp.write("uint16_t ggml_et_uberkernel_kernel_id_from_name(const char * kernel_name) {\n")
        cpp.write("    if (kernel_name == nullptr) {\n")
        cpp.write("        return GGML_ET_UBERKERNEL_KERNEL_INVALID;\n")
        cpp.write("    }\n")
        cpp.write("    static const std::unordered_map<std::string, uint16_t> kernel_id_map = {\n")
        for kernel in kernels:
            cpp.write(f'        {{"{kernel}", {enum_name(kernel)}}},\n')
        cpp.write("    };\n")
        cpp.write("    auto it = kernel_id_map.find(std::string(kernel_name));\n")
        cpp.write("    return it == kernel_id_map.end() ? GGML_ET_UBERKERNEL_KERNEL_INVALID : it->second;\n")
        cpp.write("}\n")
        cpp.write("#endif\n")


if __name__ == "__main__":
    main()
