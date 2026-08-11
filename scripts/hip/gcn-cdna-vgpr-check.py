#!/usr/bin/env python3

import sys
from collections import defaultdict
import re


def parse_log_file(filepath):
    functions = defaultdict(lambda: {'vgprs': 0, 'spill': 0, 'location': ''})
    func_stack = []

    try:
        with open(filepath, 'r') as f:
            for line in f:
                # Match function name lines
                func_match = re.search(r'remark: ([^:]+):(\d+):\d+: Function Name: (\S+)', line)
                if func_match:
                    location = func_match.group(1) + ':' + func_match.group(2)
                    func_name = func_match.group(3)
                    # Extract just the filename and line number
                    parts = location.split('/')
                    short_location = parts[-1] if len(parts) > 0 else location
                    functions[func_name]['location'] = short_location
                    # Push function onto stack with its location
                    func_stack.append({'name': func_name, 'location': location})
                    continue

                # Match VGPR usage lines (only if we have functions in stack)
                vgpr_match = re.search(r'remark: ([^:]+):(\d+):\d+:\s+VGPRs: (\d+)', line)
                if vgpr_match:
                    location = vgpr_match.group(1) + ':' + vgpr_match.group(2)
                    # Find the most recent function with matching location
                    for i in range(len(func_stack) - 1, -1, -1):
                        if func_stack[i]['location'] == location:
                            functions[func_stack[i]['name']]['vgprs'] = int(vgpr_match.group(3))
                            break
                    continue

                spill_match = re.search(r'remark: ([^:]+):(\d+):\d+:\s+VGPRs Spill: (\d+)', line)
                if spill_match:
                    location = spill_match.group(1) + ':' + spill_match.group(2)
                    # Find the most recent function with matching location
                    for i in range(len(func_stack) - 1, -1, -1):
                        if func_stack[i]['location'] == location:
                            functions[func_stack[i]['name']]['spill'] = int(spill_match.group(3))
                            break
                    continue
    except FileNotFoundError:
        print(f"Error: File {filepath} not found", file=sys.stderr)  # noqa: NP100
        sys.exit(1)

    return functions


def main():
    if len(sys.argv) < 2:
        print("Usage: ./vgpr_check.py <log_file>", file=sys.stderr)  # noqa: NP100
        sys.exit(1)

    log_file = sys.argv[1]
    ignored = {
        '_ZL21gated_linear_attn_f32ILi128EEviiiifPKfS1_S1_S1_S1_Pf',
        '_ZL13rwkv_wkv7_f32ILi128EEviiiiPKfS1_S1_S1_S1_S1_S1_Pf',
        '_ZL12rwkv_wkv_f32ILi128EEviiiiPKfS1_S1_S1_S1_S1_Pf',
        '_ZL9mul_mat_qIL9ggml_type10ELi64ELb1EEvPKcPKiS4_S4_PfS5_PKf15HIP_vector_typeIjLj3EEiiiiiS9_S9_iiiS9_S9_iiiS9_',
        '_ZL9mul_mat_qIL9ggml_type42ELi128ELb1EEvPKcPKiS4_S4_PfS5_PKf15HIP_vector_typeIjLj3EEiiiiiS9_S9_iiiS9_S9_iiiS9_',
    }

    functions = parse_log_file(log_file)
    found_issues = False

    # First print all ignored functions (deduplicated)
    printed_ignored = set()
    for func_name, data in sorted(functions.items()):
        total_vgprs = int(data['vgprs']) + int(data['spill'])
        if total_vgprs > 256 and func_name in ignored and func_name not in printed_ignored:
            location = data.get('location', log_file)
            print(f"{location}: {func_name} - Total VGPRs: {total_vgprs} ({data['vgprs']} + {data['spill']}) [IGNORED]")  # noqa: NP100
            printed_ignored.add(func_name)

    # Then print new functions with issues in red
    for func_name, data in sorted(functions.items()):
        total_vgprs = int(data['vgprs']) + int(data['spill'])
        if total_vgprs > 256 and func_name not in ignored:
            status = "[IGNORED]" if func_name in ignored else ""
            location = data.get('location', log_file)
            # Print in red if not ignored
            color_code = "\033[91m" if func_name not in ignored else ""
            reset_code = "\033[0m" if func_name not in ignored else ""
            print(f"{color_code}{location}: {func_name} - Total VGPRs: {total_vgprs} ({data['vgprs']} + {data['spill']}) {status}{reset_code}")  # noqa: NP100
            if func_name not in ignored:
                found_issues = True

    sys.exit(1 if found_issues else 0)


if __name__ == "__main__":
    main()
