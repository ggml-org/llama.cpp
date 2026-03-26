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
        '_ZL18flash_attn_ext_f16ILi64ELi64ELi16ELi2ELb0ELb0EEvPKcS1_S1_S1_S1_PKiPfP15HIP_vector_typeIfLj2EEffffjfiS5_IjLj3EEiiiiiiiiiiiliiliiiiil',
        '_ZL18flash_attn_ext_f16ILi80ELi80ELi16ELi2ELb0ELb0EEvPKcS1_S1_S1_S1_PKiPfP15HIP_vector_typeIfLj2EEffffjfiS5_IjLj3EEiiiiiiiiiiiliiliiiiil',
        '_ZL18flash_attn_ext_f16ILi96ELi96ELi16ELi2ELb0ELb0EEvPKcS1_S1_S1_S1_PKiPfP15HIP_vector_typeIfLj2EEffffjfiS5_IjLj3EEiiiiiiiiiiiliiliiiiil',
        '_ZL18flash_attn_ext_f16ILi64ELi64ELi32ELi1ELb0ELb0EEvPKcS1_S1_S1_S1_PKiPfP15HIP_vector_typeIfLj2EEffffjfiS5_IjLj3EEiiiiiiiiiiiliiliiiiil',
        '_ZL13rwkv_wkv7_f32ILi128EEviiiiPKfS1_S1_S1_S1_S1_S1_Pf',
        '_ZL18flash_attn_ext_f16ILi80ELi80ELi16ELi1ELb0ELb0EEvPKcS1_S1_S1_S1_PKiPfP15HIP_vector_typeIfLj2EEffffjfiS5_IjLj3EEiiiiiiiiiiiliiliiiiil',
        '_ZL18flash_attn_ext_f16ILi112ELi112ELi16ELi2ELb0ELb0EEvPKcS1_S1_S1_S1_PKiPfP15HIP_vector_typeIfLj2EEffffjfiS5_IjLj3EEiiiiiiiiiiiliiliiiiil',
        '_ZL18flash_attn_ext_f16ILi80ELi80ELi32ELi1ELb0ELb0EEvPKcS1_S1_S1_S1_PKiPfP15HIP_vector_typeIfLj2EEffffjfiS5_IjLj3EEiiiiiiiiiiiliiliiiiil',
        '_ZL18flash_attn_ext_f16ILi96ELi96ELi16ELi1ELb0ELb0EEvPKcS1_S1_S1_S1_PKiPfP15HIP_vector_typeIfLj2EEffffjfiS5_IjLj3EEiiiiiiiiiiiliiliiiiil',
        '_ZL18flash_attn_ext_f16ILi128ELi128ELi16ELi2ELb0ELb0EEvPKcS1_S1_S1_S1_PKiPfP15HIP_vector_typeIfLj2EEffffjfiS5_IjLj3EEiiiiiiiiiiiliiliiiiil',
        '_ZL18flash_attn_ext_f16ILi128ELi128ELi16ELi2ELb1ELb0EEvPKcS1_S1_S1_S1_PKiPfP15HIP_vector_typeIfLj2EEffffjfiS5_IjLj3EEiiiiiiiiiiiliiliiiiil',
        '_ZL18flash_attn_ext_f16ILi96ELi96ELi32ELi1ELb0ELb0EEvPKcS1_S1_S1_S1_PKiPfP15HIP_vector_typeIfLj2EEffffjfiS5_IjLj3EEiiiiiiiiiiiliiliiiiil',
        '_ZL18flash_attn_ext_f16ILi112ELi112ELi16ELi1ELb0ELb0EEvPKcS1_S1_S1_S1_PKiPfP15HIP_vector_typeIfLj2EEffffjfiS5_IjLj3EEiiiiiiiiiiiliiliiiiil',
        '_ZL18flash_attn_ext_f16ILi112ELi112ELi32ELi1ELb0ELb0EEvPKcS1_S1_S1_S1_PKiPfP15HIP_vector_typeIfLj2EEffffjfiS5_IjLj3EEiiiiiiiiiiiliiliiiiil',
        '_ZL18flash_attn_ext_f16ILi128ELi128ELi16ELi1ELb0ELb0EEvPKcS1_S1_S1_S1_PKiPfP15HIP_vector_typeIfLj2EEffffjfiS5_IjLj3EEiiiiiiiiiiiliiliiiiil',
        '_ZL18flash_attn_ext_f16ILi128ELi128ELi16ELi1ELb1ELb0EEvPKcS1_S1_S1_S1_PKiPfP15HIP_vector_typeIfLj2EEffffjfiS5_IjLj3EEiiiiiiiiiiiliiliiiiil',
        '_ZL18flash_attn_ext_f16ILi80ELi80ELi2ELi8ELb0ELb0EEvPKcS1_S1_S1_S1_PKiPfP15HIP_vector_typeIfLj2EEffffjfiS5_IjLj3EEiiiiiiiiiiiliiliiiiil',
        '_ZL18flash_attn_ext_f16ILi96ELi96ELi2ELi8ELb0ELb0EEvPKcS1_S1_S1_S1_PKiPfP15HIP_vector_typeIfLj2EEffffjfiS5_IjLj3EEiiiiiiiiiiiliiliiiiil',
        '_ZL18flash_attn_ext_f16ILi112ELi112ELi2ELi8ELb0ELb0EEvPKcS1_S1_S1_S1_PKiPfP15HIP_vector_typeIfLj2EEffffjfiS5_IjLj3EEiiiiiiiiiiiliiliiiiil',
        '_ZL18flash_attn_ext_f16ILi128ELi128ELi2ELi8ELb0ELb0EEvPKcS1_S1_S1_S1_PKiPfP15HIP_vector_typeIfLj2EEffffjfiS5_IjLj3EEiiiiiiiiiiiliiliiiiil',
        '_ZL18flash_attn_ext_f16ILi128ELi128ELi2ELi8ELb1ELb0EEvPKcS1_S1_S1_S1_PKiPfP15HIP_vector_typeIfLj2EEffffjfiS5_IjLj3EEiiiiiiiiiiiliiliiiiil',
        '_ZL18flash_attn_ext_f16ILi112ELi112ELi16ELi4ELb0ELb0EEvPKcS1_S1_S1_S1_PKiPfP15HIP_vector_typeIfLj2EEffffjfiS5_IjLj3EEiiiiiiiiiiiliiliiiiil',
        '_ZL18flash_attn_ext_f16ILi128ELi128ELi16ELi4ELb0ELb0EEvPKcS1_S1_S1_S1_PKiPfP15HIP_vector_typeIfLj2EEffffjfiS5_IjLj3EEiiiiiiiiiiiliiliiiiil',
        '_ZL18flash_attn_ext_f16ILi128ELi128ELi16ELi4ELb1ELb0EEvPKcS1_S1_S1_S1_PKiPfP15HIP_vector_typeIfLj2EEffffjfiS5_IjLj3EEiiiiiiiiiiiliiliiiiil',
        '_ZL18flash_attn_ext_f16ILi128ELi128ELi32ELi2ELb0ELb0EEvPKcS1_S1_S1_S1_PKiPfP15HIP_vector_typeIfLj2EEffffjfiS5_IjLj3EEiiiiiiiiiiiliiliiiiil',
        '_ZL18flash_attn_ext_f16ILi96ELi96ELi4ELi4ELb0ELb0EEvPKcS1_S1_S1_S1_PKiPfP15HIP_vector_typeIfLj2EEffffjfiS5_IjLj3EEiiiiiiiiiiiliiliiiiil',
        '_ZL18flash_attn_ext_f16ILi112ELi112ELi4ELi4ELb0ELb0EEvPKcS1_S1_S1_S1_PKiPfP15HIP_vector_typeIfLj2EEffffjfiS5_IjLj3EEiiiiiiiiiiiliiliiiiil',
        '_ZL18flash_attn_ext_f16ILi128ELi128ELi4ELi4ELb0ELb0EEvPKcS1_S1_S1_S1_PKiPfP15HIP_vector_typeIfLj2EEffffjfiS5_IjLj3EEiiiiiiiiiiiliiliiiiil',
        '_ZL18flash_attn_ext_f16ILi128ELi128ELi4ELi4ELb1ELb0EEvPKcS1_S1_S1_S1_PKiPfP15HIP_vector_typeIfLj2EEffffjfiS5_IjLj3EEiiiiiiiiiiiliiliiiiil',
        '_ZL18flash_attn_ext_f16ILi80ELi80ELi4ELi8ELb0ELb0EEvPKcS1_S1_S1_S1_PKiPfP15HIP_vector_typeIfLj2EEffffjfiS5_IjLj3EEiiiiiiiiiiiliiliiiiil',
        '_ZL18flash_attn_ext_f16ILi128ELi128ELi4ELi8ELb0ELb0EEvPKcS1_S1_S1_S1_PKiPfP15HIP_vector_typeIfLj2EEffffjfiS5_IjLj3EEiiiiiiiiiiiliiliiiiil',
        '_ZL18flash_attn_ext_f16ILi96ELi96ELi64ELi1ELb0ELb0EEvPKcS1_S1_S1_S1_PKiPfP15HIP_vector_typeIfLj2EEffffjfiS5_IjLj3EEiiiiiiiiiiiliiliiiiil',
        '_ZL18flash_attn_ext_f16ILi112ELi112ELi64ELi1ELb0ELb0EEvPKcS1_S1_S1_S1_PKiPfP15HIP_vector_typeIfLj2EEffffjfiS5_IjLj3EEiiiiiiiiiiiliiliiiiil',
        '_ZL18flash_attn_ext_f16ILi128ELi128ELi64ELi1ELb0ELb0EEvPKcS1_S1_S1_S1_PKiPfP15HIP_vector_typeIfLj2EEffffjfiS5_IjLj3EEiiiiiiiiiiiliiliiiiil',
        '_ZL18flash_attn_ext_f16ILi128ELi128ELi64ELi1ELb1ELb0EEvPKcS1_S1_S1_S1_PKiPfP15HIP_vector_typeIfLj2EEffffjfiS5_IjLj3EEiiiiiiiiiiiliiliiiiil',
        '_ZL18flash_attn_ext_f16ILi64ELi64ELi8ELi4ELb0ELb0EEvPKcS1_S1_S1_S1_PKiPfP15HIP_vector_typeIfLj2EEffffjfiS5_IjLj3EEiiiiiiiiiiiliiliiiiil',
        '_ZL18flash_attn_ext_f16ILi80ELi80ELi8ELi4ELb0ELb0EEvPKcS1_S1_S1_S1_PKiPfP15HIP_vector_typeIfLj2EEffffjfiS5_IjLj3EEiiiiiiiiiiiliiliiiiil',
        '_ZL18flash_attn_ext_f16ILi96ELi96ELi8ELi4ELb0ELb0EEvPKcS1_S1_S1_S1_PKiPfP15HIP_vector_typeIfLj2EEffffjfiS5_IjLj3EEiiiiiiiiiiiliiliiiiil',
        '_ZL18flash_attn_ext_f16ILi112ELi112ELi8ELi4ELb0ELb0EEvPKcS1_S1_S1_S1_PKiPfP15HIP_vector_typeIfLj2EEffffjfiS5_IjLj3EEiiiiiiiiiiiliiliiiiil',
        '_ZL18flash_attn_ext_f16ILi80ELi80ELi8ELi2ELb0ELb0EEvPKcS1_S1_S1_S1_PKiPfP15HIP_vector_typeIfLj2EEffffjfiS5_IjLj3EEiiiiiiiiiiiliiliiiiil',
        '_ZL18flash_attn_ext_f16ILi128ELi128ELi8ELi4ELb0ELb0EEvPKcS1_S1_S1_S1_PKiPfP15HIP_vector_typeIfLj2EEffffjfiS5_IjLj3EEiiiiiiiiiiiliiliiiiil',
        '_ZL18flash_attn_ext_f16ILi128ELi128ELi8ELi4ELb1ELb0EEvPKcS1_S1_S1_S1_PKiPfP15HIP_vector_typeIfLj2EEffffjfiS5_IjLj3EEiiiiiiiiiiiliiliiiiil',
        '_ZL18flash_attn_ext_f16ILi96ELi96ELi8ELi2ELb0ELb0EEvPKcS1_S1_S1_S1_PKiPfP15HIP_vector_typeIfLj2EEffffjfiS5_IjLj3EEiiiiiiiiiiiliiliiiiil',
        '_ZL18flash_attn_ext_f16ILi112ELi112ELi8ELi2ELb0ELb0EEvPKcS1_S1_S1_S1_PKiPfP15HIP_vector_typeIfLj2EEffffjfiS5_IjLj3EEiiiiiiiiiiiliiliiiiil',
        '_ZL18flash_attn_ext_f16ILi128ELi128ELi8ELi2ELb0ELb0EEvPKcS1_S1_S1_S1_PKiPfP15HIP_vector_typeIfLj2EEffffjfiS5_IjLj3EEiiiiiiiiiiiliiliiiiil',
        '_ZL18flash_attn_ext_f16ILi128ELi128ELi8ELi2ELb1ELb0EEvPKcS1_S1_S1_S1_PKiPfP15HIP_vector_typeIfLj2EEffffjfiS5_IjLj3EEiiiiiiiiiiiliiliiiiil',
        '_ZL18flash_attn_ext_f16ILi112ELi112ELi8ELi8ELb0ELb0EEvPKcS1_S1_S1_S1_PKiPfP15HIP_vector_typeIfLj2EEffffjfiS5_IjLj3EEiiiiiiiiiiiliiliiiiil',
        '_ZL18flash_attn_ext_f16ILi128ELi128ELi8ELi8ELb0ELb0EEvPKcS1_S1_S1_S1_PKiPfP15HIP_vector_typeIfLj2EEffffjfiS5_IjLj3EEiiiiiiiiiiiliiliiiiil',
        '_ZL18flash_attn_ext_f16ILi128ELi128ELi8ELi8ELb1ELb0EEvPKcS1_S1_S1_S1_PKiPfP15HIP_vector_typeIfLj2EEffffjfiS5_IjLj3EEiiiiiiiiiiiliiliiiiil',
        '_ZL24mul_mat_q_stream_k_fixupIL9ggml_type22ELi8ELb1EEvPKiS2_PfPKfiiimimimi',
        '_ZL9mul_mat_qIL9ggml_type3ELi32ELb0EEvPKcPKiS4_S4_PfS5_iiiiiiiiiiiiiiiii',
        '_ZL9mul_mat_qIL9ggml_type3ELi48ELb0EEvPKcPKiS4_S4_PfS5_iiiiiiiiiiiiiiiii',
        '_ZL9mul_mat_qIL9ggml_type20ELi32ELb1EEvPKcPKiS4_S4_PfS5_iiiiiiiiiiiiiiiii',
        '_ZL9mul_mat_qIL9ggml_type17ELi64ELb0EEvPKcPKiS4_S4_PfS5_iiiiiiiiiiiiiiiii',
        '_ZL18flash_attn_ext_f16ILi80ELi80ELi4ELi4ELb0ELb0EEvPKcS1_S1_S1_S1_PKiPfP15HIP_vector_typeIfLj2EEffffjfiS5_IjLj3EEiiiiiiiiiiiliiliiiiil',
        '_ZL15flash_attn_tileILi256ELi256ELi32ELi1ELb0EEvPKcS1_S1_S1_S1_PKiPfP15HIP_vector_typeIfLj2EEffffjfiS5_IjLj3EEiiiiiiiiiiiliiliiiiil',
        '_ZL9mul_mat_qIL9ggml_type19ELi112ELb1EEvPKcPKiS4_S4_PfS5_iiiiiiiiiiiiiiiii',
        '_ZL9mul_mat_qIL9ggml_type17ELi112ELb1EEvPKcPKiS4_S4_PfS5_iiiiiiiiiiiiiiiii',
        '_ZL9mul_mat_qIL9ggml_type22ELi112ELb1EEvPKcPKiS4_S4_PfS5_iiiiiiiiiiiiiiiii',
        '_ZL9mul_mat_qIL9ggml_type19ELi128ELb0EEvPKcPKiS4_S4_PfS5_iiiiiiiiiiiiiiiii',
        '_ZL9mul_mat_qIL9ggml_type19ELi128ELb1EEvPKcPKiS4_S4_PfS5_iiiiiiiiiiiiiiiii',
        '_ZL9mul_mat_qIL9ggml_type7ELi112ELb1EEvPKcPKiS4_S4_PfS5_iiiiiiiiiiiiiiiii',
        '_ZL9mul_mat_qIL9ggml_type3ELi128ELb0EEvPKcPKiS4_S4_PfS5_iiiiiiiiiiiiiiiii',
        '_ZL9mul_mat_qIL9ggml_type3ELi128ELb1EEvPKcPKiS4_S4_PfS5_iiiiiiiiiiiiiiiii',
        '_ZL9mul_mat_qIL9ggml_type7ELi128ELb0EEvPKcPKiS4_S4_PfS5_iiiiiiiiiiiiiiiii',
        '_ZL9mul_mat_qIL9ggml_type7ELi128ELb1EEvPKcPKiS4_S4_PfS5_iiiiiiiiiiiiiiiii',
        '_ZL9mul_mat_qIL9ggml_type11ELi112ELb0EEvPKcPKiS4_S4_PfS5_iiiiiiiiiiiiiiiii',
        '_ZL9mul_mat_qIL9ggml_type11ELi112ELb1EEvPKcPKiS4_S4_PfS5_iiiiiiiiiiiiiiiii',
        '_ZL24mul_mat_q_stream_k_fixupIL9ggml_type11ELi128ELb0EEvPKiS2_PfPKfiiimimimi',
        '_ZL18flash_attn_ext_f16ILi128ELi128ELi32ELi1ELb0ELb0EEvPKcS1_S1_S1_S1_PKiPfP15HIP_vector_typeIfLj2EEffffjfiS5_IjLj3EEiiiiiiiiiiiliiliiiiil',
        '_ZL9mul_mat_qIL9ggml_type2ELi112ELb0EEvPKcPKiS4_S4_PfS5_iiiiiiiiiiiiiiiii',
        '_ZL18flash_attn_ext_f16ILi112ELi112ELi32ELi2ELb0ELb0EEvPKcS1_S1_S1_S1_PKiPfP15HIP_vector_typeIfLj2EEffffjfiS5_IjLj3EEiiiiiiiiiiiliiliiiiil',
        '_ZL18flash_attn_ext_f16ILi112ELi112ELi4ELi8ELb0ELb0EEvPKcS1_S1_S1_S1_PKiPfP15HIP_vector_typeIfLj2EEffffjfiS5_IjLj3EEiiiiiiiiiiiliiliiiiil',
        '_ZL18flash_attn_ext_f16ILi128ELi128ELi32ELi1ELb1ELb0EEvPKcS1_S1_S1_S1_PKiPfP15HIP_vector_typeIfLj2EEffffjfiS5_IjLj3EEiiiiiiiiiiiliiliiiiil',
        '_ZL18flash_attn_ext_f16ILi128ELi128ELi32ELi2ELb1ELb0EEvPKcS1_S1_S1_S1_PKiPfP15HIP_vector_typeIfLj2EEffffjfiS5_IjLj3EEiiiiiiiiiiiliiliiiiil',
        '_ZL18flash_attn_ext_f16ILi128ELi128ELi4ELi8ELb1ELb0EEvPKcS1_S1_S1_S1_PKiPfP15HIP_vector_typeIfLj2EEffffjfiS5_IjLj3EEiiiiiiiiiiiliiliiiiil',
        '_ZL18flash_attn_ext_f16ILi96ELi96ELi4ELi8ELb0ELb0EEvPKcS1_S1_S1_S1_PKiPfP15HIP_vector_typeIfLj2EEffffjfiS5_IjLj3EEiiiiiiiiiiiliiliiiiil',
        '_ZL18flash_attn_ext_vecILi128ELi2EL9ggml_type2ELS0_2ELb0EEvPKcS2_S2_S2_S2_PKiPfP15HIP_vector_typeIfLj2EEffffjfiS6_IjLj3EEiiiiiiiiiiiliiliiiiil',
        '_ZL9mul_mat_qIL9ggml_type10ELi16ELb1EEvPKcPKiS4_S4_PfS5_iiiiiiiiiiiiiiiii',
        '_ZL9mul_mat_qIL9ggml_type12ELi128ELb1EEvPKcPKiS4_S4_PfS5_iiiiiiiiiiiiiiiii'
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
