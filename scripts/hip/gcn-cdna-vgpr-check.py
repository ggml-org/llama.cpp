#!/usr/bin/env python3

import sys
from collections import defaultdict

def parse_log_file(filepath):
    """Parse log file and extract function VGPR usage."""
    import re
    
    functions = defaultdict(lambda: {'vgprs': 0, 'spill': 0, 'location': ''})
    
    try:
        with open(filepath, 'r') as f:
            content = f.read()
            # Find all function entries with VGPR usage including location
            pattern = r'([^:]+:\d+):.*?Function Name: (\S+).*?VGPRs: (\d+).*?VGPRs Spill: (\d+)'
            matches = re.findall(pattern, content, re.DOTALL)
            
            for location, func_name, vgprs, spill in matches:
                functions[func_name]['vgprs'] = int(vgprs)
                functions[func_name]['spill'] = int(spill)
                # Extract just the filename and line number
                parts = location.split('/')
                if len(parts) > 0:
                    short_location = parts[-1]  # Get last part (filename)
                    # Check if there's a line number after filename
                    if ':' in short_location:
                        functions[func_name]['location'] = short_location
                    else:
                        functions[func_name]['location'] = location
                else:
                    functions[func_name]['location'] = location
    except FileNotFoundError:
        print(f"Error: File {filepath} not found", file=sys.stderr)
        sys.exit(1)

    return functions

def main():
    if len(sys.argv) < 2:
        print("Usage: ./vgpr_check.py <log_file>", file=sys.stderr)
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
        '_ZL9mul_mat_qIL9ggml_type2ELi112ELb0EEvPKcPKiS4_S4_PfS5_iiiiiiiiiiiiiiiii'
    }

    functions = parse_log_file(log_file)
    found_issues = False

    # First print all ignored functions (deduplicated)
    printed_ignored = set()
    for func_name, data in sorted(functions.items()):
        total_vgprs = data['vgprs'] + data['spill']
        if total_vgprs > 256 and func_name in ignored and func_name not in printed_ignored:
            location = data.get('location', log_file)
            print(f"{location}: {func_name} - Total VGPRs: {total_vgprs} ({data['vgprs']} + {data['spill']}) [IGNORED]")
            printed_ignored.add(func_name)

    # Then print new functions with issues in red
    for func_name, data in sorted(functions.items()):
        total_vgprs = data['vgprs'] + data['spill']
        if total_vgprs > 256 and func_name not in ignored:
            status = "[IGNORED]" if func_name in ignored else ""
            location = data.get('location', log_file)
            # Print in red if not ignored
            color_code = "\033[91m" if func_name not in ignored else ""
            reset_code = "\033[0m" if func_name not in ignored else ""
            print(f"{color_code}{location}: {func_name} - Total VGPRs: {total_vgprs} ({data['vgprs']} + {data['spill']}) {status}{reset_code}")
            if func_name not in ignored:
                found_issues = True

    sys.exit(1 if found_issues else 0)

if __name__ == "__main__":
    main()
