#!/usr/bin/env python3
"""Ensure routing tests detect disabled, miswired, and overbroad selection."""
import argparse
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--compiler', default='c++')
args = p.parse_args()
root = Path(__file__).resolve().parents[1]
cpp = 'ggml/src/ggml-vulkan/ggml-vulkan.cpp'
header = 'ggml/src/ggml-vulkan/ggml-vulkan-adreno-compat.hpp'
runner = 'tests/test-vulkan-adreno-routing.py'
mutations = [
    ('branch_disabled', cpp, '#if defined(__ANDROID__) && defined(GGML_VULKAN_ADRENO_750_SHMEM)', '#if 0'),
    ('wrong_pointer', cpp, 'spv_data = mul_mat_vec_f32_f32_f32_data;', 'spv_data = spv_data;'),
    ('wrong_length', cpp, 'spv_size = mul_mat_vec_f32_f32_f32_len;', 'spv_size = 0;'),
    ('guard_bypassed', header, 'return vendor ==', 'return true || vendor =='),
    ('platform_widened', cpp, '#if defined(__ANDROID__) && defined(GGML_VULKAN_ADRENO_750_SHMEM)', '#if defined(GGML_VULKAN_ADRENO_750_SHMEM)'),
]
subprocess.run([sys.executable, str(root/runner), '--compiler', args.compiler], check=True)
for name, target, before, after in mutations:
    with tempfile.TemporaryDirectory() as temp:
        temp = Path(temp)
        for file in [cpp, header, runner]:
            dst = temp/file;dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(root/file, dst)
        file = temp/target;text = file.read_text()
        assert text.count(before) == 1, name
        file.write_text(text.replace(before, after))
        result = subprocess.run([sys.executable, str(temp/runner), '--compiler', args.compiler], capture_output=True, text=True)
        if result.returncode == 0:
            raise AssertionError(f'Undetected routing mutation: {name}')
        # The deliberately valid mutations must fail in the executed routing binary,
        # not be credited for a compiler error or missing dependency.
        if "/routing']' returned non-zero exit status 1" not in result.stderr:
            raise AssertionError(f'Unexpected failure mode for {name}: {result.stderr}')
        print(f'PASS: detected {name} at runtime')
