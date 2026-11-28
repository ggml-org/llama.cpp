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
    ('branch_disabled', cpp, '#if defined(__ANDROID__) && defined(GGML_VULKAN_ADRENO_750_SHMEM)\n    // This Adreno 750', '#if 0\n    // This Adreno 750'),
    ('fallback_inverted', cpp, '        use_subgroups = false;\n', '        use_subgroups = true;\n'),
    ('guard_bypassed', header, 'return vendor ==', 'return true || vendor =='),
    ('platform_widened', cpp, '#if defined(__ANDROID__) && defined(GGML_VULKAN_ADRENO_750_SHMEM)\n    // This Adreno 750', '#if defined(GGML_VULKAN_ADRENO_750_SHMEM)\n    // This Adreno 750'),
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
