"""Exercise library version discovery without recompiling model objects."""
import hashlib
import importlib.util
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]


class RuntimeBuildTest(unittest.TestCase):
    def run_build(self, version, stale_alias=False):
        with tempfile.TemporaryDirectory(prefix='spark-runtime-test-') as directory:
            root = Path(directory)
            tools = root/'tools/spark-dspark-company'
            build = root/'build/src'
            bindir = root/'build/bin'
            tools.mkdir(parents=True)
            (root/'src').mkdir()
            (build/'CMakeFiles/llama.dir').mkdir(parents=True)
            bindir.mkdir()
            shutil.copy2(HERE/'build_runtime.py', tools/'build_runtime.py')
            shutil.copy2(HERE/'spark_dspark_prob_bench.cpp', tools/'spark_dspark_prob_bench.cpp')
            shutil.copy2(REPO/'src/llama-kv-cache.cpp', root/'src/llama-kv-cache.cpp')
            library_name = f'libllama.so.{version}'
            original = bindir/library_name
            original.write_bytes(b'original model library')
            (bindir/'libllama.so.0').symlink_to(library_name)
            (bindir/'libllama.so').symlink_to('libllama.so.0')
            (build/'CMakeFiles/llama.dir/flags.make').write_text(
                'CXX_DEFINES = \nCXX_INCLUDES = \nCXX_FLAGS = \n')
            (build/'CMakeFiles/llama.dir/link.txt').write_text(
                f'c++ -shared -Wl,-soname,libllama.so.0 -o ../bin/{library_name} '
                'CMakeFiles/llama.dir/llama-kv-cache.cpp.o\n')
            if stale_alias:
                runtime = tools/'runtime'
                runtime.mkdir()
                (runtime/'libllama.so.0.1.2').write_bytes(b'old private library')
                (runtime/'libllama.so').symlink_to('libllama.so.0.1.2')
                (runtime/'libllama.so.0').symlink_to('libllama.so.0.1.2')
            spec = importlib.util.spec_from_file_location('runtime_under_test', tools/'build_runtime.py')
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            def compiler(argv, cwd=None, **kwargs):
                output = Path(argv[argv.index('-o')+1])
                if not output.is_absolute():
                    output = Path(cwd)/output
                output.write_bytes(b'compiled artifact')

            original_hash = hashlib.sha256(original.read_bytes()).hexdigest()
            with patch.object(module.subprocess, 'run', side_effect=compiler):
                module.main()
            import json
            manifest = json.loads((tools/'build_manifest.json').read_text())
            self.assertEqual(manifest['original_library_sha256'], original_hash)
            self.assertEqual(hashlib.sha256(original.read_bytes()).hexdigest(), original_hash)
            for alias in ['libllama.so', 'libllama.so.0']:
                self.assertEqual((tools/'runtime'/alias).resolve().name, library_name)

    def test_current_library_version(self):
        self.run_build('0.7.0')

    def test_previous_library_version(self):
        self.run_build('0.1.2')

    def test_existing_runtime_aliases_are_updated(self):
        self.run_build('0.7.0', stale_alias=True)


if __name__ == '__main__':
    unittest.main()
