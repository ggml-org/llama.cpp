"""Build a private mask library from existing objects; preserve the repository build."""
import difflib
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
OLD = ROOT/'spark_dspark_prob_bench.cpp'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    build = REPO/'build/src'
    runtime = ROOT/'runtime'
    runtime.mkdir(exist_ok=True)
    link = shlex.split((build/'CMakeFiles/llama.dir/link.txt').read_text())
    output_index = link.index('-o') + 1
    original_library = (build/link[output_index]).resolve(strict=True)
    original_library_sha256 = sha(original_library)
    sonames = [arg.split(',')[-1] for arg in link if arg.startswith('-Wl,-soname,')]
    if len(sonames) != 1 or Path(sonames[0]).name != sonames[0]:
        raise RuntimeError('Expected one shared-library SONAME in the CMake link command')
    soname = sonames[0]
    path = REPO/'src/llama-kv-cache.cpp'
    original = path.read_text()
    text = original.replace('#include <algorithm>', '#include <algorithm>\n#include <atomic>\n#include <cstdlib>')
    marker = 'struct args_set_input_kq_mask {'
    counters = '''static std::atomic<uint64_t> spark_fixed_checks{0};
static std::atomic<uint64_t> spark_fixed_long_checks{0};
extern "C" uint64_t spark_company_mask_version();
extern "C" uint64_t spark_company_mask_checks();
extern "C" uint64_t spark_company_mask_long_checks();
extern "C" uint64_t spark_company_mask_version() { return 20261028; }
extern "C" uint64_t spark_company_mask_checks() { return spark_fixed_checks.load(); }
extern "C" uint64_t spark_company_mask_long_checks() { return spark_fixed_long_checks.load(); }

'''
    assert text.count(marker) == 1
    text = text.replace(marker, counters+marker)
    start = text.index('template<typename T, bool causal, bool swa, bool is_2d, bool alibi>')
    end = text.index('template<typename T, bool causal, bool swa, bool is_2d>', start)
    body = text[start:end]
    body = body.replace('    const T mask_keep',
        '    const bool fixed_history = !causal && swa && std::getenv("LLAMA_DSPARK_FIXED_HISTORY");\n\n    const T mask_keep', 1)
    body = body.replace('            const llama_pos p1 = ubatch->pos[i];',
        '            const llama_pos p1 = ubatch->pos[i];\n'
        '            const llama_pos p1_swa = fixed_history ? std::max<llama_pos>(0, seq_pos_min[seq_id] - 1) : p1;', 1)
    needle = 'llama_hparams::is_masked_swa(n_swa, swa_type, p0, p1)'
    assert body.count(needle) == 1
    body = body.replace(needle, 'llama_hparams::is_masked_swa(n_swa, swa_type, p0, p1_swa)')
    tail = '        }\n    }\n}\n\n'
    assert body.endswith(tail)
    validation = '''        }
        if (fixed_history && ubatch->token && !ubatch->embd && n_tps <= 7) {
            for (uint32_t ii = 0; ii < n_tps; ++ii) {
                const uint32_t i = s*n_tps + ii;
                const llama_seq_id sid = ubatch->seq_id[i][0];
                const llama_pos first = seq_pos_min[sid];
                const auto & cells = v_cells.at(seq_to_stream[sid]);
                const llama_pos lower = std::max<llama_pos>(0, first - n_swa);
                uint32_t visible = 0;
                for (int64_t j = 0; j < n_kv; ++j) {
                    if (data[n_kv*i + j] == mask_keep) {
                        if (cells.is_empty(j) || !cells.seq_has(j, sid) || cells.pos_get(j) < lower) {
                            throw std::runtime_error("fixed-history mask contains an invalid key");
                        }
                        ++visible;
                    }
                }
                const uint32_t expected = std::min<uint32_t>(first, n_swa) + n_tps;
                if (visible != expected) {
                    throw std::runtime_error("fixed-history cache lost a required boundary token: " +
                        std::to_string(visible) + " vs " + std::to_string(expected));
                }
                ++spark_fixed_checks;
                if (first >= (llama_pos) n_swa) ++spark_fixed_long_checks;
            }
        }
    }
}

'''
    body = body[:-len(tail)]+validation
    text = text[:start]+body+text[end:]
    # Keep the seven future query positions plus the extra fixed left-edge key.
    needle = 'llama_hparams::is_masked_swa(n_swa, swa_type, pos_cell, cells.seq_pos_max(seq_id_cell) + 1)'
    assert text.count(needle) == 1
    replacement = ('llama_hparams::is_masked_swa(n_swa + '
        '((model.arch == LLM_ARCH_DFLASH && std::getenv("LLAMA_DSPARK_FIXED_HISTORY")) ? 8 : 0), '
        'swa_type, pos_cell, cells.seq_pos_max(seq_id_cell) + 1)')
    text = text.replace(needle, replacement)
    private_source = ROOT/'llama-kv-cache-fixed.cpp'
    private_source.write_text(text)
    (ROOT/'fixed_history.patch').write_text(''.join(difflib.unified_diff(
        original.splitlines(keepends=True),text.splitlines(keepends=True),
        fromfile='a/src/llama-kv-cache.cpp',tofile='b/src/llama-kv-cache.cpp')))
    flags = {}
    for line in (build/'CMakeFiles/llama.dir/flags.make').read_text().splitlines():
        if ' = ' in line:
            key, value = line.split(' = ',1)
            flags[key] = shlex.split(value)
    obj = ROOT/'llama-kv-cache-fixed.o'
    compile_cmd = ['/usr/bin/c++', *flags['CXX_DEFINES'], *flags['CXX_INCLUDES'], *flags['CXX_FLAGS'],
                   '-c',str(private_source),'-o',str(obj)]
    subprocess.run(compile_cmd, check=True)
    library = runtime/original_library.name
    link[output_index] = str(library)
    old_object = 'CMakeFiles/llama.dir/llama-kv-cache.cpp.o'
    assert link.count(old_object) == 1
    link[link.index(old_object)] = str(obj)
    subprocess.run(link, cwd=build, check=True)
    for name in dict.fromkeys((soname, 'libllama.so')):
        if name == library.name:
            continue
        p = runtime/name
        if p.is_symlink():
            if p.readlink() == Path(library.name):
                continue
            p.unlink()
        elif p.exists():
            raise RuntimeError(f'Refusing to replace a regular runtime file: {p}')
        p.symlink_to(library.name)
    bench_original = OLD.read_text()
    bench = bench_original.replace('#include <algorithm>','#include <algorithm>\n#include <cstdlib>\n#include <dlfcn.h>')
    needle = '        ggml_backend_load_all();'
    assert bench.count(needle) == 1
    bench = bench.replace(needle,
        '        auto version = (uint64_t (*)()) dlsym(RTLD_DEFAULT, "spark_company_mask_version");\n'
        '        check(version && version() == 20261028, "private fixed-history library not loaded");\n'
        '        check(std::getenv("LLAMA_DSPARK_FIXED_HISTORY"), "fixed-history mode is not enabled");\n'+needle)
    needle = 'cp.type_k = cp.type_v = GGML_TYPE_F16;'
    assert bench.count(needle) == 1
    bench = bench.replace(needle,'cp.type_k = cp.type_v = GGML_TYPE_BF16;')
    needle = '        const auto started = Clock::now();'
    assert bench.count(needle) == 1
    bench = bench.replace(needle,
        '        auto checks = (uint64_t (*)()) dlsym(RTLD_DEFAULT, "spark_company_mask_checks");\n'
        '        auto long_checks = (uint64_t (*)()) dlsym(RTLD_DEFAULT, "spark_company_mask_long_checks");\n'
        '        const uint64_t checks_before = checks(), long_before = long_checks();\n'+needle)
    needle = '{"head_parity_max_abs", parity_error},'
    assert bench.count(needle) == 1
    bench = bench.replace(needle,needle+'\n                {"fixed_window_checked_rows", checks() - checks_before},\n'
        '                {"fixed_window_long_checked_rows", long_checks() - long_before}, {"kv_type", "bf16"},')
    benchmark = ROOT/'spark_company_prob_bench.cpp'
    benchmark.write_text(bench)
    binary = ROOT/'spark_company_prob_bench'
    bench_cmd = ['/usr/bin/c++','-std=c++17','-O2','-I'+str(REPO/'include'),'-I'+str(REPO/'src'),
        '-I'+str(REPO/'ggml/include'),'-I'+str(REPO/'vendor/nlohmann'),str(benchmark),
        '-L'+str(runtime),'-L'+str(REPO/'build/bin'),'-Wl,-rpath,'+str(runtime)+':'+str(REPO/'build/bin'),
        '-lllama','-lggml','-lggml-base','-ldl','-o',str(binary)]
    subprocess.run(bench_cmd, check=True)
    manifest = {'original_mask_source_sha256': sha(path),'private_mask_source_sha256': sha(private_source),
        'original_benchmark_source_sha256': sha(OLD),'benchmark_source_sha256': sha(benchmark),
        'benchmark_binary_sha256': sha(binary),'private_library_sha256': sha(library),
        'original_library_path': str(original_library), 'private_library_path': str(library),
        'library_soname': soname, 'original_library_sha256': sha(original_library),
        'compile_argv':compile_cmd,'link_argv':link,'benchmark_compile_argv':bench_cmd,
        'fixed_history_retention_margin':8,'query_block_max':7,'kv_type':'bf16',
        'scope':'private acceptance experiment; no repository or installed runtime overwritten'}
    if manifest['original_library_sha256'] != original_library_sha256:
        raise RuntimeError('The original library changed during the private build')
    (ROOT/'build_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('Private library and benchmark ready:',ROOT)


if __name__ == '__main__':
    main()
