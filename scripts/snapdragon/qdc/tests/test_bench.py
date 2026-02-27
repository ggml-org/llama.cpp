import pytest
import subprocess
import sys

tmp_path='/data/local/tmp'
pkg_path=f'{tmp_path}/llama.cpp'
lib_path=f'{pkg_path}/lib'
bin_path=f'{pkg_path}/bin'

model='../gguf/Llama-3.2-1B-Instruct-Q4_0.gguf'
cli_pref=f'cd {pkg_path} && LD_LIBRARY_PATH={lib_path} ADSP_LIBRARY_PATH={lib_path} {bin_path}'


def run_cmd(cmd):
    p = subprocess.run(cmd, text = True, stdout = subprocess.PIPE, stderr = subprocess.STDOUT)
    sys.stdout.write(p.stdout)
    assert(p.returncode == 0)


@pytest.mark.dependency()
def test_install():
    run_cmd(['adb', 'push', 'llama.cpp', f'{tmp_path}'])
    run_cmd(['adb', 'shell', f'chmod 755 {bin_path}/*'])


## Basic cli tests
def run_llama_cli(dev, opts):
    prompt='what is the most popular cookie in the world?\nPlease provide a very brief bullet point summary.\nBegin your answer with **BEGIN**.'
    opts = '--batch-size 128 -n 128 -no-cnv --seed 42 ' + opts
    run_cmd(['adb', 'shell', f'{cli_pref}/llama-cli -m {model} --device {dev} -ngl 99 -t 4 {opts} -p "{prompt}"'])


@pytest.mark.dependency(depends=['test_install'])
def test_llama_cli_cpu():
    run_llama_cli('none', '-ctk q8_0 -ctv q8_0 -fa on')


@pytest.mark.dependency(depends=['test_install'])
def test_llama_cli_gpu():
    run_llama_cli('GPUOpenCL', '-fa on')


@pytest.mark.dependency(depends=['test_install'])
def test_llama_cli_npu():
    run_llama_cli('HTP0', '-ctk q8_0 -ctv q8_0 -fa on')


## Basic bench tests
def run_llama_bench(dev):
    run_cmd(['adb', 'shell', f'{cli_pref}/llama-bench -m {model} --device {dev} -ngl 99 --batch-size 128 -t 4 -p 128 -n 32'])


@pytest.mark.dependency(depends=['test_install'])
def test_llama_bench_cpu():
    run_llama_bench('none')


def test_llama_bench_gpu():
    run_llama_bench('GPUOpenCL')


def test_llama_bench_npu():
    run_llama_bench('HTP0')
