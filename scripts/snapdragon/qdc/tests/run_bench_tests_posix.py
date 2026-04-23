"""
On-device bench and completion test runner for llama.cpp (CPU, GPU, NPU backends).

Executed by QDC's Appium test framework on the QDC runner.
The runner has ADB access to the allocated device.

Placeholders replaced at artifact creation time by run_qdc_jobs.py:
  <<MODEL_URL>>  Direct URL to the GGUF model file (downloaded on-device via curl)
"""

import os
import subprocess
import sys

import pytest

from utils import BIN_PATH, CMD_PREFIX, push_bundle_if_needed, run_adb_command, write_qdc_log

MODEL_PATH = "/data/local/tmp/model.gguf"
PROMPT     = "What is the capital of France?"
CLI_OPTS   = "--batch-size 128 -n 128 -no-cnv --seed 42"


@pytest.fixture(scope="session", autouse=True)
def install(driver):
    push_bundle_if_needed(f"{BIN_PATH}/llama-cli")

    # Skip model download if already present
    check = subprocess.run(
        ["adb", "shell", f"ls {MODEL_PATH}"],
        text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    if check.returncode != 0:
        run_adb_command(f'curl -L -J --output {MODEL_PATH} "<<MODEL_URL>>"')


@pytest.mark.parametrize("device,extra_flags", [
    pytest.param("none",      "-ctk q8_0 -ctv q8_0", id="cpu"),
    pytest.param("GPUOpenCL", "",                     id="gpu"),
    pytest.param("HTP0",      "-ctk q8_0 -ctv q8_0", id="npu"),
])
def test_llama_completion(device, extra_flags):
    result = run_adb_command(
        f'{CMD_PREFIX} {BIN_PATH}/llama-completion'
        f' -m {MODEL_PATH} --device {device} -ngl 99 -t 4 {CLI_OPTS} {extra_flags} -fa on'
        f' -p "{PROMPT}"',
        check=False,
    )
    write_qdc_log(f"llama_completion_{device}.log", result.stdout or "")
    assert result.returncode == 0, f"llama-completion {device} failed (exit {result.returncode})"


_DEVICE_LOG_NAME = {"none": "cpu", "GPUOpenCL": "gpu", "HTP0": "htp"}


@pytest.mark.parametrize("device", [
    pytest.param("none",      id="cpu"),
    pytest.param("GPUOpenCL", id="gpu"),
    pytest.param("HTP0",      id="npu"),
])
def test_llama_bench(device):
    result = run_adb_command(
        f"{CMD_PREFIX} {BIN_PATH}/llama-bench"
        f" -m {MODEL_PATH} --device {device} -ngl 99 --batch-size 128 -t 4 -p 128 -n 32",
        check=False,
    )
    write_qdc_log(f"llama_bench_{_DEVICE_LOG_NAME[device]}.log", result.stdout or "")
    assert result.returncode == 0, f"llama-bench {device} failed (exit {result.returncode})"


if __name__ == "__main__":
    ret = pytest.main(["-s", "--junitxml=results.xml", os.path.realpath(__file__)])
    if os.path.exists("results.xml"):
        with open("results.xml") as f:
            write_qdc_log("results.xml", f.read())
    sys.exit(ret)
