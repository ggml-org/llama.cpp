"""
On-device bench and completion test runner for llama.cpp (CPU, GPU, NPU backends).

On Android: calls upstream run-*.sh scripts from llama.cpp/scripts/snapdragon/adb/
on the QDC runner host (scripts wrap commands in ``adb shell`` internally).

On Linux: runs llama-bench directly via run_linux.sh (BASH framework).

Placeholders replaced at artifact creation time by run_qdc_jobs.py:
  <<MODEL_URL>>  Direct URL to the GGUF model file (downloaded on-device)
"""

import os
import subprocess
import sys

import pytest

from utils import (
    BIN_PATH,
    MODEL_DEVICE_PATH,
    MODEL_NAME,
    PROMPT_DIR,
    push_bundle_if_needed,
    run_adb_command,
    run_script,
    write_qdc_log,
)

MODEL_URL = "<<MODEL_URL>>"


@pytest.fixture(scope="session", autouse=True)
def install(driver):
    push_bundle_if_needed(f"{BIN_PATH}/llama-cli")
    run_adb_command(f"mkdir -p /data/local/tmp/gguf {PROMPT_DIR}")
    run_adb_command(f"echo 'What is the capital of France?' > {PROMPT_DIR}/bench_prompt.txt")
    check = subprocess.run(
        ["adb", "shell", f"ls {MODEL_DEVICE_PATH}"],
        text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    if check.returncode != 0:
        run_adb_command(f'curl -L -J --output {MODEL_DEVICE_PATH} "{MODEL_URL}"')


@pytest.mark.parametrize(
    "device",
    [
        pytest.param("none", id="cpu"),
        pytest.param("GPUOpenCL", id="gpu"),
        pytest.param("HTP0", id="npu"),
    ],
)
def test_llama_completion(device):
    result = run_script(
        "run-completion.sh",
        extra_env={"D": device, "M": MODEL_NAME},
        extra_args=["--batch-size", "128", "-n", "128", "--seed", "42",
                    "-f", f"{PROMPT_DIR}/bench_prompt.txt"],
    )
    write_qdc_log(f"llama_completion_{device}.log", result.stdout or "")
    assert result.returncode == 0, (
        f"llama-completion {device} failed (exit {result.returncode})"
    )


_DEVICE_LOG_NAME = {"none": "cpu", "GPUOpenCL": "gpu", "HTP0": "htp"}


@pytest.mark.parametrize(
    "device",
    [
        pytest.param("none", id="cpu"),
        pytest.param("GPUOpenCL", id="gpu"),
        pytest.param("HTP0", id="npu"),
    ],
)
def test_llama_bench(device):
    result = run_script(
        "run-bench.sh",
        extra_env={"D": device, "M": MODEL_NAME},
        extra_args=["--batch-size", "128", "-p", "128", "-n", "32"],
    )
    write_qdc_log(f"llama_bench_{_DEVICE_LOG_NAME[device]}.log", result.stdout or "")
    assert result.returncode == 0, (
        f"llama-bench {device} failed (exit {result.returncode})"
    )


if __name__ == "__main__":
    ret = pytest.main(["-s", "--junitxml=results.xml", os.path.realpath(__file__)])
    if os.path.exists("results.xml"):
        with open("results.xml") as f:
            write_qdc_log("results.xml", f.read())
    sys.exit(ret)
