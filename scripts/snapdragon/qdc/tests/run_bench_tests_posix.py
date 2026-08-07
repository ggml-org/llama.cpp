"""
Bench and completion test runner for llama.cpp (CPU, GPU, NPU backends).

On Android: calls upstream run-*.sh scripts from llama.cpp/scripts/snapdragon/adb/
on the QDC runner host (scripts wrap commands in ``adb shell`` internally).

On Windows: runs llama-completion.exe and llama-bench.exe directly via subprocess.

Placeholders replaced at artifact creation time by run_qdc_jobs.py:
  <<MODEL_URL>>  Direct URL to the GGUF model file (downloaded on-device)
"""

import os
import ssl
import subprocess
import sys
from pathlib import Path

import pytest

from utils import (
    IS_WINDOWS,
    BIN_PATH,
    MODEL_DEVICE_PATH,
    MODEL_NAME,
    PROMPT_DIR,
    ensure_bundle,
    write_qdc_log,
)

if IS_WINDOWS:
    from utils import run_exe
else:
    from utils import push_bundle_if_needed, run_adb_command, run_script

MODEL_URL = "<<MODEL_URL>>"


@pytest.fixture(scope="session", autouse=True)
def install(driver):
    if IS_WINDOWS:
        ensure_bundle()
        model_path = Path(MODEL_DEVICE_PATH)
        prompt_dir = Path(PROMPT_DIR)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        prompt_dir.mkdir(parents=True, exist_ok=True)
        (prompt_dir / "bench_prompt.txt").write_text(
            "What is the capital of France?\n", encoding="utf-8"
        )
        if not model_path.is_file() or model_path.stat().st_size < 1024 * 1024:
            print(f"Downloading model from {MODEL_URL} ...")
            import urllib.request
            ctx = ssl.create_default_context()
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
            with urllib.request.urlopen(MODEL_URL, context=ctx) as resp:
                model_path.write_bytes(resp.read())
    else:
        push_bundle_if_needed(f"{BIN_PATH}/llama-cli")
        run_adb_command(f"mkdir -p /data/local/tmp/gguf {PROMPT_DIR}")
        run_adb_command(f"echo 'What is the capital of France?' > {PROMPT_DIR}/bench_prompt.txt")
        check = subprocess.run(
            ["adb", "shell", f"ls {MODEL_DEVICE_PATH}"],
            text=True, errors="replace", stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
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
    if IS_WINDOWS:
        args = [
            "--model", MODEL_DEVICE_PATH,
            "-n", "128", "--seed", "42",
            "-f", str(Path(PROMPT_DIR) / "bench_prompt.txt"),
            "--no-display-prompt", "-no-cnv",
            "--batch-size", "128",
        ]
        if device != "none":
            args += ["--device", device]
        result = run_exe("llama-completion.exe", args, device)
    else:
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
    if IS_WINDOWS:
        args = [
            "--model", MODEL_DEVICE_PATH,
            "-p", "128", "-n", "32",
            "--batch-size", "128",
        ]
        if device != "none":
            args += ["--device", device]
        result = run_exe("llama-bench.exe", args, device)
    else:
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
