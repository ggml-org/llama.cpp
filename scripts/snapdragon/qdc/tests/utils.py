"""Shared helpers for QDC on-device test runners."""

from __future__ import annotations

import logging
import os
import subprocess
import tempfile

from appium.options.common import AppiumOptions

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# On-device paths
# ---------------------------------------------------------------------------

BUNDLE_PATH = "/data/local/tmp/llama.cpp"
BIN_PATH = f"{BUNDLE_PATH}/bin"
LIB_PATH = f"{BUNDLE_PATH}/lib"
QDC_LOGS_PATH = "/data/local/tmp/QDC_logs"
SCRIPTS_DIR = "/qdc/appium"
MODEL_NAME = "model.gguf"
MODEL_DEVICE_PATH = "/data/local/tmp/gguf/model.gguf"
PROMPT_DIR = "/data/local/tmp/scorecard_prompts"

# ---------------------------------------------------------------------------
# Appium session options
# ---------------------------------------------------------------------------

options = AppiumOptions()
options.set_capability("automationName", "UiAutomator2")
options.set_capability("platformName", "Android")
options.set_capability("deviceName", os.getenv("ANDROID_DEVICE_VERSION"))

# ---------------------------------------------------------------------------
# Shell / process helpers
# ---------------------------------------------------------------------------


def write_qdc_log(filename: str, content: str) -> None:
    """Write content as a log file for QDC log collection."""
    subprocess.run(
        ["adb", "shell", f"mkdir -p {QDC_LOGS_PATH}"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    with tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=False) as f:
        f.write(content)
        tmp_path = f.name
    try:
        subprocess.run(
            ["adb", "push", tmp_path, f"{QDC_LOGS_PATH}/{filename}"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
    finally:
        os.unlink(tmp_path)


def ensure_bundle(check_binary: str | None = None) -> None:
    """Ensure the llama_cpp_bundle is available on the target device."""
    push_bundle_if_needed(check_binary or f"{BIN_PATH}/llama-cli")


# ---------------------------------------------------------------------------
# Android / Linux host helpers
# ---------------------------------------------------------------------------


def run_adb_command(cmd: str, *, check: bool = True) -> subprocess.CompletedProcess:
    """Run a command on-device via ``adb shell`` with exit-code sentinel."""
    raw = subprocess.run(
        ["adb", "shell", f"{cmd}; echo __RC__:$?"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    stdout = raw.stdout
    returncode = raw.returncode
    if stdout:
        lines = stdout.rstrip("\n").split("\n")
        if lines and lines[-1].startswith("__RC__:"):
            try:
                returncode = int(lines[-1][7:])
                stdout = "\n".join(lines[:-1]) + "\n"
            except ValueError:
                pass
    log.info(stdout)
    result = subprocess.CompletedProcess(raw.args, returncode, stdout=stdout)
    if check:
        assert returncode == 0, f"Command failed (exit {returncode})"
    return result


def run_script(
    script: str,
    extra_env: dict[str, str] | None = None,
    extra_args: list[str] | None = None,
) -> subprocess.CompletedProcess:
    """Run an upstream shell script from /qdc/appium/ on the QDC runner host."""
    env = os.environ.copy()
    env["GGML_HEXAGON_EXPERIMENTAL"] = "1"
    if extra_env:
        env.update(extra_env)
    cmd = [f"{SCRIPTS_DIR}/{script}"] + (extra_args or [])
    result = subprocess.run(
        cmd, env=env,
        text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    log.info(result.stdout)
    return result


def adb_shell(cmd: str) -> None:
    """Run a command via adb shell (fire-and-forget, no error check)."""
    subprocess.run(
        ["adb", "shell", "sh", "-c", cmd],
        capture_output=True, encoding="utf-8", errors="replace", check=False,
    )


def push_bundle_if_needed(check_binary: str) -> None:
    """Push llama_cpp_bundle to the device if check_binary is not already present."""
    result = subprocess.run(
        ["adb", "shell", f"ls {check_binary}"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if result.returncode != 0:
        subprocess.run(
            ["adb", "push", "/qdc/appium/llama_cpp_bundle/", BUNDLE_PATH],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        subprocess.run(
            ["adb", "shell", f"find {BUNDLE_PATH}/bin -type f -exec chmod 755 {{}} +"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
