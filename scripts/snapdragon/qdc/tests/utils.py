"""Shared helpers for QDC on-device test runners."""

import logging
import os
import subprocess
import tempfile

from appium.options.common import AppiumOptions

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# On-device paths
# ---------------------------------------------------------------------------

BUNDLE_PATH  = "/data/local/tmp/llama_cpp_bundle"
QDC_LOGS_PATH = "/data/local/tmp/QDC_logs"
LIB_PATH    = f"{BUNDLE_PATH}/lib"
BIN_PATH    = f"{BUNDLE_PATH}/bin"
ENV_PREFIX  = (
    f"export LD_LIBRARY_PATH={LIB_PATH} && "
    f"export ADSP_LIBRARY_PATH={LIB_PATH} && "
    f"chmod +x {BIN_PATH}/* &&"
)
CMD_PREFIX  = f"cd {BUNDLE_PATH} && {ENV_PREFIX}"

# ---------------------------------------------------------------------------
# Appium session options
# ---------------------------------------------------------------------------

options = AppiumOptions()
options.set_capability("automationName", "UiAutomator2")
options.set_capability("platformName", "Android")
options.set_capability("deviceName", os.getenv("ANDROID_DEVICE_VERSION"))

# ---------------------------------------------------------------------------
# ADB helpers
# ---------------------------------------------------------------------------


def run_adb_command(cmd: str, *, check: bool = True) -> subprocess.CompletedProcess:
    # Append exit-code sentinel because `adb shell` doesn't reliably propagate
    # the on-device exit code (older ADB versions always return 0).
    raw = subprocess.run(
        ["adb", "shell", f"{cmd}; echo __RC__:$?"],
        text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
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
    log.info("%s", stdout)
    result = subprocess.CompletedProcess(raw.args, returncode, stdout=stdout)
    if check:
        assert returncode == 0, f"Command failed (exit {returncode})"
    return result


def write_qdc_log(filename: str, content: str) -> None:
    """Push content as a log file to QDC_LOGS_PATH on the device for QDC log collection."""
    subprocess.run(
        ["adb", "shell", f"mkdir -p {QDC_LOGS_PATH}"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    with tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=False) as f:
        f.write(content)
        tmp_path = f.name
    try:
        subprocess.run(
            ["adb", "push", tmp_path, f"{QDC_LOGS_PATH}/{filename}"],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        )
    finally:
        os.unlink(tmp_path)


def push_bundle_if_needed(check_binary: str) -> None:
    """Push llama_cpp_bundle to the device if check_binary is not already present."""
    result = subprocess.run(
        ["adb", "shell", f"ls {check_binary}"],
        text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    if result.returncode != 0:
        subprocess.run(
            ["adb", "push", "/qdc/appium/llama_cpp_bundle/", "/data/local/tmp"],
            text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        )
