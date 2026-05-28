# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""Shared helpers for QDC on-device test runners (Android and Windows)."""

from __future__ import annotations

import logging
import os
import subprocess
import tempfile
from pathlib import Path

PLATFORM = "<<PLATFORM>>"
IS_WINDOWS = PLATFORM == "windows"

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# On-device paths
# ---------------------------------------------------------------------------

if IS_WINDOWS:
    TESTS_DIR = Path(__file__).resolve().parent
    WORKSPACE_PATH = TESTS_DIR.parent
    BUNDLE_PATH = str(WORKSPACE_PATH / "llama_cpp_bundle")
    BIN_PATH = str(WORKSPACE_PATH / "llama_cpp_bundle" / "bin")
    LIB_PATH = str(WORKSPACE_PATH / "llama_cpp_bundle" / "lib")
    QDC_LOGS_PATH = "C:\\Temp\\QDC_Logs"
    MODEL_NAME = "model.gguf"
    MODEL_DEVICE_PATH = "C:\\Temp\\gguf\\model.gguf"
    PROMPT_DIR = "C:\\Temp\\prompts"
else:
    BUNDLE_PATH = "/data/local/tmp/llama.cpp"
    BIN_PATH = f"{BUNDLE_PATH}/bin"
    LIB_PATH = f"{BUNDLE_PATH}/lib"
    QDC_LOGS_PATH = "/data/local/tmp/QDC_logs"
    SCRIPTS_DIR = "/qdc/appium"
    MODEL_NAME = "model.gguf"
    MODEL_DEVICE_PATH = "/data/local/tmp/gguf/model.gguf"
    PROMPT_DIR = "/data/local/tmp/scorecard_prompts"

# ---------------------------------------------------------------------------
# Appium session options (Android only)
# ---------------------------------------------------------------------------

if not IS_WINDOWS:
    from appium.options.common import AppiumOptions

    options = AppiumOptions()
    options.set_capability("automationName", "UiAutomator2")
    options.set_capability("platformName", "Android")
    options.set_capability("deviceName", os.getenv("ANDROID_DEVICE_VERSION"))

# ---------------------------------------------------------------------------
# Shell / process helpers
# ---------------------------------------------------------------------------


def write_qdc_log(filename: str, content: str) -> None:
    """Write content as a log file for QDC log collection."""
    if IS_WINDOWS:
        logs = Path(QDC_LOGS_PATH)
        logs.mkdir(parents=True, exist_ok=True)
        (logs / filename).write_text(content, encoding="utf-8")
    else:
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
    if IS_WINDOWS:
        bin_dir = Path(BIN_PATH)
        if not bin_dir.is_dir():
            raise FileNotFoundError(
                f"llama_cpp_bundle/bin not found at {bin_dir}; "
                f"workspace contents: "
                f"{os.listdir(Path(BUNDLE_PATH).parent) if Path(BUNDLE_PATH).parent.is_dir() else 'missing'}"
            )
    else:
        push_bundle_if_needed(check_binary or f"{BIN_PATH}/llama-cli")


# ---------------------------------------------------------------------------
# Windows helpers
# ---------------------------------------------------------------------------


def run_exe(
    exe_name: str, args: list[str], device: str
) -> subprocess.CompletedProcess:
    """Run a Windows executable from the bundle with proper env setup."""
    exe = Path(BIN_PATH) / exe_name
    cmd = [str(exe)] + args
    proc = subprocess.Popen(
        cmd,
        text=True,
        bufsize=1,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        errors="replace",
        env=_build_win_env(device),
        cwd=BUNDLE_PATH,
    )
    assert proc.stdout is not None
    chunks: list[str] = []
    for line in proc.stdout:
        print(line, end="", flush=True)
        chunks.append(line)
    proc.wait()
    return subprocess.CompletedProcess(
        args=cmd, returncode=proc.returncode, stdout="".join(chunks)
    )


def _build_win_env(device: str) -> dict[str, str]:
    """Build environment dict for running llama.cpp on Windows."""
    env = os.environ.copy()
    env["PATH"] = f"{BIN_PATH};{LIB_PATH};{env.get('PATH', '')}"
    env["ADSP_LIBRARY_PATH"] = LIB_PATH
    env["NO_COLOR"] = "1"
    if device == "HTP0":
        env["GGML_HEXAGON_NDEV"] = "1"
    else:
        env["GGML_HEXAGON_NDEV"] = "0"
    return env


def log_environment() -> None:
    """Log environment info for debugging (Windows)."""
    bp = Path(BIN_PATH)
    lp = Path(LIB_PATH)
    lines = [
        f"__file__: {Path(__file__).resolve()}",
        f"WORKSPACE_PATH: {WORKSPACE_PATH}",
        f"BUNDLE_PATH exists: {Path(BUNDLE_PATH).is_dir()}",
        f"BIN_PATH contents: "
        f"{os.listdir(bp) if bp.is_dir() else 'missing'}",
        f"LIB_PATH contents: "
        f"{os.listdir(lp) if lp.is_dir() else 'missing'}",
        f"PATH: {os.environ.get('PATH', '')}",
    ]
    for line in lines:
        log.info("[env] %s", line)
    write_qdc_log("environment.log", "\n".join(lines) + "\n")


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
        errors="replace",
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
    print(stdout)
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
    if extra_env:
        env.update(extra_env)
    cmd = [f"{SCRIPTS_DIR}/{script}"] + (extra_args or [])
    result = subprocess.run(
        cmd, env=env,
        text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        errors="replace",
    )
    print(result.stdout)
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
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if result.returncode != 0:
        subprocess.run(
            ["adb", "push", "/qdc/appium/llama_cpp_bundle/", BUNDLE_PATH],
            text=True,
            errors="replace",
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        subprocess.run(
            ["adb", "shell", f"find {BUNDLE_PATH}/bin -type f -exec chmod 755 {{}} +"],
            text=True,
            errors="replace",
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
