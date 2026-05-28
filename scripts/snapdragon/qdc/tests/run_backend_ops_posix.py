# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""
test-backend-ops runner for llama.cpp (HTP0 backend).

On Android: calls upstream run-tool.sh from llama.cpp/scripts/snapdragon/adb/
on the QDC runner host (script wraps commands in ``adb shell`` internally).

On Windows: runs test-backend-ops.exe directly via subprocess.
"""

import os
import sys

import pytest

from utils import (
    IS_WINDOWS,
    BIN_PATH,
    ensure_bundle,
    write_qdc_log,
)

if IS_WINDOWS:
    from utils import run_exe
else:
    from utils import push_bundle_if_needed, run_script


@pytest.fixture(scope="session", autouse=True)
def install(driver):
    if IS_WINDOWS:
        ensure_bundle()
    else:
        push_bundle_if_needed(f"{BIN_PATH}/test-backend-ops")


@pytest.mark.parametrize("type_a", ["mxfp4", "fp16", "q4_0"])
def test_backend_ops_htp0(type_a):
    if type_a == "q4_0":
        pattern = r'^(?=.*type_a=q4_0)(?!.*type_b=f32,m=576,n=512,k=576).*$'
    else:
        pattern = f"type_a={type_a}"

    if IS_WINDOWS:
        result = run_exe(
            "test-backend-ops.exe",
            ["-b", "HTP0", "-o", "MUL_MAT", "-p", pattern],
            "HTP0",
        )
    else:
        quoted_pattern = f'"{pattern}"' if type_a == "q4_0" else pattern
        result = run_script(
            "run-tool.sh",
            extra_env={"HB": "0"},
            extra_args=["test-backend-ops", "-b", "HTP0", "-o", "MUL_MAT", "-p", quoted_pattern],
        )
    write_qdc_log(f"backend_ops_{type_a}.log", result.stdout or "")
    assert result.returncode == 0, (
        f"test-backend-ops type_a={type_a} failed (exit {result.returncode})"
    )


if __name__ == "__main__":
    ret = pytest.main(["-s", "--junitxml=results.xml", os.path.realpath(__file__)])
    if os.path.exists("results.xml"):
        with open("results.xml") as f:
            write_qdc_log("results.xml", f.read())
    sys.exit(ret)
