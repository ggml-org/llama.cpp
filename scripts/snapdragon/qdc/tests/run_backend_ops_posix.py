"""
On-device test-backend-ops runner for llama.cpp (HTP0 backend).

On Android: executed by QDC's Appium test framework on the QDC runner.
The runner has ADB access to the allocated device.
On Linux: runs test-backend-ops directly via run_linux.sh (BASH framework).
"""

import os
import sys

import pytest

from utils import (
    BIN_PATH,
    push_bundle_if_needed,
    run_script,
    write_qdc_log,
)


@pytest.fixture(scope="session", autouse=True)
def install(driver):
    push_bundle_if_needed(f"{BIN_PATH}/test-backend-ops")


@pytest.mark.parametrize("type_a", ["mxfp4", "fp16", "q4_0"])
def test_backend_ops_htp0(type_a):
    if type_a == "q4_0":
        pattern = r'^(?=.*type_a=q4_0)(?!.*type_b=f32,m=576,n=512,k=576).*$'
    else:
        pattern = f"type_a={type_a}"

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
