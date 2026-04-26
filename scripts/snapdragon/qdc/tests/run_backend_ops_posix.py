"""
On-device test-backend-ops runner for llama.cpp (HTP0 backend).

Executed by QDC's Appium test framework on the QDC runner.
The runner has ADB access to the allocated device.
"""

import os
import sys

import pytest

from utils import BIN_PATH, CMD_PREFIX, push_bundle_if_needed, run_adb_command, write_qdc_log


@pytest.fixture(scope="session", autouse=True)
def install(driver):
    push_bundle_if_needed(f"{BIN_PATH}/test-backend-ops")


@pytest.mark.parametrize("type_a", ["mxfp4", "fp16", "q4_0"])
def test_backend_ops_htp0(type_a):
    cmd = f"{CMD_PREFIX} GGML_HEXAGON_HOSTBUF=0 GGML_HEXAGON_EXPERIMENTAL=1 {BIN_PATH}/test-backend-ops -b HTP0 -o MUL_MAT"
    if type_a == "q4_0":
        cmd += r' -p "^(?=.*type_a=q4_0)(?!.*type_b=f32,m=576,n=512,k=576).*$"'
    else:
        cmd += f" -p type_a={type_a}"
    result = run_adb_command(
        cmd,
        check=False,
    )
    write_qdc_log(f"backend_ops_{type_a}.log", result.stdout or "")
    assert result.returncode == 0, f"test-backend-ops type_a={type_a} failed (exit {result.returncode})"


if __name__ == "__main__":
    ret = pytest.main(["-s", "--junitxml=results.xml", os.path.realpath(__file__)])
    if os.path.exists("results.xml"):
        with open("results.xml") as f:
            write_qdc_log("results.xml", f.read())
    sys.exit(ret)
