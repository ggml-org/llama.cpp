"""test-backend-ops runner for llama.cpp (HTP0 backend) on Windows.

Executed by pytest inside the QDC POWERSHELL job (see run_tests.ps1) and
dispatched through llama.cpp\\run.py.
"""

import subprocess
import sys
from pathlib import Path

import pytest

WORKSPACE = Path(__file__).resolve().parent.parent
RUN_PY = WORKSPACE / "llama.cpp" / "run.py"
QDC_LOGS = Path("C:\\Temp\\QDC_Logs")


@pytest.mark.parametrize("type_a", ["mxfp4", "fp16", "q4_0"])
def test_backend_ops_htp0(type_a):
    if type_a == "q4_0":
        pattern = r'^(?=.*type_a=q4_0)(?!.*type_b=f32,m=576,n=512,k=576).*$'
    else:
        pattern = f"type_a={type_a}"

    result = subprocess.run(
        [sys.executable, str(RUN_PY), "--install-dir", ".", "-d", "HTP0", "--hex-hostbuf", "0",
         "--", "test-backend-ops", "-o", "MUL_MAT", "-p", pattern],
        cwd=WORKSPACE, text=True, errors="replace",
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    sys.stdout.buffer.write((result.stdout or "").encode("utf-8", "replace"))
    QDC_LOGS.mkdir(parents=True, exist_ok=True)
    (QDC_LOGS / f"backend_ops_{type_a}.log").write_text(result.stdout or "", encoding="utf-8")
    assert result.returncode == 0, (
        f"test-backend-ops type_a={type_a} failed (exit {result.returncode})"
    )
