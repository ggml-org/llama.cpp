import pytest
from utils import *

server = ServerPreset.tinyllama2()


@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.tinyllama2()


def test_stream_conv_id_cannot_forge_log_lines(tmp_path):
    """conv_id from the query string is URL-decoded before logging; without
    escaping it can inject forged lines into the server log (CWE-117)."""
    global server
    log_path = str(tmp_path / "server.log")
    server.log_path = log_path
    server.start()

    forged = "FORGED-LOG-LINE-SHOULD-NOT-STAND-ALONE"
    res = server.make_request("DELETE", f"/v1/stream?conv_id=%0D%0A{forged}")
    assert res.status_code in (200, 204, 404)

    with open(log_path, encoding="utf-8", errors="replace") as f:
        lines = f.read().splitlines()

    # the injected marker must never start its own log line
    assert not any(line.startswith(forged) for line in lines)
    # the payload must appear escaped on the genuine log line instead
    assert any("\\x0d\\x0a" in line and forged in line for line in lines)
