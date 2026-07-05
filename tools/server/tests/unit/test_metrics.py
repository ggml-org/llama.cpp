import pytest
from utils import *

server = ServerPreset.tinyllama2()


@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.tinyllama2()


def test_model_resource_metrics():
    global server
    server.server_metrics = True
    server.start()

    res = server.make_request("GET", "/metrics")
    assert res.status_code == 200
    assert match_regex(r"llamacpp:model_size_bytes\s+[1-9]", res.body)
    assert match_regex(r"llamacpp:model_n_params\s+[1-9]", res.body)
    assert match_regex(r"llamacpp:n_ctx\s+[1-9]", res.body)
