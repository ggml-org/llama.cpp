import os
import pytest
from utils import *

server = ServerPreset.tinyllama2()

# To run the cvector-file tests, set both env vars to matching paths:
#   CVECTOR_TEST_MODEL=/path/to/model.gguf
#   CVECTOR_TEST_FILE=/path/to/control_vector.gguf
CVECTOR_FILE  = os.environ.get("CVECTOR_TEST_FILE", "")
CVECTOR_MODEL = os.environ.get("CVECTOR_TEST_MODEL", "")
HAS_CVEC = bool(CVECTOR_FILE and CVECTOR_MODEL
                and os.path.exists(CVECTOR_FILE)
                and os.path.exists(CVECTOR_MODEL))


@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.tinyllama2()


def test_get_cvectors_empty():
    global server
    server.start()
    res = server.make_request("GET", "/cvectors")
    assert res.status_code == 200
    assert res.body == []


def test_post_cvectors_invalid_id():
    global server
    server.start()
    res = server.make_request("POST", "/cvectors", data=[{"id": 0, "scale": 1.0}])
    assert res.status_code == 400


def test_post_cvectors_non_array():
    global server
    server.start()
    res = server.make_request("POST", "/cvectors", data={"id": 0, "scale": 1.0})
    assert res.status_code == 400


@pytest.mark.skipif(not HAS_CVEC, reason="set CVECTOR_TEST_FILE and CVECTOR_TEST_MODEL to run")
def test_get_cvectors_lists_startup():
    global server
    server.model_hf_repo = None
    server.model_hf_file = None
    server.model_file = CVECTOR_MODEL
    server.control_vector_files = [CVECTOR_FILE]
    server.start()
    res = server.make_request("GET", "/cvectors")
    assert res.status_code == 200
    assert isinstance(res.body, list)
    assert len(res.body) == 1
    entry = res.body[0]
    assert entry["id"] == 0
    assert "path" in entry
    assert entry["scale"] == pytest.approx(1.0)


@pytest.mark.skipif(not HAS_CVEC, reason="set CVECTOR_TEST_FILE and CVECTOR_TEST_MODEL to run")
def test_post_cvectors_changes_scale():
    global server
    server.model_hf_repo = None
    server.model_hf_file = None
    server.model_file = CVECTOR_MODEL
    server.control_vector_files = [CVECTOR_FILE]
    server.start()
    res = server.make_request("POST", "/cvectors", data=[{"id": 0, "scale": 0.5}])
    assert res.status_code == 200
    assert res.body.get("success") is True
    res = server.make_request("GET", "/cvectors")
    assert res.status_code == 200
    assert res.body[0]["scale"] == pytest.approx(0.5)
