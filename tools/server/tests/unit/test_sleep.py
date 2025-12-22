import pytest
import time
from utils import *

server = ServerPreset.tinyllama2()


@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.tinyllama2()


def test_server_sleep():
    global server
    server.sleep_idle_seconds = 1
    server.start()

    # wait a bit so that server can go to sleep
    time.sleep(2)

    # make sure these endpoints are still responsive after sleep
    res = server.make_request("GET", "/health")
    assert res.status_code == 200
    res = server.make_request("GET", "/props")
    assert res.status_code == 200
    assert res.body["is_sleeping"] == True

    # make a generation request to wake up the server
    res = server.make_request("POST", "/completion", data={
        "n_predict": 1,
        "prompt": "Hello",
    })
    assert res.status_code == 200

    # it should no longer be sleeping
    res = server.make_request("GET", "/props")
    assert res.status_code == 200
    assert res.body["is_sleeping"] == False
