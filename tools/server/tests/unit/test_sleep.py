import pytest
import socket
import tempfile
import time
from utils import *

server = ServerPreset.tinyllama2()


@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.tinyllama2()


def is_sleeping(server: ServerProcess) -> bool:
    res = server.make_request("GET", "/props")
    assert res.status_code == 200
    return res.body["is_sleeping"]


def wait_for_sleep(server: ServerProcess, timeout: float = 10.0):
    start = time.time()
    while time.time() - start < timeout:
        if is_sleeping(server):
            return
        time.sleep(0.1)
    raise TimeoutError("server did not go to sleep")


def fetch_metrics(server: ServerProcess) -> str:
    res = server.make_request("GET", "/metrics")
    assert res.status_code == 200
    assert isinstance(res.body, str)
    return res.body


def get_metric(text: str, name: str) -> float:
    prefix = f"llamacpp:{name} "
    values = [ln for ln in text.splitlines() if ln.startswith(prefix)]
    assert len(values) == 1, f"{name} not found in metrics"
    return float(values[0][len(prefix):])


def test_server_sleep():
    global server
    server.sleep_idle_seconds = 1
    server.start()

    wait_for_sleep(server)

    # make sure these endpoints are still responsive after sleep
    res = server.make_request("GET", "/health")
    assert res.status_code == 200
    res = server.make_request("GET", "/props")
    assert res.status_code == 200
    assert res.body["is_sleeping"] == True
    res = server.make_request("GET", "/models")
    assert res.status_code == 200
    assert len(res.body["data"]) == 1
    assert res.body["data"][0]["id"] == server.model_alias

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


def test_server_sleep_read_only_endpoints():
    global server
    server.sleep_idle_seconds = 1
    server.server_metrics = True
    server.start()

    res = server.make_request("POST", "/completion", data={
        "n_predict": 4,
        "prompt": "Hello",
    })
    assert res.status_code == 200

    # the first scrape resets the throughput buckets, so that the second one reports
    # the same zero rates as the snapshot taken on entering sleep
    fetch_metrics(server)
    metrics_awake = fetch_metrics(server)
    assert get_metric(metrics_awake, "tokens_predicted_total") > 0

    wait_for_sleep(server)

    # during sleep, metrics are served from the snapshot taken right before sleeping
    assert fetch_metrics(server) == metrics_awake

    # scraping /metrics must not wake the server up
    assert is_sleeping(server)


def test_server_sleep_metrics_buckets():
    global server
    server.sleep_idle_seconds = 1
    server.server_metrics = True
    server.start()

    res = server.make_request("POST", "/completion", data={
        "n_predict": 8,
        "prompt": "Hello",
    })
    assert res.status_code == 200

    wait_for_sleep(server)

    # the first scrape reports the throughput of the last generation
    assert get_metric(fetch_metrics(server), "predicted_tokens_seconds") > 0

    # nothing runs while sleeping, so the next scrapes report an empty window
    assert get_metric(fetch_metrics(server), "predicted_tokens_seconds") == 0
    assert is_sleeping(server)

    # waking up must not report the buckets again
    res = server.make_request("POST", "/tokenize", data={"content": "Hello"})
    assert res.status_code == 200
    assert is_sleeping(server) == False
    assert get_metric(fetch_metrics(server), "predicted_tokens_seconds") == 0


def test_server_sleep_token_counting_wake():
    global server
    server.sleep_idle_seconds = 1
    server.start()

    wait_for_sleep(server)
    assert is_sleeping(server)

    res = server.make_request("POST", "/chat/completions/input_tokens", data={
        "messages": [
            {"role": "user", "content": "Hello world"}
        ]
    })
    assert res.status_code == 200
    assert res.body["input_tokens"] > 0
    assert is_sleeping(server) == False

    wait_for_sleep(server)
    assert is_sleeping(server)

    res = server.make_request("POST", "/v1/responses/input_tokens", data={
        "input": "Hello world"
    })
    assert res.status_code == 200
    assert res.body["input_tokens"] > 0
    assert is_sleeping(server) == False

    wait_for_sleep(server)
    assert is_sleeping(server)

    res = server.make_request("POST", "/v1/messages/count_tokens", data={
        "messages": [
            {"role": "user", "content": "Hello world"}
        ]
    })
    assert res.status_code == 200
    assert res.body["input_tokens"] > 0
    assert is_sleeping(server) == False


@pytest.fixture
def rpc_servers():
    server_bin = os.environ.get("LLAMA_SERVER_BIN_PATH", "../../../build/bin/llama-server")
    rpc_bin = os.path.join(os.path.dirname(server_bin), "ggml-rpc-server")
    if not os.path.exists(rpc_bin):
        pytest.skip("needs ggml-rpc-server, build with -DGGML_RPC=ON")
    socks = [socket.socket() for _ in range(2)]
    for sock in socks:
        sock.bind(("127.0.0.1", 0))
    ports = [sock.getsockname()[1] for sock in socks]
    for sock in socks:
        sock.close()
    procs = [subprocess.Popen([rpc_bin, "--port", str(port)]) for port in ports]
    for port in ports:
        start = time.time()
        while True:
            try:
                socket.create_connection(("127.0.0.1", port), timeout=1).close()
                break
            except OSError:
                if time.time() - start > 10:
                    raise
                time.sleep(0.1)
    yield ",".join(f"127.0.0.1:{port}" for port in ports)
    for proc in procs:
        proc.terminate()
        proc.wait()


def test_server_sleep_fit_again(rpc_servers):
    global server
    server.rpc_servers = rpc_servers
    server.fit_target = "100000000,0"
    server.sleep_idle_seconds = 1
    fd, server.log_path = tempfile.mkstemp(suffix=".log")
    os.close(fd)
    server.start()

    wait_for_sleep(server)
    res = server.make_request("POST", "/completion", data={
        "n_predict": 1,
        "prompt": "Hello",
    })
    assert res.status_code == 200

    with open(server.log_path) as f:
        assert "failed to fit params" not in f.read()
