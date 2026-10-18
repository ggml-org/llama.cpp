import os
import time

import pytest
import requests
from utils import *

server = ServerPreset.tinyllama2()


@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.tinyllama2()


def get_perf(server: ServerProcess, headers: dict | None = None) -> dict:
    res = server.make_request("GET", "/perf", headers=headers)
    assert res.status_code == 200, res.body
    return res.body


def sampler_switches(pid: int) -> int:
    """voluntary context switches of the sampler thread: one or more per sample, none while it sleeps"""
    for tid in os.listdir(f"/proc/{pid}/task"):
        with open(f"/proc/{pid}/task/{tid}/comm") as f:
            if f.read().strip() != "srv-perf":
                continue
        with open(f"/proc/{pid}/task/{tid}/status") as f:
            for line in f:
                if line.startswith("voluntary_ctxt_switches:"):
                    return int(line.split()[1])
    raise AssertionError("sampler thread srv-perf not found")


def test_perf_schema():
    global server
    server.start()
    body = get_perf(server)
    for key in ["ts_ms", "window_s", "cpu", "gpu", "mem", "gen", "prompt", "nvme", "predict", "preload"]:
        assert key in body, key
    assert body["window_s"] == 60
    assert abs(time.time()*1000 - body["ts_ms"]) < 2500

    if body["cpu"] is not None:
        assert 0 <= body["cpu"]["busy_pct"] <= 100
        assert 0 <= body["cpu"]["proc_pct"] <= 100
        assert body["cpu"]["n_cpu"] >= 1
    if body["mem"] is not None:
        mem = body["mem"]
        assert 0 < mem["occupied_bytes"] <= mem["total_bytes"]
        assert mem["used_bytes"] <= mem["occupied_bytes"]
        assert mem["free_bytes"] == mem["total_bytes"] - mem["occupied_bytes"]
    if body["gpu"] is not None:
        assert 0 <= body["gpu"]["busy_pct"] <= 100
        assert 0 < body["gpu"]["mem_used_bytes"] <= body["gpu"]["mem_total_bytes"]

    # no request yet: no speeds, never 0
    assert body["gen"]["tps_headline"] is None and body["gen"]["generating"] is False
    assert body["prompt"]["tps_headline"] is None and body["prompt"]["processing"] is False

    # a dense model has no predictor
    assert body["predict"] is None
    assert body["preload"] is None


def test_perf_api_key():
    global server
    server.api_key = "perf-key"
    server.start()
    res = requests.get(server.make_url("/perf"), timeout=10)
    assert res.status_code == 401
    res = requests.get(server.make_url("/perf"), headers={"Authorization": "Bearer wrong"}, timeout=10)
    assert res.status_code == 401
    body = get_perf(server, headers={"Authorization": "Bearer perf-key"})
    assert "gen" in body


@pytest.mark.parametrize("n_predict", [8, 32])
def test_perf_after_completion(n_predict: int):
    global server
    server.start()
    get_perf(server)
    res = server.make_request("POST", "/completion", data={
        "prompt": "Once upon a time there was a little girl",
        "n_predict": n_predict,
        "ignore_eos": True,
    })
    assert res.status_code == 200
    timings = res.body["timings"]
    assert timings["predicted_n"] == n_predict

    # the slot is released right after the response is sent
    deadline = time.time() + 3
    while True:
        body = get_perf(server)
        if body["gen"]["generating"] is False and body["gen"]["n_tokens"] == n_predict:
            break
        assert time.time() < deadline, body["gen"]
        time.sleep(0.1)

    gen = body["gen"]
    assert gen["n_tokens"] == n_predict
    assert gen["tps_last"] == pytest.approx(timings["predicted_per_second"], rel=0.05)
    assert gen["tps_headline"] == pytest.approx(timings["predicted_per_second"], rel=0.05)
    assert gen["tps_total"] == pytest.approx(timings["predicted_per_second"], rel=0.05)

    prompt = body["prompt"]
    assert prompt["n_tokens"] == timings["prompt_n"]
    assert prompt["tps_last"] == pytest.approx(timings["prompt_per_second"], rel=0.05)
    assert prompt["tps_headline"] == prompt["tps_last"]


def test_perf_first_poll_after_idle():
    global server
    server.start()

    def first_poll():
        t0 = time.time()
        body = get_perf(server)
        dt = time.time() - t0
        assert dt <= 1.5, f"first /perf took {dt:.2f} s"
        assert abs(time.time()*1000 - body["ts_ms"]) < 2500
        assert body["cpu"] is not None and body["cpu"]["busy_pct"] is not None
        if body["nvme"] is not None:
            assert body["nvme"]["read_bps"] is not None
        return body

    # the sampler has never run
    first_poll()
    # it stops 10 s after the last poll; the next poll wakes it again
    time.sleep(11)
    first_poll()


def test_perf_sampler_idle_without_polling():
    global server
    server.start()
    pid = server.process.pid

    # never polled: the sampler sleeps
    n0 = sampler_switches(pid)
    time.sleep(5)
    assert sampler_switches(pid) == n0, "the sampler ran without polling"

    # polled: about one sample per second
    for _ in range(4):
        get_perf(server)
        time.sleep(1)
    assert sampler_switches(pid) - n0 >= 3

    # 10 s after the last poll it sleeps again
    time.sleep(11)
    n1 = sampler_switches(pid)
    time.sleep(3)
    assert sampler_switches(pid) == n1, "the sampler kept running after polling stopped"
