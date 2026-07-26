import re

import pytest
import requests
from utils import *

server = ServerPreset.tinyllama2()


@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.tinyllama2()
    server.server_metrics = True
    server.server_memory = True


def fetch_metrics() -> str:
    global server
    server.start()
    res = requests.get(f"http://{server.server_host}:{server.server_port}/metrics")
    assert res.status_code == 200
    return res.text


def parse_series(text: str) -> dict[str, str]:
    # "llamacpp:name{label="value"} 123" -> {'name{label="value"}': "123"}
    series = {}
    for line in text.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        key, _, value = line.partition(" ")
        series[key.removeprefix("llamacpp:")] = value
    return series


@pytest.mark.parametrize("path", ["/metrics", "/memory"])
def test_endpoints_disabled_by_default(path: str):
    global server
    server.server_metrics = False
    server.server_memory = False
    server.start()
    res = server.make_request("GET", path)
    assert res.status_code == 501


@pytest.mark.parametrize("metrics_on,memory_on", [(True, False), (False, True)])
def test_gates_are_independent(metrics_on: bool, memory_on: bool):
    global server
    server.server_metrics = metrics_on
    server.server_memory = memory_on
    server.start()
    assert server.make_request("GET", "/metrics").status_code == (200 if metrics_on else 501)
    assert server.make_request("GET", "/memory").status_code == (200 if memory_on else 501)


def test_memory_endpoint_reports_devices_as_json():
    global server
    server.start()
    res = server.make_request("GET", "/memory")
    assert res.status_code == 200
    assert isinstance(res.body["n_layer"], int) and res.body["n_layer"] > 0
    rows = res.body["data"]
    assert isinstance(rows, list) and rows
    for row in rows:
        assert isinstance(row["name"], str), row
        for field in ["model", "context", "compute"]:
            assert isinstance(row[field], int), row
        for field in ["total", "free"]:
            if field in row:
                assert isinstance(row[field], int), row
        assert ("total" in row) == ("free" in row), row
    # the host row is always present and describes no whole device
    host = next(row for row in rows if row["name"] == "Host")
    assert "total" not in host and "free" not in host


def test_metrics_reports_memory_per_device():
    series = parse_series(fetch_metrics())

    devices = set()
    for key in series:
        match = re.fullmatch(r'memory_model_bytes\{device="([^"]+)"\}', key)
        if match:
            devices.add(match.group(1))

    # a CPU-only build still allocates the weights on the host
    assert devices
    assert re.fullmatch(r"\d+", series["model_n_layer"]) and int(series["model_n_layer"]) > 0
    for device in devices:
        for field in ["model", "context", "compute"]:
            assert f'memory_{field}_bytes{{device="{device}"}}' in series


def test_metrics_byte_values_are_exact_integers():
    # a byte count streamed as double is rounded to 6 significant digits
    series = parse_series(fetch_metrics())
    byte_series = {k: v for k, v in series.items() if k.startswith("memory_")}
    assert byte_series
    for key, value in byte_series.items():
        assert re.fullmatch(r"-?\d+", value), f"{key} is not an exact integer: {value}"


def test_metrics_describe_each_name_once():
    # the exposition format allows only one HELP/TYPE line per metric name; strict parsers reject duplicates
    text = fetch_metrics()
    help_names = re.findall(r"^# HELP (\S+)", text, re.MULTILINE)
    type_names = re.findall(r"^# TYPE (\S+)", text, re.MULTILINE)
    assert help_names
    assert len(help_names) == len(set(help_names))
    assert help_names == type_names


def test_memory_reports_mmproj():
    global server
    server = ServerPreset.tinygemma3()
    server.server_memory = True
    server.start(timeout_seconds=60)
    res = server.make_request("GET", "/memory")
    assert res.status_code == 200
    rows = res.body["data"]
    assert any(isinstance(row.get("mmproj"), int) and row["mmproj"] > 0 for row in rows), rows
