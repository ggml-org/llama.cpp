import pytest
from utils import *

server = ServerPreset.tinyllama2()

LONG_PROMPT = "I believe the meaning of life is " * 6  # ~108 tokens, fits a 256-token slot


@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.tinyllama2()
    server.server_metrics = True


def budget_steps() -> float:
    res = server.make_request("GET", "/metrics")
    assert res.status_code == 200
    for line in res.body.splitlines():
        if line.startswith("llamacpp:prefill_budget_steps_total "):
            return float(line.split(" ", 1)[1])
    raise AssertionError("llamacpp:prefill_budget_steps_total not exported")


@pytest.mark.parametrize("budget", [0, 8])
def test_prefill_budget_while_generating(budget: int):
    global server
    server.prefill_budget_busy = budget
    server.start()
    tasks = [
        (server.make_request, ("POST", "/completion", {"prompt": "I believe", "n_predict": 64, "temperature": 0.0})),
        (server.make_request, ("POST", "/completion", {"prompt": LONG_PROMPT, "n_predict": 8, "temperature": 0.0})),
    ]
    for res in parallel_function_calls(tasks):
        assert res.status_code == 200
        assert len(res.body["content"]) > 0
    if budget == 0:
        assert budget_steps() == 0
    else:
        assert budget_steps() > 0
