import pytest
from utils import *

server = ServerPreset.tinygemma3()


@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.tinygemma3()


def gate_chat_payload(candidates: list, **gate_fields) -> dict:
    return {
        "messages": [{"role": "user", "content": "1, 2, 3,"}],
        "max_tokens": 8,
        "logit_gate": {"candidates": candidates, **gate_fields},
    }

def test_logit_gate_gate_only():
    global server
    server.start()
    res = server.make_request("POST", "/v1/chat/completions", data=gate_chat_payload(
        [{"label": "A", "text": "A"}, {"label": "B", "text": "B"}],
        mode="gate_only",
    ))
    assert res.status_code == 200
    gate = res.body["logit_gate"]
    assert gate["fired"] is True
    assert gate["best"] in ("A", "B")
    probs = {d["label"]: d["prob"] for d in gate["distribution"]}
    assert set(probs.keys()) == {"A", "B"}
    assert abs(sum(probs.values()) - 1.0) < 1e-3
    assert abs(gate["prob"] - max(probs.values())) < 1e-6
    assert gate["threshold"] == 0.5
    assert res.body["choices"][0]["message"]["content"] == ""
    assert res.body["choices"][0]["finish_reason"] == "stop"
    assert res.body["usage"]["completion_tokens"] == 0


def test_logit_gate_single_candidate():
    global server
    server.start()
    res = server.make_request("POST", "/v1/chat/completions", data=gate_chat_payload(
        [{"label": "A", "text": "A"}],
        mode="gate_only",
    ))
    assert res.status_code == 200
    gate = res.body["logit_gate"]
    assert gate["best"] == "A"
    assert gate["prob"] == 1.0
    assert gate["confidence"] == 1.0
    assert len(gate["distribution"]) == 1


def test_logit_gate_gate_then_chat_fires_on_low_threshold():
    global server
    server.start()
    res = server.make_request("POST", "/v1/chat/completions", data=gate_chat_payload(
        [{"label": "A", "text": "A"}, {"label": "B", "text": "B"}],
        mode="gate_then_chat", threshold=0.0,
    ))
    assert res.status_code == 200
    assert res.body["logit_gate"]["fired"] is True
    assert res.body["usage"]["completion_tokens"] == 0


def test_logit_gate_gate_then_chat_fall_through():
    # best/p stay double through the threshold comparison, so the best
    # probability is strictly below 1.0 for any finite logits and threshold
    # 1.0 never fires
    global server
    server.start()
    res = server.make_request("POST", "/v1/chat/completions", data=gate_chat_payload(
        [{"label": "A", "text": "A"}, {"label": "B", "text": "B"}],
        mode="gate_then_chat", threshold=1.0,
    ))
    assert res.status_code == 200, res.body
    assert "logit_gate" not in res.body, res.body.get("logit_gate")
    assert res.body["usage"]["completion_tokens"] > 0, res.body["usage"]
    assert res.body["choices"][0]["finish_reason"] in ("stop", "length"), res.body["choices"]


def test_logit_gate_gate_then_chat_sharpened_temperature():
    # sharpening (low temperature) widens the logit gap; threshold 1.0 must
    # still never fire even when the winning probability rounds to 1.0
    global server
    server.start()
    res = server.make_request("POST", "/v1/chat/completions", data=gate_chat_payload(
        [{"label": "A", "text": "A"}, {"label": "B", "text": "B"}],
        mode="gate_then_chat", threshold=1.0,
        options={"temperature": 0.05},
    ))
    assert res.status_code == 200, res.body
    assert "logit_gate" not in res.body, res.body.get("logit_gate")
    assert res.body["usage"]["completion_tokens"] > 0, res.body["usage"]


def test_logit_gate_rejected_on_anthropic_endpoint():
    # the anthropic converter drops unknown fields, so logit_gate would be
    # silently ignored — it must be rejected explicitly like on /v1/responses
    global server
    server.start()
    res = server.make_request("POST", "/v1/messages", data={
        "model": "test",
        "max_tokens": 8,
        "messages": [{"role": "user", "content": "1, 2, 3,"}],
        "logit_gate": {
            "candidates": [{"label": "A", "text": "A"}, {"label": "B", "text": "B"}],
            "mode": "gate_only",
        },
    })
    assert res.status_code == 400, res.body
    assert "only supported" in str(res.body)


def test_logit_gate_single_candidate_thresholds():
    # a single candidate's 1.0 is vacuous certainty: threshold 1.0 never fires,
    # any threshold below 1 fires as usual
    global server
    server.start()
    res = server.make_request("POST", "/v1/chat/completions", data=gate_chat_payload(
        [{"label": "A", "text": "A"}],
        mode="gate_then_chat", threshold=1.0,
    ))
    assert res.status_code == 200, res.body
    assert "logit_gate" not in res.body, res.body.get("logit_gate")
    assert res.body["usage"]["completion_tokens"] > 0, res.body["usage"]
    res = server.make_request("POST", "/v1/chat/completions", data=gate_chat_payload(
        [{"label": "A", "text": "A"}],
        mode="gate_then_chat", threshold=0.5,
    ))
    assert res.status_code == 200, res.body
    assert res.body["logit_gate"]["fired"] is True
    assert res.body["usage"]["completion_tokens"] == 0


def test_logit_gate_options_temperature_and_return_logits():
    global server
    server.start()
    res = server.make_request("POST", "/v1/chat/completions", data=gate_chat_payload(
        [{"label": "A", "text": "A"}, {"label": "B", "text": "B"}],
        mode="gate_only",
        options={"temperature": 0.5, "return_logits": True},
    ))
    assert res.status_code == 200
    gate = res.body["logit_gate"]
    assert gate["temperature"] == 0.5
    assert set(gate["label_logits"].keys()) == {"A", "B"}
    probs = [d["prob"] for d in gate["distribution"]]
    assert abs(sum(probs) - 1.0) < 1e-3
    assert 0.0 <= gate["confidence"] <= 1.0


def test_logit_gate_completions_endpoint():
    global server
    server.start()
    res = server.make_request("POST", "/v1/completions", data={
        "prompt": "1, 2, 3,",
        "max_tokens": 8,
        "logit_gate": {
            "candidates": [{"label": "A", "text": "A"}, {"label": "B", "text": "B"}],
            "mode": "gate_only",
        },
    })
    assert res.status_code == 200
    assert res.body["logit_gate"]["fired"] is True
    assert res.body["usage"]["completion_tokens"] == 0


def test_logit_gate_rejected_on_responses_endpoint():
    # /v1/responses shares the completion handler but renders no gate result;
    # a fired gate there would silently vanish, so it is rejected instead
    global server
    server.start()
    res = server.make_request("POST", "/v1/responses", data={
        "input": "1, 2, 3,",
        "max_tokens": 8,
        "logit_gate": {
            "candidates": [{"label": "A", "text": "A"}, {"label": "B", "text": "B"}],
            "mode": "gate_only",
        },
    })
    assert res.status_code == 400, res.body
    assert "only supported" in res.body["error"]["message"]


def test_logit_gate_rejects_n_gt_1():
    # tinygemma3 runs with n_slots = 2, so n = 2 passes the field's own limits
    # and reaches the gate validation, which must reject it
    global server
    server.start()
    gate = {
        "candidates": [{"label": "A", "text": "A"}, {"label": "B", "text": "B"}],
        "mode": "gate_only",
    }
    res = server.make_request("POST", "/v1/chat/completions", data={
        "messages": [{"role": "user", "content": "1, 2, 3,"}],
        "max_tokens": 8,
        "n": 2,
        "logit_gate": gate,
    })
    assert res.status_code == 400, res.body
    assert "requires n = 1" in res.body["error"]["message"], res.body
    # "n_cmpl" is the primary field ("n" is only its alias) — must be caught too
    res = server.make_request("POST", "/v1/chat/completions", data={
        "messages": [{"role": "user", "content": "1, 2, 3,"}],
        "max_tokens": 8,
        "n_cmpl": 2,
        "logit_gate": gate,
    })
    assert res.status_code == 400, res.body
    assert "requires n = 1" in res.body["error"]["message"], res.body


@pytest.mark.parametrize("candidates,gate_fields,expected", [
    ([], {}, "non-empty array"),
    ([{"label": "A"}], {}, "needs either 'text' or 'ids'"),
    ([{"label": "", "text": "A"}], {}, "label must not be empty"),
    ([{"label": "A", "text": "A"}, {"label": "A", "text": "B"}], {}, "duplicate candidate label"),
    ([{"label": "A", "text": "hello world"}], {}, "exactly 1 token"),
    ([{"label": "A", "ids": [999999999]}], {}, "out of range"),
    ([{"label": "A", "ids": []}], {}, "ids must not be empty"),
    ([{"label": "A", "text": "yes"}, {"label": "B", "text": "yes"}], {}, "used by multiple candidates"),
    ([{"label": "A", "text": "A"}, {"label": "B", "text": "B"}], {"mode": "bogus"}, "mode must be"),
    ([{"label": "A", "text": "A"}], {"options": {"temperature": 0}}, "must be > 0"),
    ([{"label": "A", "text": "A"}, {"label": "B", "text": "B"}], {"threshold": 5}, "between 0 and 1"),
])
def test_logit_gate_validation_errors(candidates, gate_fields, expected):
    global server
    server.start()
    res = server.make_request("POST", "/v1/chat/completions", data=gate_chat_payload(
        candidates, **gate_fields,
    ))
    assert res.status_code == 400
    assert expected in res.body["error"]["message"]
