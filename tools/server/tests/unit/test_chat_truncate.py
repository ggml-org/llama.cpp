import pytest
from utils import *

server: ServerProcess

SYSTEM = "You are a helpful assistant."
FINAL_USER = "[U Last]This is the most recent user message."
N_TURNS_OVERFLOW = 128

def _user_msg(i: int) -> str:
    return f"[U{i:02d}] Please explain topic {i} in detail."

def _asst_msg(i: int) -> str:
    return f"[A{i:02d}] Here is my explanation of topic {i}."


def _get_messages(n_turns: int = 128, include_final_user: bool = True) -> list[dict]:
    msgs = [{"role": "system", "content": SYSTEM}]
    for i in range(1, n_turns + 1):
        msgs.append({"role": "user",      "content": _user_msg(i)})
        msgs.append({"role": "assistant", "content": _asst_msg(i)})
    if include_final_user:
        msgs.append({"role": "user", "content": FINAL_USER})
    return msgs


def _short_messages() -> list[dict]:
    return _get_messages(n_turns=1)

def assert_turns_consistency_in_prompt(prompt: str, include_final_user: bool = True):
    """
    Verify that the user and assistant turns in the rendered prompt are consistent
    with the chat template, meaning:
        - no assistant message appears without its preceding user message
        - final user message is present.
        - system prompt is present.
    """
    turns = prompt.split("\n")
    user_turns = [t for t in turns if t.startswith("[U")]
    asst_turns = [t for t in turns if t.startswith("[A")]
    assert SYSTEM in prompt
    if include_final_user:
        assert len(user_turns) == len(asst_turns) + 1, "Each assistant turn should be preceded by a user turn, and there should be one extra user turn at the end"
        assert FINAL_USER in prompt
    else:
        assert len(user_turns) == len(asst_turns), "Each assistant turn should be preceded by a user turn, and there should be no extra user turn at the end"
    for i in range(len(asst_turns)):
        expected_asst = _asst_msg(i + 1)
        if expected_asst in prompt:
            expected_user = _user_msg(i + 1)
            assert expected_user in prompt, f"User turn {i+1} should be present in the prompt as it precedes assistant turn {i+1}"


@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.tinyllama2()
    # tinyllama2 preset: n_ctx=512, n_slots=2, n_predict=64
    # so per-slot context = 512 // 2 = 256 tokens
    server.jinja = True
    server.chat_template = "chatml"


# Tests

def test_chat_truncate_overflow_without_chat_truncate_flag():
    """
    Baseline: a long conversation exceeds the per-slot context when
    --chat-truncate is not set, returning a 400 exceed_context_size_error.
    """
    global server
    server.start()
    res = server.make_request("POST", "/chat/completions", data={
        "max_completion_tokens": 5,
        "messages": _get_messages(),
    })
    assert res.status_code == 400
    assert res.body["error"]["type"] == "exceed_context_size_error"


def test_chat_truncate_prevents_overflow():
    """
    With --chat-truncate set, long conversation succeeds.
    """
    global server
    server.chat_truncate = 0.8
    server.start()
    res = server.make_request("POST", "/chat/completions", data={
        "max_completion_tokens": 5,
        "messages": _get_messages(N_TURNS_OVERFLOW),
    })
    assert res.status_code == 200


def test_chat_truncate_no_op():
    """
    A short conversation already below the truncation target is left untouched
    """
    global server
    server.chat_truncate = 0.8
    server.debug = True
    server.start()
    res = server.make_request("POST", "/chat/completions", data={
        "max_completion_tokens": 5,
        "messages": _short_messages(),
    })
    assert res.status_code == 200
    assert "__verbose" in res.body
    prompt = res.body["__verbose"]["prompt"]
    assert "[U01]" in prompt, "No turn should not be dropped"
    assert_turns_consistency_in_prompt(prompt)


def test_chat_truncate_prompt_within_budget():
    """
    After truncation, prompt_tokens must be strictly less than the truncation
    target: floor(fraction * per_slot_ctx).
    """
    global server
    server.chat_truncate = 0.8
    server.start()
    res = server.make_request("POST", "/chat/completions", data={
        "max_completion_tokens": 5,
        "messages": _get_messages(n_turns=N_TURNS_OVERFLOW),
    })
    assert res.status_code == 200
    assert server.n_ctx is not None and server.n_slots is not None
    per_slot_ctx = server.n_ctx // server.n_slots
    target = int(server.chat_truncate * per_slot_ctx)
    assert res.body["usage"]["prompt_tokens"] < target


def test_chat_truncate_drops_oldest_keeps_newest():
    global server
    server.chat_truncate = 0.8
    server.debug = True
    server.start()
    res = server.make_request("POST", "/chat/completions", data={
        "max_completion_tokens": 5,
        "messages": _get_messages(n_turns=N_TURNS_OVERFLOW),
    })
    assert res.status_code == 200
    assert "__verbose" in res.body
    prompt = res.body["__verbose"]["prompt"]
    assert _user_msg(1) not in prompt
    assert_turns_consistency_in_prompt(prompt)

def test_chat_truncate_non_user_newest_preserved():
    global server
    server.chat_truncate = 0.8
    server.debug = True
    server.start()
    res = server.make_request("POST", "/chat/completions", data={
        "max_completion_tokens": 5,
        "messages": _get_messages(n_turns=N_TURNS_OVERFLOW, include_final_user=False),
    })
    assert res.status_code == 200
    assert "__verbose" in res.body
    prompt = res.body["__verbose"]["prompt"]
    assert _user_msg(1) not in prompt
    assert_turns_consistency_in_prompt(prompt, include_final_user=False)

def test_chat_truncate_n_predict_threshold_vs_max_completion_tokens():
    """
    The truncation threshold differs depending on whether max_completion_tokens is in the request:

        no max_completion_tokens  -> n_predict = server.n_predict = 64  -> threshold = 256 - 64  = 192
        max_completion_tokens=5   -> n_predict = 5                      -> threshold = 256 - 5   = 251

    We first probe to find n_turns so that the true token count T satisfies:

        target (192) < T < threshold_with_max5 (251)

    Then, with T turns:
    - Without max_completion_tokens: threshold=192 < T -> truncation fires, prompt < target
    - With max_completion_tokens=5:  threshold=251 > T -> truncation silent, prompt unchanged
    """
    global server
    server.chat_truncate = 0.8
    server.start()

    assert server.n_ctx is not None and server.n_slots is not None and server.n_predict is not None
    max_completion_tokens = 5
    per_slot_ctx = server.n_ctx // server.n_slots
    threshold_no_max_completion_tokens = per_slot_ctx - server.n_predict
    threshold_with_max5 = per_slot_ctx - max_completion_tokens
    target = int(server.chat_truncate * per_slot_ctx)

    assert threshold_no_max_completion_tokens < target < threshold_with_max5

    # Find n_turns so that target < T < threshold_with_max5
    found_n_turns  = None
    found_n_tokens = None
    for n_turns in range(0, 10):
        probe = server.make_request("POST", "/chat/completions", data={
            "max_completion_tokens": 5,
            "messages": _get_messages(n_turns),
        })
        if probe.status_code != 200:
            break
        pt = probe.body["usage"]["prompt_tokens"]
        if target < pt < threshold_with_max5:
            found_n_turns  = n_turns
            found_n_tokens = pt
            break

    assert found_n_turns is not None

    msgs = _get_messages(found_n_turns)

    # Without max_completion_tokens: threshold=192 < T, then truncation fires
    res_no_max = server.make_request("POST", "/chat/completions", data={
        "messages": msgs,
    })
    assert res_no_max.status_code == 200
    assert res_no_max.body["usage"]["prompt_tokens"] < target

    # With max_completion_tokens=5: threshold=251 > T, then truncation silent
    res_max5 = server.make_request("POST", "/chat/completions", data={
        "max_completion_tokens": 5,
        "messages": msgs,
    })
    assert res_max5.status_code == 200
    assert res_max5.body["usage"]["prompt_tokens"] == found_n_tokens


def test_chat_truncate_target_capped_to_budget():
    """
    When chat_truncate * n_ctx > n_ctx - n_predict (the generation budget),
    chat_truncate_target_tokens caps the truncation target at the budget.
    """
    global server

    n_predict_req = 128
    server.chat_truncate = 0.99
    server.n_predict = n_predict_req
    server.start()

    assert server.n_ctx is not None and server.n_slots is not None
    per_slot_ctx    = server.n_ctx // server.n_slots
    fraction_target = int(server.chat_truncate * per_slot_ctx)
    budget          = per_slot_ctx - n_predict_req

    assert fraction_target > budget

    res = server.make_request("POST", "/chat/completions", data={
        "max_completion_tokens": n_predict_req,
        "messages": _get_messages(),
    })
    assert res.status_code == 200

    assert res.body["usage"]["prompt_tokens"] < budget


def test_chat_truncate_very_low_fraction_preserves_last_user_msg():
    """
    Even with small chat_truncate fraction, truncation loop drops keeps system + 1 user prompt
    """
    global server
    server.chat_truncate = 0.001
    server.n_predict = -1
    server.debug = True
    server.start()

    res = server.make_request("POST", "/chat/completions", data={
        "messages": _get_messages(),
    })
    assert res.status_code == 200
    assert "__verbose" in res.body
    prompt = res.body["__verbose"]["prompt"]
    templated_sys_prompt = "<s> <|im_start|>system\n" + SYSTEM + "<|im_end|>\n"
    templated_last_message = "<|im_start|>user\n" + FINAL_USER + "<|im_end|>\n<|im_start|>assistant\n"
    assert templated_sys_prompt + templated_last_message == prompt


def test_chat_truncate_negative_value_rejected():
    global server
    server.chat_truncate = 0.0
    with pytest.raises(RuntimeError):
        server.start()


def test_chat_truncate_above_one_rejected():
    global server
    server.chat_truncate = 1.0
    with pytest.raises(RuntimeError):
        server.start()
