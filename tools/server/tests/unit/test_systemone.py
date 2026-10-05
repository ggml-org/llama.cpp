import os
import re
import time
from concurrent.futures import ThreadPoolExecutor

import pytest
import requests
from utils import *
from test_vision_api import get_img_url

server = ServerPreset.tinylaya()


@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.tinylaya()


TEST_STATE = "I was charged twice for my order last week and nobody has replied."

TEST_QUESTIONS = {
    "route": {
        "type": "choice",
        "instructions": "Which team should handle this?",
        "criteria": {"billing": "payments and refunds", "shipping": None, "technical": None},
    },
    "urgency": {
        "type": "score",
        "instructions": "How urgent is this?",
        "criteria": ["can wait", "this week", "today", "right now"],
    },
    "angry": {
        "type": "noul",
        "instructions": "Is the customer angry?",
    },
}


def post(**kwargs):
    return server.make_request("POST", "/v1/systemone", data={
        "state": TEST_STATE,
        **kwargs,
    })


def choice_confidence(probabilities: dict) -> float:
    """the confidence Jev publishes for a choice: (N * p_max - 1) / (N - 1)"""
    n = len(probabilities)
    if n < 2:
        return 1.0
    p_max = max(probabilities.values())
    return max(0.0, (p_max - 1 / n) / (1 - 1 / n))


def score_confidence(probabilities: dict) -> float:
    """the confidence of a score: the mean distance to the most likely level, relative to the same
    distance for a uniform distribution. Jev documents this only for 2 and 3 levels."""
    n = len(probabilities)
    if n < 2:
        return 1.0
    probs = [probabilities[key] for key in sorted(probabilities, key=int)]
    mode = probs.index(max(probs))
    dist = sum(p * abs(i - mode) for i, p in enumerate(probs))
    dist_uniform = sum(abs(i - (n - 1) / 2) for i in range(n)) / n
    return max(0.0, 1.0 - dist / dist_uniform)


def read_metrics(server: ServerProcess) -> dict[str, int]:
    """every llamacpp counter and gauge the running server reports"""
    res = server.make_request("GET", "/metrics")
    assert res.status_code == 200
    values = {}
    for line in res.body.splitlines():
        if line.startswith("llamacpp:"):
            name, value = line.split(" ")
            values[name] = int(float(value))
    return values


def get_prompt_metrics(server: ServerProcess) -> tuple[int, int]:
    """returns the number of prompt tokens (processed, cached) since the server started"""
    values = read_metrics(server)
    return values["llamacpp:prompt_tokens_total"], values["llamacpp:prompt_tokens_cached_total"]


def count_tokens(server: ServerProcess, text: str) -> int:
    """the number of tokens the running model gives a piece of text"""
    res = server.make_request("POST", "/tokenize", data={"content": text})
    assert res.status_code == 200
    return len(res.body["tokens"])


def slots(server: ServerProcess) -> list[dict]:
    """one entry per slot of the running server. Needs --slots, see ServerProcess.server_slots"""
    res = server.make_request("GET", "/slots")
    assert res.status_code == 200
    return res.body


def slot_states(server: ServerProcess) -> list[bool]:
    """is each slot of the running server holding a task"""
    return [slot["is_processing"] for slot in slots(server)]


def n_ctx_slot(server: ServerProcess) -> int:
    """the context one slot of the running server has. It is min(llama_n_ctx_seq,
    llama_model_n_ctx_train), so --ctx-size and --parallel do not determine it, the server reports it"""
    return slots(server)[0]["n_ctx"]


@pytest.mark.parametrize("preset", ["tinylaya", "tinyopenjev"])
def test_systemone(preset: str):
    global server
    server = getattr(ServerPreset, preset)()
    server.start()
    res = post(questions=TEST_QUESTIONS)
    assert res.status_code == 200
    assert res.body["model"]  # the model that answered, never an alias
    assert res.body["usage"]["input_tokens"] > 0
    assert res.body["usage"]["output_tokens"] == 0
    assert set(res.body["usage"].keys()) == {"input_tokens", "output_tokens"}

    answers = res.body["answers"]
    assert list(answers.keys()) == ["route", "urgency", "angry"]

    route = answers["route"]
    assert route["type"] == "choice"
    assert list(route["probabilities"].keys()) == ["billing", "shipping", "technical"]
    assert abs(sum(route["probabilities"].values()) - 1.0) < 1e-4
    assert route["choice"] == max(route["probabilities"], key=route["probabilities"].get)
    assert abs(route["confidence"] - choice_confidence(route["probabilities"])) < 1e-6

    urgency = answers["urgency"]
    assert urgency["type"] == "score"
    assert urgency["legend"] == {"0": "can wait", "1": "this week", "2": "today", "3": "right now"}
    assert list(urgency["probabilities"].keys()) == ["0", "1", "2", "3"]
    assert abs(sum(urgency["probabilities"].values()) - 1.0) < 1e-4
    assert abs(urgency["score"] - sum(i * p for i, p in enumerate(urgency["probabilities"].values()))) < 1e-4
    # 4 levels, where Jev does not publish a confidence: the local rule is reported
    assert abs(urgency["confidence"] - score_confidence(urgency["probabilities"])) < 1e-6

    angry = answers["angry"]
    assert angry["type"] == "noul"
    assert 0.0 <= angry["noul"] <= 1.0
    assert "confidence" not in angry


@pytest.mark.parametrize("preset", ["tinylaya", "tinyopenjev"])
def test_systemone_option_limit_is_the_models_own(preset: str, tmp_path):
    """the number of options one question may carry comes from the model, not from the API ceiling.

    laya derives it from the token window its head reads (tinylaya: 44 of a 192 token window), the
    types that name options with a single character derive it from their tokenizer. The number is
    read from the startup log so this test holds for any model.
    """
    global server
    server = getattr(ServerPreset, preset)()
    server.log_path = str(tmp_path / "server.log")
    server.start()

    reported = re.search(r"at most (\d+) options per question", read_log(server))
    assert reported, read_log(server)
    n_max = int(reported.group(1))
    assert 0 < n_max <= 255

    def ask(n_options: int):
        return post(questions={"q": {
            "type": "choice",
            "instructions": "which one of these describes the ticket",
            "criteria": {f"o{i}": f"option number {i}" for i in range(n_options)},
        }})

    # a question at the limit is answered, and one option more is refused with the model's own number
    assert ask(n_max).status_code == 200
    res = ask(n_max + 1)
    assert res.status_code == 422
    assert res.body["error"]["type"] == "unprocessable_entity_error"
    assert f"at most {n_max} are supported" in res.body["error"]["message"]


def test_systemone_model_field():
    server.start()

    # the answer reports the model that answered, whatever the request says
    res_absent = post(questions=TEST_QUESTIONS)
    assert res_absent.status_code == 200
    loaded = res_absent.body["model"]
    for model in ["jev-latest", "jev-preview", "some-router-name", ""]:
        res = server.make_request("POST", "/v1/systemone", data={
            "model": model,
            "state": TEST_STATE,
            "questions": TEST_QUESTIONS,
        })
        assert res.status_code == 200
        assert res.body["model"] == loaded

    # a value that is not a string is a client error
    res = server.make_request("POST", "/v1/systemone", data={
        "model": 123,
        "state": TEST_STATE,
        "questions": TEST_QUESTIONS,
    })
    assert res.status_code == 422
    assert "model" in res.body["error"]["message"]


def test_systemone_question_id_shown():
    server.start()
    # the key of a question is given to the model, so nothing pins an answer to it: renaming every
    # question changes the prompts and the answers may move. What must hold is that the answers come
    # back under the keys the caller sent, and that each is a well formed answer of its own type.
    renamed = {"question_18_" + qid: q for qid, q in TEST_QUESTIONS.items()}
    res = post(questions=TEST_QUESTIONS)
    res_renamed = post(questions=renamed)
    assert res.status_code == 200
    assert res_renamed.status_code == 200
    for key, body in ((TEST_QUESTIONS, res), (renamed, res_renamed)):
        assert list(body.body["answers"].keys()) == list(key.keys())
        for qid, question in key.items():
            answer = body.body["answers"][qid]
            assert answer["type"] == question["type"]
            if question["type"] == "noul":
                assert 0.0 <= answer["noul"] <= 1.0
            else:
                assert answer["probabilities"]
                assert abs(sum(answer["probabilities"].values()) - 1.0) < 1e-3


def test_systemone_json_state():
    server.start()
    questions = {
        "refund": {
            "type": "noul",
            "instructions": "Is a refund requested?",
            "criteria": {"false": "no refund is asked", "true": "a refund is asked"},
        },
    }
    res_obj = post(state={"ticket": TEST_STATE, "plan": "pro"}, questions=questions)
    assert res_obj.status_code == 200
    # an object is given to the model as JSON text
    res_str = post(state='{"ticket": "' + TEST_STATE + '", "plan": "pro"}', questions=questions)
    assert res_str.status_code == 200
    assert res_obj.body["usage"] == res_str.body["usage"]
    assert abs(res_obj.body["answers"]["refund"]["noul"] - res_str.body["answers"]["refund"]["noul"]) < 1e-4


def test_systemone_instructions_must_carry_content():
    """only a non-empty string, object or array is a question. Jev lists null as allowed and the
    check is stricter, so that no template can fall back to another value when this one is absent."""
    server.start()
    for instructions in ["", {}, [], 0, 1.5, False, True, None]:
        res = post(questions={"q": {"type": "noul", "instructions": instructions}})
        assert res.status_code == 422, instructions
        assert res.body["error"]["type"] == "unprocessable_entity_error"
        assert "instructions" in res.body["error"]["message"]


def test_systemone_invalid_request():
    # the instructions domain has its own table above, which also names the field in the message, so
    # it is not repeated here. A refused request never reaches a slot, so one server answers each
    body = [
        {"questions": TEST_QUESTIONS},  # no state
        {"state": TEST_STATE},  # no questions
        {"state": TEST_STATE, "questions": {}},
        {"state": TEST_STATE, "model": 123, "questions": TEST_QUESTIONS},
        {"state": TEST_STATE, "questions": {"q": {"type": "unknown", "instructions": "x"}}},
        {"state": TEST_STATE, "questions": {"q": {"type": "choice", "instructions": "x"}}},
        {"state": TEST_STATE, "questions": {"q": {"type": "choice", "instructions": "x", "criteria": {}}}},
        {"state": TEST_STATE, "questions": {"q": {"type": "score", "instructions": "x", "criteria": ["only one"]}}},
        {"state": TEST_STATE, "questions": {"q": {"type": "score", "instructions": "x", "criteria": ["a"] * 11}}},
    ]
    server.start()
    for case in body:
        res = server.make_request("POST", "/v1/systemone", data=case)
        assert res.status_code == 422, case
        assert res.body["error"]["type"] == "unprocessable_entity_error"
        assert res.body["error"]["message"]


def test_systemone_malformed_body():
    server.start()
    # make_request sends JSON, so the malformed body needs a raw request
    assert requests.post(server.make_url("/v1/systemone"),
                         data='{"state": "unterminated').status_code == 400
    # and an empty body is malformed too, not a well-formed one the route refuses
    assert server.make_request("POST", "/v1/systemone").status_code == 400


def test_systemone_shared_prompt():
    global server
    server = ServerPreset.tinyopenjev()
    server.server_metrics = True
    server.start()
    res = post(questions=TEST_QUESTIONS)
    assert res.status_code == 200

    # the first question evaluates the shared prefix, the 2 others start from it
    n_processed, n_cached = get_prompt_metrics(server)
    assert n_cached > 0
    # The 3 questions group as one parent and two children, so the shared prefix is cached once per
    # child and the cached total is even. That parity is a fixture property, not a server one.
    assert n_cached % 2 == 0
    # the tokens of the shared prefix are evaluated once, so the usage is the number of tokens
    # that were processed, the cached ones are the tokens the children did not evaluate again
    assert n_processed == res.body["usage"]["input_tokens"]

    # with one slot the prompt cannot be shared, the answers must be the same
    server.stop()
    server = ServerPreset.tinyopenjev()
    server.n_slots = 1
    server.server_metrics = True
    server.start()
    res_single = post(questions=TEST_QUESTIONS)
    assert res_single.status_code == 200
    # nothing is cached without a parent to copy from, so the counter above is the sharing and only
    # the sharing
    assert get_prompt_metrics(server)[1] == 0
    # the shared prefix is only paid once, so the shared run costs fewer input tokens
    assert res.body["usage"]["input_tokens"] < res_single.body["usage"]["input_tokens"]
    for qid in ["route", "urgency"]:
        probs_shared = res.body["answers"][qid]["probabilities"]
        probs_single = res_single.body["answers"][qid]["probabilities"]
        for key in probs_shared:
            assert abs(probs_shared[key] - probs_single[key]) < 0.01
    assert abs(res.body["answers"]["angry"]["noul"] - res_single.body["answers"]["angry"]["noul"]) < 0.01


def test_systemone_input_tokens_warm_cache():
    """input_tokens describes the request, not the work done, so it is identical on a cold and a warm
    run. That the server-side processed count is smaller on the second run is not observable with the
    test models available here and is deliberately not asserted."""
    global server
    server = ServerPreset.tinyopenjev()
    server.server_metrics = True
    server.start()

    res_cold = post(questions=TEST_QUESTIONS)
    assert res_cold.status_code == 200
    processed_cold, cached_cold = get_prompt_metrics(server)

    # the identical request again
    res_warm = post(questions=TEST_QUESTIONS)
    assert res_warm.status_code == 200
    processed_warm, cached_warm = get_prompt_metrics(server)

    # the reported usage is a property of the request, so it is identical either way
    assert res_warm.body["usage"]["input_tokens"] == res_cold.body["usage"]["input_tokens"]
    assert res_warm.body["answers"].keys() == res_cold.body["answers"].keys()

    # the counters are cumulative, so it is the delta that says anything: the second request reused
    # the prefix the first one left in the slots
    assert cached_warm - cached_cold > 0, "the identical request reused nothing"
    assert processed_warm - processed_cold == res_warm.body["usage"]["input_tokens"]


def test_systemone_images():
    global server
    server = ServerPreset.tinyopenjev()
    server.start()
    image = get_img_url("IMG_BASE64_URI_0")

    res_text = post(questions=TEST_QUESTIONS)
    assert res_text.status_code == 200

    res = post(questions=TEST_QUESTIONS, images=[image])
    assert res.status_code == 200
    assert list(res.body["answers"].keys()) == ["route", "urgency", "angry"]
    assert res.body["usage"]["input_tokens"] > res_text.body["usage"]["input_tokens"]

    # same image, given as a part of a chat message
    res_part = post(questions=TEST_QUESTIONS, state=[{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": image}},
        {"type": "text", "text": TEST_STATE},
    ]}])
    assert res_part.status_code == 200
    assert res_part.body["usage"]["input_tokens"] > res_text.body["usage"]["input_tokens"]

    # too many images. The limit is 8 and it is the server's own, so the refusal names it and a client
    # can discover the number from the answer instead of having to guess it
    res_over = post(questions=TEST_QUESTIONS, images=[image] * 9)
    assert res_over.status_code == 422
    assert "the maximum is 8" in res_over.body["error"]["message"]

    # an image URL that is not a data URL, and a data URL that cannot be decoded, are semantic
    # refusals, not malformed bodies
    for bad in ["https://example.com/image.png", "data:image/png;base64"]:
        res_bad = post(questions=TEST_QUESTIONS, images=[bad])
        assert res_bad.status_code == 422
        assert res_bad.body["error"]["type"] == "unprocessable_entity_error"


#
# lev and kev
#
# There is no tiny model for either, so these skip unless a local copy is pointed at. The lev
# arithmetic is covered without a model by the averaging unit tests in test-server-decision.cpp.
#

LEV_AVAILABLE = os.environ.get("LEV_LOCAL_MODEL") is not None
KEV_AVAILABLE = os.environ.get("KEV_LOCAL_MODEL") is not None

LEV_SKIP_REASON = (
    "no tiny lev model exists; the prompt/option encoding and the answer math are covered "
    "model-free by tools/server/tests/test-server-decision.cpp, set LEV_LOCAL_MODEL to a local "
    "copy only for the tokenizer-dependent end-to-end check"
)
KEV_SKIP_REASON = (
    "no tiny kev model exists; the prompt/option encoding and the answer math are covered "
    "model-free by tools/server/tests/test-server-decision.cpp, set KEV_LOCAL_MODEL to a local "
    "copy only for the tokenizer-dependent end-to-end check"
)


@pytest.mark.skipif(not LEV_AVAILABLE, reason=LEV_SKIP_REASON)
def test_systemone_lev():
    global server
    server = ServerPreset.lev()
    server.start()

    res = post(questions=TEST_QUESTIONS)
    assert res.status_code == 200
    assert list(res.body["answers"].keys()) == ["route", "urgency", "angry"]

    # lev runs every choice question in both option orders and averages, so the answer must not
    # change when the caller renames the options
    renamed = {
        "route": {**TEST_QUESTIONS["route"], "criteria": {"z": "payments", "y": "postage"}},
        "urgency": TEST_QUESTIONS["urgency"],
        "angry": TEST_QUESTIONS["angry"],
    }
    res_renamed = post(questions=renamed)
    assert res_renamed.status_code == 200
    assert res_renamed.body["answers"]["route"]["choice"] in ("z", "y")
    assert abs(sum(res_renamed.body["answers"]["route"]["probabilities"].values()) - 1.0) < 1e-4


@pytest.mark.skipif(not KEV_AVAILABLE, reason=KEV_SKIP_REASON)
def test_systemone_kev():
    global server
    server = ServerPreset.kev()
    server.start()

    res = post(questions=TEST_QUESTIONS)
    assert res.status_code == 200
    assert list(res.body["answers"].keys()) == ["route", "urgency", "angry"]

    # kev reads a noul answer as the weighted index of the levels, so it is in [0, 1]
    assert 0.0 <= res.body["answers"]["angry"]["noul"] <= 1.0
    assert 0 <= res.body["answers"]["urgency"]["score"] <= 2


def test_systemone_head_window_over_batch_is_refused():
    """the question and its options are read from one batch, so a --ubatch-size below the window
    the head reads refuses the request with its own 400 kind, not the generic one."""
    global server
    server = ServerPreset.tinylaya()
    # tinylaya's head reads a 192 token window, so a batch below it cannot hold the answer
    server.n_ubatch = 64
    server.start()

    res = post(questions=WIDE_CHOICE)
    assert res.status_code == 400
    assert res.body["error"]["type"] == "exceed_batch_size_error"
    assert "batch" in res.body["error"]["message"]

    # the same request is answered once the batch is large enough to hold the window
    server.stop()
    server.n_ubatch = 512
    server.start()
    assert post(questions=WIDE_CHOICE).status_code == 200


# a parent released before its prompt is done leaves its children unable to run, so the overflow
# must land on the first question: the questions-object order decides which task is the parent.
OVERFLOW_SENTENCE = "The customer wrote in again about the same billing problem that nobody has answered. "
OVERFLOW_OPTION = "a very long option description that takes many tokens to say out loud in full. "
OVERFLOW_SMALL = {"type": "noul", "instructions": "Is the customer angry?"}
OVERFLOW_SMALL2 = {"type": "noul", "instructions": "Is a refund requested?"}
OVERFLOW_BIG = {
    "type": "choice",
    "instructions": "Which of these twenty options describes the ticket best?",
    "criteria": {f"team_{i:02d}": OVERFLOW_OPTION for i in range(20)},
}


def test_systemone_refused_group_leaves_no_slot_behind(tmp_path):
    global server
    server = ServerPreset.tinyopenjev()
    server.server_slots = True
    server.log_path = str(tmp_path / "server.log")
    server.start()

    n_ctx = n_ctx_slot(server)
    assert not any(slot_states(server))

    state = OVERFLOW_SENTENCE
    while count_tokens(server, state) < int(0.85 * n_ctx):
        state += OVERFLOW_SENTENCE

    # the state and the two small questions fit a slot, so the group below can only overflow on the
    # first question. If this ever stops holding, the refusal below would not be testing the group
    res = post(state=state, questions={"a": OVERFLOW_SMALL, "b": OVERFLOW_SMALL2})
    assert res.status_code == 200
    assert res.body["usage"]["input_tokens"] < n_ctx

    res = post(state=state, questions={"big": OVERFLOW_BIG, "a": OVERFLOW_SMALL, "b": OVERFLOW_SMALL2},
               timeout=30)
    assert res.status_code == 400
    assert res.body["error"]["type"] == "exceed_context_size_error"
    # the message reports the size of the prompt that was refused, so the sizing above is checked
    # against what the server measured and not against the arithmetic of this test
    refused = re.search(r"request \((\d+) tokens\)", res.body["error"]["message"])
    assert refused, res.body["error"]["message"]
    assert int(refused.group(1)) >= n_ctx

    # the children of the refused group waited on a parent that is gone. No slot may still hold one,
    # or the loop keeps looking for work that can never arrive and the server spins on one core
    deadline = time.time() + 10
    while any(slot_states(server)):
        assert time.time() < deadline, "a slot is still holding a task of the refused group"
        time.sleep(0.1)
    assert server.make_request("GET", "/health").status_code == 200

    # and the server keeps answering
    res = post(questions=TEST_QUESTIONS)
    assert res.status_code == 200
    assert not any(slot_states(server))


def test_systemone_answers_429_when_the_queue_is_full():
    global server
    server = ServerPreset.tinylaya()
    server.server_metrics = True
    # the cap counts queued decision tasks, not requests: a noul question is one task on every
    # preset, so one request weighs n_questions. Two requests' worth admits one on an idle server.
    n_questions = 8
    cap = 2 * n_questions
    server.decision_max_queued = cap
    server.start()

    questions = burst_questions(n_questions)
    n_burst = server.n_slots * 4 + 8
    responses = fire_burst(server, n_burst, questions)

    refused = [r for r in responses if r.status_code == 429]
    assert refused, f"no 429 in {n_burst} concurrent requests: " \
        f"{sorted({r.status_code for r in responses})}"
    for r in refused:
        assert r.body["error"]["type"] == "rate_limit_error"
        assert r.body["error"]["code"] == 429
        # the table says a limited request is retried after a delay, so the delay is on the wire
        assert int(r.headers["Retry-After"]) >= 1

    # the server counted every refusal it answered, so the gate is what refused them
    counted = read_metrics(server)["llamacpp:decision_requests_refused_total"]
    assert counted == len(refused), f"{len(refused)} refusals but {counted} counted"

    # everything the server did accept is a real answer, not a partial body
    admitted = [r for r in responses if r.status_code != 429]
    assert admitted, "the whole burst was refused, which means the cap is below one request"
    for r in admitted:
        assert r.status_code == 200, r.body
        assert list(r.body["answers"].keys()) == sorted(questions.keys())
        for answer in r.body["answers"].values():
            assert 0.0 <= answer["noul"] <= 1.0

    # once the burst drains the server takes work again
    assert post(questions=TEST_QUESTIONS).status_code == 200

    # the calibration record: a future change to the default shows up as a diff in this line
    histogram = {}
    for r in responses:
        histogram[r.status_code] = histogram.get(r.status_code, 0) + 1
    print(f"\n/v1/systemone admission: cap={cap} tasks, admitted={len(admitted)}, "
          f"refused={len(refused)}, status_histogram={dict(sorted(histogram.items()))}")


def read_log(server: ServerProcess) -> str:
    """everything the running server has written so far"""
    with open(server.log_path) as log:
        return log.read()


REFUSAL_MESSAGE = re.compile(
    r'^The request renders (?P<total>\d+) prompt tokens, the maximum is (?P<budget>\d+)\.'
    r' Give a shorter "state", fewer questions, or raise the limit with'
    r' --decision-max-prompt-tokens$')


def refusal(res) -> tuple[int, int]:
    """the (total, budget) a refusal names, with the message format pinned to the documented one"""
    assert res.status_code == 413
    assert res.body["error"]["type"] == "request_too_large_error"
    assert res.body["error"]["code"] == 413
    # 413 is a fault of the request's own size, not of the load, so unlike 429 it carries no Retry-After
    assert "Retry-After" not in res.headers
    assert "answers" not in res.body
    named = REFUSAL_MESSAGE.match(res.body["error"]["message"])
    assert named, res.body["error"]["message"]
    return int(named["total"]), int(named["budget"])


def uniform_questions(n_questions: int) -> dict:
    """n noul questions with byte-identical instructions, so every task renders to the same token count and
    the number of tasks the renderer built is derivable from a refusal."""
    return {
        f"q{i}": {"type": "noul", "instructions": "Is this statement about the ticket true?"}
        for i in range(n_questions)
    }


def at_prompt_budget(server: ServerProcess, budget: int):
    """restart at this --decision-max-prompt-tokens. negative is the derived default, 0 is unlimited. Every
    budget in these tests is set this way, so none can measure a budget it did not set."""
    server.stop()
    server.decision_max_prompt_tokens = budget
    server.start()


def derived_prompt_budget(server: ServerProcess) -> int:
    """what decision_prompt_budget derives when the flag is negative, spelled out so the calibration
    tests check the derivation instead of taking it on trust"""
    return 8 * len(slots(server)) * n_ctx_slot(server)


def request_rendering_exactly(target_total: int, per_task: int) -> dict:
    """uniform questions that render exactly target_total prompt tokens.

    per_task rarely divides a budget: tinylaya's derived 8192 leaves 32 over 170 tasks, tinyopenjev's
    32768 leaves 60 over 442. Padding the last task with one-token words absorbs the remainder, so the
    request lands exactly on the budget rather than a whole task under it.
    """
    n_tasks, remainder = divmod(target_total, per_task)
    assert remainder, f"per_task {per_task} divides {target_total}, nothing to pad"

    questions = uniform_questions(n_tasks)
    last = list(questions.keys())[-1]
    questions[last] = dict(questions[last])
    questions[last]["instructions"] += " " + " ".join(["pad"] * remainder)
    return questions


def refused_prompt_total(server: ServerProcess, budget: int, questions: dict) -> int:
    """the prompt total the server reports when it refuses this request at this budget"""
    at_prompt_budget(server, budget)
    return refusal(post(questions=questions))[0]


def exact_prompt_total(server: ServerProcess, questions: dict) -> int:
    """the exact sum the budget measures for this request.

    A budget below the first task names that task alone, and one past the reported sum moves the refusal
    on to the next, so the request is admitted exactly when the budget spans every task."""
    total = refused_prompt_total(server, 1, questions)
    while True:
        at_prompt_budget(server, total + 1)
        res = post(questions=questions)
        if res.status_code == 200:
            return total
        total = refusal(res)[0]


def test_systemone_answers_413_when_one_request_is_too_large(tmp_path):
    """a request over the budget is refused whole, before anything is queued.

    The budget counts every task before grouping, which usage.input_tokens does not: that discounts the
    shared prefix a parent and its children inherit, so the total is read from the refusal and never
    from the usage of an admitted run.
    """
    global server
    server = ServerPreset.tinyopenjev()
    server.server_slots = True
    server.server_metrics = True
    server.log_path = str(tmp_path / "server.log")

    one = {"angry": TEST_QUESTIONS["angry"]}

    # a budget of one token refuses the request and names what it measured
    server.decision_max_prompt_tokens = 1
    server.start()
    counters_before = get_prompt_metrics(server)

    res = post(questions=one)
    assert res.status_code == 413
    assert res.body["error"]["type"] == "request_too_large_error"
    assert res.body["error"]["code"] == 413
    # 413 is a fault of the request's own size, not of the load, so unlike 429 it carries no Retry-After
    assert "Retry-After" not in res.headers

    # one question is one task, so the total the server refused on is exactly that task
    total = int(re.search(r"renders (\d+) prompt tokens", res.body["error"]["message"]).group(1))
    assert total > 0

    # the refusal is total: no prompt was evaluated, no task was queued and no slot kept the work
    assert get_prompt_metrics(server) == counters_before
    assert not any(slot_states(server))
    assert server.make_request("GET", "/health").status_code == 200

    # at the measured total the request is admitted, and one token under it is refused again
    server.stop()
    server.decision_max_prompt_tokens = total
    server.start()
    res = post(questions=one)
    assert res.status_code == 200, res.body
    assert not any(slot_states(server))

    assert refused_prompt_total(server, total - 1, one) == total

    # the refusal left the server serving: with the gate lifted the regular request is answered
    server.stop()
    server.decision_max_prompt_tokens = 0
    server.start()
    res = post(questions=TEST_QUESTIONS)
    assert res.status_code == 200
    assert not any(slot_states(server))


def test_systemone_zero_prompt_budget_is_unlimited():
    """a zero budget turns the gate off, which is how a request larger than the default is run.

    The flag and the resolved value share one convention: 0 reaches build_tasks as the 0 that means
    unlimited, not as a budget of nothing.
    """
    global server
    server = ServerPreset.tinyopenjev()

    # many small questions: each prompt fits its slot, and the request as a whole is far larger. This
    # is the shape the budget exists for, and the one the derived default has to be sized against
    questions = burst_questions(24)

    server.decision_max_prompt_tokens = 0
    server.start()
    res = post(questions=questions)
    assert res.status_code == 200
    assert list(res.body["answers"].keys()) == list(questions.keys())

    # the same request is refused under a budget below what it renders
    server.stop()
    server.decision_max_prompt_tokens = max(1, res.body["usage"]["input_tokens"] // 2)
    server.start()
    assert post(questions=questions).status_code == 413


def test_systemone_413_is_not_counted_as_a_load_refusal():
    """413 and 429 are different faults and the metrics must not merge them.

    429 says the server is busy, so it carries Retry-After and is counted by the refusal counter. 413
    says this request is too large to render, which is a property of the request alone: it never
    reached the admission gate, so it must not be counted there.
    """
    global server
    server = ServerPreset.tinyopenjev()
    server.server_metrics = True
    server.decision_max_prompt_tokens = 1
    server.start()

    res = post(questions=TEST_QUESTIONS)
    assert res.status_code == 413
    assert read_metrics(server)["llamacpp:decision_requests_refused_total"] == 0

    # with the budget lifted the very same request is admitted, so it was its own size that was refused
    server.stop()
    server.decision_max_prompt_tokens = 0
    server.start()
    assert post(questions=TEST_QUESTIONS).status_code == 200


@pytest.mark.parametrize("preset", ["tinylaya", "tinyopenjev"])
def test_systemone_the_derived_default_admits_the_largest_request_it_should(preset: str):
    """the derived default admits the largest request it should and refuses one task more.

    No single task can reach the budget: a prompt over a slot's context is refused earlier with 400
    exceed_context_size_error, so a request gets there only as many tasks that each fit. This is the
    control for every refusal below.
    """
    global server
    server = getattr(ServerPreset, preset)()
    server.server_slots = True
    at_prompt_budget(server, -1)

    derived = derived_prompt_budget(server)
    per_task = exact_prompt_total(server, uniform_questions(1))
    n_max = derived // per_task
    assert n_max * per_task <= derived < (n_max + 1) * per_task

    at_prompt_budget(server, -1)
    admitted = uniform_questions(n_max)
    res = post(questions=admitted)
    assert res.status_code == 200, res.body
    assert list(res.body["answers"].keys()) == list(admitted.keys())
    assert not any(slot_states(server))

    # one task more and the sum is over the budget. The refusal names the whole request here, because
    # the budget spans every task but the last
    total, budget = refusal(post(questions=uniform_questions(n_max + 1)))
    assert (total, budget) == ((n_max + 1) * per_task, derived)
    assert not any(slot_states(server))


@pytest.mark.parametrize("preset", ["tinylaya", "tinyopenjev"])
def test_systemone_a_total_of_exactly_the_budget_is_admitted(preset: str):
    """total == budget is served and total == budget - 1 is refused, explicit budget and derived alike.

    The comparison is >, not >=, and this is the only place that shows: the request is padded to land
    exactly on the budget, so an off-by-one would refuse it. A request one task short would be served
    under either comparison and prove nothing.
    """
    global server
    server = getattr(ServerPreset, preset)()
    server.server_slots = True
    at_prompt_budget(server, -1)

    derived = derived_prompt_budget(server)
    per_task = exact_prompt_total(server, uniform_questions(1))
    at_budget = request_rendering_exactly(derived, per_task)

    # one token under, the same request is refused and the refusal names that exact total
    assert refused_prompt_total(server, derived - 1, at_budget) == derived

    # at the budget it is admitted, whether the number is spelled out or derived from the server
    for budget in (derived, -1):
        at_prompt_budget(server, budget)
        res = post(questions=at_budget)
        assert res.status_code == 200, (budget, res.body)
        assert len(res.body["answers"]) == derived // per_task


@pytest.mark.parametrize("preset", ["tinylaya", "tinyopenjev"])
def test_systemone_the_derived_default_leaves_the_control_corpus_alone(preset: str):
    """no realistic request comes near the derived default, so the gate never fires on normal work.

    If a default were tighter than the corpus, every refusal test would still pass while the endpoint
    had stopped answering real requests, so the margin is asserted and not just the admission.
    """
    global server
    server = getattr(ServerPreset, preset)()
    server.server_slots = True
    at_prompt_budget(server, -1)

    derived = derived_prompt_budget(server)
    corpus = {
        "TEST_QUESTIONS": TEST_QUESTIONS,
        "single noul": {"q": TEST_QUESTIONS["angry"]},
        "one question per slot, 4 deep": uniform_questions(4 * len(slots(server))),
    }
    for name, questions in corpus.items():
        res = post(questions=questions)
        assert res.status_code == 200, (name, res.body)

    # a wide request, the shape that is meant to hit the gate, is still comfortably inside it
    wide = uniform_questions(8 * len(slots(server)))
    assert post(questions=wide).status_code == 200

    # and the realistic fixtures keep a margin wide enough that a longer state or a few more questions
    # does not cross it. Measured: TEST_QUESTIONS spans 62x of tinylaya's default and 140x of
    # tinyopenjev's; a single question spans 182x and 461x
    for name in ("TEST_QUESTIONS", "single noul"):
        total = exact_prompt_total(server, corpus[name])
        assert derived >= 32 * total, f"{name}: {total} of {derived}, under 32x the margin"


@pytest.mark.parametrize("preset", ["tinylaya", "tinyopenjev"])
def test_systemone_the_renderer_stops_one_task_past_the_budget(preset: str):
    """an oversized request is abandoned after the budget plus one task, not after all of them.

    The tasks are uniform, so the total the refusal names divided by the one-task total is the number
    of tasks the renderer actually built. That number is the bound on the work a refused request can
    cause: without it, refusing a huge request would cost as much as running it.
    """
    global server
    server = getattr(ServerPreset, preset)()
    server.server_slots = True
    server.server_metrics = True
    at_prompt_budget(server, -1)

    per_task = exact_prompt_total(server, uniform_questions(1))
    n_tasks = 16 * len(slots(server))

    for n_built in (1, 2):
        at_prompt_budget(server, n_built * per_task)
        counters = get_prompt_metrics(server)

        total, budget = refusal(post(questions=uniform_questions(n_tasks)))
        assert (total, budget) == ((n_built + 1) * per_task, n_built * per_task)

        # the budget, plus the single task that broke it. A renderer that kept going would report more
        assert total // per_task == n_built + 1
        assert n_built + 1 < n_tasks

        # nothing of the request was posted: no prompt evaluated, no slot holding work, none queued
        assert get_prompt_metrics(server) == counters
        assert not any(slot_states(server))
        assert read_metrics(server)["llamacpp:requests_processing"] == 0

        # the server still serves. The follow-up has to fit the budget in force, so it is one task
        assert post(questions=uniform_questions(1)).status_code == 200


@pytest.mark.parametrize("preset", ["tinylaya", "tinyopenjev"])
def test_systemone_an_overshoot_of_any_width_is_refused(preset: str):
    """overshooting the budget by 1 and by 4096 tokens is refused the same way.
    The widths are synthetic: a gate catching only a large overshoot would pass the boundary tests
    and still let a single-token overshoot through.
    """
    global server
    server = getattr(ServerPreset, preset)()
    server.server_slots = True
    server.server_metrics = True
    at_prompt_budget(server, -1)

    per_task = exact_prompt_total(server, uniform_questions(1))
    # enough tasks that the budget stays positive at the widest margin
    n_tasks = 4096 // per_task + 2
    total = n_tasks * per_task

    for margin in (1, 4096):
        at_prompt_budget(server, total - margin)
        counters = get_prompt_metrics(server)

        named, budget = refusal(post(questions=uniform_questions(n_tasks)))
        assert budget == total - margin
        # refused at the first task over the budget, and never reported more than the whole request
        assert budget < named <= total
        assert named % per_task == 0

        assert get_prompt_metrics(server) == counters
        assert not any(slot_states(server))
        assert read_metrics(server)["llamacpp:requests_processing"] == 0

        # the server still serves. The follow-up has to fit the budget in force, so it is one task
        assert post(questions=uniform_questions(1)).status_code == 200


def test_systemone_queue_cap_startup_log(tmp_path):
    """the startup log reports the cap the gate will use, explicit or unlimited.

    The cap is resolved once at startup, so the log names the same value the gate uses. The derived
    value itself is decision_queue_cap() and is pinned by tools/server/tests/test-server-decision.cpp.
    """
    global server

    # a positive value is reported as itself
    server = ServerPreset.tinylaya()
    server.log_path = str(tmp_path / "explicit.log")
    server.decision_max_queued = 7
    server.start()
    assert re.search(r"max queued = 7 tasks", read_log(server)), read_log(server)
    server.stop()

    # 0 is unlimited, not the derived default
    server = ServerPreset.tinylaya()
    server.log_path = str(tmp_path / "unlimited.log")
    server.decision_max_queued = 0
    server.start()
    assert re.search(r"max queued = unlimited", read_log(server)), read_log(server)
    server.stop()


def test_systemone_validation_precedes_the_load_gate():
    """validation is checked before the admission gate, so a body this route refuses answers 400 or
    422 even while the decision queue is full. A 429 there would blame the request's merit on load.
    """
    global server
    server = ServerPreset.tinylaya()
    server.decision_max_queued = 1
    server.start()

    n_burst = server.n_slots * 4 + 8

    def send():
        return server.make_request("POST", "/v1/systemone",
                                   data={"state": TEST_STATE, "questions": burst_questions(1)},
                                   timeout=DEFAULT_REQUEST_TIMEOUT)

    with ThreadPoolExecutor(max_workers=n_burst) as pool:
        futures = [pool.submit(send) for _ in range(n_burst)]

        # a valid request with an empty instructions, and a malformed body. Both are refused before
        # the cap is read, whatever the queue depth is at that moment.
        assert post(questions={"q": {"type": "noul", "instructions": ""}}).status_code == 422
        assert requests.post(server.make_url("/v1/systemone"),
                             data='{"state": "unterminated').status_code == 400

        results = [f.result() for f in futures]

    # the gate really was reached, otherwise the precedence above proves nothing
    assert any(r.status_code == 429 for r in results), sorted({r.status_code for r in results})


def answer_keys(answer: dict) -> set:
    return set(answer.keys())


# the answer shape each question type publishes. See ANSWER_KEYS["noul"] for why noul publishes
# neither confidence nor probabilities
ANSWER_KEYS = {
    # a noul's distribution is a single number, so there is nothing left to summarize: the probability
    # it publishes *is* the confidence. choice and score return a distribution over their options, so
    # they carry a `confidence` summarizing how concentrated it is
    "noul":   {"type", "noul"},
    "choice": {"type", "choice", "probabilities", "confidence"},
    "score":  {"type", "score", "legend", "probabilities", "confidence"},
}

# the status and error type each fault of a request is answered with. error_type_info is the only
# place these are mapped, so this table is the contract a caller can rely on: same fault, same pair,
# whichever preset or option count produced it. 400 is shared by the size faults, which pass their
# own type to contract_of
ERROR_CONTRACT = {
    400: "invalid_request_error",
    413: "request_too_large_error",
    422: "unprocessable_entity_error",
    429: "rate_limit_error",
    501: "not_supported_error",
}


@pytest.mark.parametrize("preset", ["tinylaya", "tinyopenjev"])
def test_systemone_wire_contract(preset: str, tmp_path):
    """the published shape of every answer, and of every fault, on one server per preset.

    This is the golden: envelope keys, per-type answer keys, confidence ranges, status/error pairs.
    Changing any of it fails here and the diff is the review.
    """
    global server
    server = getattr(ServerPreset, preset)()
    server.server_slots = True
    server.server_metrics = True
    server.log_path = str(tmp_path / "server.log")
    at_prompt_budget(server, 0)   # unlimited, so the budget cannot turn a fault into a 413

    # the envelope: exactly these three keys, and usage exactly these two
    res = post(questions=TEST_QUESTIONS)
    assert res.status_code == 200, res.body
    assert set(res.body.keys()) == {"model", "answers", "usage"}
    assert set(res.body["usage"].keys()) == {"input_tokens", "output_tokens"}
    # the model echoed is the one that answered, not the one named in the body. test_systemone_model_field
    # covers that it ignores the body's "model"; here it only has to be a non-empty string
    assert isinstance(res.body["model"], str) and res.body["model"]

    # answers keep the request's keys, in the request's order, and use no other key
    assert list(res.body["answers"].keys()) == list(TEST_QUESTIONS.keys())

    for key, question in TEST_QUESTIONS.items():
        answer = res.body["answers"][key]
        assert answer["type"] == question["type"]
        assert answer_keys(answer) == ANSWER_KEYS[question["type"]], (key, answer)

        if question["type"] == "choice":
            # the published choice is one of the criteria keys, and the distribution covers them all
            assert answer["choice"] in question["criteria"]
            assert set(answer["probabilities"].keys()) == set(question["criteria"].keys())
            # confidence is relative to uniform over the options, so it is a fraction
            assert 0.0 <= answer["confidence"] <= 1.0
        elif question["type"] == "score":
            # a score takes its levels as an array, and the server names them by index: the level
            # keys are "0".."N-1" and the legend carries the caller's level strings
            levels = question["criteria"]
            level_keys = {str(i) for i in range(len(levels))}
            assert set(answer["probabilities"].keys()) == level_keys, answer
            assert set(answer["legend"].keys()) == level_keys, answer
            assert list(answer["legend"].values()) == levels, answer
            # the score is the expected level index, so it lands in the levels' range
            assert 0.0 <= answer["score"] <= len(levels) - 1
            assert 0.0 <= answer["confidence"] <= 1.0
        else:
            assert 0.0 <= answer["noul"] <= 1.0
            assert "confidence" not in answer

    # every distribution sums to one, or the confidence above it means nothing
    for answer in res.body["answers"].values():
        if "probabilities" in answer:
            assert abs(sum(answer["probabilities"].values()) - 1.0) < 1e-3

    # each fault of a request carries its own status and error type, and no other one
    def contract_of(res, status: int, error_type: str | None = None):
        assert res.status_code == status, res.body
        assert res.body["error"]["type"] == (error_type or ERROR_CONTRACT[status])
        assert res.body["error"]["code"] == status
        assert res.body["error"]["message"]
        assert set(res.body.keys()) == {"error"}, res.body
        assert set(res.body["error"].keys()) == {"message", "type", "code"}

    # 400: a body that is not JSON at all. This one needs a raw request, so it comes back as a
    # requests.Response and not as the harness's ServerResponse
    raw = requests.post(server.make_url("/v1/systemone"), data='{"state": "unterminated')
    assert raw.status_code == 400
    assert raw.json()["error"]["type"] == ERROR_CONTRACT[400]
    # 422: well-formed JSON, but a field the route will not accept. An unknown field is not one of
    # them: the route ignores what it does not know, so only the fields it reads can be refused
    contract_of(post(questions={"q": {"type": "noul", "instructions": ""}}), 422)
    contract_of(server.make_request("POST", "/v1/systemone", data={"questions": TEST_QUESTIONS}), 422)
    contract_of(server.make_request("POST", "/v1/systemone",
                                    data={"state": TEST_STATE, "questions": TEST_QUESTIONS,
                                          "model": 123}), 422)
    # 501: images a model without image input cannot take. tinyopenjev has a projector, so this is
    # tinylaya's 501; the loaded-model 501 is test_systemone_not_a_decision_model_is_501
    if preset == "tinylaya":
        contract_of(post(questions=TEST_QUESTIONS, images=[get_img_url("IMG_BASE64_URI_0")]), 501)

    # 413: over the prompt budget
    at_prompt_budget(server, 1)
    contract_of(post(questions=TEST_QUESTIONS), 413)

    # 429: the queue at its cap, which is the one fault that is the server's load and not the
    # request's shape. It is the only one that carries Retry-After. The prompt budget goes back to
    # unlimited first: at 1 every request would be refused 413 before it ever reached the queue gate,
    # which would make the 429 below prove nothing
    server.stop()
    server.decision_max_queued = 1
    server.decision_max_prompt_tokens = 0
    server.start()
    # the cap counts tasks, so a cap of 1 admits one task at a time. One task is one noul question,
    # so two questions in a request already weigh more than the cap and are refused outright; a burst
    # of those is refused for the same reason. This mirrors test_systemone_answers_429_when_the_queue_is_full
    questions = uniform_questions(2)
    cap = server.n_slots * 4 + 8
    responses = fire_burst(server, cap, questions)
    refused = [r for r in responses if r.status_code == 429]
    # the requests are cheap and the slots few, so a burst can drain before it saturates the queue.
    # Widen it until the gate is actually reached, or the assertion below would pass on a 200-only run
    for n in (2 * cap, 4 * cap):
        if refused:
            break
        responses = fire_burst(server, n, questions)
        refused = [r for r in responses if r.status_code == 429]
    assert refused, f"no 429 in up to {4 * cap} concurrent requests: " \
                     f"{sorted({r.status_code for r in responses})}"
    for r in refused:
        assert r.body["error"]["type"] == ERROR_CONTRACT[429]
        assert int(r.headers["Retry-After"]) >= 1

    # 400 exceed_batch_size_error: the window the head reads does not fit the batch the server
    # decodes, so the request is refused with its own kind, not the generic invalid_request_error.
    # test_systemone_head_window_over_batch_is_refused owns the 400-then-200 recovery.
    # Only tinylaya's head reads a window wide enough for 64 to be too small: tinyopenjev answers the
    # same request at that batch, so the pair is only pinned where the fault can be produced
    if preset == "tinylaya":
        server.stop()
        server.decision_max_queued = -1
        server.n_ubatch = 64
        server.start()
        contract_of(post(questions=WIDE_CHOICE), 400, "exceed_batch_size_error")


def test_systemone_not_a_decision_model_is_501():
    """a server holding a model that cannot decide answers refuses the route with 501.

    This is the one 501 the body cannot ask for: "model" in the request is accepted and ignored, so
    the fault is a property of the loaded model and can only be produced by loading another one.
    """
    global server
    server = ServerPreset.tinyllama2()
    server.start()

    res = post(questions=TEST_QUESTIONS)
    assert res.status_code == 501
    assert res.body["error"]["type"] == ERROR_CONTRACT[501]
    assert res.body["error"]["code"] == 501
    assert set(res.body.keys()) == {"error"}


# tinylaya's head reads a 192 token window, so a --ubatch-size below it cannot hold the answer
WIDE_CHOICE = {
    "q": {
        "type": "choice",
        "instructions": "which of these twenty options describes the ticket",
        "criteria": {f"team_{i:02d}": "a description long enough to take real tokens to say"
                     for i in range(20)},
    },
}


def burst_questions(n_questions: int) -> dict:
    """n noul questions for the admission tests: one decision task each, and grouping preserves the
    total, so the queued weight of the request is exactly n_questions"""
    return {
        f"q{i}": {
            "type": "noul",
            "instructions": f"Is statement {i} about this ticket true?",
        }
        for i in range(n_questions)
    }


def fire_burst(server: ServerProcess, n_burst: int, questions: dict) -> list:
    def send():
        return server.make_request("POST", "/v1/systemone",
                                   data={"state": TEST_STATE, "questions": questions},
                                   timeout=DEFAULT_REQUEST_TIMEOUT)

    with ThreadPoolExecutor(max_workers=n_burst) as pool:
        return list(pool.map(lambda _: send(), range(n_burst)))


def test_systemone_idle_server_is_never_refused():
    """the control group of the admission cap: an idle server answers every request it is given.

    Without this, the burst test above would pass even if the gate refused everything.
    """
    global server
    server = ServerPreset.tinylaya()
    server.server_metrics = True
    server.start()  # the derived cap

    # the queue drains between sequential requests, so the depth is 0 every time the gate is read
    for _ in range(2 * server.n_slots):
        assert post(questions=TEST_QUESTIONS).status_code == 200

    # a burst that all arrives at once: every request is admitted while the queue is still below the
    # derived cap, and the cap is the sum of a whole number of request weights
    per_request = 4
    n_requests = (8 * server.n_slots) // per_request
    responses = fire_burst(server, n_requests, burst_questions(per_request))
    assert [r.status_code for r in responses] == [200] * n_requests, \
        [r.status_code for r in responses]

    assert read_metrics(server)["llamacpp:decision_requests_refused_total"] == 0
