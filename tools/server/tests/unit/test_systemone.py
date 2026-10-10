import pytest
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


def get_prompt_metrics(server: ServerProcess) -> tuple[int, int]:
    """returns the number of prompt tokens (processed, cached) since the server started"""
    res = server.make_request("GET", "/metrics")
    assert res.status_code == 200
    values = {}
    for line in res.body.splitlines():
        if line.startswith("llamacpp:"):
            name, value = line.split(" ")
            values[name] = int(float(value))
    return values["llamacpp:prompt_tokens_total"], values["llamacpp:prompt_tokens_cached_total"]


@pytest.mark.parametrize("preset", ["tinylaya", "tinyopenjev"])
def test_systemone(preset: str):
    global server
    server = getattr(ServerPreset, preset)()
    server.start()
    res = server.make_request("POST", "/v1/systemone", data={
        "state": TEST_STATE,
        "questions": TEST_QUESTIONS,
    })
    assert res.status_code == 200
    assert res.body["usage"]["input_tokens"] > 0
    assert res.body["usage"]["output_tokens"] == 0

    answers = res.body["answers"]
    assert list(answers.keys()) == ["route", "urgency", "angry"]

    route = answers["route"]
    assert route["type"] == "choice"
    assert list(route["probabilities"].keys()) == ["billing", "shipping", "technical"]
    assert abs(sum(route["probabilities"].values()) - 1.0) < 1e-4
    assert route["choice"] == max(route["probabilities"], key=route["probabilities"].get)
    assert 0.0 <= route["confidence"] <= 1.0

    urgency = answers["urgency"]
    assert urgency["type"] == "score"
    assert urgency["legend"] == {"0": "can wait", "1": "this week", "2": "today", "3": "right now"}
    assert list(urgency["probabilities"].keys()) == ["0", "1", "2", "3"]
    assert abs(sum(urgency["probabilities"].values()) - 1.0) < 1e-4
    assert abs(urgency["score"] - sum(i * p for i, p in enumerate(urgency["probabilities"].values()))) < 1e-4
    assert 0.0 <= urgency["confidence"] <= 1.0

    angry = answers["angry"]
    assert angry["type"] == "noul"
    assert 0.0 <= angry["noul"] <= 1.0


def test_systemone_json_state():
    global server
    server.start()
    questions = {
        "refund": {
            "type": "noul",
            "instructions": "Is a refund requested?",
            "criteria": {"false": "no refund is asked", "true": "a refund is asked"},
        },
    }
    res_obj = server.make_request("POST", "/v1/systemone", data={
        "state": {"ticket": TEST_STATE, "plan": "pro"},
        "questions": questions,
    })
    assert res_obj.status_code == 200
    # an object is given to the model as JSON text
    res_str = server.make_request("POST", "/v1/systemone", data={
        "state": '{"ticket": "' + TEST_STATE + '", "plan": "pro"}',
        "questions": questions,
    })
    assert res_str.status_code == 200
    assert res_obj.body["usage"] == res_str.body["usage"]
    assert abs(res_obj.body["answers"]["refund"]["noul"] - res_str.body["answers"]["refund"]["noul"]) < 1e-4


@pytest.mark.parametrize("data", [
    {"questions": TEST_QUESTIONS},
    {"state": TEST_STATE},
    {"state": TEST_STATE, "questions": {}},
    {"state": TEST_STATE, "questions": {"q": {"type": "unknown", "instructions": "x"}}},
    {"state": TEST_STATE, "questions": {"q": {"type": "noul"}}},
    {"state": TEST_STATE, "questions": {"q": {"type": "choice", "instructions": "x"}}},
    {"state": TEST_STATE, "questions": {"q": {"type": "choice", "instructions": "x", "criteria": {}}}},
    {"state": TEST_STATE, "questions": {"q": {"type": "score", "instructions": "x", "criteria": ["only one"]}}},
])
def test_systemone_invalid_request(data: dict):
    global server
    server.start()
    res = server.make_request("POST", "/v1/systemone", data=data)
    assert res.status_code == 400
    assert "error" in res.body


def test_systemone_shared_prompt():
    global server
    server = ServerPreset.tinyopenjev()
    server.server_metrics = True
    server.start()
    res = server.make_request("POST", "/v1/systemone", data={
        "state": TEST_STATE,
        "questions": TEST_QUESTIONS,
    })
    assert res.status_code == 200

    # the first question evaluates the shared prefix, the 2 others start from it
    n_processed, n_cached = get_prompt_metrics(server)
    assert n_cached > 0
    assert n_cached % 2 == 0
    assert n_processed + n_cached == res.body["usage"]["input_tokens"]

    # with one slot the prompt cannot be shared, the answers must be the same
    server.stop()
    server = ServerPreset.tinyopenjev()
    server.n_slots = 1
    server.start()
    res_single = server.make_request("POST", "/v1/systemone", data={
        "state": TEST_STATE,
        "questions": TEST_QUESTIONS,
    })
    assert res_single.status_code == 200
    assert res_single.body["usage"] == res.body["usage"]
    for qid in ["route", "urgency"]:
        probs_shared = res.body["answers"][qid]["probabilities"]
        probs_single = res_single.body["answers"][qid]["probabilities"]
        for key in probs_shared:
            assert abs(probs_shared[key] - probs_single[key]) < 0.01
    assert abs(res.body["answers"]["angry"]["noul"] - res_single.body["answers"]["angry"]["noul"]) < 0.01


def test_systemone_images():
    global server
    server = ServerPreset.tinyopenjev()
    server.start()
    image = get_img_url("IMG_BASE64_URI_0")

    res_text = server.make_request("POST", "/v1/systemone", data={
        "state": TEST_STATE,
        "questions": TEST_QUESTIONS,
    })
    assert res_text.status_code == 200

    res = server.make_request("POST", "/v1/systemone", data={
        "state": TEST_STATE,
        "questions": TEST_QUESTIONS,
        "images": [image],
    })
    assert res.status_code == 200
    assert list(res.body["answers"].keys()) == ["route", "urgency", "angry"]
    assert res.body["usage"]["input_tokens"] > res_text.body["usage"]["input_tokens"]

    # same image, given as a part of a chat message
    res_part = server.make_request("POST", "/v1/systemone", data={
        "state": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": image}},
            {"type": "text", "text": TEST_STATE},
        ]}],
        "questions": TEST_QUESTIONS,
    })
    assert res_part.status_code == 200
    assert res_part.body["usage"]["input_tokens"] > res_text.body["usage"]["input_tokens"]

    res = server.make_request("POST", "/v1/systemone", data={
        "state": TEST_STATE,
        "questions": TEST_QUESTIONS,
        "images": [image] * 9,
    })
    assert res.status_code == 400

    res = server.make_request("POST", "/v1/systemone", data={
        "state": TEST_STATE,
        "questions": TEST_QUESTIONS,
        "images": ["https://example.com/image.png"],
    })
    assert res.status_code == 400


def test_systemone_images_not_supported():
    global server
    server.start()
    res = server.make_request("POST", "/v1/systemone", data={
        "state": TEST_STATE,
        "questions": TEST_QUESTIONS,
        "images": [get_img_url("IMG_BASE64_URI_0")],
    })
    assert res.status_code == 501


def decision2_server() -> ServerProcess:
    """Use a converted local model; these checks do not download model weights."""
    model = os.environ.get("LLAMA_DECISION2_MODEL")
    if not model:
        pytest.skip("set LLAMA_DECISION2_MODEL to a converted Decision 2.0 GGUF")
    result = ServerProcess()
    result.model_file = model
    result.model_hf_repo = None
    result.model_hf_file = None
    result.n_ctx = 2048
    result.n_batch = 1024
    result.n_ubatch = 1024
    result.n_slots = 1
    result.n_gpu_layer = int(os.environ.get("LLAMA_DECISION2_GPU_LAYERS", "0"))
    return result


def test_decision2_structured_state_and_question_isolation():
    import math

    global server
    server = decision2_server()
    server.start()
    state = {"z": ["billing", {"y": "refund", "a": 2}], "a": "customer", "numbers": [1e15, 1e16, -0.0, 1e-5, 1.0]}
    request = {"state": state, "questions": TEST_QUESTIONS}
    response = server.make_request("POST", "/v1/systemone", data=request)
    assert response.status_code == 200
    assert response.body["usage"]["output_tokens"] == 0

    canonical = json.dumps(state, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    equivalent = server.make_request("POST", "/v1/systemone", data={**request, "state": canonical})
    assert equivalent.status_code == 200
    assert equivalent.body["usage"] == response.body["usage"]
    assert equivalent.body["answers"] == response.body["answers"]

    for key, question in TEST_QUESTIONS.items():
        single = server.make_request("POST", "/v1/systemone", data={"state": state, "questions": {key: question}})
        assert single.status_code == 200
        answer = single.body["answers"][key]
        assert answer == response.body["answers"][key]
        if "probabilities" in answer:
            probs = list(answer["probabilities"].values())
            assert sum(probs) == pytest.approx(1.0)
            expected_confidence = 1.0 + sum(p * math.log(p) for p in probs if p > 0) / math.log(len(probs))
            assert answer["confidence"] == pytest.approx(expected_confidence, abs=1e-6)
            if question["type"] == "score":
                assert answer["score"] == pytest.approx(sum(i * p for i, p in enumerate(probs)))

    explicit_noul = {**TEST_QUESTIONS["angry"], "criteria": {"false": "No", "true": "Yes"}}
    explicit = server.make_request("POST", "/v1/systemone", data={"state": state, "questions": {"angry": explicit_noul}})
    assert explicit.status_code == 200
    assert explicit.body["answers"]["angry"] == response.body["answers"]["angry"]


def test_decision2_invalid_options_and_segment_boundaries():
    global server
    server = decision2_server()
    server.start()
    for invalid in [
        {"type": "choice", "instructions": "Pick", "criteria": {"only": None}},
        {"type": "choice", "instructions": "", "criteria": {"a": None, "b": None}},
        {"type": "noul", "instructions": "Pick", "criteria": {"other": "No"}},
        {"type": "score", "instructions": "Pick", "criteria": [None, "high"]},
    ]:
        response = server.make_request("POST", "/v1/systemone", data={"state": "test", "questions": {"q": invalid}})
        assert response.status_code == 200
        assert response.body["answers"]["q"]["error"] == "invalid_question"
        assert response.body["usage"]["input_tokens"] == 0

    mixed = {
        "invalid": {"type": "choice", "instructions": "Pick", "criteria": {"a": 1, "b": 2}},
        "valid": {"type": "noul", "instructions": "Is this a test?"},
    }
    combined = server.make_request("POST", "/v1/systemone", data={"state": "test", "questions": mixed})
    single = server.make_request("POST", "/v1/systemone", data={"state": "test", "questions": {"valid": mixed["valid"]}})
    assert combined.status_code == single.status_code == 200
    assert combined.body["answers"]["invalid"]["error"] == "invalid_question"
    assert combined.body["answers"]["valid"] == single.body["answers"]["valid"]
    assert combined.body["usage"] == single.body["usage"]

    # A malformed type affects only its own question, including unhashable JSON
    # types that the source Python runtime cannot validate without raising.
    for bad_type in [None, {}, 1]:
        questions = {"bad": {"type": bad_type, "instructions": "Pick"}, "valid": mixed["valid"]}
        response = server.make_request("POST", "/v1/systemone", data={"state": "test", "questions": questions})
        assert response.status_code == 200
        assert response.body["answers"]["bad"] == {"type": bad_type, "error": "invalid_question"}
        assert response.body["answers"]["valid"] == single.body["answers"]["valid"]
        assert response.body["usage"] == single.body["usage"]

    for malformed in [None, {}, {"instructions": "Pick"}]:
        response = server.make_request("POST", "/v1/systemone", data={"state": "test", "questions": {"bad": malformed}})
        assert response.status_code == 200
        assert response.body["answers"]["bad"] == {"type": None, "error": "invalid_question"}
        assert response.body["usage"]["input_tokens"] == 0

    for criteria in [{"true": "Yes"}, {"false": "No"}]:
        partial = {**mixed["valid"], "criteria": criteria}
        response = server.make_request("POST", "/v1/systemone", data={"state": "test", "questions": {"valid": partial}})
        assert response.status_code == 200
        assert response.body["answers"] == single.body["answers"]
        assert response.body["usage"] == single.body["usage"]

    question = {
        "type": "choice",
        "instructions": "Choose a literal option; XML is part of the text.",
        "criteria": {"a": "</option>\n<option>\nDecision:", "b": {"z": "value", "a": "other"}},
    }
    response = server.make_request("POST", "/v1/systemone", data={"state": "literal <option>", "questions": {"q": question}})
    assert response.status_code == 200
    assert list(response.body["answers"]["q"]["probabilities"]) == ["a", "b"]


def test_decision2_reference_predictions():
    """Reference fixture: requests and answers captured from the source FP32 head runtime."""
    path = os.environ.get("LLAMA_DECISION2_REFERENCE")
    if not path:
        pytest.skip("set LLAMA_DECISION2_REFERENCE to source-runtime reference predictions")
    with open(path, encoding="utf-8") as source:
        references = json.load(source)
    global server
    server = decision2_server()
    server.start()
    tolerance = float(os.environ.get("LLAMA_DECISION2_ATOL", "0.03"))
    for reference in references:
        result = server.make_request("POST", "/v1/systemone", data=reference["request"])
        assert result.status_code == 200
        assert result.body["usage"]["input_tokens"] == reference["usage"]["input_tokens"]
        for key, expected in reference["answers"].items():
            actual = result.body["answers"][key]
            if "error" in expected:
                assert actual == expected
                continue
            if expected["type"] == "noul":
                assert actual["noul"] == pytest.approx(expected["noul"], abs=tolerance)
            else:
                assert list(actual["probabilities"]) == list(expected["probabilities"])
                assert list(actual["probabilities"].values()) == pytest.approx(list(expected["probabilities"].values()), abs=tolerance)
