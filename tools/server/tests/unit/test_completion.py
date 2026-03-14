import pytest
import requests
import time
import random

from openai import OpenAI
from utils import *

server = ServerPreset.tinyllama2()

JSON_MULTIMODAL_KEY = "multimodal_data"
JSON_PROMPT_STRING_KEY = "prompt_string"

@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.tinyllama2()

@pytest.mark.parametrize("prompt,n_predict,re_content,n_prompt,n_predicted,truncated,return_tokens", [
    ("I believe the meaning of life is", 8, "(going|bed)+", 18, 8, False, False),
    ("Write a joke about AI from a very long prompt which will not be truncated", 64, "(princesses|everyone|kids|Anna|forest)+", 46, 64, False, True),
])
def test_completion(prompt: str, n_predict: int, re_content: str, n_prompt: int, n_predicted: int, truncated: bool, return_tokens: bool):
    global server
    server.start()
    res = server.make_request("POST", "/completion", data={
        "n_predict": n_predict,
        "prompt": prompt,
        "return_tokens": return_tokens,
    })
    assert res.status_code == 200
    assert res.body["timings"]["prompt_n"] == n_prompt
    assert res.body["timings"]["predicted_n"] == n_predicted
    assert res.body["truncated"] == truncated
    assert type(res.body["has_new_line"]) == bool
    assert match_regex(re_content, res.body["content"])
    if return_tokens:
        assert len(res.body["tokens"]) > 0
        assert all(type(tok) == int for tok in res.body["tokens"])
    else:
        assert res.body["tokens"] == []


@pytest.mark.parametrize("prompt,n_predict,re_content,n_prompt,n_predicted,truncated", [
    ("I believe the meaning of life is", 8, "(going|bed)+", 18, 8, False),
    ("Write a joke about AI from a very long prompt which will not be truncated", 64, "(princesses|everyone|kids|Anna|forest)+", 46, 64, False),
])
def test_completion_stream(prompt: str, n_predict: int, re_content: str, n_prompt: int, n_predicted: int, truncated: bool):
    global server
    server.start()
    res = server.make_stream_request("POST", "/completion", data={
        "n_predict": n_predict,
        "prompt": prompt,
        "stream": True,
    })
    content = ""
    for data in res:
        assert "stop" in data and type(data["stop"]) == bool
        if data["stop"]:
            assert data["timings"]["prompt_n"] == n_prompt
            assert data["timings"]["predicted_n"] == n_predicted
            assert data["truncated"] == truncated
            assert data["stop_type"] == "limit"
            assert type(data["has_new_line"]) == bool
            assert "generation_settings" in data
            assert server.n_predict is not None
            assert data["generation_settings"]["n_predict"] == min(n_predict, server.n_predict)
            assert data["generation_settings"]["seed"] == server.seed
            assert match_regex(re_content, content)
        else:
            assert len(data["tokens"]) > 0
            assert all(type(tok) == int for tok in data["tokens"])
            content += data["content"]


def test_completion_stream_vs_non_stream():
    global server
    server.start()
    res_stream = server.make_stream_request("POST", "/completion", data={
        "n_predict": 8,
        "prompt": "I believe the meaning of life is",
        "stream": True,
    })
    res_non_stream = server.make_request("POST", "/completion", data={
        "n_predict": 8,
        "prompt": "I believe the meaning of life is",
    })
    content_stream = ""
    for data in res_stream:
        content_stream += data["content"]
    assert content_stream == res_non_stream.body["content"]


def test_completion_with_openai_library():
    global server
    server.start()
    client = OpenAI(api_key="dummy", base_url=f"http://{server.server_host}:{server.server_port}/v1")
    res = client.completions.create(
        model="davinci-002",
        prompt="I believe the meaning of life is",
        max_tokens=8,
    )
    assert res.system_fingerprint is not None and res.system_fingerprint.startswith("b")
    assert res.choices[0].finish_reason == "length"
    assert res.choices[0].text is not None
    assert match_regex("(going|bed)+", res.choices[0].text)


def test_completion_stream_with_openai_library():
    global server
    server.start()
    client = OpenAI(api_key="dummy", base_url=f"http://{server.server_host}:{server.server_port}/v1")
    res = client.completions.create(
        model="davinci-002",
        prompt="I believe the meaning of life is",
        max_tokens=8,
        stream=True,
    )
    output_text = ''
    for data in res:
        choice = data.choices[0]
        if choice.finish_reason is None:
            assert choice.text is not None
            output_text += choice.text
    assert match_regex("(going|bed)+", output_text)


# Test case from https://github.com/ggml-org/llama.cpp/issues/13780
@pytest.mark.slow
def test_completion_stream_with_openai_library_stops():
    global server
    server.model_hf_repo = "bartowski/Phi-3.5-mini-instruct-GGUF:Q4_K_M"
    server.model_hf_file = None
    server.start()
    client = OpenAI(api_key="dummy", base_url=f"http://{server.server_host}:{server.server_port}/v1")
    res = client.completions.create(
        model="davinci-002",
        prompt="System: You are helpfull assistant.\nAssistant:\nHey! How could I help?\nUser:\nTell me a joke.\nAssistant:\n",
        stop=["User:\n", "Assistant:\n"],
        max_tokens=200,
        stream=True,
    )
    output_text = ''
    for data in res:
        choice = data.choices[0]
        if choice.finish_reason is None:
            assert choice.text is not None
            output_text += choice.text
    assert match_regex("Sure, here's one for[\\s\\S]*", output_text), f'Unexpected output: {output_text}'


@pytest.mark.parametrize("n_slots", [1, 2])
def test_consistent_result_same_seed(n_slots: int):
    global server
    server.n_slots = n_slots
    server.start()
    last_res = None
    for _ in range(4):
        res = server.make_request("POST", "/completion", data={
            "prompt": "I believe the meaning of life is",
            "seed": 42,
            "temperature": 0.0,
            "cache_prompt": False,  # TODO: remove this once test_cache_vs_nocache_prompt is fixed
        })
        if last_res is not None:
            assert res.body["content"] == last_res.body["content"]
        last_res = res


@pytest.mark.parametrize("n_slots", [1, 2])
def test_different_result_different_seed(n_slots: int):
    global server
    server.n_slots = n_slots
    server.start()
    last_res = None
    for seed in range(4):
        res = server.make_request("POST", "/completion", data={
            "prompt": "I believe the meaning of life is",
            "seed": seed,
            "temperature": 1.0,
            "cache_prompt": False,  # TODO: remove this once test_cache_vs_nocache_prompt is fixed
        })
        if last_res is not None:
            assert res.body["content"] != last_res.body["content"]
        last_res = res

# TODO figure why it don't work with temperature = 1
# @pytest.mark.parametrize("temperature", [0.0, 1.0])
@pytest.mark.parametrize("n_batch", [16, 32])
@pytest.mark.parametrize("temperature", [0.0])
def test_consistent_result_different_batch_size(n_batch: int, temperature: float):
    global server
    server.n_batch = n_batch
    server.start()
    last_res = None
    for _ in range(4):
        res = server.make_request("POST", "/completion", data={
            "prompt": "I believe the meaning of life is",
            "seed": 42,
            "temperature": temperature,
            "cache_prompt": False,  # TODO: remove this once test_cache_vs_nocache_prompt is fixed
        })
        if last_res is not None:
            assert res.body["content"] == last_res.body["content"]
        last_res = res


@pytest.mark.skip(reason="This test fails on linux, need to be fixed")
def test_cache_vs_nocache_prompt():
    global server
    server.start()
    res_cache = server.make_request("POST", "/completion", data={
        "prompt": "I believe the meaning of life is",
        "seed": 42,
        "temperature": 1.0,
        "cache_prompt": True,
    })
    res_no_cache = server.make_request("POST", "/completion", data={
        "prompt": "I believe the meaning of life is",
        "seed": 42,
        "temperature": 1.0,
        "cache_prompt": False,
    })
    assert res_cache.body["content"] == res_no_cache.body["content"]


def test_nocache_long_input_prompt():
    global server
    server.start()
    res = server.make_request("POST", "/completion", data={
        "prompt": "I believe the meaning of life is"*32,
        "seed": 42,
        "temperature": 1.0,
        "cache_prompt": False,
    })
    assert res.status_code == 400

def test_json_prompt_no_mtmd():
    global server
    server.start()
    res = server.make_request("POST", "/completion", data={
        "prompt": { JSON_PROMPT_STRING_KEY: "I believe the meaning of life is" },
        "seed": 42,
        "temperature": 1.0,
        "cache_prompt": False,
    })
    assert res.status_code == 200

def test_json_prompt_mtm_error_when_not_supported():
    global server
    server.start()
    res = server.make_request("POST", "/completion", data={
        "prompt": { JSON_PROMPT_STRING_KEY: "I believe the meaning of life is <__media__>", JSON_MULTIMODAL_KEY: "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=" },
        "seed": 42,
        "temperature": 1.0,
        "cache_prompt": False,
    })
    # MTMD is disabled on this model, so this should fail.
    assert res.status_code != 200

def test_completion_with_tokens_input():
    global server
    server.temperature = 0.0
    server.start()
    prompt_str = "I believe the meaning of life is"
    res = server.make_request("POST", "/tokenize", data={
        "content": prompt_str,
        "add_special": True,
    })
    assert res.status_code == 200
    tokens = res.body["tokens"]

    # single completion
    res = server.make_request("POST", "/completion", data={
        "prompt": tokens,
    })
    assert res.status_code == 200
    assert type(res.body["content"]) == str

    # batch completion
    res = server.make_request("POST", "/completion", data={
        "prompt": [tokens, tokens],
    })
    assert res.status_code == 200
    assert type(res.body) == list
    assert len(res.body) == 2
    assert res.body[0]["content"] == res.body[1]["content"]

    # mixed string and tokens
    res = server.make_request("POST", "/completion", data={
        "prompt": [tokens, prompt_str],
    })
    assert res.status_code == 200
    assert type(res.body) == list
    assert len(res.body) == 2
    assert res.body[0]["content"] == res.body[1]["content"]

    # mixed JSON and tokens
    res = server.make_request("POST", "/completion", data={
        "prompt": [
            tokens,
            {
                JSON_PROMPT_STRING_KEY: "I believe the meaning of life is",
            },
        ],
    })
    assert res.status_code == 200
    assert type(res.body) == list
    assert len(res.body) == 2
    assert res.body[0]["content"] == res.body[1]["content"]

    # mixed string and tokens in one sequence
    res = server.make_request("POST", "/completion", data={
        "prompt": [1, 2, 3, 4, 5, 6, prompt_str, 7, 8, 9, 10, prompt_str],
    })
    assert res.status_code == 200
    assert type(res.body["content"]) == str


@pytest.mark.parametrize("n_slots,n_requests", [
    (1, 3),
    (2, 2),
    (2, 4),
    (4, 2), # some slots must be idle
    (4, 6),
])
def test_completion_parallel_slots(n_slots: int, n_requests: int):
    global server
    server.n_slots = n_slots
    server.temperature = 0.0
    server.start()

    PROMPTS = [
        ("Write a very long book.", "(very|special|big)+"),
        ("Write another a poem.", "(small|house)+"),
        ("What is LLM?", "(Dad|said)+"),
        ("The sky is blue and I love it.", "(climb|leaf)+"),
        ("Write another very long music lyrics.", "(friends|step|sky)+"),
        ("Write a very long joke.", "(cat|Whiskers)+"),
    ]
    def check_slots_status():
        should_all_slots_busy = n_requests >= n_slots
        time.sleep(0.1)
        res = server.make_request("GET", "/slots")
        n_busy = sum([1 for slot in res.body if slot["is_processing"]])
        if should_all_slots_busy:
            assert n_busy == n_slots
        else:
            assert n_busy <= n_slots

    tasks = []
    for i in range(n_requests):
        prompt, re_content = PROMPTS[i % len(PROMPTS)]
        tasks.append((server.make_request, ("POST", "/completion", {
            "prompt": prompt,
            "seed": 42,
            "temperature": 1.0,
        })))
    tasks.append((check_slots_status, ()))
    results = parallel_function_calls(tasks)

    # check results
    for i in range(n_requests):
        prompt, re_content = PROMPTS[i % len(PROMPTS)]
        res = results[i]
        assert res.status_code == 200
        assert type(res.body["content"]) == str
        assert len(res.body["content"]) > 10
        # FIXME: the result is not deterministic when using other slot than slot 0
        # assert match_regex(re_content, res.body["content"])


@pytest.mark.parametrize(
    "n_ctx,n_slots,n_predict_vals,expected_success",
    [
        (256, 4, [80, 40, 80, 80], [True,  True,  True,  True]),
        (256, 4, [70, 70, 70, 70], [False, False, False, False]),
        (256, 4, [90, 90, 40, 90], [False, False, True,  False]),
        (256, 4, [90, 90, 40, 75], [True,  True,  True,  True]),
    ],
)
def test_completion_unified(n_ctx, n_slots, n_predict_vals, expected_success):
    global server
    server.n_slots = n_slots
    server.kv_unified = True
    server.n_ctx = n_ctx
    server.start()
    prompt = "A"
    tasks = []
    for n_predict in n_predict_vals:
        tasks.append((server.make_request, ("POST", "/completion", {"prompt": prompt, "n_predict": n_predict})))
    results = parallel_function_calls(tasks)
    for res, n_predict, expect_ok in zip(results, n_predict_vals, expected_success):
        if expect_ok:
            assert res.status_code == 200

        # note: https://github.com/ggml-org/llama.cpp/pull/18700#issuecomment-3728695581
        if res.status_code == 200:
            assert "content" in res.body
            if "timings" in res.body:
                assert res.body["timings"]["predicted_n"] == n_predict


@pytest.mark.parametrize(
    "prompt,n_predict,response_fields",
    [
        ("I believe the meaning of life is", 8, []),
        ("I believe the meaning of life is", 32, ["content", "generation_settings/n_predict", "prompt"]),
    ],
)
def test_completion_response_fields(
    prompt: str, n_predict: int, response_fields: list[str]
):
    global server
    server.start()
    res = server.make_request(
        "POST",
        "/completion",
        data={
            "n_predict": n_predict,
            "prompt": prompt,
            "response_fields": response_fields,
        },
    )
    assert res.status_code == 200
    assert "content" in res.body
    assert len(res.body["content"])
    if len(response_fields):
        assert res.body["generation_settings/n_predict"] == n_predict
        assert res.body["prompt"] == "<s> " + prompt
        assert isinstance(res.body["content"], str)
        assert len(res.body) == len(response_fields)
    else:
        assert len(res.body)
        assert "generation_settings" in res.body


def test_n_probs():
    global server
    server.start()
    res = server.make_request("POST", "/completion", data={
        "prompt": "I believe the meaning of life is",
        "n_probs": 10,
        "temperature": 0.0,
        "n_predict": 5,
    })
    assert res.status_code == 200
    assert "completion_probabilities" in res.body
    assert len(res.body["completion_probabilities"]) == 5
    for tok in res.body["completion_probabilities"]:
        assert "id" in tok and tok["id"] > 0
        assert "token" in tok and type(tok["token"]) == str
        assert "logprob" in tok and tok["logprob"] <= 0.0
        assert "bytes" in tok and type(tok["bytes"]) == list
        assert len(tok["top_logprobs"]) == 10
        for prob in tok["top_logprobs"]:
            assert "id" in prob and prob["id"] > 0
            assert "token" in prob and type(prob["token"]) == str
            assert "logprob" in prob and prob["logprob"] <= 0.0
            assert "bytes" in prob and type(prob["bytes"]) == list


def test_n_probs_stream():
    global server
    server.start()
    res = server.make_stream_request("POST", "/completion", data={
        "prompt": "I believe the meaning of life is",
        "n_probs": 10,
        "temperature": 0.0,
        "n_predict": 5,
        "stream": True,
    })
    for data in res:
        if data["stop"] == False:
            assert "completion_probabilities" in data
            assert len(data["completion_probabilities"]) == 1
            for tok in data["completion_probabilities"]:
                assert "id" in tok and tok["id"] > 0
                assert "token" in tok and type(tok["token"]) == str
                assert "logprob" in tok and tok["logprob"] <= 0.0
                assert "bytes" in tok and type(tok["bytes"]) == list
                assert len(tok["top_logprobs"]) == 10
                for prob in tok["top_logprobs"]:
                    assert "id" in prob and prob["id"] > 0
                    assert "token" in prob and type(prob["token"]) == str
                    assert "logprob" in prob and prob["logprob"] <= 0.0
                    assert "bytes" in prob and type(prob["bytes"]) == list


def test_n_probs_post_sampling():
    global server
    server.start()
    res = server.make_request("POST", "/completion", data={
        "prompt": "I believe the meaning of life is",
        "n_probs": 10,
        "temperature": 0.0,
        "n_predict": 5,
        "post_sampling_probs": True,
    })
    assert res.status_code == 200
    assert "completion_probabilities" in res.body
    assert len(res.body["completion_probabilities"]) == 5
    for tok in res.body["completion_probabilities"]:
        assert "id" in tok and tok["id"] > 0
        assert "token" in tok and type(tok["token"]) == str
        assert "prob" in tok and 0.0 < tok["prob"] <= 1.0
        assert "bytes" in tok and type(tok["bytes"]) == list
        assert len(tok["top_probs"]) == 10
        for prob in tok["top_probs"]:
            assert "id" in prob and prob["id"] > 0
            assert "token" in prob and type(prob["token"]) == str
            assert "prob" in prob and 0.0 <= prob["prob"] <= 1.0
            assert "bytes" in prob and type(prob["bytes"]) == list
        # because the test model usually output token with either 100% or 0% probability, we need to check all the top_probs
        assert any(prob["prob"] == 1.0 for prob in tok["top_probs"])


@pytest.mark.parametrize("tokenize,openai_style", [(False, False), (False, True), (True, False), (True, True)])
def test_logit_bias(tokenize, openai_style):
    global server
    server.start()

    exclude = ["i", "I", "the", "The", "to", "a", "an", "be", "is", "was", "but", "But", "and", "And", "so", "So", "you", "You", "he", "He", "she", "She", "we", "We", "they", "They", "it", "It", "his", "His", "her", "Her", "book", "Book"]

    logit_bias = []
    if tokenize:
        res = server.make_request("POST", "/tokenize", data={
            "content": " " + " ".join(exclude) + " ",
        })
        assert res.status_code == 200
        tokens = res.body["tokens"]
        logit_bias = [[tok, -100] for tok in tokens]

    else:
        logit_bias = [[" " + tok + " ", -100] for tok in exclude]

    if openai_style:
        logit_bias = {el[0]: -100 for el in logit_bias}

    res = server.make_request("POST", "/completion", data={
        "n_predict": 64,
        "prompt": "What is the best book",
        "logit_bias": logit_bias,
        "temperature": 0.0
    })
    assert res.status_code == 200
    output_text = res.body["content"]
    assert all(output_text.find(" " + tok + " ") == -1 for tok in exclude)


def test_cancel_request():
    global server
    server.n_ctx = 4096
    server.n_predict = -1
    server.n_slots = 1
    server.server_slots = True
    server.start()
    # send a request that will take a long time, but cancel it before it finishes
    try:
        server.make_request("POST", "/completion", data={
            "prompt": "I believe the meaning of life is",
        }, timeout=0.1)
    except requests.exceptions.ReadTimeout:
        pass # expected
    # make sure the slot is free
    time.sleep(1) # wait for HTTP_POLLING_SECONDS
    res = server.make_request("GET", "/slots")
    assert res.body[0]["is_processing"] == False


# this test exercises the host-memory prompt cache
# ref: https://github.com/ggml-org/llama.cpp/pull/16391
# ref: https://github.com/ggml-org/llama.cpp/pull/17078
def test_completion_prompt_cache():
    global server
    server.n_slots = 2
    server.kv_unified = True
    server.start()

    for _ in range(16):
        # generate alternating random prompts with variable lengths in order to get them in and out of the cache
        r = random.randint(0, 4)
        prompt = (" Hello " +  str(r)) * (40 + r)
        n_prompt = (40 + r)*5 + 2
        n_predict = random.randint(1, 8)

        res = server.make_request(
            "POST",
            "/completion",
            data={
                "prompt": prompt,
                "n_predict": n_predict,
            },
        )

        assert res.status_code == 200
        assert "content" in res.body
        content = res.body["content"]
        assert isinstance(content, str)
        assert len(content) > 0

        assert type(res.body["has_new_line"]) == bool
        assert "timings" in res.body
        timings = res.body["timings"]

        assert "prompt_n" in timings and timings["prompt_n"] + timings["cache_n"] == n_prompt
        assert "predicted_n" in timings and timings["predicted_n"] == n_predict
        assert "tokens" in res.body and isinstance(res.body["tokens"], list)
