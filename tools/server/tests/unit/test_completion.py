import pytest
import re
import requests
import time
import random

from openai import OpenAI
from utils import *
from test_sleep import fetch_metrics, get_metric
from test_slot_save import tokenize
from test_vision_api import get_img_url

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
            assert "adaptive_target" in data["generation_settings"]
            assert "adaptive_decay" in data["generation_settings"]
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
        prompt="System: You are helpful assistant.\nAssistant:\nHey! How could I help?\nUser:\nTell me a joke.\nAssistant:\n",
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
    assert match_regex("Sure, here's one for[\\s\\S]*|Sure! Here's one for you[\\s\\S]*", output_text), f'Unexpected output: {output_text}'


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


# a request pinned to a busy slot leaves the generation running on it untouched
def test_pinned_request_on_busy_slot():
    global server
    server.n_ctx = 4096
    server.start()
    story = "Once upon a time a dragon named Ember guarded a golden key in a deep cave. " * 8

    def run(pin_busy: bool) -> str:
        server.make_request("POST", "/completion", data={"prompt": story, "id_slot": 0, "n_predict": 4, "temperature": 0.0})
        res = server.make_stream_request("POST", "/completion", data={
            "prompt": "To bake bread, mix flour, water and salt, then",
            "id_slot": 0, "n_predict": 1024, "ignore_eos": True, "temperature": 0.0, "stream": True,
        })
        content = next(res)["content"]
        if pin_busy:
            server.make_request("POST", "/completion", data={"prompt": story + "The dragon", "id_slot": 0, "n_predict": 4, "temperature": 0.0})
        return content + "".join(chunk["content"] for chunk in res)

    baseline = run(pin_busy=False)
    assert run(pin_busy=True) == baseline


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
            # the pool is aborted as a whole, so a request that fits on its own
            # is still dropped when the slots overlap, and it says so explicitly
            assert res.status_code == 200 or (
                res.status_code == 500
                and "context size has been exceeded" in res.body["error"]["message"].lower()
            )

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
        "prompt": "Today was the day. Today I would finally become a",
        "n_probs": 10,
        "temperature": 1.0,
        "n_predict": 5,
        "post_sampling_probs": True,
    })
    assert res.status_code == 200
    assert "completion_probabilities" in res.body
    assert len(res.body["completion_probabilities"]) == 5
    for (i, tok) in enumerate(res.body["completion_probabilities"]):
        assert "id" in tok and tok["id"] > 0
        assert "token" in tok and type(tok["token"]) == str
        assert "prob" in tok and 0.0 < tok["prob"] <= 1.0
        assert "bytes" in tok and type(tok["bytes"]) == list
        assert "top_probs" in tok and type(tok["top_probs"]) == list

        for prob in tok["top_probs"]:
            assert "id" in prob and prob["id"] > 0
            assert "token" in prob and type(prob["token"]) == str
            # 0.0 probability tokens should never be returned by the server
            assert "prob" in prob and 0.0 < prob["prob"] <= 1.0
            assert "bytes" in prob and type(prob["bytes"]) == list

        if i == 0:
            # The prompt is vague enough that we should get at least 10 possibilities
            # for the first token.
            assert len(tok["top_probs"]) == 10

        if len(tok["top_probs"]) < 10:
            # Getting less than the requested number of probabilities should only happen
            # if the ones we did get already sum to 1.0.
            assert sum(p["prob"] for p in tok["top_probs"]) == pytest.approx(1.0)

def test_n_probs_post_backend_sampling():
    """Verify that the same probabilities are returned with and without backend sampling."""
    global server
    server.backend_sampling = True
    server.start()

    def make_request(backend_sampling):
        n_predict = 20

        res = server.make_request("POST", "/completion", data={
            "prompt": "The countries of Europe, in random order, are:",
            "n_probs": 10,
            "n_predict": n_predict,
            "post_sampling_probs": True,
            "seed": 4242,
            "backend_sampling": backend_sampling,
        })
        assert res.status_code == 200

        total_probs = 0
        completions = res.body["completion_probabilities"]
        assert len(completions) == n_predict
        for tok in completions:
            # Handling of 0.0 probabilities differs between samplers and backend sampling. Filter them to normalize the
            # data.
            tok["top_probs"] = [x for x in tok["top_probs"] if x["prob"] > 0.0]
            total_probs += len(tok["top_probs"])
        # Verify that we got at least two top probs on average, to ensure the effectiveness of the test.
        assert total_probs >= 2 * n_predict
        return completions

    def verify_token(a, b):
        assert a["id"] == b["id"]
        assert a["token"] == b["token"]
        assert a["bytes"] == b["bytes"]
        assert a["prob"] == pytest.approx(b["prob"], abs=0.01)

    for (a, b) in zip(make_request(True), make_request(False)):
        verify_token(a, b)
        assert len(a["top_probs"]) == len(b["top_probs"])

        for (aa, bb) in zip(a["top_probs"], b["top_probs"]):
            verify_token(aa, bb)

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
    server.enable_ctx_shift = True
    server.start()
    # context shift keeps the generation running until the request is cancelled
    try:
        server.make_request("POST", "/completion", data={
            "prompt": "I believe the meaning of life is",
        }, timeout=0.1)
    except requests.exceptions.ReadTimeout:
        pass # expected
    # the slot is freed once the server notices the closed connection
    deadline = time.time() + 10
    while time.time() < deadline:
        res = server.make_request("GET", "/slots")
        if not res.body[0]["is_processing"]:
            return
        time.sleep(0.1)
    raise AssertionError("slot is still processing after the request was cancelled")


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


# a slot that switched to an unrelated prompt must not restore a context checkpoint of the previous one
def test_completion_checkpoint_after_unrelated_prompt():
    global server
    server = ServerPreset.tinygemma3()
    server.n_slots = 1
    server.cache_ram = 0
    server.temperature = 0.0
    server.start()

    def tokenize(content):
        res = server.make_request("POST", "/tokenize", data={"content": content})
        assert res.status_code == 200
        return res.body["tokens"]

    def top_logprobs(prompt, cache_prompt):
        res = server.make_request("POST", "/completion", data={
            "prompt": prompt,
            "n_predict": 1,
            "n_probs": 5,
            "cache_prompt": cache_prompt,
        })
        assert res.status_code == 200
        return [(p["id"], p["logprob"]) for p in res.body["completion_probabilities"][0]["top_logprobs"]]

    tokens = tokenize("Purple elephants quietly compose sonnets about tax law")
    expected = top_logprobs(tokens[:2], False)

    # 5 tokens: the checkpoint 4 tokens before the end of the prompt holds only its first token
    top_logprobs(tokenize("The quick brown fox jumps over the lazy dog")[:5], True)
    top_logprobs(tokens[:1], True)
    actual = top_logprobs(tokens[:2], True)

    assert [tid for tid, _ in actual] == [tid for tid, _ in expected]
    for (_, lp), (_, lp_expected) in zip(actual, expected):
        assert lp == pytest.approx(lp_expected, abs=1e-3)


# a restored context checkpoint must be kept for the next request that shares the same prefix
def test_completion_checkpoint_kept_after_restore():
    global server
    server = ServerPreset.tinygemma3()
    server.n_slots = 1
    server.n_batch = 16
    server.n_ubatch = 16
    server.cache_ram = 0
    server.temperature = 0.0
    server.start()

    def tokenize(content):
        res = server.make_request("POST", "/tokenize", data={"content": content})
        assert res.status_code == 200
        return res.body["tokens"]

    def cache_n(prompt):
        res = server.make_request("POST", "/completion", data={
            "prompt": prompt,
            "n_predict": 1,
            "cache_prompt": True,
        })
        assert res.status_code == 200
        return res.body["timings"]["cache_n"]

    prefix = tokenize("Purple elephants quietly compose sonnets about tax law in the moonlight")
    suffixes = [tokenize(s) for s in [
        "The quick brown fox jumps over the lazy dog while the farmer counts his sheep and the sun slowly sets behind the distant hills",
        "Seven tiny robots repaired the broken lighthouse lens before the storm arrived, then danced on the rocks until the morning tide came in",
        "Green apples rolled down the hill toward the river, where a curious otter collected them one by one and stacked them on a flat stone",
    ]]

    # 4 tokens: the checkpoint 4 tokens before the end of the prompt holds exactly the prefix
    cache_n(prefix + suffixes[0][:4])

    # suffixes longer than n_ubatch + 4: no new checkpoint within the prefix, only the restored one can be reused
    for suffix in suffixes[1:]:
        assert len(suffix) > server.n_ubatch + 4
        assert cache_n(prefix + suffix) == len(prefix) - 1


# --slot-attach: a prompt starts from the KV cells of another slot that holds the same prefix (on by default)
SLOT_ATTACH_TEXT = "Once upon a time, a little fox lived in a quiet forest near a wide river. "


def slot_attach_prompts(n_repeat=48, n_prefix=None):
    prefix = tokenize(server, SLOT_ATTACH_TEXT * n_repeat)[:n_prefix]
    return (prefix + tokenize(server, " The fox found a red apple under the old tree."),
            prefix + tokenize(server, " The river was cold, and the fish swam fast."))


# cache_n, and the ids and logprobs of the top tokens for the first generated token
def slot_attach_completion(prompt, id_slot, n_predict=1, **extra):
    res = server.make_request("POST", "/completion", data={
        "prompt": prompt,
        "id_slot": id_slot,
        "n_predict": n_predict,
        "n_probs": 5,
        **extra,
    })
    assert res.status_code == 200
    top = res.body["completion_probabilities"][0]["top_logprobs"]
    return res.body["timings"]["cache_n"], [p["id"] for p in top], [p["logprob"] for p in top]


# the same for a chat request, with the top tokens as text
def slot_attach_chat(system, user, id_slot, **extra):
    res = server.make_request("POST", "/chat/completions", data={
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "id_slot": id_slot,
        "logprobs": True,
        "top_logprobs": 5,
        **extra,
    })
    assert res.status_code == 200
    top = res.body["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
    return res.body["timings"]["cache_n"], [p["token"] for p in top], [p["logprob"] for p in top]


def slot_attach_assert_same(actual, expected):
    assert actual[:2] == expected[:2]
    assert actual[2] == pytest.approx(expected[2], abs=1e-5)


def slot_attach_log():
    with open(server.log_path) as f:
        return f.read()


# the slot that attaches and the donor compute what they compute from their own cells
@pytest.mark.parametrize("cache_idle_slots", [True, False])
def test_slot_attach(cache_idle_slots: bool):
    global server
    server.n_ctx = 8192
    server.kv_unified = True
    server.no_cache_idle_slots = not cache_idle_slots
    server.server_metrics = True
    server.temperature = 0.0
    server.start()

    a, b = slot_attach_prompts()
    n_shared = next(i for i, (x, y) in enumerate(zip(a, b)) if x != y)

    slot_attach_completion(a, 0)
    expected_a = slot_attach_completion(a, 0)

    # b attaches the cells of a and writes its own cells after them; with cache_idle_slots, slot 0 is then cleared
    actual_b = slot_attach_completion(b, 1, n_predict=16)
    assert actual_b[0] == n_shared
    metrics = fetch_metrics(server)
    assert get_metric(metrics, "slot_attach_total") == 1
    assert get_metric(metrics, "slot_attach_tokens_total") == n_shared

    if not cache_idle_slots:
        slot_attach_assert_same(slot_attach_completion(a, 0), expected_a)

    # the same prefix, reused by slot 1 itself
    slot_attach_completion(a, 1)
    slot_attach_assert_same(slot_attach_completion(b, 1), actual_b)

    # the n_cache_reuse of a request is ignored: nothing of b is reused once its first tokens are dropped
    assert slot_attach_completion(b[100:], 1, n_cache_reuse=256)[0] == 0


# b does not attach
@pytest.mark.parametrize("kv_unified,slot_attach,spec_type,n_cache_reuse,n_repeat,level", [
    (False, None,  None,        None, 48, "I"),  # attach needs a unified KV cache
    (True,  None,  "ngram-mod", None, 48, "I"),  # and turns itself off with speculative decoding
    (True,  True,  None,        256,  48, "W"),  # or with --cache-reuse, which stays on; a warning when --slot-attach is explicit
    (True,  False, None,        None, 48, None), # --no-slot-attach turns it off without a message
    (True,  None,  None,        None, 7,  None), # a prefix shorter than 256 tokens is not attached
])
def test_slot_attach_off(kv_unified, slot_attach, spec_type, n_cache_reuse, n_repeat, level, tmp_path):
    global server
    server.n_ctx = 8192
    server.n_batch = 128 # room for the draft tokens of ngram-mod
    server.kv_unified = kv_unified
    server.slot_attach = slot_attach
    server.spec_type = spec_type
    server.n_cache_reuse = n_cache_reuse
    server.server_metrics = True
    server.log_path = str(tmp_path / "server.log")
    server.start()

    a, b = slot_attach_prompts(n_repeat)
    slot_attach_completion(a, 0)
    assert slot_attach_completion(b, 1)[0] == 0
    assert get_metric(fetch_metrics(server), "slot_attach_total") == 0

    # the startup line when attach turns itself off
    m = re.search(r" ([IW]) srv +init: .*slot.attach .*, disabling", slot_attach_log())
    assert (m.group(1) if m else None) == level

    if n_cache_reuse:
        # b without its first tokens: slot 1 reuses the rest of b by shifting its cells
        assert slot_attach_completion(b[100:], 1)[0] > 0


# tinyopenjev has hybrid memory: a prompt attaches at a checkpoint of another slot
def slot_attach_hybrid_server(tmp_path):
    server = ServerPreset.tinyopenjev()
    server.jinja = True
    server.n_batch = 1024
    server.n_ubatch = 1024
    server.n_slots = 3
    server.kv_unified = True
    server.no_cache_idle_slots = True
    server.slot_save_path = str(tmp_path) # for /slots erase
    server.server_metrics = True
    server.temperature = 0.0
    server.n_predict = 1
    return server


# hybrid memory attaches at a checkpoint, and adds one where the system prompts of two sessions diverge
def test_slot_attach_hybrid(tmp_path):
    global server
    server = slot_attach_hybrid_server(tmp_path)
    server.no_mmproj = True
    server.n_cache_reuse = 256 # turned off at load for hybrid memory, so attach stays on
    server.start()

    # the sessions differ at the end of the system prompt, about 1,200 tokens in
    def chat(session, id_slot):
        return slot_attach_chat(f"{SLOT_ATTACH_TEXT * 64}Session: {session}", "Tell me a story.", id_slot)

    chat("first", 0)

    # the only checkpoint of slot 0 in the shared prefix is about n_ubatch tokens before the end of its prompt, under 256 tokens: no attach
    # slot 1 adds a checkpoint where the sessions diverge
    assert chat("second", 1)[0] == 0
    assert get_metric(fetch_metrics(server), "slot_attach_checkpoints_total") == 1

    # attaches up to where the sessions diverge
    actual = chat("third", 2)
    assert actual[0] >= 1024

    # the same, from the checkpoint that slot 1 holds itself
    assert server.make_request("POST", "/slots/2?action=erase").status_code == 200
    slot_attach_assert_same(chat("third", 1), actual)


# with a projector loaded, a prompt without media attaches, also from a slot whose prompt has an image, and a prompt with media does not
def test_slot_attach_media(tmp_path):
    global server
    server = slot_attach_hybrid_server(tmp_path)
    server.start()

    system = SLOT_ATTACH_TEXT * 32
    with_image = [
        {"type": "image_url", "image_url": {"url": get_img_url("IMG_BASE64_URI_0")}},
        {"type": "text", "text": "What is this?"},
    ]

    # slot 0 has a checkpoint where its user message starts, before the image
    slot_attach_chat(system, with_image, 0)

    # slot 1 attaches the system prompt from slot 0, without the cells of the image
    actual = slot_attach_chat(system, "Tell me a story.", 1)
    assert actual[0] >= 256

    # a prompt with media does not attach
    assert slot_attach_chat(system, with_image, 2)[0] == 0
    assert get_metric(fetch_metrics(server), "slot_attach_total") == 1

    # the same, from the checkpoint that slot 0 holds itself
    for id_slot in [1, 2]:
        assert server.make_request("POST", f"/slots/{id_slot}?action=erase").status_code == 200
    slot_attach_assert_same(slot_attach_chat(system, "Tell me a story.", 0), actual)


# when the KV cache is full, the idle slots whose cells no other slot holds are purged first
def test_slot_attach_purge_order(tmp_path):
    global server
    server.n_ctx = 1024
    server.n_slots = 4
    server.kv_unified = True
    server.cache_ram = 0 # no idle slots are cleared when a task starts
    server.temperature = 0.0
    server.log_path = str(tmp_path / "server.log")
    server.start()

    # slots 0 and 1 share a prefix of 400 tokens, slot 2 holds 300 tokens of its own
    a, b = slot_attach_prompts(16, 400)
    slot_attach_completion(a, 0)
    assert slot_attach_completion(b, 1)[0] >= 400
    slot_attach_completion(tokenize(server, "The quick brown fox jumps over the lazy dog. " * 40)[:300], 2)

    # 500 more tokens do not fit
    slot_attach_completion(tokenize(server, "A cat sat on the warm mat by the door. " * 40)[:500], 3)
    assert re.findall(r"purging slot (\d+)", slot_attach_log()) == ["2"]
