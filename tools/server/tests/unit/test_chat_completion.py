import pytest
from openai import OpenAI
from utils import *

server: ServerProcess

@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.tinyllama2()


@pytest.mark.parametrize(
    "model,system_prompt,user_prompt,max_tokens,re_content,n_prompt,n_predicted,finish_reason,jinja,chat_template",
    [
        (None, "Book", "Hey", 8, "But she couldn't", 69, 8, "length", False, None),
        (None, "Book", "Hey", 8, "But she couldn't", 69, 8, "length", True, None),
        (None, "Book", "What is the best book", 8, "(Suddenly)+|\\{ \" Sarax.", 77, 8, "length", False, None),
        (None, "Book", "What is the best book", 8, "(Suddenly)+|\\{ \" Sarax.", 77, 8, "length", True,  None),
        (None, "Book", "What is the best book", 8, "(Suddenly)+|\\{ \" Sarax.", 77, 8, "length", True, 'chatml'),
        (None, "Book", "What is the best book", 8, "^ blue",                    23, 8, "length", True, "This is not a chat template, it is"),
        ("codellama70b", "You are a coding assistant.", "Write the fibonacci function in c++.", 128, "(Aside|she|felter|alonger)+", 104, 128, "length", False, None),
        ("codellama70b", "You are a coding assistant.", "Write the fibonacci function in c++.", 128, "(Aside|she|felter|alonger)+", 104, 128, "length", True, None),
        (None, "Book", [{"type": "text", "text": "What is"}, {"type": "text", "text": "the best book"}], 8, "Whillicter", 79, 8, "length", False, None),
        (None, "Book", [{"type": "text", "text": "What is"}, {"type": "text", "text": "the best book"}], 8, "Whillicter", 79, 8, "length", True, None),
    ]
)
def test_chat_completion(model, system_prompt, user_prompt, max_tokens, re_content, n_prompt, n_predicted, finish_reason, jinja, chat_template):
    global server
    server.jinja = jinja
    server.chat_template = chat_template
    server.start()
    res = server.make_request("POST", "/chat/completions", data={
        "model": model,
        "max_tokens": max_tokens,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    })
    assert res.status_code == 200
    assert "cmpl" in res.body["id"] # make sure the completion id has the expected format
    assert res.body["system_fingerprint"].startswith("b")
    assert res.body["model"] == model if model is not None else server.model_alias
    assert res.body["usage"]["prompt_tokens"] == n_prompt
    assert res.body["usage"]["completion_tokens"] == n_predicted
    choice = res.body["choices"][0]
    assert "assistant" == choice["message"]["role"]
    assert match_regex(re_content, choice["message"]["content"]), f'Expected {re_content}, got {choice["message"]["content"]}'
    assert choice["finish_reason"] == finish_reason


@pytest.mark.parametrize(
    "system_prompt,user_prompt,max_tokens,re_content,n_prompt,n_predicted,finish_reason",
    [
        ("Book", "What is the best book", 8, "(Suddenly)+", 77, 8, "length"),
        ("You are a coding assistant.", "Write the fibonacci function in c++.", 128, "(Aside|she|felter|alonger)+", 104, 128, "length"),
    ]
)
def test_chat_completion_stream(system_prompt, user_prompt, max_tokens, re_content, n_prompt, n_predicted, finish_reason):
    global server
    server.model_alias = None # try using DEFAULT_OAICOMPAT_MODEL
    server.start()
    res = server.make_stream_request("POST", "/chat/completions", data={
        "max_tokens": max_tokens,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "stream": True,
    })
    content = ""
    last_cmpl_id = None
    for i, data in enumerate(res):
        if data["choices"]:
            choice = data["choices"][0]
            if i == 0:
                # Check first role message for stream=True
                assert choice["delta"]["content"] is None
                assert choice["delta"]["role"] == "assistant"
            else:
                assert "role" not in choice["delta"]
            assert data["system_fingerprint"].startswith("b")
            assert "gpt-3.5" in data["model"] # DEFAULT_OAICOMPAT_MODEL, maybe changed in the future
            if last_cmpl_id is None:
                last_cmpl_id = data["id"]
            assert last_cmpl_id == data["id"] # make sure the completion id is the same for all events in the stream
            if choice["finish_reason"] in ["stop", "length"]:
                assert "content" not in choice["delta"]
                assert match_regex(re_content, content)
                assert choice["finish_reason"] == finish_reason
            else:
                assert choice["finish_reason"] is None
                content += choice["delta"]["content"] or ''
        else:
            assert data["usage"]["prompt_tokens"] == n_prompt
            assert data["usage"]["completion_tokens"] == n_predicted


def test_chat_completion_with_openai_library():
    global server
    server.start()
    client = OpenAI(api_key="dummy", base_url=f"http://{server.server_host}:{server.server_port}/v1")
    res = client.chat.completions.create(
        model="gpt-3.5-turbo-instruct",
        messages=[
            {"role": "system", "content": "Book"},
            {"role": "user", "content": "What is the best book"},
        ],
        max_tokens=8,
        seed=42,
        temperature=0.8,
    )
    assert res.system_fingerprint is not None and res.system_fingerprint.startswith("b")
    assert res.choices[0].finish_reason == "length"
    assert res.choices[0].message.content is not None
    assert match_regex("(Suddenly)+", res.choices[0].message.content)


def test_chat_template():
    global server
    server.chat_template = "llama3"
    server.debug = True  # to get the "__verbose" object in the response
    server.start()
    res = server.make_request("POST", "/chat/completions", data={
        "max_tokens": 8,
        "messages": [
            {"role": "system", "content": "Book"},
            {"role": "user", "content": "What is the best book"},
        ]
    })
    assert res.status_code == 200
    assert "__verbose" in res.body
    assert res.body["__verbose"]["prompt"] == "<s> <|start_header_id|>system<|end_header_id|>\n\nBook<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nWhat is the best book<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"


@pytest.mark.parametrize("prefill,re_prefill", [
    ("Whill", "Whill"),
    ([{"type": "text", "text": "Wh"}, {"type": "text", "text": "ill"}], "Whill"),
])
def test_chat_template_assistant_prefill(prefill, re_prefill):
    global server
    server.chat_template = "llama3"
    server.debug = True  # to get the "__verbose" object in the response
    server.start()
    res = server.make_request("POST", "/chat/completions", data={
        "max_tokens": 8,
        "messages": [
            {"role": "system", "content": "Book"},
            {"role": "user", "content": "What is the best book"},
            {"role": "assistant", "content": prefill},
        ]
    })
    assert res.status_code == 200
    assert "__verbose" in res.body
    assert res.body["__verbose"]["prompt"] == f"<s> <|start_header_id|>system<|end_header_id|>\n\nBook<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nWhat is the best book<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{re_prefill}"


def test_apply_chat_template():
    global server
    server.chat_template = "command-r"
    server.start()
    res = server.make_request("POST", "/apply-template", data={
        "messages": [
            {"role": "system", "content": "You are a test."},
            {"role": "user", "content":"Hi there"},
        ]
    })
    assert res.status_code == 200
    assert "prompt" in res.body
    assert res.body["prompt"] == "<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>You are a test.<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|USER_TOKEN|>Hi there<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>"


@pytest.mark.parametrize("response_format,n_predicted,re_content", [
    ({"type": "json_object", "schema": {"const": "42"}}, 6, "\"42\""),
    ({"type": "json_object", "schema": {"items": [{"type": "integer"}]}}, 10, "[ -3000 ]"),
    ({"type": "json_schema", "json_schema": {"schema": {"const": "foooooo"}}}, 10, "\"foooooo\""),
    ({"type": "json_object"}, 10, "(\\{|John)+"),
    ({"type": "sound"}, 0, None),
    # invalid response format (expected to fail)
    ({"type": "json_object", "schema": 123}, 0, None),
    ({"type": "json_object", "schema": {"type": 123}}, 0, None),
    ({"type": "json_object", "schema": {"type": "hiccup"}}, 0, None),
])
def test_completion_with_response_format(response_format: dict, n_predicted: int, re_content: str | None):
    global server
    server.start()
    res = server.make_request("POST", "/chat/completions", data={
        "max_tokens": n_predicted,
        "messages": [
            {"role": "system", "content": "You are a coding assistant."},
            {"role": "user", "content": "Write an example"},
        ],
        "response_format": response_format,
    })
    if re_content is not None:
        assert res.status_code == 200
        choice = res.body["choices"][0]
        assert match_regex(re_content, choice["message"]["content"])
    else:
        assert res.status_code != 200
        assert "error" in res.body


@pytest.mark.parametrize("jinja,json_schema,n_predicted,re_content", [
    (False, {"const": "42"}, 6, "\"42\""),
    (True, {"const": "42"}, 6, "\"42\""),
])
def test_completion_with_json_schema(jinja: bool, json_schema: dict, n_predicted: int, re_content: str):
    global server
    server.jinja = jinja
    server.start()
    res = server.make_request("POST", "/chat/completions", data={
        "max_tokens": n_predicted,
        "messages": [
            {"role": "system", "content": "You are a coding assistant."},
            {"role": "user", "content": "Write an example"},
        ],
        "json_schema": json_schema,
    })
    assert res.status_code == 200, f'Expected 200, got {res.status_code}'
    choice = res.body["choices"][0]
    assert match_regex(re_content, choice["message"]["content"]), f'Expected {re_content}, got {choice["message"]["content"]}'


@pytest.mark.parametrize("jinja,grammar,n_predicted,re_content", [
    (False, 'root ::= "a"{5,5}', 6, "a{5,5}"),
    (True, 'root ::= "a"{5,5}', 6, "a{5,5}"),
])
def test_completion_with_grammar(jinja: bool, grammar: str, n_predicted: int, re_content: str):
    global server
    server.jinja = jinja
    server.start()
    res = server.make_request("POST", "/chat/completions", data={
        "max_tokens": n_predicted,
        "messages": [
            {"role": "user", "content": "Does not matter what I say, does it?"},
        ],
        "grammar": grammar,
    })
    assert res.status_code == 200, res.body
    choice = res.body["choices"][0]
    assert match_regex(re_content, choice["message"]["content"]), choice["message"]["content"]


@pytest.mark.parametrize("messages", [
    None,
    "string",
    [123],
    [{}],
    [{"role": 123}],
    [{"role": "system", "content": 123}],
    # [{"content": "hello"}], # TODO: should not be a valid case
    [{"role": "system", "content": "test"}, {}],
    [{"role": "user", "content": "test"}, {"role": "assistant", "content": "test"}, {"role": "assistant", "content": "test"}],
])
def test_invalid_chat_completion_req(messages):
    global server
    server.start()
    res = server.make_request("POST", "/chat/completions", data={
        "messages": messages,
    })
    assert res.status_code == 400 or res.status_code == 500
    assert "error" in res.body


def test_chat_completion_with_timings_per_token():
    global server
    server.start()
    res = server.make_stream_request("POST", "/chat/completions", data={
        "max_tokens": 10,
        "messages": [{"role": "user", "content": "test"}],
        "stream": True,
        "stream_options": {"include_usage": True},
        "timings_per_token": True,
    })
    stats_received = False
    for i, data in enumerate(res):
        if i == 0:
            # Check first role message for stream=True
            assert data["choices"][0]["delta"]["content"] is None
            assert data["choices"][0]["delta"]["role"] == "assistant"
            assert "timings" not in data, f'First event should not have timings: {data}'
        else:
            if data["choices"]:
                assert "role" not in data["choices"][0]["delta"]
            else:
                assert "timings" in data
                assert "prompt_per_second" in data["timings"]
                assert "predicted_per_second" in data["timings"]
                assert "predicted_n" in data["timings"]
                assert data["timings"]["predicted_n"] <= 10
                stats_received = True
    assert stats_received


def test_logprobs():
    global server
    server.start()
    client = OpenAI(api_key="dummy", base_url=f"http://{server.server_host}:{server.server_port}/v1")
    res = client.chat.completions.create(
        model="gpt-3.5-turbo-instruct",
        temperature=0.0,
        messages=[
            {"role": "system", "content": "Book"},
            {"role": "user", "content": "What is the best book"},
        ],
        max_tokens=5,
        logprobs=True,
        top_logprobs=10,
    )
    output_text = res.choices[0].message.content
    aggregated_text = ''
    assert res.choices[0].logprobs is not None
    assert res.choices[0].logprobs.content is not None
    for token in res.choices[0].logprobs.content:
        aggregated_text += token.token
        assert token.logprob <= 0.0
        assert token.bytes is not None
        assert len(token.top_logprobs) > 0
    assert aggregated_text == output_text


def test_logprobs_stream():
    global server
    server.start()
    client = OpenAI(api_key="dummy", base_url=f"http://{server.server_host}:{server.server_port}/v1")
    res = client.chat.completions.create(
        model="gpt-3.5-turbo-instruct",
        temperature=0.0,
        messages=[
            {"role": "system", "content": "Book"},
            {"role": "user", "content": "What is the best book"},
        ],
        max_tokens=5,
        logprobs=True,
        top_logprobs=10,
        stream=True,
    )
    output_text = ''
    aggregated_text = ''
    for i, data in enumerate(res):
        if data.choices:
            choice = data.choices[0]
            if i == 0:
                # Check first role message for stream=True
                assert choice.delta.content is None
                assert choice.delta.role == "assistant"
            else:
                assert choice.delta.role is None
                if choice.finish_reason is None:
                    if choice.delta.content:
                        output_text += choice.delta.content
                    assert choice.logprobs is not None
                    assert choice.logprobs.content is not None
                    for token in choice.logprobs.content:
                        aggregated_text += token.token
                        assert token.logprob <= 0.0
                        assert token.bytes is not None
                        assert token.top_logprobs is not None
                        assert len(token.top_logprobs) > 0
    assert aggregated_text == output_text


def test_logit_bias():
    global server
    server.start()

    exclude = ["i", "I", "the", "The", "to", "a", "an", "be", "is", "was", "but", "But", "and", "And", "so", "So", "you", "You", "he", "He", "she", "She", "we", "We", "they", "They", "it", "It", "his", "His", "her", "Her", "book", "Book"]

    res = server.make_request("POST", "/tokenize", data={
        "content": " " + " ".join(exclude) + " ",
    })
    assert res.status_code == 200
    tokens = res.body["tokens"]
    logit_bias = {tok: -100 for tok in tokens}

    client = OpenAI(api_key="dummy", base_url=f"http://{server.server_host}:{server.server_port}/v1")
    res = client.chat.completions.create(
        model="gpt-3.5-turbo-instruct",
        temperature=0.0,
        messages=[
            {"role": "system", "content": "Book"},
            {"role": "user", "content": "What is the best book"},
        ],
        max_tokens=64,
        logit_bias=logit_bias
    )
    output_text = res.choices[0].message.content
    assert output_text
    assert all(output_text.find(" " + tok + " ") == -1 for tok in exclude)

def test_context_size_exceeded():
    global server
    server.start()
    res = server.make_request("POST", "/chat/completions", data={
        "messages": [
            {"role": "system", "content": "Book"},
            {"role": "user", "content": "What is the best book"},
        ] * 100, # make the prompt too long
    })
    assert res.status_code == 400
    assert "error" in res.body
    assert res.body["error"]["type"] == "exceed_context_size_error"
    assert res.body["error"]["n_prompt_tokens"] > 0
    assert server.n_ctx is not None
    assert server.n_slots is not None
    assert res.body["error"]["n_ctx"] == server.n_ctx // server.n_slots


def test_context_size_exceeded_stream():
    global server
    server.start()
    try:
        for _ in server.make_stream_request("POST", "/chat/completions", data={
            "messages": [
                {"role": "system", "content": "Book"},
                {"role": "user", "content": "What is the best book"},
            ] * 100, # make the prompt too long
            "stream": True}):
                pass
        assert False, "Should have failed"
    except ServerError as e:
        assert e.code == 400
        assert "error" in e.body
        assert e.body["error"]["type"] == "exceed_context_size_error"
        assert e.body["error"]["n_prompt_tokens"] > 0
        assert server.n_ctx is not None
        assert server.n_slots is not None
        assert e.body["error"]["n_ctx"] == server.n_ctx // server.n_slots


@pytest.mark.parametrize(
    "n_batch,batch_count,reuse_cache",
    [
        (64, 15, False),
        (64, 1, True),
    ]
)
def test_return_progresssss(n_batch, batch_count, reuse_cache):
    global server
    server.n_batch = n_batch
    server.n_ctx = 2048
    server.n_slots = 1
    server.start()
    def make_cmpl_request():
        return server.make_stream_request("POST", "/chat/completions", data={
            "max_tokens": 10,
            "messages": [
                {"role": "user", "content": "This is a test" * 100},
            ],
            "stream": True,
            "return_progress": True,
        })
    if reuse_cache:
        # make a first request to populate the cache
        res0 = make_cmpl_request()
        for _ in res0:
            pass # discard the output

    res = make_cmpl_request()
    last_progress = None
    total_batch_count = 0
    for data in res:
        cur_progress = data.get("prompt_progress", None)
        if cur_progress is None:
            continue
        if last_progress is not None:
            assert cur_progress["total"] == last_progress["total"]
            assert cur_progress["cache"] == last_progress["cache"]
            assert cur_progress["processed"] > last_progress["processed"]
        total_batch_count += 1
        last_progress = cur_progress

    assert last_progress is not None
    assert last_progress["total"] > 0
    assert last_progress["processed"] == last_progress["total"]
    assert total_batch_count == batch_count
