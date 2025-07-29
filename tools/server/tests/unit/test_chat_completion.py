import pytest
from openai import OpenAI
from utils import *

server: ServerProcess

@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.tinyllama2()
    server.server_port = 8080  


@pytest.mark.parametrize(
    "model,system_prompt,user_prompt,max_tokens,re_content,n_prompt,n_predicted,finish_reason,jinja,chat_template",
    [
        (None, "Book", "Hey", 8, "But she couldn't", 69, 8, "length", False, None),
        (None, "Book", "Hey", 8, "But she couldn't", 69, 8, "length", True, None),
        (None, "Book", "What is the best book", 8, "(Suddenly)+|\\{ \" Sarax.", 77, 8, "length", False, None),
        (None, "Book", "What is the best book", 8, "(Suddenly)+|\\{ \" Sarax.", 77, 8, "length", True,  None),
        (None, "Book", "What is the best book", 8, "(Suddenly)+|\\{ \" Sarax.", 77, 8, "length", True, 'chatml'),
        (None, "Book", "What is the best book", 8, "^ blue",                    23, 8, "length", True, "This is not a chat template, it is"),
        ("codellama70b", "You are a coding assistant.", "Write the fibonacci function in c++.", 128, "(Aside|she|felter|alonger)+", 104, 64, "length", False, None),
        ("codellama70b", "You are a coding assistant.", "Write the fibonacci function in c++.", 128, "(Aside|she|felter|alonger)+", 104, 64, "length", True, None),
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
        ("You are a coding assistant.", "Write the fibonacci function in c++.", 128, "(Aside|she|felter|alonger)+", 104, 64, "length"),
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
            assert data["usage"]["prompt_tokens"] == n_prompt
            assert data["usage"]["completion_tokens"] == n_predicted
            assert "content" not in choice["delta"]
            assert match_regex(re_content, content)
            assert choice["finish_reason"] == finish_reason
        else:
            assert choice["finish_reason"] is None
            content += choice["delta"]["content"] or ''


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
        "timings_per_token": True,
    })
    for i, data in enumerate(res):
        if i == 0:
            # Check first role message for stream=True
            assert data["choices"][0]["delta"]["content"] is None
            assert data["choices"][0]["delta"]["role"] == "assistant"
            assert "timings" not in data, f'First event should not have timings: {data}'
        else:
            assert "role" not in data["choices"][0]["delta"]
            assert "timings" in data
            assert "prompt_per_second" in data["timings"]
            assert "predicted_per_second" in data["timings"]
            assert "predicted_n" in data["timings"]
            assert data["timings"]["predicted_n"] <= 10


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


def test_progress_feature_enabled():
    """Test progress feature when return_progress is enabled"""
    global server
    server.start()
    
    # Create a long prompt to ensure multiple batches are processed
    long_prompt = "This is a comprehensive test prompt designed to verify the progress functionality thoroughly. " * 100
    
    res = server.make_stream_request("POST", "/chat/completions", data={
        "max_tokens": 10,
        "messages": [
            {"role": "user", "content": long_prompt},
        ],
        "stream": True,
        "return_progress": True,
    })
    
    progress_responses = []
    content_responses = []
    
    for data in res:
        choice = data["choices"][0]
        
        # Check for progress responses (they can be at root level or in delta)
        if "prompt_processing" in data:
            progress_responses.append(data["prompt_processing"])
        elif "delta" in choice and "prompt_processing" in choice["delta"]:
            progress_responses.append(choice["delta"]["prompt_processing"])
        elif "delta" in choice and "content" in choice["delta"] and choice["delta"]["content"]:
            content_responses.append(data)
    
    # Verify we received progress responses
    assert len(progress_responses) > 0, "No progress responses received"
    
    # Verify the last progress response shows 100% completion
    last_progress = progress_responses[-1]
    assert last_progress["progress"] >= 0.99, f"Progress did not reach 100% (last: {last_progress['progress']*100:.1f}%)"
    
    # Verify we received content responses
    assert len(content_responses) > 0, "No content responses received"


def test_progress_feature_disabled():
    """Test that progress is not sent when return_progress is disabled"""
    global server
    server.start()
    
    # Create a long prompt
    long_prompt = "This is a comprehensive test prompt designed to verify the progress functionality thoroughly. " * 100
    
    res = server.make_stream_request("POST", "/chat/completions", data={
        "max_tokens": 10,
        "messages": [
            {"role": "user", "content": long_prompt},
        ],
        "stream": True,
        "return_progress": False,  # Disable progress
    })
    
    progress_responses = []
    content_responses = []
    
    for data in res:
        choice = data["choices"][0]
        
        # Check for progress responses (they can be at root level or in delta)
        if "prompt_processing" in data:
            progress_responses.append(data["prompt_processing"])
        elif "delta" in choice and "prompt_processing" in choice["delta"]:
            progress_responses.append(choice["delta"]["prompt_processing"])
        elif "delta" in choice and "content" in choice["delta"] and choice["delta"]["content"]:
            content_responses.append(data)
    
    # Verify no progress responses were received
    assert len(progress_responses) == 0, f"Progress responses received when disabled: {len(progress_responses)}"
    
    # Verify we still received content responses
    assert len(content_responses) > 0, "No content responses received"


def test_progress_feature_completion_endpoint():
    """Test progress feature on /completion endpoint"""
    global server
    server.start()
    
    # Create a long prompt
    long_prompt = "This is a comprehensive test prompt designed to verify the progress functionality thoroughly. " * 100
    
    res = server.make_stream_request("POST", "/completion", data={
        "prompt": long_prompt,
        "stream": True,
        "return_progress": True,
        "max_tokens": 10,
    })
    
    progress_responses = []
    content_responses = []
    
    for data in res:
        # Check for progress responses in /completion format
        if "prompt_processing" in data:
            progress_responses.append(data["prompt_processing"])
        elif "content" in data and data["content"]:
            content_responses.append(data)
    
    # Verify we received progress responses
    assert len(progress_responses) > 0, "No progress responses received from /completion endpoint"
    
    # Verify the last progress response shows 100% completion
    last_progress = progress_responses[-1]
    assert last_progress["progress"] >= 0.99, f"Progress did not reach 100% (last: {last_progress['progress']*100:.1f}%)"
    
    # Verify we received content responses
    assert len(content_responses) > 0, "No content responses received from /completion endpoint"


def test_progress_feature_with_different_batch_sizes():
    """Test progress feature behavior with different batch processing scenarios"""
    global server
    server.start()
    
    # Test with different prompt lengths to simulate different batch processing
    test_cases = [
        ("Short prompt", "Short test prompt"),
        ("Medium prompt", "This is a medium length test prompt designed to test progress functionality. " * 20),
        ("Long prompt", "This is a comprehensive test prompt designed to verify the progress functionality thoroughly. " * 100),
    ]
    
    for test_name, prompt in test_cases:
        res = server.make_stream_request("POST", "/chat/completions", data={
            "max_tokens": 5,
            "messages": [
                {"role": "user", "content": prompt},
            ],
            "stream": True,
            "return_progress": True,
        })
        
        progress_responses = []
        content_responses = []
        
        for data in res:
            choice = data["choices"][0]
            
            # Check for progress responses (they can be at root level or in delta)
            if "prompt_processing" in data:
                progress_responses.append(data["prompt_processing"])
            elif "delta" in choice and "prompt_processing" in choice["delta"]:
                progress_responses.append(choice["delta"]["prompt_processing"])
            elif "delta" in choice and "content" in choice["delta"] and choice["delta"]["content"]:
                content_responses.append(data)
        
        # Verify progress functionality works for all prompt lengths
        assert len(progress_responses) > 0, f"No progress responses for {test_name}"
        assert len(content_responses) > 0, f"No content responses for {test_name}"
        
        # Verify progress reaches 100%
        if progress_responses:
            last_progress = progress_responses[-1]
            assert last_progress["progress"] >= 0.99, f"Progress did not reach 100% for {test_name} (last: {last_progress['progress']*100:.1f}%)"
