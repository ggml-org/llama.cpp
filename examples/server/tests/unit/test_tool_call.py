import pytest
from utils import *

server: ServerProcess

TIMEOUT_SERVER_START = 15*60
TIMEOUT_HTTP_REQUEST = 60

@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.tinyllama2()
    server.model_alias = "tinyllama-2-tool-call"
    server.server_port = 8081


TEST_TOOL = {
    "type":"function",
    "function": {
        "name": "test",
        "description": "",
        "parameters": {
            "type": "object",
            "properties": {
                "success": {"type": "boolean", "const": True},
            },
            "required": ["success"]
        }
    }
}

PYTHON_TOOL = {
    "type": "function",
    "function": {
        "name": "python",
        "description": "Runs code in an ipython interpreter and returns the result of the execution after 60 seconds.",
        "parameters": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "The code to run in the ipython interpreter."
                }
            },
            "required": ["code"]
        }
    }
}

WEATHER_TOOL = {
  "type":"function",
  "function":{
    "name":"get_current_weather",
    "description":"Get the current weather in a given location",
    "parameters":{
      "type":"object",
      "properties":{
        "location":{
          "type":"string",
          "description":"The city and country/state, e.g. 'San Francisco, CA', or 'Paris, France'"
        }
      },
      "required":["location"]
    }
  }
}


def do_test_completion_with_required_tool_tiny(template_name: str, tool: dict, argument_key: str | None):
    global server
    n_predict = 512
    # server = ServerPreset.stories15m_moe()
    server.jinja = True
    server.n_predict = n_predict
    server.chat_template_file = f'../../../models/templates/{template_name}.jinja'
    server.start(timeout_seconds=TIMEOUT_SERVER_START)
    res = server.make_request("POST", "/chat/completions", data={
        "max_tokens": n_predict,
        "messages": [
            {"role": "system", "content": "You are a coding assistant."},
            {"role": "user", "content": "Write an example"},
        ],
        "tool_choice": "required",
        "tools": [tool],
        "parallel_tool_calls": False,
        "temperature": 0.0,
        "top_k": 1,
        "top_p": 1.0,
    })
    assert res.status_code == 200, f"Expected status code 200, got {res.status_code}"
    choice = res.body["choices"][0]
    tool_calls = choice["message"].get("tool_calls")
    assert tool_calls and len(tool_calls) == 1, f'Expected 1 tool call in {choice["message"]}'
    tool_call = tool_calls[0]
    expected_function_name = "python" if tool["type"] == "code_interpreter" else tool["function"]["name"]
    assert expected_function_name == tool_call["function"]["name"]
    actual_arguments = tool_call["function"]["arguments"]
    assert isinstance(actual_arguments, str)
    if argument_key is not None:
        actual_arguments = json.loads(actual_arguments)
        assert argument_key in actual_arguments, f"tool arguments: {json.dumps(actual_arguments)}, expected: {argument_key}"


@pytest.mark.parametrize("template_name,tool,argument_key", [
    ("google-gemma-2-2b-it",                          TEST_TOOL,            "success"),
    ("meta-llama-Llama-3.3-70B-Instruct",             TEST_TOOL,            "success"),
    ("meta-llama-Llama-3.3-70B-Instruct",             PYTHON_TOOL,          "code"),
])
def test_completion_with_required_tool_tiny_fast(template_name: str, tool: dict, argument_key: str | None):
    do_test_completion_with_required_tool_tiny(template_name, tool, argument_key)


@pytest.mark.slow
@pytest.mark.parametrize("template_name,tool,argument_key", [
    ("meta-llama-Llama-3.1-8B-Instruct",              TEST_TOOL,            "success"),
    ("meta-llama-Llama-3.1-8B-Instruct",              PYTHON_TOOL,          "code"),
    ("meetkai-functionary-medium-v3.1",               TEST_TOOL,            "success"),
    ("meetkai-functionary-medium-v3.1",               PYTHON_TOOL,          "code"),
    ("meetkai-functionary-medium-v3.2",               TEST_TOOL,            "success"),
    ("meetkai-functionary-medium-v3.2",               PYTHON_TOOL,          "code"),
    ("NousResearch-Hermes-2-Pro-Llama-3-8B-tool_use", TEST_TOOL,            "success"),
    ("NousResearch-Hermes-2-Pro-Llama-3-8B-tool_use", PYTHON_TOOL,          "code"),
    ("meta-llama-Llama-3.2-3B-Instruct",              TEST_TOOL,            "success"),
    ("meta-llama-Llama-3.2-3B-Instruct",              PYTHON_TOOL,          "code"),
    ("mistralai-Mistral-Nemo-Instruct-2407",          TEST_TOOL,            "success"),
    ("mistralai-Mistral-Nemo-Instruct-2407",          PYTHON_TOOL,          "code"),
    ("NousResearch-Hermes-3-Llama-3.1-8B-tool_use",   TEST_TOOL,            "success"),
    ("NousResearch-Hermes-3-Llama-3.1-8B-tool_use",   PYTHON_TOOL,          "code"),
    ("deepseek-ai-DeepSeek-R1-Distill-Llama-8B",      TEST_TOOL,            "success"),
    ("deepseek-ai-DeepSeek-R1-Distill-Llama-8B",      PYTHON_TOOL,          "code"),
    ("fireworks-ai-llama-3-firefunction-v2",          TEST_TOOL,            "success"),
    ("fireworks-ai-llama-3-firefunction-v2",          PYTHON_TOOL,          "code"),
])
def test_completion_with_required_tool_tiny_slow(template_name: str, tool: dict, argument_key: str | None):
    do_test_completion_with_required_tool_tiny(template_name, tool, argument_key)


@pytest.mark.slow
@pytest.mark.parametrize("tool,argument_key,hf_repo,template_override", [
    (TEST_TOOL,    "success",  "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF:Q4_K_M", None),
    (PYTHON_TOOL,  "code",     "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF:Q4_K_M", None),
    (PYTHON_TOOL,  "code",     "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF:Q4_K_M", "chatml"),

    # Note: gemma-2-2b-it knows itself as "model", not "assistant", so we don't test the ill-suited chatml on it.
    (TEST_TOOL,    "success",  "bartowski/gemma-2-2b-it-GGUF:Q4_K_M",              None),
    (PYTHON_TOOL,  "code",     "bartowski/gemma-2-2b-it-GGUF:Q4_K_M",              None),

    (TEST_TOOL,    "success",  "bartowski/Phi-3.5-mini-instruct-GGUF:Q4_K_M",      None),
    (PYTHON_TOOL,  "code",     "bartowski/Phi-3.5-mini-instruct-GGUF:Q4_K_M",      None),
    (PYTHON_TOOL,  "code",     "bartowski/Phi-3.5-mini-instruct-GGUF:Q4_K_M",      "chatml"),

    (TEST_TOOL,    "success",  "bartowski/Qwen2.5-7B-Instruct-GGUF:Q4_K_M",        None),
    (PYTHON_TOOL,  "code",     "bartowski/Qwen2.5-7B-Instruct-GGUF:Q4_K_M",        None),
    (PYTHON_TOOL,  "code",     "bartowski/Qwen2.5-7B-Instruct-GGUF:Q4_K_M",        "chatml"),

    (TEST_TOOL,    "success",  "bartowski/Hermes-2-Pro-Llama-3-8B-GGUF:Q4_K_M", ("NousResearch/Hermes-2-Pro-Llama-3-8B", "tool_use")),
    (PYTHON_TOOL,  "code",     "bartowski/Hermes-2-Pro-Llama-3-8B-GGUF:Q4_K_M", ("NousResearch/Hermes-2-Pro-Llama-3-8B", "tool_use")),
    (PYTHON_TOOL,  "code",     "bartowski/Hermes-2-Pro-Llama-3-8B-GGUF:Q4_K_M", "chatml"),

    (TEST_TOOL,    "success",  "bartowski/Hermes-3-Llama-3.1-8B-GGUF:Q4_K_M",   ("NousResearch/Hermes-3-Llama-3.1-8B", "tool_use")),
    (PYTHON_TOOL,  "code",     "bartowski/Hermes-3-Llama-3.1-8B-GGUF:Q4_K_M",   ("NousResearch/Hermes-3-Llama-3.1-8B", "tool_use")),
    (PYTHON_TOOL,  "code",     "bartowski/Hermes-3-Llama-3.1-8B-GGUF:Q4_K_M",   "chatml"),

    (TEST_TOOL,    "success",  "bartowski/Mistral-Nemo-Instruct-2407-GGUF:Q4_K_M", None),
    (PYTHON_TOOL,  "code",     "bartowski/Mistral-Nemo-Instruct-2407-GGUF:Q4_K_M", None),
    (PYTHON_TOOL,  "code",     "bartowski/Mistral-Nemo-Instruct-2407-GGUF:Q4_K_M", "chatml"),

    (TEST_TOOL,    "success",  "bartowski/functionary-small-v3.2-GGUF:Q4_K_M",       ("meetkai/functionary-medium-v3.2", None)),
    (PYTHON_TOOL,  "code",     "bartowski/functionary-small-v3.2-GGUF:Q4_K_M",       ("meetkai/functionary-medium-v3.2", None)),
    (PYTHON_TOOL,  "code",     "bartowski/functionary-small-v3.2-GGUF:Q4_K_M",       "chatml"),

    (TEST_TOOL,    "success",  "bartowski/Llama-3.2-3B-Instruct-GGUF:Q4_K_M",      ("meta-llama/Llama-3.2-3B-Instruct", None)),
    (PYTHON_TOOL,  "code",     "bartowski/Llama-3.2-3B-Instruct-GGUF:Q4_K_M",      ("meta-llama/Llama-3.2-3B-Instruct", None)),
    (PYTHON_TOOL,  "code",     "bartowski/Llama-3.2-3B-Instruct-GGUF:Q4_K_M",      "chatml"),

    (TEST_TOOL,    "success",  "bartowski/Llama-3.2-1B-Instruct-GGUF:Q4_K_M",      ("meta-llama/Llama-3.2-3B-Instruct", None)),
    (PYTHON_TOOL,  "code",     "bartowski/Llama-3.2-1B-Instruct-GGUF:Q4_K_M",      ("meta-llama/Llama-3.2-3B-Instruct", None)),
    (PYTHON_TOOL,  "code",     "bartowski/Llama-3.2-1B-Instruct-GGUF:Q4_K_M",      "chatml"),
    # TODO: fix these
    # (TEST_TOOL,    "success",  "bartowski/DeepSeek-R1-Distill-Qwen-7B-GGUF:Q4_K_M", None),
    # (PYTHON_TOOL,  "code",     "bartowski/DeepSeek-R1-Distill-Qwen-7B-GGUF:Q4_K_M", None),
])
def test_completion_with_required_tool_real_model(tool: dict, argument_key: str | None, hf_repo: str, template_override: str | Tuple[str, str | None] | None):
    global server
    n_predict = 512
    server.n_slots = 1
    server.jinja = True
    server.n_ctx = 8192
    server.n_predict = n_predict
    server.model_hf_repo = hf_repo
    server.model_hf_file = None
    if isinstance(template_override, tuple):
        (template_hf_repo, template_variant) = template_override
        server.chat_template_file = f"../../../models/templates/{template_hf_repo.replace('/', '-') + ('-' + template_variant if template_variant else '')}.jinja"
        assert os.path.exists(server.chat_template_file), f"Template file {server.chat_template_file} does not exist. Run `python scripts/get_chat_template.py {template_hf_repo} {template_variant} > {server.chat_template_file}` to download the template."
    elif isinstance(template_override, str):
        server.chat_template = template_override
    server.start(timeout_seconds=TIMEOUT_SERVER_START)
    res = server.make_request("POST", "/chat/completions", data={
        "max_tokens": n_predict,
        "messages": [
            {"role": "system", "content": "You are a coding assistant."},
            {"role": "user", "content": "Write an example"},
        ],
        "tool_choice": "required",
        "tools": [tool],
        "parallel_tool_calls": False,
        "temperature": 0.0,
        "top_k": 1,
        "top_p": 1.0,
    }, timeout=TIMEOUT_HTTP_REQUEST)
    assert res.status_code == 200, f"Expected status code 200, got {res.status_code}"
    choice = res.body["choices"][0]
    tool_calls = choice["message"].get("tool_calls")
    assert tool_calls and len(tool_calls) == 1, f'Expected 1 tool call in {choice["message"]}'
    tool_call = tool_calls[0]
    expected_function_name = "python" if tool["type"] == "code_interpreter" else tool["function"]["name"]
    assert expected_function_name == tool_call["function"]["name"]
    actual_arguments = tool_call["function"]["arguments"]
    assert isinstance(actual_arguments, str)
    if argument_key is not None:
        actual_arguments = json.loads(actual_arguments)
        assert argument_key in actual_arguments, f"tool arguments: {json.dumps(actual_arguments)}, expected: {argument_key}"


def do_test_completion_without_tool_call(template_name: str, n_predict: int, tools: list[dict], tool_choice: str | None):
    global server
    server.jinja = True
    server.n_predict = n_predict
    server.chat_template_file = f'../../../models/templates/{template_name}.jinja'
    server.start(timeout_seconds=TIMEOUT_SERVER_START)
    res = server.make_request("POST", "/chat/completions", data={
        "max_tokens": n_predict,
        "messages": [
            {"role": "system", "content": "You are a coding assistant."},
            {"role": "user", "content": "say hello world with python"},
        ],
        "tools": tools if tools else None,
        "tool_choice": tool_choice,
        "temperature": 0.0,
        "top_k": 1,
        "top_p": 1.0,
    }, timeout=TIMEOUT_HTTP_REQUEST)
    assert res.status_code == 200, f"Expected status code 200, got {res.status_code}"
    choice = res.body["choices"][0]
    assert choice["message"].get("tool_calls") is None, f'Expected no tool call in {choice["message"]}'


@pytest.mark.parametrize("template_name,n_predict,tools,tool_choice", [
    ("meta-llama-Llama-3.3-70B-Instruct",         128, [],            None),
    ("meta-llama-Llama-3.3-70B-Instruct",         128, [TEST_TOOL],   None),
    ("meta-llama-Llama-3.3-70B-Instruct",         128, [PYTHON_TOOL], 'none'),
])
def test_completion_without_tool_call_fast(template_name: str, n_predict: int, tools: list[dict], tool_choice: str | None):
    do_test_completion_without_tool_call(template_name, n_predict, tools, tool_choice)


@pytest.mark.slow
@pytest.mark.parametrize("template_name,n_predict,tools,tool_choice", [
    ("meetkai-functionary-medium-v3.2",               256, [],            None),
    ("meetkai-functionary-medium-v3.2",               256, [TEST_TOOL],   None),
    ("meetkai-functionary-medium-v3.2",               256, [PYTHON_TOOL], 'none'),
    ("meetkai-functionary-medium-v3.1",               256, [],            None),
    ("meetkai-functionary-medium-v3.1",               256, [TEST_TOOL],   None),
    ("meetkai-functionary-medium-v3.1",               256, [PYTHON_TOOL], 'none'),
    ("meta-llama-Llama-3.2-3B-Instruct",              256, [],            None),
    ("meta-llama-Llama-3.2-3B-Instruct",              256, [TEST_TOOL],   None),
    ("meta-llama-Llama-3.2-3B-Instruct",              256, [PYTHON_TOOL], 'none'),
])
def test_completion_without_tool_call_slow(template_name: str, n_predict: int, tools: list[dict], tool_choice: str | None):
    do_test_completion_without_tool_call(template_name, n_predict, tools, tool_choice)


@pytest.mark.slow
@pytest.mark.parametrize("hf_repo,template_override", [
    ("bartowski/c4ai-command-r7b-12-2024-GGUF:Q4_K_M",   ("CohereForAI/c4ai-command-r7b-12-2024", "tool_use")),
    ("bartowski/Meta-Llama-3.1-8B-Instruct-GGUF:Q4_K_M", None),
    ("bartowski/Meta-Llama-3.1-8B-Instruct-GGUF:Q4_K_M", "chatml"),

    ("bartowski/Phi-3.5-mini-instruct-GGUF:Q4_K_M",      None),
    ("bartowski/Phi-3.5-mini-instruct-GGUF:Q4_K_M",      "chatml"),

    ("bartowski/Qwen2.5-7B-Instruct-GGUF:Q4_K_M",        None),
    ("bartowski/Qwen2.5-7B-Instruct-GGUF:Q4_K_M",        "chatml"),

    ("bartowski/Hermes-2-Pro-Llama-3-8B-GGUF:Q4_K_M",    ("NousResearch/Hermes-2-Pro-Llama-3-8B", "tool_use")),
    ("bartowski/Hermes-2-Pro-Llama-3-8B-GGUF:Q4_K_M",    "chatml"),

    ("bartowski/Hermes-3-Llama-3.1-8B-GGUF:Q4_K_M",      ("NousResearch/Hermes-3-Llama-3.1-8B", "tool_use")),
    ("bartowski/Hermes-3-Llama-3.1-8B-GGUF:Q4_K_M",      "chatml"),

    ("bartowski/Mistral-Nemo-Instruct-2407-GGUF:Q4_K_M", None),
    ("bartowski/Mistral-Nemo-Instruct-2407-GGUF:Q4_K_M", "chatml"),

    ("bartowski/functionary-small-v3.2-GGUF:Q8_0",       ("meetkai/functionary-medium-v3.2", None)),
    ("bartowski/functionary-small-v3.2-GGUF:Q8_0",       "chatml"),

    ("bartowski/Llama-3.2-3B-Instruct-GGUF:Q4_K_M",      ("meta-llama/Llama-3.2-3B-Instruct", None)),
    ("bartowski/Llama-3.2-3B-Instruct-GGUF:Q4_K_M",      "chatml"),

    # Note: gemma-2-2b-it knows itself as "model", not "assistant", so we don't test the ill-suited chatml on it.
    ("bartowski/gemma-2-2b-it-GGUF:Q4_K_M",              None),

    # ("bartowski/Llama-3.2-1B-Instruct-GGUF:Q4_K_M", ("meta-llama/Llama-3.2-3B-Instruct", None)),
    # ("bartowski/DeepSeek-R1-Distill-Qwen-7B-GGUF:Q4_K_M", None),
])
def test_weather(hf_repo: str, template_override: Tuple[str, str | None] | None):
    global server
    n_predict = 512
    server.n_slots = 1
    server.jinja = True
    server.n_ctx = 8192
    server.n_predict = n_predict
    server.model_hf_repo = hf_repo
    server.model_hf_file = None
    if isinstance(template_override, tuple):
        (template_hf_repo, template_variant) = template_override
        server.chat_template_file = f"../../../models/templates/{template_hf_repo.replace('/', '-') + ('-' + template_variant if template_variant else '')}.jinja"
        assert os.path.exists(server.chat_template_file), f"Template file {server.chat_template_file} does not exist. Run `python scripts/get_chat_template.py {template_hf_repo} {template_variant} > {server.chat_template_file}` to download the template."
    elif isinstance(template_override, str):
        server.chat_template = template_override
    server.start(timeout_seconds=TIMEOUT_SERVER_START)
    res = server.make_request("POST", "/chat/completions", data={
        "max_tokens": n_predict,
        "messages": [
            {"role": "user", "content": "What is the weather in Istanbul?"},
        ],
        "tools": [WEATHER_TOOL],
    }, timeout=TIMEOUT_HTTP_REQUEST)
    assert res.status_code == 200, f"Expected status code 200, got {res.status_code}"
    choice = res.body["choices"][0]
    tool_calls = choice["message"].get("tool_calls")
    assert tool_calls and len(tool_calls) == 1, f'Expected 1 tool call in {choice["message"]}'
    tool_call = tool_calls[0]
    assert tool_call["function"]["name"] == WEATHER_TOOL["function"]["name"]
    actual_arguments = json.loads(tool_call["function"]["arguments"])
    assert 'location' in actual_arguments, f"location not found in {json.dumps(actual_arguments)}"
    location = actual_arguments["location"]
    assert isinstance(location, str), f"Expected location to be a string, got {type(location)}: {json.dumps(location)}"
    assert re.match('^Istanbul(, (TR|Turkey|Türkiye))?$', location), f'Expected Istanbul for location, got {location}'


@pytest.mark.slow
@pytest.mark.parametrize("expected_arguments_override,hf_repo,template_override", [
    (None,                 "bartowski/Phi-3.5-mini-instruct-GGUF:Q4_K_M",      None),
    (None,                 "bartowski/Phi-3.5-mini-instruct-GGUF:Q4_K_M",      "chatml"),

    (None,                 "bartowski/functionary-small-v3.2-GGUF:Q8_0",       ("meetkai-functionary-medium-v3.2", None)),
    (None,                 "bartowski/functionary-small-v3.2-GGUF:Q8_0",       "chatml"),

    (None,                 "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF:Q4_K_M", None),
    ('{"code":"print("}',  "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF:Q4_K_M", "chatml"),

    ('{"code":"print("}',  "bartowski/Llama-3.2-1B-Instruct-GGUF:Q4_K_M",      ("meta-llama-Llama-3.2-3B-Instruct", None)),
    (None,                 "bartowski/Llama-3.2-1B-Instruct-GGUF:Q4_K_M",      "chatml"),

    ('{"code":"print("}',  "bartowski/Llama-3.2-3B-Instruct-GGUF:Q4_K_M",      ("meta-llama-Llama-3.2-3B-Instruct", None)),
    ('{"code":"print("}',  "bartowski/Llama-3.2-3B-Instruct-GGUF:Q4_K_M",      "chatml"),

    (None,                 "bartowski/Qwen2.5-7B-Instruct-GGUF:Q4_K_M",        None),
    (None,                 "bartowski/Qwen2.5-7B-Instruct-GGUF:Q4_K_M",        "chatml"),

    (None,                 "bartowski/Hermes-2-Pro-Llama-3-8B-GGUF:Q4_K_M",    ("NousResearch/Hermes-2-Pro-Llama-3-8B", "tool_use")),
    (None,                 "bartowski/Hermes-2-Pro-Llama-3-8B-GGUF:Q4_K_M",    "chatml"),

    (None,                 "bartowski/Hermes-3-Llama-3.1-8B-GGUF:Q4_K_M",      ("NousResearch-Hermes-3-Llama-3.1-8B", "tool_use")),
    (None,                 "bartowski/Hermes-3-Llama-3.1-8B-GGUF:Q4_K_M",      "chatml"),

    (None,                 "bartowski/Mistral-Nemo-Instruct-2407-GGUF:Q4_K_M", None),
    (None,                 "bartowski/Mistral-Nemo-Instruct-2407-GGUF:Q4_K_M", "chatml"),

    # Note: gemma-2-2b-it knows itself as "model", not "assistant", so we don't test the ill-suited chatml on it.
    (None,                 "bartowski/gemma-2-2b-it-GGUF:Q4_K_M",              None),

    # (None,                 "bartowski/DeepSeek-R1-Distill-Qwen-7B-GGUF:Q4_K_M", None),
])
def test_hello_world_tool_call(expected_arguments_override: str | None, hf_repo: str, template_override: str | Tuple[str, str | None] | None):
    global server
    server.n_slots = 1
    server.jinja = True
    server.n_ctx = 8192
    server.n_predict = 128
    server.model_hf_repo = hf_repo
    server.model_hf_file = None
    if isinstance(template_override, tuple):
        (template_hf_repo, template_variant) = template_override
        server.chat_template_file = f"../../../models/templates/{template_hf_repo.replace('/', '-') + ('-' + template_variant if template_variant else '')}.jinja"
        assert os.path.exists(server.chat_template_file), f"Template file {server.chat_template_file} does not exist. Run `python scripts/get_chat_template.py {template_hf_repo} {template_variant} > {server.chat_template_file}` to download the template."
    elif isinstance(template_override, str):
        server.chat_template = template_override
    server.start(timeout_seconds=TIMEOUT_SERVER_START)
    res = server.make_request("POST", "/chat/completions", data={
        "max_tokens": 256,
        "messages": [
            {"role": "system", "content": "You are a coding assistant."},
            {"role": "user", "content": "say hello world with python"},
        ],
        "tools": [PYTHON_TOOL],
        # Note: without these greedy params, Functionary v3.2 writes `def hello_world():\n    print("Hello, World!")\nhello_world()` which is correct but a pain to test.
        "temperature": 0.0,
        "top_k": 1,
        "top_p": 1.0,
    }, timeout=TIMEOUT_HTTP_REQUEST)
    assert res.status_code == 200, f"Expected status code 200, got {res.status_code}"
    choice = res.body["choices"][0]
    tool_calls = choice["message"].get("tool_calls")
    assert tool_calls and len(tool_calls) == 1, f'Expected 1 tool call in {choice["message"]}'
    tool_call = tool_calls[0]
    assert tool_call["function"]["name"] == PYTHON_TOOL["function"]["name"]
    actual_arguments = tool_call["function"]["arguments"]
    if expected_arguments_override is not None:
        assert actual_arguments == expected_arguments_override
    else:
        actual_arguments = json.loads(actual_arguments)
        assert 'code' in actual_arguments, f"code not found in {json.dumps(actual_arguments)}"
        code = actual_arguments["code"]
        assert isinstance(code, str), f"Expected code to be a string, got {type(code)}: {json.dumps(code)}"
        assert re.match(r'''print\(("[Hh]ello,? [Ww]orld!?"|'[Hh]ello,? [Ww]orld!?')\)''', code), f'Expected hello world, got {code}'
