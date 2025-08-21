#!/usr/bin/env python
import pytest

# ensure grandparent path is in sys.path
from pathlib import Path
import sys
path = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(path))

from utils import *
from enum import Enum

server: ServerProcess

TIMEOUT_SERVER_START = 15*60
TIMEOUT_HTTP_REQUEST = 60

@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.tinyllama2()
    server.model_alias = "tinyllama-2-tool-call"
    server.server_port = 8081
    server.n_slots = 1
    server.n_ctx = 8192
    server.n_batch = 2048

class CompletionMode(Enum):
    NORMAL = "normal"
    STREAMED = "streamed"

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

def do_test_completion_with_required_tool_tiny(server: ServerProcess, tool: dict, argument_key: str | None, n_predict, **kwargs):
    body = server.make_any_request("POST", "/v1/chat/completions", data={
        "max_tokens": n_predict,
        "messages": [
            {"role": "system", "content": "You are a coding assistant."},
            {"role": "user", "content": "Write an example"},
        ],
        "tool_choice": "required",
        "tools": [tool],
        "parallel_tool_calls": False,
        **kwargs,
    })
    # assert res.status_code == 200, f"Expected status code 200, got {res.status_code}"
    choice = body["choices"][0]
    tool_calls = choice["message"].get("tool_calls")
    assert tool_calls and len(tool_calls) == 1, f'Expected 1 tool call in {choice["message"]}'
    tool_call = tool_calls[0]
    assert choice["message"].get("content") in (None, ""), f'Expected no content in {choice["message"]}'
    # assert len(tool_call.get("id", "")) > 0, f'Expected non empty tool call id in {tool_call}'
    expected_function_name = "python" if tool["type"] == "code_interpreter" else tool["function"]["name"]
    assert expected_function_name == tool_call["function"]["name"]
    actual_arguments = tool_call["function"]["arguments"]
    assert isinstance(actual_arguments, str)
    if argument_key is not None:
        actual_arguments = json.loads(actual_arguments)
        assert argument_key in actual_arguments, f"tool arguments: {json.dumps(actual_arguments)}, expected: {argument_key}"


@pytest.mark.parametrize("stream", [CompletionMode.NORMAL, CompletionMode.STREAMED])
@pytest.mark.parametrize("template_name,tool,argument_key", [
    ("google-gemma-2-2b-it",                          TEST_TOOL,            "success"),
    ("google-gemma-2-2b-it",                          TEST_TOOL,            "success"),
    ("meta-llama-Llama-3.3-70B-Instruct",             TEST_TOOL,            "success"),
    ("meta-llama-Llama-3.3-70B-Instruct",             TEST_TOOL,            "success"),
    ("meta-llama-Llama-3.3-70B-Instruct",             PYTHON_TOOL,          "code"),
    ("meta-llama-Llama-3.3-70B-Instruct",             PYTHON_TOOL,          "code"),
])
def test_completion_with_required_tool_tiny_fast(template_name: str, tool: dict, argument_key: str | None, stream: CompletionMode):
    global server
    n_predict = 1024
    # server = ServerPreset.stories15m_moe()
    server.jinja = True
    server.n_predict = n_predict
    server.chat_template_file = f'../../../models/templates/{template_name}.jinja'
    server.start(timeout_seconds=TIMEOUT_SERVER_START)
    do_test_completion_with_required_tool_tiny(server, tool, argument_key, n_predict, stream=stream == CompletionMode.STREAMED, temperature=0.0, top_k=1, top_p=1.0)


@pytest.mark.slow
@pytest.mark.parametrize("stream", [CompletionMode.NORMAL, CompletionMode.STREAMED])
@pytest.mark.parametrize("template_name,tool,argument_key", [
    ("meta-llama-Llama-3.1-8B-Instruct",              TEST_TOOL,            "success"),
    ("meta-llama-Llama-3.1-8B-Instruct",              PYTHON_TOOL,          "code"),

    ("meetkai-functionary-medium-v3.1",               TEST_TOOL,            "success"),
    ("meetkai-functionary-medium-v3.1",               PYTHON_TOOL,          "code"),

    ("meetkai-functionary-medium-v3.2",               TEST_TOOL,            "success"),
    # Functionary v3.2 format supports raw python content, which w/ a dummy stories model will never end on its own.
    # ("meetkai-functionary-medium-v3.2",               PYTHON_TOOL,          "code"),

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
    # ("fireworks-ai-llama-3-firefunction-v2",          PYTHON_TOOL,          "codeFalse), True),
    # ("fireworks-ai-llama-3-firefunction-v2",          PYTHON_TOOL,          "code"),

])
def test_completion_with_required_tool_tiny_slow(template_name: str, tool: dict, argument_key: str | None, stream: CompletionMode):
    global server
    n_predict = 512
    # server = ServerPreset.stories15m_moe()
    server.jinja = True
    server.n_predict = n_predict
    server.chat_template_file = f'../../../models/templates/{template_name}.jinja'
    server.start(timeout_seconds=TIMEOUT_SERVER_START)
    do_test_completion_with_required_tool_tiny(server, tool, argument_key, n_predict, stream=stream == CompletionMode.STREAMED)


@pytest.mark.slow
@pytest.mark.parametrize("stream", [CompletionMode.NORMAL, CompletionMode.STREAMED])
@pytest.mark.parametrize("tool,argument_key,hf_repo,template_override", [
    (TEST_TOOL,    "success",  "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF:Q4_K_M", None),
    (PYTHON_TOOL,  "code",     "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF:Q4_K_M", None),
    (PYTHON_TOOL,  "code",     "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF:Q4_K_M", "chatml"),

    (TEST_TOOL,    "success",  "bartowski/gemma-2-2b-it-GGUF:Q4_K_M",              None),
    (PYTHON_TOOL,  "code",     "bartowski/gemma-2-2b-it-GGUF:Q4_K_M",              None),
    (PYTHON_TOOL,  "code",     "bartowski/gemma-2-2b-it-GGUF:Q4_K_M",              "chatml"),

    (TEST_TOOL,    "success",  "bartowski/Phi-3.5-mini-instruct-GGUF:Q4_K_M",      None),
    (PYTHON_TOOL,  "code",     "bartowski/Phi-3.5-mini-instruct-GGUF:Q4_K_M",      None),
    (PYTHON_TOOL,  "code",     "bartowski/Phi-3.5-mini-instruct-GGUF:Q4_K_M",      "chatml"),

    (TEST_TOOL,    "success",  "bartowski/Qwen2.5-1.5B-Instruct-GGUF:Q4_K_M",      None),
    (PYTHON_TOOL,  "code",     "bartowski/Qwen2.5-1.5B-Instruct-GGUF:Q4_K_M",      None),
    (PYTHON_TOOL,  "code",     "bartowski/Qwen2.5-1.5B-Instruct-GGUF:Q4_K_M",      "chatml"),

    (TEST_TOOL,    "success",  "bartowski/Qwen2.5-Coder-3B-Instruct-GGUF:Q4_K_M",      None),
    (PYTHON_TOOL,  "code",     "bartowski/Qwen2.5-Coder-3B-Instruct-GGUF:Q4_K_M",      None),
    (PYTHON_TOOL,  "code",     "bartowski/Qwen2.5-Coder-3B-Instruct-GGUF:Q4_K_M",      "chatml"),

    (TEST_TOOL,    "success",  "bartowski/Qwen2.5-7B-Instruct-GGUF:Q4_K_M",        None),
    (PYTHON_TOOL,  "code",     "bartowski/Qwen2.5-7B-Instruct-GGUF:Q4_K_M",        None),
    (PYTHON_TOOL,  "code",     "bartowski/Qwen2.5-7B-Instruct-GGUF:Q4_K_M",        "chatml"),

    (TEST_TOOL,    "success",  "bartowski/Hermes-2-Pro-Llama-3-8B-GGUF:Q4_K_M", ("NousResearch/Hermes-2-Pro-Llama-3-8B", "tool_use")),
    (PYTHON_TOOL,  "code",     "bartowski/Hermes-2-Pro-Llama-3-8B-GGUF:Q4_K_M", ("NousResearch/Hermes-2-Pro-Llama-3-8B", "tool_use")),
    (PYTHON_TOOL,  "code",     "bartowski/Hermes-2-Pro-Llama-3-8B-GGUF:Q4_K_M", "chatml"),

    (TEST_TOOL,    "success",  "bartowski/Hermes-3-Llama-3.1-8B-GGUF:Q4_K_M",   ("NousResearch/Hermes-3-Llama-3.1-8B", "tool_use")),
    (PYTHON_TOOL,  "code",     "bartowski/Hermes-3-Llama-3.1-8B-GGUF:Q4_K_M",   ("NousResearch/Hermes-3-Llama-3.1-8B", "tool_use")),
    (PYTHON_TOOL,  "code",     "bartowski/Hermes-3-Llama-3.1-8B-GGUF:Q4_K_M",   "chatml"),

    # (TEST_TOOL,    "success",  "bartowski/Mistral-Nemo-Instruct-2407-GGUF:Q4_K_M", None),
    # (PYTHON_TOOL,  "code",     "bartowski/Mistral-Nemo-Instruct-2407-GGUF:Q4_K_M", None),
    # (PYTHON_TOOL,  "code",     "bartowski/Mistral-Nemo-Instruct-2407-GGUF:Q4_K_M", "chatml"),

    (TEST_TOOL,    "success",  "bartowski/functionary-small-v3.2-GGUF:Q4_K_M",       ("meetkai/functionary-medium-v3.2", None)),
    (PYTHON_TOOL,  "code",     "bartowski/functionary-small-v3.2-GGUF:Q4_K_M",       ("meetkai/functionary-medium-v3.2", None)),
    (PYTHON_TOOL,  "code",     "bartowski/functionary-small-v3.2-GGUF:Q4_K_M",       "chatml"),

    (TEST_TOOL,    "success",  "bartowski/Llama-3.2-3B-Instruct-GGUF:Q4_K_M",      ("meta-llama/Llama-3.2-3B-Instruct", None)),
    (PYTHON_TOOL,  "code",     "bartowski/Llama-3.2-3B-Instruct-GGUF:Q4_K_M",      ("meta-llama/Llama-3.2-3B-Instruct", None)),
    (PYTHON_TOOL,  "code",     "bartowski/Llama-3.2-3B-Instruct-GGUF:Q4_K_M",      "chatml"),

    (TEST_TOOL,    "success",  "bartowski/Llama-3.2-1B-Instruct-GGUF:Q4_K_M",      ("meta-llama/Llama-3.2-3B-Instruct", None)),
    (PYTHON_TOOL,  "code",     "bartowski/Llama-3.2-1B-Instruct-GGUF:Q4_K_M",      ("meta-llama/Llama-3.2-3B-Instruct", None)),
    (PYTHON_TOOL,  "code",     "bartowski/Llama-3.2-1B-Instruct-GGUF:Q4_K_M",      "chatml"),

    (TEST_TOOL,    "success",  "bartowski/DeepSeek-R1-Distill-Qwen-7B-GGUF:Q4_K_M", None),
    (PYTHON_TOOL,  "code",     "bartowski/DeepSeek-R1-Distill-Qwen-7B-GGUF:Q4_K_M", None),
])
def test_completion_with_required_tool_real_model(tool: dict, argument_key: str | None, hf_repo: str, template_override: str | Tuple[str, str | None] | None, stream: CompletionMode):
    global server
    n_predict = 512
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
    body = server.make_any_request("POST", "/v1/chat/completions", data={
        "max_tokens": n_predict,
        "messages": [
            {"role": "system", "content": "You are a coding assistant."},
            {"role": "user", "content": "Write an example"},
        ],
        "tool_choice": "required",
        "tools": [tool],
        "parallel_tool_calls": False,
        "stream": stream == CompletionMode.STREAMED,
        "temperature": 0.0,
        "top_k": 1,
        "top_p": 1.0,
    }, timeout=TIMEOUT_HTTP_REQUEST)
    choice = body["choices"][0]
    tool_calls = choice["message"].get("tool_calls")
    assert tool_calls and len(tool_calls) == 1, f'Expected 1 tool call in {choice["message"]}'
    tool_call = tool_calls[0]
    # assert choice["message"].get("content") in (None, ""), f'Expected no content in {choice["message"]}'
    expected_function_name = "python" if tool["type"] == "code_interpreter" else tool["function"]["name"]
    assert expected_function_name == tool_call["function"]["name"]
    actual_arguments = tool_call["function"]["arguments"]
    assert isinstance(actual_arguments, str)
    if argument_key is not None:
        actual_arguments = json.loads(actual_arguments)
        assert argument_key in actual_arguments, f"tool arguments: {json.dumps(actual_arguments)}, expected: {argument_key}"


def do_test_completion_without_tool_call(server: ServerProcess, n_predict: int, tools: list[dict], tool_choice: str | None, **kwargs):
    body = server.make_any_request("POST", "/v1/chat/completions", data={
        "max_tokens": n_predict,
        "messages": [
            {"role": "system", "content": "You are a coding assistant."},
            {"role": "user", "content": "say hello world with python"},
        ],
        "tools": tools if tools else None,
        "tool_choice": tool_choice,
        **kwargs,
    }, timeout=TIMEOUT_HTTP_REQUEST)
    choice = body["choices"][0]
    assert choice["message"].get("tool_calls") is None, f'Expected no tool call in {choice["message"]}'


@pytest.mark.parametrize("stream", [CompletionMode.NORMAL, CompletionMode.STREAMED])
@pytest.mark.parametrize("template_name,n_predict,tools,tool_choice", [
    ("meta-llama-Llama-3.3-70B-Instruct",         128, [],            None),
    ("meta-llama-Llama-3.3-70B-Instruct",         128, [TEST_TOOL],   None),
    ("meta-llama-Llama-3.3-70B-Instruct",         128, [PYTHON_TOOL], 'none'),
])
def test_completion_without_tool_call_fast(template_name: str, n_predict: int, tools: list[dict], tool_choice: str | None, stream: CompletionMode):
    global server
    server.n_predict = n_predict
    server.jinja = True
    server.chat_template_file = f'../../../models/templates/{template_name}.jinja'
    server.start(timeout_seconds=TIMEOUT_SERVER_START)
    do_test_completion_without_tool_call(server, n_predict, tools, tool_choice, stream=stream == CompletionMode.STREAMED)


@pytest.mark.slow
@pytest.mark.parametrize("stream", [CompletionMode.NORMAL, CompletionMode.STREAMED])
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
def test_completion_without_tool_call_slow(template_name: str, n_predict: int, tools: list[dict], tool_choice: str | None, stream: CompletionMode):
    global server
    server.n_predict = n_predict
    server.jinja = True
    server.chat_template_file = f'../../../models/templates/{template_name}.jinja'
    server.start(timeout_seconds=TIMEOUT_SERVER_START)
    do_test_completion_without_tool_call(server, n_predict, tools, tool_choice, stream=stream == CompletionMode.STREAMED)


@pytest.mark.slow
@pytest.mark.parametrize("stream", [CompletionMode.NORMAL, CompletionMode.STREAMED])
@pytest.mark.parametrize("hf_repo,template_override", [
    ("bartowski/Meta-Llama-3.1-8B-Instruct-GGUF:Q4_K_M", None),
    ("bartowski/Meta-Llama-3.1-8B-Instruct-GGUF:Q4_K_M", "chatml"),

    ("bartowski/Phi-3.5-mini-instruct-GGUF:Q4_K_M",      None),
    ("bartowski/Phi-3.5-mini-instruct-GGUF:Q4_K_M",      "chatml"),

    ("bartowski/Qwen2.5-1.5B-Instruct-GGUF:Q4_K_M",      None),
    ("bartowski/Qwen2.5-1.5B-Instruct-GGUF:Q4_K_M",      "chatml"),

    ("bartowski/Qwen2.5-Coder-3B-Instruct-GGUF:Q4_K_M",      None),
    ("bartowski/Qwen2.5-Coder-3B-Instruct-GGUF:Q4_K_M",      "chatml"),

    ("bartowski/Qwen2.5-7B-Instruct-GGUF:Q4_K_M",        None),
    ("bartowski/Qwen2.5-7B-Instruct-GGUF:Q4_K_M",        "chatml"),

    ("bartowski/Hermes-2-Pro-Llama-3-8B-GGUF:Q4_K_M",    ("NousResearch/Hermes-2-Pro-Llama-3-8B", "tool_use")),
    ("bartowski/Hermes-2-Pro-Llama-3-8B-GGUF:Q4_K_M",    "chatml"),

    ("bartowski/Hermes-3-Llama-3.1-8B-GGUF:Q4_K_M",      ("NousResearch/Hermes-3-Llama-3.1-8B", "tool_use")),
    ("bartowski/Hermes-3-Llama-3.1-8B-GGUF:Q4_K_M",      "chatml"),

    # ("bartowski/Mistral-Nemo-Instruct-2407-GGUF:Q4_K_M", None),
    # ("bartowski/Mistral-Nemo-Instruct-2407-GGUF:Q4_K_M", "chatml"),

    # ("bartowski/functionary-small-v3.2-GGUF:Q8_0",       ("meetkai/functionary-medium-v3.2", None)),
    # ("bartowski/functionary-small-v3.2-GGUF:Q8_0",       "chatml"),

    ("bartowski/Llama-3.2-3B-Instruct-GGUF:Q4_K_M",      ("meta-llama/Llama-3.2-3B-Instruct", None)),
    ("bartowski/Llama-3.2-3B-Instruct-GGUF:Q4_K_M",      "chatml"),

    ("bartowski/c4ai-command-r7b-12-2024-GGUF:Q6_K_L",   ("CohereForAI/c4ai-command-r7b-12-2024", "tool_use")),

    ("bartowski/DeepSeek-R1-Distill-Qwen-7B-GGUF:Q4_K_M", None),

    # Note: gemma-2-2b-it knows itself as "model", not "assistant", so we don't test the ill-suited chatml on it.
    ("bartowski/gemma-2-2b-it-GGUF:Q4_K_M",              None),

    # ("bartowski/Llama-3.2-1B-Instruct-GGUF:Q4_K_M", ("meta-llama/Llama-3.2-3B-Instruct", None)),
])
def test_weather(hf_repo: str, template_override: str | Tuple[str, str | None] | None, stream: CompletionMode):
    global server
    n_predict = 512
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
    do_test_weather(server, stream=stream == CompletionMode.STREAMED, max_tokens=n_predict)


def do_test_weather(server: ServerProcess, **kwargs):
    body = server.make_any_request("POST", "/v1/chat/completions", data={
        "messages": [
            {"role": "system", "content": "You are a chatbot that uses tools/functions. Dont overthink things."},
            {"role": "user", "content": "What is the weather in Istanbul?"},
        ],
        "tools": [WEATHER_TOOL],
        **kwargs,
    }, timeout=TIMEOUT_HTTP_REQUEST)
    choice = body["choices"][0]
    tool_calls = choice["message"].get("tool_calls")
    assert tool_calls and len(tool_calls) == 1, f'Expected 1 tool call in {choice["message"]}'
    tool_call = tool_calls[0]
    # assert choice["message"].get("content") in (None, ""), f'Expected no content in {choice["message"]}'
    assert tool_call["function"]["name"] == WEATHER_TOOL["function"]["name"], f'Expected weather tool call, got {tool_call["function"]["name"]}'
    # assert len(tool_call.get("id", "")) > 0, f'Expected non empty tool call id in {tool_call}'
    actual_arguments = json.loads(tool_call["function"]["arguments"])
    assert 'location' in actual_arguments, f"location not found in {json.dumps(actual_arguments)}"
    location = actual_arguments["location"]
    assert isinstance(location, str), f"Expected location to be a string, got {type(location)}: {json.dumps(location)}"
    assert re.match('^Istanbul(( |, ?)(TR|Turkey|Türkiye))?$', location), f'Expected Istanbul for location, got {location}'


@pytest.mark.slow
@pytest.mark.parametrize("stream", [CompletionMode.NORMAL, CompletionMode.STREAMED])
@pytest.mark.parametrize("result_override,n_predict,hf_repo,template_override", [
    (None,                                           128,  "bartowski/Phi-3.5-mini-instruct-GGUF:Q4_K_M",       "chatml"),
    (None,                                           128,  "bartowski/Qwen2.5-Coder-3B-Instruct-GGUF:Q4_K_M", None),
    (None,                                           128,  "bartowski/Qwen2.5-Coder-3B-Instruct-GGUF:Q4_K_M", "chatml"),
    (None,                                           128,  "bartowski/Qwen2.5-7B-Instruct-GGUF:Q4_K_M",         "chatml"),
    (None,                                           128,  "bartowski/Hermes-2-Pro-Llama-3-8B-GGUF:Q4_K_M",     ("NousResearch/Hermes-2-Pro-Llama-3-8B", "tool_use")),
    (None,                                           128,  "bartowski/Hermes-3-Llama-3.1-8B-GGUF:Q4_K_M",       ("NousResearch/Hermes-3-Llama-3.1-8B", "tool_use")),
    (None,                                           128,  "bartowski/functionary-small-v3.2-GGUF:Q8_0",        ("meetkai/functionary-medium-v3.2", None)),
    (None,                                           128,  "bartowski/Mistral-Nemo-Instruct-2407-GGUF:Q4_K_M",  None),
    (None,                                           128,  "bartowski/Mistral-Nemo-Instruct-2407-GGUF:Q4_K_M",  "chatml"),
    (None,                                           128,  "bartowski/Phi-3.5-mini-instruct-GGUF:Q4_K_M",       None),
    ("[\\s\\S]*?\\*\\*\\s*0.5($|\\*\\*)",            8192, "bartowski/DeepSeek-R1-Distill-Qwen-7B-GGUF:Q4_K_M", ("llama-cpp-deepseek-r1", None)),

    # TODO: fix these (wrong results, either didn't respect decimal instruction or got wrong value)
    # (None,                                           128,  "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF:Q4_K_M",  None),
    # ("[\\s\\S]*?\\*\\*\\s*0.5($|\\*\\*)",            8192, "bartowski/DeepSeek-R1-Distill-Qwen-7B-GGUF:Q4_K_M", None),
])
def test_calc_result(result_override: str | None, n_predict: int, hf_repo: str, template_override: str | Tuple[str, str | None] | None, stream: CompletionMode):
    global server
    server.jinja = True
    server.n_ctx = 8192 * 2
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
    do_test_calc_result(server, result_override, n_predict, stream=stream == CompletionMode.STREAMED)


def do_test_calc_result(server: ServerProcess, result_override: str | None, n_predict: int, **kwargs):
    body = server.make_any_request("POST", "/v1/chat/completions", data={
        "max_tokens": n_predict,
        "messages": [
            {"role": "system", "content": "You are a tools-calling assistant. You express numerical values with at most two decimals."},
            {"role": "user", "content": "What's the y coordinate of a point on the unit sphere at angle 30 degrees?"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_6789",
                        "type": "function",
                        "function": {
                            "name": "calculate",
                            "arguments": "{\"expression\":\"sin(30 * pi / 180)\"}"
                        }
                    }
                ]
            },
            {
                "role": "tool",
                "name": "calculate",
                "content": "0.55644242476",
                "tool_call_id": "call_6789"
            }
        ],
        "tools": [
            {
                "type":"function",
                "function":{
                    "name":"calculate",
                    "description":"A calculator function that computes values of arithmetic expressions in the Python syntax",
                    "parameters":{
                        "type":"object",
                        "properties":{
                            "expression":{
                            "type":"string",
                            "description":"An arithmetic expression to compute the value of (Python syntad, assuming all floats)"
                            }
                        },
                        "required":["expression"]
                    }
                }
            }
        ],
        **kwargs,
    }, timeout=TIMEOUT_HTTP_REQUEST)
    choice = body["choices"][0]
    tool_calls = choice["message"].get("tool_calls")
    assert tool_calls is None, f'Expected no tool call in {choice["message"]}'
    content = choice["message"].get("content")
    assert content is not None, f'Expected content in {choice["message"]}'
    if result_override is not None:
        assert re.match(result_override, content), f'Expected {result_override}, got {content}'
    else:
        assert re.match('^[\\s\\S]*?((That\'s|\\bis) (approximately )?)?\\b0\\.(5\\b|56\\b|556)', content), \
            f'Expected something like "The y coordinate is 0.56.", got {content}'


@pytest.mark.slow
@pytest.mark.parametrize("stream", [CompletionMode.NORMAL, CompletionMode.STREAMED])
@pytest.mark.parametrize("n_predict,reasoning_format,expect_reasoning_content,expect_content,hf_repo,template_override", [
    (128, 'deepseek',   None, "^The sum of 102 and 7 is 109[\\s\\S]*",                                       "bartowski/Phi-3.5-mini-instruct-GGUF:Q4_K_M",       None),
    (128,  None,        None, "^The sum of 102 and 7 is 109[\\s\\S]*",                                       "bartowski/Phi-3.5-mini-instruct-GGUF:Q4_K_M",       None),
    (1024, 'deepseek',  "I need to calculate the sum of 102 and 7[\\s\\S]*", "To find the sum of[\\s\\S]*",  "bartowski/DeepSeek-R1-Distill-Qwen-7B-GGUF:Q4_K_M", None),
    (1024, 'deepseek',  "First, I [\\s\\S]*", "To find the sum of[\\s\\S]*",                                 "bartowski/DeepSeek-R1-Distill-Qwen-7B-GGUF:Q4_K_M", ("llama-cpp-deepseek-r1", None)),
    # (1024, 'none',      CompletionMode.NORMAL,   None, "^(<think>\\s*)?I need[\\s\\S]*?</think>\\s*To find[\\s\\S]*",                 "bartowski/DeepSeek-R1-Distill-Qwen-7B-GGUF:Q4_K_M", None),
    # (128,  'deepseek',  None, "^Okay, let me figure out the sum of 102 and 7[\\s\\S]*",                      "bartowski/Qwen_QwQ-32B-GGUF:Q4_K_M",                None),
])
def test_thoughts(n_predict: int, reasoning_format: Literal['deepseek', 'none'] | None, expect_content: str | None, expect_reasoning_content: str | None, hf_repo: str, template_override: str | Tuple[str, str | None] | None, stream: CompletionMode):
    global server
    server.reasoning_format = reasoning_format
    server.jinja = True
    server.n_ctx = 8192 * 2
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
    body = server.make_any_request("POST", "/v1/chat/completions", data={
        "max_tokens": n_predict,
        "messages": [
            {"role": "user", "content": "What's the sum of 102 and 7?"},
        ],
        "stream": stream == CompletionMode.STREAMED,
    }, timeout=TIMEOUT_HTTP_REQUEST)
    choice = body["choices"][0]
    assert choice["message"].get("tool_calls") is None, f'Expected no tool call in {choice["message"]}'

    content = choice["message"].get("content")
    if expect_content is None:
        assert choice["message"].get("content") in (None, ""), f'Expected no content in {choice["message"]}'
    else:
        assert re.match(expect_content, content), f'Expected {expect_content}, got {content}'

    reasoning_content = choice["message"].get("reasoning_content")
    if expect_reasoning_content is None:
        assert reasoning_content is None, f'Expected no reasoning content in {choice["message"]}'
    else:
        assert re.match(expect_reasoning_content, reasoning_content), f'Expected {expect_reasoning_content}, got {reasoning_content}'


@pytest.mark.slow
@pytest.mark.parametrize("stream", [CompletionMode.NORMAL, CompletionMode.STREAMED])
@pytest.mark.parametrize("hf_repo,template_override", [
    ("bartowski/DeepSeek-R1-Distill-Qwen-7B-GGUF:Q4_K_M", None),

    ("bartowski/Phi-3.5-mini-instruct-GGUF:Q4_K_M",      None),
    ("bartowski/Phi-3.5-mini-instruct-GGUF:Q4_K_M",      "chatml"),

    ("bartowski/functionary-small-v3.2-GGUF:Q8_0",       ("meetkai-functionary-medium-v3.2", None)),
    ("bartowski/functionary-small-v3.2-GGUF:Q8_0",       "chatml"),

    # ("bartowski/Meta-Llama-3.1-8B-Instruct-GGUF:Q4_K_M", None),
    ("bartowski/Meta-Llama-3.1-8B-Instruct-GGUF:Q4_K_M", "chatml"),

    ("bartowski/Llama-3.2-1B-Instruct-GGUF:Q4_K_M",      ("meta-llama-Llama-3.2-3B-Instruct", None)),
    ("bartowski/Llama-3.2-1B-Instruct-GGUF:Q4_K_M",      None),

    ("bartowski/Llama-3.2-3B-Instruct-GGUF:Q4_K_M",      ("meta-llama-Llama-3.2-3B-Instruct", None)),
    ("bartowski/Llama-3.2-3B-Instruct-GGUF:Q4_K_M",      None),

    ("bartowski/Qwen2.5-7B-Instruct-GGUF:Q4_K_M",        None),
    ("bartowski/Qwen2.5-7B-Instruct-GGUF:Q4_K_M",        "chatml"),

    ("bartowski/Hermes-2-Pro-Llama-3-8B-GGUF:Q4_K_M",    ("NousResearch/Hermes-2-Pro-Llama-3-8B", "tool_use")),
    ("bartowski/Hermes-2-Pro-Llama-3-8B-GGUF:Q4_K_M",    "chatml"),

    ("bartowski/Hermes-3-Llama-3.1-8B-GGUF:Q4_K_M",      ("NousResearch-Hermes-3-Llama-3.1-8B", "tool_use")),
    ("bartowski/Hermes-3-Llama-3.1-8B-GGUF:Q4_K_M",      "chatml"),

    ("bartowski/Mistral-Nemo-Instruct-2407-GGUF:Q4_K_M", None),
    ("bartowski/Mistral-Nemo-Instruct-2407-GGUF:Q4_K_M", "chatml"),

    ("bartowski/gemma-2-2b-it-GGUF:Q4_K_M",              None),
    ("bartowski/gemma-2-2b-it-GGUF:Q4_K_M",              "chatml"),
])
def test_hello_world(hf_repo: str, template_override: str | Tuple[str, str | None] | None, stream: CompletionMode):
    global server
    n_predict = 512 # High because of DeepSeek R1
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

    do_test_hello_world(server, stream=stream == CompletionMode.STREAMED, max_tokens=n_predict)


def do_test_hello_world(server: ServerProcess, **kwargs):
    body = server.make_any_request("POST", "/v1/chat/completions", data={
        "messages": [
            {"role": "system", "content": "You are a tool-calling agent."},
            {"role": "user", "content": "say hello world with python"},
        ],
        "tools": [PYTHON_TOOL],
        **kwargs,
    }, timeout=TIMEOUT_HTTP_REQUEST)
    choice = body["choices"][0]
    tool_calls = choice["message"].get("tool_calls")
    assert tool_calls and len(tool_calls) == 1, f'Expected 1 tool call in {choice["message"]}'
    tool_call = tool_calls[0]
    # assert choice["message"].get("content") in (None, ""), f'Expected no content in {choice["message"]}'
    assert tool_call["function"]["name"] == PYTHON_TOOL["function"]["name"]
    # assert len(tool_call.get("id", "")) > 0, f'Expected non empty tool call id in {tool_call}'
    actual_arguments = json.loads(tool_call["function"]["arguments"])
    assert 'code' in actual_arguments, f"code not found in {json.dumps(actual_arguments)}"
    code = actual_arguments["code"]
    assert isinstance(code, str), f"Expected code to be a string, got {type(code)}: {json.dumps(code)}"
    assert re.match(r'''print\(("[Hh]ello,? [Ww]orld!?"|'[Hh]ello,? [Ww]orld!?')\)''', re.sub(r'#.*\n?', '', code)), f'Expected hello world, got {code}'
