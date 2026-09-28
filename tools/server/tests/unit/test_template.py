#!/usr/bin/env python
import pytest

# ensure grandparent path is in sys.path
from pathlib import Path
import sys

from unit.test_tool_call import TEST_TOOL
path = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(path))

import datetime
from utils import *
from typing import Literal

server: ServerProcess

@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.tinyllama2()
    server.model_alias = "tinyllama-2"
    server.n_slots = 1


@pytest.mark.parametrize("tools", [None, [], [TEST_TOOL]])
@pytest.mark.parametrize("template_name,reasoning,expected_end", [
    ("deepseek-ai-DeepSeek-R1-Distill-Qwen-32B",  "on", "<think>\n"),
    ("deepseek-ai-DeepSeek-R1-Distill-Qwen-32B","auto", "<think>\n"),
    ("deepseek-ai-DeepSeek-R1-Distill-Qwen-32B", "off", "<think>\n</think>"),

    ("Qwen-Qwen3-0.6B","auto", "<|im_start|>assistant\n"),
    ("Qwen-Qwen3-0.6B", "off", "<|im_start|>assistant\n<think>\n\n</think>\n\n"),

    ("Qwen-QwQ-32B","auto", "<|im_start|>assistant\n<think>\n"),
    ("Qwen-QwQ-32B", "off", "<|im_start|>assistant\n<think>\n</think>"),

    ("CohereForAI-c4ai-command-r7b-12-2024-tool_use","auto", "<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>"),
    ("CohereForAI-c4ai-command-r7b-12-2024-tool_use", "off", "<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|><|START_THINKING|><|END_THINKING|>"),
])
def test_reasoning(template_name: str, reasoning: Literal['on', 'off', 'auto'] | None, expected_end: str, tools: list[dict]):
    global server
    server.jinja = True
    server.reasoning = reasoning
    server.chat_template_file = f'../../../models/templates/{template_name}.jinja'
    server.start()

    res = server.make_request("POST", "/apply-template", data={
        "messages": [
            {"role": "user", "content": "What is today?"},
        ],
        "tools": tools,
    })
    assert res.status_code == 200
    prompt = res.body["prompt"]

    assert prompt.endswith(expected_end), f"Expected prompt to end with '{expected_end}', got '{prompt}'"


# Gate A: --no-jinja --reasoning off must match Jinja disable suffixes (legacy post-process).
# tools require --jinja; keep tools=None here.
@pytest.mark.parametrize("template_name,expected_end", [
    ("deepseek-ai-DeepSeek-R1-Distill-Qwen-32B", "<think>\n</think>"),
    ("Qwen-Qwen3-0.6B", "<|im_start|>assistant\n<think>\n\n</think>\n\n"),
    ("Qwen-QwQ-32B", "<|im_start|>assistant\n<think>\n</think>"),
    ("CohereForAI-c4ai-command-r7b-12-2024-tool_use",
     "<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|><|START_THINKING|><|END_THINKING|>"),
])
def test_reasoning_no_jinja_off(template_name: str, expected_end: str):
    global server
    server.jinja = False
    server.reasoning = "off"
    server.chat_template_file = f'../../../models/templates/{template_name}.jinja'
    server.start()

    res = server.make_request("POST", "/apply-template", data={
        "messages": [
            {"role": "user", "content": "What is today?"},
        ],
    })
    assert res.status_code == 200
    prompt = res.body["prompt"]
    assert prompt.endswith(expected_end), f"Expected prompt to end with '{expected_end}', got '{prompt}'"


# Gate B: chat_template_kwargs / reasoning_effort disable on legacy path
@pytest.mark.parametrize("body_extra", [
    {"chat_template_kwargs": {"enable_thinking": False}},
    {"reasoning_effort": "none"},
])
def test_reasoning_no_jinja_kwargs_off(body_extra: dict):
    global server
    server.jinja = False
    server.reasoning = "on"  # default on; request must override to off
    server.chat_template_file = '../../../models/templates/Qwen-Qwen3-0.6B.jinja'
    server.start()

    data = {
        "messages": [{"role": "user", "content": "What is today?"}],
        **body_extra,
    }
    res = server.make_request("POST", "/apply-template", data=data)
    assert res.status_code == 200
    expected_end = "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    assert res.body["prompt"].endswith(expected_end), f"got '{res.body['prompt']}'"


# Gate: explicit thinking on must not append Qwen3 empty-close pair under --no-jinja
def test_reasoning_no_jinja_on_no_false_disable():
    global server
    server.jinja = False
    server.reasoning = "on"
    server.chat_template_file = '../../../models/templates/Qwen-Qwen3-0.6B.jinja'
    server.start()

    res = server.make_request("POST", "/apply-template", data={
        "messages": [{"role": "user", "content": "What is today?"}],
    })
    assert res.status_code == 200
    prompt = res.body["prompt"]
    assert prompt.endswith("<|im_start|>assistant\n"), f"got '{prompt}'"
    assert "<think>\n\n</think>\n\n" not in prompt


# Gate D: non-thinking ChatML-like / Llama template must not gain <think> under --no-jinja --reasoning off
def test_reasoning_no_jinja_non_think_template():
    global server
    server.jinja = False
    server.reasoning = "off"
    server.chat_template_file = '../../../models/templates/meta-llama-Llama-3.3-70B-Instruct.jinja'
    server.start()

    res = server.make_request("POST", "/apply-template", data={
        "messages": [{"role": "user", "content": "What is today?"}],
    })
    assert res.status_code == 200
    prompt = res.body["prompt"]
    assert "<think>" not in prompt, f"unexpected think markers in non-think template: {prompt}"


@pytest.mark.parametrize("tools", [None, [], [TEST_TOOL]])
@pytest.mark.parametrize("template_name,format", [
    ("meta-llama-Llama-3.3-70B-Instruct",    "%d %b %Y"),
    ("fireworks-ai-llama-3-firefunction-v2", "%b %d %Y"),
])
def test_date_inside_prompt(template_name: str, format: str, tools: list[dict]):
    global server
    server.jinja = True
    server.chat_template_file = f'../../../models/templates/{template_name}.jinja'
    server.start()

    res = server.make_request("POST", "/apply-template", data={
        "messages": [
            {"role": "user", "content": "What is today?"},
        ],
        "tools": tools,
    })
    assert res.status_code == 200
    prompt = res.body["prompt"]

    today_str = datetime.date.today().strftime(format)
    assert today_str in prompt, f"Expected today's date ({today_str}) in content ({prompt})"


@pytest.mark.parametrize("add_generation_prompt", [False, True])
@pytest.mark.parametrize("template_name,expected_generation_prompt", [
    ("meta-llama-Llama-3.3-70B-Instruct",    "<|start_header_id|>assistant<|end_header_id|>"),
])
def test_add_generation_prompt(template_name: str, expected_generation_prompt: str, add_generation_prompt: bool):
    global server
    server.jinja = True
    server.chat_template_file = f'../../../models/templates/{template_name}.jinja'
    server.start()

    res = server.make_request("POST", "/apply-template", data={
        "messages": [
            {"role": "user", "content": "What is today?"},
        ],
        "add_generation_prompt": add_generation_prompt,
    })
    assert res.status_code == 200
    prompt = res.body["prompt"]

    if add_generation_prompt:
        assert expected_generation_prompt in prompt, f"Expected generation prompt ({expected_generation_prompt}) in content ({prompt})"
    else:
        assert expected_generation_prompt not in prompt, f"Did not expect generation prompt ({expected_generation_prompt}) in content ({prompt})"
