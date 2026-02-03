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

server: ServerProcess

@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.tinyllama2()
    server.model_alias = "tinyllama-2"
    server.n_slots = 1


@pytest.mark.parametrize("tools", [None, [], [TEST_TOOL]])
@pytest.mark.parametrize("template_name,reasoning_budget,expected_end", [
    ("deepseek-ai-DeepSeek-R1-Distill-Qwen-32B", None, "<think>\n"),
    ("deepseek-ai-DeepSeek-R1-Distill-Qwen-32B",   -1, "<think>\n"),
    ("deepseek-ai-DeepSeek-R1-Distill-Qwen-32B",    0, "<think>\n</think>"),

    ("Qwen-Qwen3-0.6B", -1, "<|im_start|>assistant\n"),
    ("Qwen-Qwen3-0.6B",  0, "<|im_start|>assistant\n<think>\n\n</think>\n\n"),

    ("Qwen-QwQ-32B", -1, "<|im_start|>assistant\n<think>\n"),
    ("Qwen-QwQ-32B",  0, "<|im_start|>assistant\n<think>\n</think>"),

    ("CohereForAI-c4ai-command-r7b-12-2024-tool_use", -1, "<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>"),
    ("CohereForAI-c4ai-command-r7b-12-2024-tool_use",  0, "<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|><|START_THINKING|><|END_THINKING|>"),
])
def test_reasoning_budget(template_name: str, reasoning_budget: int | None, expected_end: str, tools: list[dict]):
    global server
    server.jinja = True
    server.reasoning_budget = reasoning_budget
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
