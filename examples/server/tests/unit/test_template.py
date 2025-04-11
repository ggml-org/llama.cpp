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

TIMEOUT_SERVER_START = 15*60

@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.tinyllama2()
    server.model_alias = "tinyllama-2"
    server.server_port = 8081
    server.n_slots = 1


@pytest.mark.parametrize("tools", [None, [], [TEST_TOOL]])
@pytest.mark.parametrize("template_name,format", [
    ("meta-llama-Llama-3.3-70B-Instruct",    "%d %b %Y"),
    ("fireworks-ai-llama-3-firefunction-v2", "%b %d %Y"),
])
def test_date_inside_prompt(template_name: str, format: str, tools: list[dict]):
    global server
    server.jinja = True
    server.chat_template_file = f'../../../models/templates/{template_name}.jinja'
    server.start(timeout_seconds=TIMEOUT_SERVER_START)

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
