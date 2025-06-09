import json
import os
import tempfile
from pathlib import Path
import sys

import pytest

# ensure grandparent path is in sys.path
path = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(path))

from utils import *

server = ServerPreset.stories15m_moe()

LORA_FILE_URL = "https://huggingface.co/ggml-org/stories15M_MOE/resolve/main/moe_shakespeare15M.gguf"

@pytest.fixture(scope="module", autouse=True)
def create_server():
    global server
    server = ServerPreset.stories15m_moe()
    server.lora_files = [download_file(LORA_FILE_URL)]


def test_alias_presets_per_request():
    global server
    server.n_slots = 4

    preset_data = {
        "bedtime-stories": {
            "lora": [{"id": 0, "scale": 0.0}]
        },
        "shakespeare-light": {
            "lora": [{"id": 0, "scale": 0.3}]
        },
        "shakespeare-medium": {
            "lora": [{"id": 0, "scale": 0.7}]
        },
        "shakespeare-full": {
            "lora": [{"id": 0, "scale": 1.0}]
        }
    }

    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(preset_data, f)
        preset_file_path = f.name

    try:
        server.alias_presets_file = preset_file_path
        server.start()

        # running the same prompt with different model aliases, all in parallel
        # each prompt will be processed by a different slot
        prompt = "Look in thy glass"
        alias_config = [
            ("bedtime-stories", "(bright|day|many|happy)+"),
            ("bedtime-stories", "(bright|day|many|happy)+"),
            ("shakespeare-light", "(special|thing|gifted)+"),
            ("shakespeare-medium", "(far|from|home|away)+"),
            ("shakespeare-full", "(eye|love|glass|sun)+"),
            ("shakespeare-full", "(eye|love|glass|sun)+"),
        ]

        tasks = [(
            server.make_request,
            ("POST", "/completions", {
                "model": model_alias,
                "prompt": prompt,
                "seed": 42,
                "temperature": 0.0,
                "cache_prompt": False,
            })
        ) for model_alias, _ in alias_config]
        results = parallel_function_calls(tasks)

        assert all([res.status_code == 200 for res in results])
        for res, (_, re_test) in zip(results, alias_config):
            assert match_regex(re_test, res.body["content"])

    finally:
        server.stop()
        os.unlink(preset_file_path)

def test_alias_override():
    # test whether we honor the user's override even in case a preset is set
    global server
    server.n_slots = 2

    # Use the same preset data as test_alias_presets_per_request
    preset_data = {
        "bedtime-stories": {
            "lora": [{"id": 0, "scale": 0.0}]
        },
        "shakespeare-light": {
            "lora": [{"id": 0, "scale": 0.3}]
        },
        "shakespeare-medium": {
            "lora": [{"id": 0, "scale": 0.7}]
        },
        "shakespeare-full": {
            "lora": [{"id": 0, "scale": 1.0}]
        }
    }

    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(preset_data, f)
        preset_file_path = f.name

    try:
        server.alias_presets_file = preset_file_path
        server.start()

        prompt = "Look in thy glass"

        res1 = server.make_request("POST", "/completions", {
            "model": "bedtime-stories",
            "prompt": prompt,
            "cache_prompt": False,
        })

        # override to shakespeare
        res2 = server.make_request("POST", "/completions", {
            "model": "bedtime-stories",
            "prompt": prompt,
            "cache_prompt": False,
            "lora": [{"id": 0, "scale": 1.0}],
        })

        assert res1.status_code == 200
        assert res2.status_code == 200

        assert match_regex("(bright|day|many|happy)+", res1.body["content"])
        assert match_regex("(eye|love|glass|sun)+", res2.body["content"])
        assert res1.body["content"] != res2.body["content"]

    finally:
        server.stop()
        os.unlink(preset_file_path)
