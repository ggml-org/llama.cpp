import os
import pytest
from utils import *
import base64
import requests
import time


server = ServerPreset.tinygemma3()


def get_img_base64(url: str) -> str:
    resp = requests.get(url)
    resp.raise_for_status()
    return base64.b64encode(resp.content).decode("utf-8")


@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.tinygemma3()
    server.server_slots = True
    server.slot_save_path = "./tmp"
    server.temperature = 0.0


def test_mtmd_embd_cache_hit_and_persistence(capfd):
    global server

    cache_path = os.path.join(server.slot_save_path, "mtmd_embd_cache.bin")
    for p in (cache_path, cache_path + ".tmp"):
        try:
            os.remove(p)
        except FileNotFoundError:
            pass

    server.start()

    # Wait for model to fully load and verify multimodal support
    # The model may take a moment to initialize mmproj
    for _ in range(10):
        res = server.make_request("GET", "/models", data={})
        if res.status_code == 200:
            model_info = res.body["models"][0]
            if "multimodal" in model_info.get("capabilities", []):
                break
        time.sleep(0.5)
    else:
        pytest.skip("Model does not support multimodal - may need mmproj file or model not fully loaded")

    img_url = "https://huggingface.co/ggml-org/tinygemma3-GGUF/resolve/main/test/11_truck.png"
    img_b64 = get_img_base64(img_url)

    prompt = {
        "prompt_string": "What is this: <__media__>\n",
        "multimodal_data": [img_b64],
    }

    # First request should populate the cache (miss -> encode)
    res = server.make_request("POST", "/completions", data={
        "temperature": 0.0,
        "top_k": 1,
        "prompt": prompt,
        "id_slot": 0,
        "cache_prompt": False,
    })
    assert res.status_code == 200

    assert os.path.exists(cache_path)
    assert os.path.getsize(cache_path) > 0

    # Clear captured output before the hit request
    capfd.readouterr()

    # Second request with same image should hit the embedding cache
    res = server.make_request("POST", "/completions", data={
        "temperature": 0.0,
        "top_k": 1,
        "prompt": prompt,
        "id_slot": 0,
        "cache_prompt": False,
    })
    assert res.status_code == 200

    out, err = capfd.readouterr()
    assert "MTMD_EMBD_CACHE_HIT" in (out + err)

    # Restart server and ensure persisted cache is used
    server.stop()
    capfd.readouterr()

    server = ServerPreset.tinygemma3()
    server.server_slots = True
    server.slot_save_path = "./tmp"
    server.temperature = 0.0
    server.start()

    # Wait for multimodal support again after restart
    for _ in range(10):
        res = server.make_request("GET", "/models", data={})
        if res.status_code == 200:
            model_info = res.body["models"][0]
            if "multimodal" in model_info.get("capabilities", []):
                break
        time.sleep(0.5)
    else:
        pytest.skip("Model does not support multimodal after restart")

    capfd.readouterr()

    res = server.make_request("POST", "/completions", data={
        "temperature": 0.0,
        "top_k": 1,
        "prompt": prompt,
        "id_slot": 0,
        "cache_prompt": False,
    })
    assert res.status_code == 200

    out, err = capfd.readouterr()
    assert "MTMD_EMBD_CACHE_HIT" in (out + err)

