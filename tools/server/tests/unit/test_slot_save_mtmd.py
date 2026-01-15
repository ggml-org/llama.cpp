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


def test_slot_save_restore_mtmd_image():
    global server
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

    # First prompt in slot 1 should be fully processed (including image)
    res = server.make_request("POST", "/completions", data={
        "temperature": 0.0,
        "top_k": 1,
        "prompt": prompt,
        "id_slot": 1,
        "cache_prompt": True,
    })
    assert res.status_code == 200
    prompt_n_full = res.body["timings"]["prompt_n"]
    assert prompt_n_full > 1

    # Save state of slot 1
    res = server.make_request("POST", "/slots/1?action=save", data={
        "filename": "slot_mtmd.bin",
    })
    assert res.status_code == 200

    # Restore into slot 0
    res = server.make_request("POST", "/slots/0?action=restore", data={
        "filename": "slot_mtmd.bin",
    })
    assert res.status_code == 200

    # After restore, same prompt in slot 0 should reuse cached KV (including image)
    res = server.make_request("POST", "/completions", data={
        "temperature": 0.0,
        "top_k": 1,
        "prompt": prompt,
        "id_slot": 0,
        "cache_prompt": True,
    })
    assert res.status_code == 200
    prompt_n_cached = res.body["timings"]["prompt_n"]
    assert prompt_n_cached < prompt_n_full

