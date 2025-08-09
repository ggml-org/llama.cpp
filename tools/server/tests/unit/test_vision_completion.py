import pytest
from utils import *
import base64
import requests

server: ServerProcess

IMG_URL_0 = "https://huggingface.co/ggml-org/tinygemma3-GGUF/resolve/main/test/11_truck.png"
IMG_URL_1 = "https://huggingface.co/ggml-org/tinygemma3-GGUF/resolve/main/test/91_cat.png"

response = requests.get(IMG_URL_0)
response.raise_for_status() # Raise an exception for bad status codes
IMG_BASE64_0 = base64.b64encode(response.content).decode("utf-8")
response = requests.get(IMG_URL_1)
response.raise_for_status() # Raise an exception for bad status codes
IMG_BASE64_1 = base64.b64encode(response.content).decode("utf-8")

@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.tinygemma3()


@pytest.mark.parametrize(
    "prompt, image_data, success, re_content",
    [
        # test model is trained on CIFAR-10, but it's quite dumb due to small size
        ("What is this: <__media__>\n", IMG_BASE64_0,           True, "(cat)+"), # exceptional, so that we don't cog up the log
        ("What is this: <__media__>\n", IMG_BASE64_1,           True, "(frog)+"),
        ("What is this: <__media__>\n", "malformed",              False, None),
        ("What is this:\n", "base64",        False, None), # non-image data
    ]
)
def test_vision_completion(prompt, image_data, success, re_content):
    global server
    server.start(timeout_seconds=60) # vision model may take longer to load due to download size
    res = server.make_request("POST", "/completions", data={
        "temperature": 0.0,
        "top_k": 1,
        "prompt": { "prompt": prompt, "multimodal_data": [ image_data ] },
    })
    if success:
        assert res.status_code == 200
        content = res.body["content"]
        assert match_regex(re_content, content)
    else:
        assert res.status_code != 200

