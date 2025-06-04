import pytest
from utils import *
import base64
import requests

server: ServerProcess

IMG_URL_0 = "https://huggingface.co/ggml-org/tinygemma3-GGUF/resolve/main/test/11_truck.png"
IMG_URL_1 = "https://huggingface.co/ggml-org/tinygemma3-GGUF/resolve/main/test/91_cat.png"

response = requests.get(IMG_URL_0)
response.raise_for_status() # Raise an exception for bad status codes
IMG_BASE64_0 = "data:image/png;base64," + base64.b64encode(response.content).decode("utf-8")


@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.tinygemma3()


@pytest.mark.parametrize(
    "prompt, image_url, success, re_content",
    [
        # test model is trained on CIFAR-10, but it's quite dumb due to small size
        ("What is this:\n", IMG_URL_0,                True, "(cat)+"),
        ("What is this:\n", "IMG_BASE64_0",           True, "(cat)+"), # exceptional, so that we don't cog up the log
        ("What is this:\n", IMG_URL_1,                True, "(frog)+"),
        ("Test test\n",     IMG_URL_1,                True, "(frog)+"), # test invalidate cache
        ("What is this:\n", "malformed",              False, None),
        ("What is this:\n", "https://google.com/404", False, None), # non-existent image
        ("What is this:\n", "https://ggml.ai",        False, None), # non-image data
        # TODO @ngxson : test with multiple images, no images and with audio
    ]
)
def test_vision_chat_completion(prompt, image_url, success, re_content):
    global server
    server.start(timeout_seconds=60) # vision model may take longer to load due to download size
    if image_url == "IMG_BASE64_0":
        image_url = IMG_BASE64_0
    res = server.make_request("POST", "/chat/completions", data={
        "temperature": 0.0,
        "top_k": 1,
        "messages": [
            {"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {
                    "url": image_url,
                }},
            ]},
        ],
    })
    if success:
        assert res.status_code == 200
        choice = res.body["choices"][0]
        assert "assistant" == choice["message"]["role"]
        assert match_regex(re_content, choice["message"]["content"])
    else:
        assert res.status_code != 200

