import pytest
from utils import *
import base64
import requests

server: ServerProcess

IMG_URL_0 = "https://huggingface.co/ggml-org/tinygemma3-GGUF/resolve/main/test/11_truck.png"
IMG_URL_1 = "https://huggingface.co/ggml-org/tinygemma3-GGUF/resolve/main/test/91_cat.png"

response = requests.get(IMG_URL_0)
response.raise_for_status() # Raise an exception for bad status codes
IMG_BASE64_URI_0 = "data:image/png;base64," + base64.b64encode(response.content).decode("utf-8")
IMG_BASE64_0 = base64.b64encode(response.content).decode("utf-8")

response = requests.get(IMG_URL_1)
response.raise_for_status() # Raise an exception for bad status codes
IMG_BASE64_URI_1 = "data:image/png;base64," + base64.b64encode(response.content).decode("utf-8")
IMG_BASE64_1 = base64.b64encode(response.content).decode("utf-8")

JSON_MULTIMODAL_KEY = "multimodal_data"
JSON_PROMPT_STRING_KEY = "prompt_string"

@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.tinygemma3()

def test_models_supports_multimodal_capability():
    global server
    server.start() # vision model may take longer to load due to download size
    res = server.make_request("GET", "/models", data={})
    assert res.status_code == 200
    model_info = res.body["models"][0]
    print(model_info)
    assert "completion" in model_info["capabilities"]
    assert "multimodal" in model_info["capabilities"]

def test_v1_models_supports_multimodal_capability():
    global server
    server.start() # vision model may take longer to load due to download size
    res = server.make_request("GET", "/v1/models", data={})
    assert res.status_code == 200
    model_info = res.body["models"][0]
    print(model_info)
    assert "completion" in model_info["capabilities"]
    assert "multimodal" in model_info["capabilities"]

@pytest.mark.parametrize(
    "prompt, image_url, success, re_content",
    [
        # test model is trained on CIFAR-10, but it's quite dumb due to small size
        ("What is this:\n", IMG_URL_0,                True, "(cat)+"),
        ("What is this:\n", "IMG_BASE64_URI_0",       True, "(cat)+"), # exceptional, so that we don't cog up the log
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
    if image_url == "IMG_BASE64_URI_0":
        image_url = IMG_BASE64_URI_0
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


@pytest.mark.parametrize(
    "prompt, image_data, success, re_content",
    [
        # test model is trained on CIFAR-10, but it's quite dumb due to small size
        ("What is this: <__media__>\n", IMG_BASE64_0,           True, "(cat)+"),
        ("What is this: <__media__>\n", IMG_BASE64_1,           True, "(frog)+"),
        ("What is this: <__media__>\n", "malformed",            False, None), # non-image data
        ("What is this:\n",             "",                     False, None), # empty string
    ]
)
def test_vision_completion(prompt, image_data, success, re_content):
    global server
    server.start() # vision model may take longer to load due to download size
    res = server.make_request("POST", "/completions", data={
        "temperature": 0.0,
        "top_k": 1,
        "prompt": { JSON_PROMPT_STRING_KEY: prompt, JSON_MULTIMODAL_KEY: [ image_data ] },
    })
    if success:
        assert res.status_code == 200
        content = res.body["content"]
        assert match_regex(re_content, content)
    else:
        assert res.status_code != 200


@pytest.mark.parametrize(
    "prompt, image_data, success",
    [
        # test model is trained on CIFAR-10, but it's quite dumb due to small size
        ("What is this: <__media__>\n", IMG_BASE64_0,           True), # exceptional, so that we don't cog up the log
        ("What is this: <__media__>\n", IMG_BASE64_1,           True),
        ("What is this: <__media__>\n", "malformed",            False), # non-image data
        ("What is this:\n",             "base64",               False), # non-image data
    ]
)
def test_vision_embeddings(prompt, image_data, success):
    global server
    server.server_embeddings=True
    server.n_batch=512
    server.start() # vision model may take longer to load due to download size
    res = server.make_request("POST", "/embeddings", data={
        "content": [
            { JSON_PROMPT_STRING_KEY: prompt, JSON_MULTIMODAL_KEY: [ image_data ] },
            { JSON_PROMPT_STRING_KEY: prompt, JSON_MULTIMODAL_KEY: [ image_data ] },
            { JSON_PROMPT_STRING_KEY: prompt, },
        ],
    })
    if success:
        assert res.status_code == 200
        content = res.body
        # Ensure embeddings are stable when multimodal.
        assert content[0]['embedding'] == content[1]['embedding']
        # Ensure embeddings without multimodal but same prompt do not match multimodal embeddings.
        assert content[0]['embedding'] != content[2]['embedding']
    else:
        assert res.status_code != 200
