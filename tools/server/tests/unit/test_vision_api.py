import pytest
from utils import *
import base64
import requests
import subprocess

server: ServerProcess

def get_img_url(id: str) -> str:
    IMG_URL_0 = "https://huggingface.co/ggml-org/tinygemma3-GGUF/resolve/main/test/11_truck.png"
    IMG_URL_1 = "https://huggingface.co/ggml-org/tinygemma3-GGUF/resolve/main/test/91_cat.png"
    if id == "IMG_URL_0":
        return IMG_URL_0
    elif id == "IMG_URL_1":
        return IMG_URL_1
    elif id == "IMG_BASE64_URI_0":
        response = requests.get(IMG_URL_0)
        response.raise_for_status() # Raise an exception for bad status codes
        return "data:image/png;base64," + base64.b64encode(response.content).decode("utf-8")
    elif id == "IMG_BASE64_0":
        response = requests.get(IMG_URL_0)
        response.raise_for_status() # Raise an exception for bad status codes
        return base64.b64encode(response.content).decode("utf-8")
    elif id == "IMG_BASE64_URI_1":
        response = requests.get(IMG_URL_1)
        response.raise_for_status() # Raise an exception for bad status codes
        return "data:image/png;base64," + base64.b64encode(response.content).decode("utf-8")
    elif id == "IMG_BASE64_1":
        response = requests.get(IMG_URL_1)
        response.raise_for_status() # Raise an exception for bad status codes
        return base64.b64encode(response.content).decode("utf-8")
    else:
        return id

JSON_MULTIMODAL_KEY = "multimodal_data"
JSON_PROMPT_STRING_KEY = "prompt_string"

@pytest.fixture(autouse=True)
def create_server():
    global server
    os.environ['LLAMA_MEDIA_MARKER'] = '<__media__>'
    server = ServerPreset.tinygemma3()

@pytest.mark.skipif("LLAMA_CLI_BIN_PATH" not in os.environ, reason="LLAMA_CLI_BIN_PATH is not set")
@pytest.mark.parametrize(
    "media,error",
    [
        ("missing.png", "file does not exist or cannot be opened"),
        ("invalid.png", "Failed to load image or audio file"),
        (None, None),
    ],
    ids=["missing", "invalid", "success"],
)
def test_cli_single_turn_exit_code(tmp_path, media, error):
    """llama-cli --single-turn must exit with 1 on media errors; reuses the vision server via --server-base"""
    server.start()
    transcript = tmp_path / "output.txt"
    args = [
        os.environ["LLAMA_CLI_BIN_PATH"],
        "--server-base", f"http://{server.server_host}:{server.server_port}",
        "--simple-io", "--single-turn", "--prompt", "test",
        "--output-file", str(transcript),
    ]
    if media is not None:
        path = tmp_path / media
        if media == "invalid.png":
            path.write_bytes(b"not an image")
        args.extend(["--image", str(path)])
    result = subprocess.run(args, input="/exit\n", encoding="utf-8", errors="replace", capture_output=True, timeout=30)
    output = result.stdout + result.stderr
    assert result.returncode == (1 if error else 0), output
    if error:
        assert error in output
    else:
        assert "Error:" not in output
        assert transcript.read_text(encoding="utf-8").partition("Assistant:\n")[2].strip(), output

def test_models_supports_multimodal_capability():
    global server
    server.start()
    res = server.make_request("GET", "/models", data={})
    assert res.status_code == 200
    model_info = res.body["models"][0]
    print(model_info)
    assert "completion" in model_info["capabilities"]
    assert "multimodal" in model_info["capabilities"]

def test_v1_models_supports_multimodal_capability():
    global server
    server.start()
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
        ("What is this:\n", "IMG_URL_0",              True, "(cat)+"),
        ("What is this:\n", "IMG_BASE64_URI_0",       True, "(cat)+"),
        ("What is this:\n", "IMG_URL_1",              True, "(frog)+"),
        ("Test test\n",     "IMG_URL_1",              True, "(frog)+"), # test invalidate cache
        ("What is this:\n", "malformed",              False, None),
        ("What is this:\n", "https://google.com/404", False, None), # non-existent image
        ("What is this:\n", "https://ggml.ai",        False, None), # non-image data
        ("What is this:\n", "data:text/html;base64,aGVsbG8=", False, None), # unsupported data uri mime
        # TODO @ngxson : test with multiple images, no images and with audio
    ]
)
def test_vision_chat_completion(prompt, image_url, success, re_content):
    global server
    server.start()
    res = server.make_request("POST", "/chat/completions", data={
        "temperature": 0.0,
        "top_k": 1,
        "messages": [
            {"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {
                    "url": get_img_url(image_url),
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


def test_vision_chat_completion_token_count():
    global server
    server.start()
    res = server.make_request("POST", "/chat/completions/input_tokens", data={
        "temperature": 0.0,
        "top_k": 1,
        "messages": [
            {"role": "user", "content": [
                {"type": "text", "text": "What is this:"},
                {"type": "image_url", "image_url": {
                    "url": get_img_url("IMG_URL_0"),
                }},
            ]},
        ],
    })
    assert res.status_code == 200
    assert res.body["input_tokens"] > 10


@pytest.mark.parametrize(
    "prompt, image_data, success, re_content",
    [
        # test model is trained on CIFAR-10, but it's quite dumb due to small size
        ("What is this: <__media__>\n", "IMG_BASE64_0",         True, "(cat)+|(automobile)+"),
        ("What is this: <__media__>\n", "IMG_BASE64_1",         True, "(frog)+"),
        ("What is this: <__media__>\n", "malformed",            False, None), # non-image data
        ("What is this:\n",             "",                     False, None), # empty string
    ]
)
def test_vision_completion(prompt, image_data, success, re_content):
    global server
    server.start()
    res = server.make_request("POST", "/completions", data={
        "temperature": 0.0,
        "top_k": 1,
        "prompt": {
            JSON_PROMPT_STRING_KEY: prompt,
            JSON_MULTIMODAL_KEY: [ get_img_url(image_data) ],
        },
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
        ("What is this: <__media__>\n", "IMG_BASE64_0",         True),
        ("What is this: <__media__>\n", "IMG_BASE64_1",         True),
        ("What is this: <__media__>\n", "malformed",            False), # non-image data
        ("What is this:\n",             "base64",               False), # non-image data
    ]
)
def test_vision_embeddings(prompt, image_data, success):
    global server
    server.server_embeddings = True
    server.n_batch = 512
    server.start()
    image_data = get_img_url(image_data)
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


def test_vision_embeddings_oai_content():
    global server
    server.server_embeddings = True
    server.pooling = 'mean'
    server.n_batch = 512
    server.start()
    res = server.make_request("POST", "/v1/embeddings", data={
        "input": [
            {"content": [
                {"type": "text", "text": "What is this: "},
                {"type": "image_url", "image_url": {"url": get_img_url("IMG_BASE64_URI_0")}},
                {"type": "text", "text": "\n"},
            ]},
            {JSON_PROMPT_STRING_KEY: "What is this: <__media__>\n", JSON_MULTIMODAL_KEY: [get_img_url("IMG_BASE64_0")]},
            "What is this: \n",
        ],
    })
    assert res.status_code == 200
    data = res.body["data"]
    assert len(data) == 3
    # same prompt and image in both formats
    assert data[0]["embedding"] == data[1]["embedding"]
    assert data[0]["embedding"] != data[2]["embedding"]
