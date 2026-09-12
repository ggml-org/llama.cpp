import pytest
import base64
import io
from utils import *
from unit.test_tool_call import TIMEOUT_HTTP_REQUEST, CompletionMode, TEST_TOOL, PYTHON_TOOL, WEATHER_TOOL, do_test_completion_with_required_tool_tiny, do_test_completion_without_tool_call, do_test_weather, do_test_calc_result, do_test_hello_world

# one ~95M multimodal fixture exercising chat, tool calling, OCR and MTP drafting
server: ServerProcess

GREEDY = {"temperature": 0.0, "top_k": 1, "top_p": 1.0}


@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.small_test()


def ocr_image(text: str) -> str:
    # a PNG with black text on white in a common truetype font, the kind of image the fixture is trained on
    from PIL import Image, ImageDraw, ImageFont
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 28)
    img = Image.new("RGB", (360, 80), "white")
    ImageDraw.Draw(img).text((16, 20), text, font=font, fill="black")
    buf = io.BytesIO(); img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


@pytest.mark.parametrize("stream", [CompletionMode.NORMAL, CompletionMode.STREAMED])
@pytest.mark.parametrize("tool,argument_key", [(TEST_TOOL, "success"), (PYTHON_TOOL, "code")])
def test_required_tool(tool: dict, argument_key: str, stream: CompletionMode):
    global server
    server.start()
    do_test_completion_with_required_tool_tiny(server, tool, argument_key, 256, stream=stream == CompletionMode.STREAMED, **GREEDY)


@pytest.mark.parametrize("stream", [CompletionMode.NORMAL, CompletionMode.STREAMED])
def test_weather(stream: CompletionMode):
    global server
    server.start()
    do_test_weather(server, stream=stream == CompletionMode.STREAMED, max_tokens=256, **GREEDY)


@pytest.mark.parametrize("stream", [CompletionMode.NORMAL, CompletionMode.STREAMED])
def test_hello_world(stream: CompletionMode):
    global server
    server.start()
    do_test_hello_world(server, stream=stream == CompletionMode.STREAMED, max_tokens=256, **GREEDY)


@pytest.mark.parametrize("stream", [CompletionMode.NORMAL, CompletionMode.STREAMED])
def test_calc_result(stream: CompletionMode):
    global server
    server.start()
    do_test_calc_result(server, None, 256, stream=stream == CompletionMode.STREAMED, **GREEDY)


@pytest.mark.parametrize("stream", [CompletionMode.NORMAL, CompletionMode.STREAMED])
@pytest.mark.parametrize("tools,tool_choice", [(None, None), ([], None), ([TEST_TOOL], "none")])
def test_without_tool_call(tools, tool_choice, stream: CompletionMode):
    global server
    server.start()
    do_test_completion_without_tool_call(server, 64, tools, tool_choice, stream=stream == CompletionMode.STREAMED, **GREEDY)


@pytest.mark.parametrize("text", ["HELLO WORLD", "Invoice 2026"])
def test_ocr(text: str):
    global server
    server.start()
    body = server.make_any_request("POST", "/v1/chat/completions", data={
        "max_tokens": 32,
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": ocr_image(text)}},
            {"type": "text", "text": "What is written in this image?"},
        ]}],
        **GREEDY,
    }, timeout=TIMEOUT_HTTP_REQUEST)
    content = body["choices"][0]["message"]["content"]
    assert text.lower() in content.lower(), f"expected {text!r} in {content!r}"


def test_mtp_draft_matches_target():
    # greedy tokens are identical with and without the MTP draft, and the draft is actually used
    global server
    server.start()
    req = {"prompt": "<|im_start|>user\nList three colors.<|im_end|>\n<|im_start|>assistant\n", "n_predict": 48, "temperature": 0.0, "top_k": 1, "return_tokens": True}
    res = server.make_request("POST", "/completion", data=req)
    assert res.status_code == 200
    tokens_no_draft = res.body["tokens"]
    server.stop()
    server.spec_type = "draft-mtp"
    server.start()
    res = server.make_request("POST", "/completion", data=req)
    assert res.status_code == 200
    assert res.body["timings"]["draft_n"] > 0
    assert res.body["tokens"] == tokens_no_draft
