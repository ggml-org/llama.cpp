import pytest
from utils import *
import base64
import requests

server: ServerProcess

IMG_URL = "https://huggingface.co/ggml-org/tinygemma3-GGUF/resolve/main/test/11_truck.png"

JSON_MULTIMODAL_KEY = "multimodal_data"
JSON_PROMPT_STRING_KEY = "prompt_string"

# three distinct parts; each one is well above the reuse chunk size (32 tokens)
HEAD = "The quick brown fox jumps over the lazy dog near the river bank. " * 4
MIDDLE = "Seven silver spoons were found under the old wooden table in the kitchen. " * 4
TAIL = "A small red boat sailed across the calm blue lake at dawn, carrying three fishermen. " * 4


@pytest.fixture(autouse=True)
def create_server():
    global server
    os.environ['LLAMA_MEDIA_MARKER'] = '<__media__>'
    server = ServerPreset.tinygemma3()
    server.n_slots = 1
    server.swa_full = True # iSWA memory can only be shifted with a full-size SWA cache


def complete(prompt):
    res = server.make_request("POST", "/completion", data={
        "prompt": prompt,
        "temperature": 0.0,
        "top_k": 1,
        "seed": 42,
        "cache_prompt": True,
    })
    assert res.status_code == 200
    return res.body


def n_tokens(text: str) -> int:
    res = server.make_request("POST", "/tokenize", data={"content": text})
    assert res.status_code == 200
    return len(res.body["tokens"])


def run_edit(cache_reuse: int, no_mmproj: bool = False):
    # prompt A, then B = A without its middle part; returns the result of B and the token count of TAIL
    global server
    server.cache_reuse = cache_reuse
    server.no_mmproj = no_mmproj
    server.start()
    n_tail = n_tokens(TAIL)
    complete(HEAD + MIDDLE + TAIL)
    res = complete(HEAD + TAIL)
    server.stop()
    return res, n_tail


def test_cache_reuse_text_prompt_with_mmproj():
    ref, n_tail = run_edit(cache_reuse=0)
    res, _ = run_edit(cache_reuse=32)
    txt, _ = run_edit(cache_reuse=32, no_mmproj=True)

    # without reuse, TAIL is evaluated again after the removed MIDDLE; with reuse, its KV cells are shifted
    assert ref["timings"]["prompt_n"] >= n_tail
    assert res["timings"]["prompt_n"] < n_tail

    # with a projector loaded, a text prompt is reused exactly as on a text-only server
    # (shifted KV is not bit-identical to a fresh prefill, so the reference also uses reuse)
    assert res["timings"]["prompt_n"] == txt["timings"]["prompt_n"]
    assert res["content"] == txt["content"]


def test_cache_reuse_skipped_for_media_prompt():
    global server
    server.cache_reuse = 32
    server.start()
    img = base64.b64encode(requests.get(IMG_URL).content).decode("utf-8")
    res = server.make_request("POST", "/completion", data={
        "prompt": {
            JSON_PROMPT_STRING_KEY: HEAD + "<__media__>" + MIDDLE + TAIL,
            JSON_MULTIMODAL_KEY: [img],
        },
        "temperature": 0.0,
        "top_k": 1,
        "cache_prompt": True,
    })
    assert res.status_code == 200

    # the cached prompt has media: nothing after it is reused, the text prompt is evaluated from the end of HEAD
    res = complete(HEAD + MIDDLE + TAIL)
    assert res["timings"]["prompt_n"] > n_tokens(TAIL)

    # the media is gone from the cache: the next text prompt reuses TAIL again
    res = complete(HEAD + TAIL)
    assert res["timings"]["prompt_n"] < n_tokens(TAIL)
