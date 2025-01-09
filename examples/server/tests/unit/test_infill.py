import pytest
from utils import *

server = ServerPreset.tinyllama_infill()

@pytest.fixture(scope="module", autouse=True)
def create_server():
    global server
    server = ServerPreset.tinyllama_infill()


def test_infill_without_input_extra():
    global server
    server.start()
    res = server.make_request("POST", "/infill", data={
        "input_prefix": "#include <cstdio>\n#include \"llama.h\"\n\nint main() {\n",
        "prompt": "    int n_threads = llama_",
        "input_suffix": "}\n",
    })
    assert res.status_code == 200
    assert match_regex("(Ann|small|shiny|Daddy)+", res.body["content"])


def test_infill_with_input_extra():
    global server
    server.start()
    res = server.make_request("POST", "/infill", data={
        "input_extra": [{
            "filename": "llama.h",
            "text": "LLAMA_API int32_t llama_n_threads();\n"
        }],
        "input_prefix": "#include <cstdio>\n#include \"llama.h\"\n\nint main() {\n",
        "prompt": "    int n_threads = llama_",
        "input_suffix": "}\n",
    })
    assert res.status_code == 200
    assert match_regex("(Dad|excited|park)+", res.body["content"])


@pytest.mark.parametrize("input_extra", [
    {},
    {"filename": "ok"},
    {"filename": 123},
    {"filename": 123, "text": "abc"},
    {"filename": 123, "text": 456},
])
def test_invalid_input_extra_req(input_extra):
    global server
    server.start()
    res = server.make_request("POST", "/infill", data={
        "input_extra": [input_extra],
        "input_prefix": "#include <cstdio>\n#include \"llama.h\"\n\nint main() {\n",
        "prompt": "    int n_threads = llama_",
        "input_suffix": "}\n",
    })
    assert res.status_code == 400
    assert "error" in res.body


@pytest.mark.skipif(not is_slow_test_allowed(), reason="skipping slow test")
def test_with_qwen_model():
    global server
    server.model_file = None
    server.model_hf_repo = "ggml-org/Qwen2.5-Coder-1.5B-IQ3_XXS-GGUF"
    server.model_hf_file = "qwen2.5-coder-1.5b-iq3_xxs-imat.gguf"
    server.start(timeout_seconds=600)
    res = server.make_request("POST", "/infill", data={
        "input_extra": [{
            "filename": "llama.h",
            "text": "LLAMA_API int32_t llama_n_threads();\n"
        }],
        "input_prefix": "#include <cstdio>\n#include \"llama.h\"\n\nint main() {\n",
        "prompt": "    int n_threads = llama_",
        "input_suffix": "}\n",
    })
    assert res.status_code == 200
    assert res.body["content"] == "n_threads();\n    printf(\"Number of threads: %d\\n\", n_threads);\n    return 0;\n"
