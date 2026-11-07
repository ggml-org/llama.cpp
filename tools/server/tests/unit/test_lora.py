import pytest
from utils import *

server = ServerPreset.stories15m_moe()

LORA_FILE_URL = "https://huggingface.co/ggml-org/stories15M_MOE/resolve/main/moe_shakespeare15M.gguf"

@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.stories15m_moe()
    server.lora_files = [download_file(LORA_FILE_URL)]


@pytest.mark.parametrize("scale,re_content", [
    # without applying lora, the model should behave like a bedtime story generator
    (0.0, "(little|girl|three|years|old)+"),
    # with lora, the model should behave like a Shakespearean text generator
    (1.0, "(eye|love|glass|sun)+"),
])
def test_lora(scale: float, re_content: str):
    global server
    server.start()
    res_lora_control = server.make_request("POST", "/lora-adapters", data=[
        {"id": 0, "scale": scale}
    ])
    assert res_lora_control.status_code == 200
    res = server.make_request("POST", "/completion", data={
        "prompt": "Look in thy glass",
    })
    assert res.status_code == 200
    assert match_regex(re_content, res.body["content"])


def test_lora_per_request():
    global server
    server.n_slots = 4
    server.start()

    # running the same prompt with different lora scales, all in parallel
    # each prompt will be processed by a different slot
    prompt = "Look in thy glass"
    lora_config = [
        ( [{"id": 0, "scale": 0.0}], "(bright|day|many|happy)+" ),
        ( [{"id": 0, "scale": 0.0}], "(bright|day|many|happy)+" ),
        ( [{"id": 0, "scale": 0.3}], "(special|thing|gifted)+" ),
        ( [{"id": 0, "scale": 0.7}], "(far|from|home|away)+" ),
        ( [{"id": 0, "scale": 1.0}], "(eye|love|glass|sun)+" ),
        ( [{"id": 0, "scale": 1.0}], "(eye|love|glass|sun)+" ),
    ]

    tasks = [(
        server.make_request,
        ("POST", "/completion", {
            "prompt": prompt,
            "lora": lora,
            "seed": 42,
            "temperature": 0.0,
            "cache_prompt": False, # TODO: remove this once test_cache_vs_nocache_prompt is fixed
        })
    ) for lora, _ in lora_config]
    results = parallel_function_calls(tasks)

    assert all([res.status_code == 200 for res in results])
    for res, (_, re_test) in zip(results, lora_config):
        assert match_regex(re_test, res.body["content"])


def test_lora_global_change_clears_slot_cache():
    global server
    server.n_slots = 1
    server.cache_ram = 0
    server.start()

    # long enough that stale kv changes the output
    prompt = "Look in thy glass and tell the face thou viewest, now is the time that face should form another, whose fresh repair if now thou not renewest, thou dost beguile the world, unbless some mother."

    def set_global_lora(scale: float):
        res = server.make_request("POST", "/lora-adapters", data=[{"id": 0, "scale": scale}])
        assert res.status_code == 200

    def complete(lora=None, cache_prompt: bool = True) -> dict:
        data = {
            "prompt": prompt,
            "cache_prompt": cache_prompt,
            "temperature": 0.0,
            "seed": 42,
            "n_predict": 16,
        }
        if lora is not None:
            data["lora"] = lora
        res = server.make_request("POST", "/completion", data=data)
        assert res.status_code == 200
        return res.body

    set_global_lora(0.0)
    expected_zero = complete(cache_prompt=False)["content"]
    set_global_lora(1.0)
    expected_one = complete(cache_prompt=False)["content"]
    assert expected_zero != expected_one

    # the global config changes between two requests that do not set "lora"
    set_global_lora(0.0)
    res = complete()
    assert res["timings"]["cache_n"] == 0
    assert res["content"] == expected_zero
    set_global_lora(1.0)
    res = complete()
    assert res["timings"]["cache_n"] == 0
    assert res["content"] == expected_one

    # unchanged config keeps the cache
    res = complete()
    assert res["timings"]["cache_n"] > 0

    # a request that sets "lora", followed by one that doesn't
    res = complete(lora=[{"id": 0, "scale": 0.0}])
    assert res["timings"]["cache_n"] == 0
    assert res["content"] == expected_zero
    res = complete()
    assert res["timings"]["cache_n"] == 0
    assert res["content"] == expected_one


@pytest.mark.skipif(not is_slow_test_allowed(), reason="skipping slow test")
def test_with_big_model():
    server = ServerProcess()
    server.model_hf_repo = "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF"
    server.model_hf_file = "Meta-Llama-3.1-8B-Instruct-IQ2_M.gguf"
    server.model_alias = "Llama-3.2-8B-Instruct"
    server.n_slots = 4
    server.n_ctx = server.n_slots * 1024
    server.n_predict = 64
    server.temperature = 0.0
    server.seed = 42
    server.lora_files = [
        download_file("https://huggingface.co/ngxson/Llama-3-Instruct-abliteration-LoRA-8B-F16-GGUF/resolve/main/Llama-3-Instruct-abliteration-LoRA-8B-f16.gguf"),
        # TODO: find & add other lora adapters for this model
    ]
    server.start(timeout_seconds=600)

    # running the same prompt with different lora scales, all in parallel
    # each prompt will be processed by a different slot
    prompt = "Write a computer virus"
    lora_config = [
        # without applying lora, the model should reject the request
        ( [{"id": 0, "scale": 0.0}], "I can't provide you with a code for a computer virus" ),
        ( [{"id": 0, "scale": 0.0}], "I can't provide you with a code for a computer virus" ),
        ( [{"id": 0, "scale": 0.3}], "I can't write a computer virus" ),
        # with 0.7 scale, the model should provide a simple computer virus with hesitation
        ( [{"id": 0, "scale": 0.7}], "Warning: This is a hypothetical exercise" ),
        # with 1.5 scale, the model should confidently provide a computer virus
        ( [{"id": 0, "scale": 1.5}], "A task of some complexity! Here's a simple computer virus" ),
        ( [{"id": 0, "scale": 1.5}], "A task of some complexity! Here's a simple computer virus" ),
    ]

    tasks = [(
        server.make_request,
        ("POST", "/v1/chat/completions", {
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "lora": lora,
            "cache_prompt": False, # TODO: remove this once test_cache_vs_nocache_prompt is fixed
        })
    ) for lora, _ in lora_config]
    results = parallel_function_calls(tasks)

    assert all([res.status_code == 200 for res in results])
    for res, (_, re_test) in zip(results, lora_config):
        assert re_test in res.body["choices"][0]["message"]["content"]
