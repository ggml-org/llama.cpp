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


def test_lora_init_without_apply_starts_at_zero():
    global server
    server.lora_init_without_apply = True
    server.start()
    res = server.make_request("GET", "/lora-adapters")
    assert res.status_code == 200
    assert len(res.body) >= 1
    assert all(adapter["scale"] == 0.0 for adapter in res.body)

    res = server.make_request("POST", "/completion", data={
        "prompt": "Look in thy glass",
        "temperature": 0.0,
        "seed": 42,
        "cache_prompt": False,
    })
    assert res.status_code == 200
    assert match_regex("(little|girl|three|years|old)+", res.body["content"])


def test_lora_empty_list_disables_adapters():
    global server
    server.start()
    res = server.make_request("POST", "/completion", data={
        "prompt": "Look in thy glass",
        "lora": [],
        "temperature": 0.0,
        "seed": 42,
        "cache_prompt": False,
    })
    assert res.status_code == 200
    assert match_regex("(little|girl|three|years|old)+", res.body["content"])


def test_lora_omitted_after_empty_does_not_reuse_cache():
    global server
    server.n_slots = 1
    server.start()
    prompt = "Look in thy glass"
    res_off = server.make_request("POST", "/completion", data={
        "prompt": prompt,
        "lora": [],
        "temperature": 0.0,
        "seed": 42,
        "cache_prompt": True,
    })
    assert res_off.status_code == 200
    assert match_regex("(little|girl|three|years|old)+", res_off.body["content"])

    # omitted lora restores the global scale-1 adapters; the no-adapter prompt
    # cache from the previous request must not be reused
    res_on = server.make_request("POST", "/completion", data={
        "prompt": prompt,
        "temperature": 0.0,
        "seed": 42,
        "cache_prompt": True,
    })
    assert res_on.status_code == 200
    assert match_regex("(eye|love|glass|sun)+", res_on.body["content"])


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
