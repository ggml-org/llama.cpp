import pytest
from utils import *
import base64
import requests

server = ServerPreset.tinyllama2()

@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.tinyllama2()
    server.slot_save_path = "./tmp"
    server.temperature = 0.0


def test_slot_save_restore():
    global server
    server.start()

    # First prompt in slot 1 should be fully processed
    res = server.make_request("POST", "/completion", data={
        "prompt": "What is the capital of France?",
        "id_slot": 1,
        "cache_prompt": True,
    })
    assert res.status_code == 200
    assert match_regex("(Whiskers|Flana)+", res.body["content"])
    assert res.body["timings"]["prompt_n"] == 21  # all tokens are processed

    # Save state of slot 1
    res = server.make_request("POST", "/slots/1?action=save", data={
        "filename": "slot1.bin",
    })
    assert res.status_code == 200
    assert res.body["n_saved"] == 84

    # Since we have cache, this should only process the last tokens
    res = server.make_request("POST", "/completion", data={
        "prompt": "What is the capital of Germany?",
        "id_slot": 1,
        "cache_prompt": True,
    })
    assert res.status_code == 200
    assert match_regex("(Jack|said)+", res.body["content"])
    assert res.body["timings"]["prompt_n"] == 6  # only different part is processed

    # Loading the saved cache into slot 0
    res = server.make_request("POST", "/slots/0?action=restore", data={
        "filename": "slot1.bin",
    })
    assert res.status_code == 200
    assert res.body["n_restored"] == 84

    # Since we have cache, slot 0 should only process the last tokens
    res = server.make_request("POST", "/completion", data={
        "prompt": "What is the capital of Germany?",
        "id_slot": 0,
        "cache_prompt": True,
    })
    assert res.status_code == 200
    assert match_regex("(Jack|said)+", res.body["content"])
    assert res.body["timings"]["prompt_n"] == 6  # only different part is processed

    # For verification that slot 1 was not corrupted during slot 0 load, same thing should work
    res = server.make_request("POST", "/completion", data={
        "prompt": "What is the capital of Germany?",
        "id_slot": 1,
        "cache_prompt": True,
    })
    assert res.status_code == 200
    assert match_regex("(Jack|said)+", res.body["content"])
    assert res.body["timings"]["prompt_n"] == 1


def test_slot_erase():
    global server
    server.start()

    res = server.make_request("POST", "/completion", data={
        "prompt": "What is the capital of France?",
        "id_slot": 1,
        "cache_prompt": True,
    })
    assert res.status_code == 200
    assert match_regex("(Whiskers|Flana)+", res.body["content"])
    assert res.body["timings"]["prompt_n"] == 21  # all tokens are processed

    # erase slot 1
    res = server.make_request("POST", "/slots/1?action=erase")
    assert res.status_code == 200

    # re-run the same prompt, it should process all tokens again
    res = server.make_request("POST", "/completion", data={
        "prompt": "What is the capital of France?",
        "id_slot": 1,
        "cache_prompt": True,
    })
    assert res.status_code == 200
    assert match_regex("(Whiskers|Flana)+", res.body["content"])
    assert res.body["timings"]["prompt_n"] == 21  # all tokens are processed


#
# Multimodal server (mmproj loaded) slot save/restore.
#
# Regression coverage for issue #21133: slot save/restore/erase must be gated on
# the slot's CONTENT (does it actually hold image/audio tokens) rather than the
# model's CAPABILITY (is an mmproj loaded). A pure-text slot on a multimodal
# server must save/restore/erase normally; a slot that actually holds an image
# must be rejected with ERROR_TYPE_NOT_SUPPORTED (HTTP 501).
#

IMG_URL_CAT = "https://huggingface.co/ggml-org/tinygemma3-GGUF/resolve/main/test/91_cat.png"


def _get_img_base64(url: str) -> str:
    response = requests.get(url)
    response.raise_for_status()  # Raise an exception for bad status codes
    return base64.b64encode(response.content).decode("utf-8")


@pytest.fixture
def mmproj_server():
    # tinygemma3 is a small multimodal model: the mmproj is provided by the HF
    # registry API and auto-downloaded on first run.
    os.environ['LLAMA_MEDIA_MARKER'] = '<__media__>'
    mm_server = ServerPreset.tinygemma3()
    mm_server.slot_save_path = "./tmp"
    mm_server.temperature = 0.0
    return mm_server


def test_slot_save_restore_text_only_on_multimodal(mmproj_server):
    server = mmproj_server
    server.start()

    # A pure-text prompt processed on slot 1 of a multimodal server.
    res = server.make_request("POST", "/completion", data={
        "prompt": "The quick brown fox jumps over the lazy dog.",
        "id_slot": 1,
        "cache_prompt": True,
    })
    assert res.status_code == 200
    prompt_n = res.body["timings"]["prompt_n"]
    assert prompt_n > 0  # all tokens are processed

    # Saving a pure-text slot must succeed even though an mmproj is loaded.
    res = server.make_request("POST", "/slots/1?action=save", data={
        "filename": "mm_slot1.bin",
    })
    assert res.status_code == 200
    n_saved = res.body["n_saved"]
    assert n_saved > 0  # the slot KV (prompt + generated tokens) was written

    # Restore the saved state into slot 0; it must round-trip exactly.
    res = server.make_request("POST", "/slots/0?action=restore", data={
        "filename": "mm_slot1.bin",
    })
    assert res.status_code == 200
    assert res.body["n_restored"] == n_saved

    # The restored slot is usable for a follow-up completion. We do NOT assert
    # prefix reuse here (it depends on the restored checkpoint positions vs the
    # re-sent prompt) - checkpoint preservation across save/restore is covered
    # by test_slot_restore_preserves_context_checkpoints below.
    res = server.make_request("POST", "/completion", data={
        "prompt": "The quick brown fox jumps over the lazy dog.",
        "id_slot": 0,
        "cache_prompt": True,
    })
    assert res.status_code == 200


def test_slot_save_rejected_when_slot_holds_image(mmproj_server):
    server = mmproj_server
    server.start()

    # Process a prompt that actually contains an image on slot 1.
    res = server.make_request("POST", "/completions", data={
        "temperature": 0.0,
        "top_k": 1,
        "id_slot": 1,
        "cache_prompt": True,
        "prompt": {
            "prompt_string": "What is this: <__media__>\n",
            "multimodal_data": [ _get_img_base64(IMG_URL_CAT) ],
        },
    })
    assert res.status_code == 200

    # Saving a slot that holds image tokens must be rejected (HTTP 501,
    # not_supported_error).
    res = server.make_request("POST", "/slots/1?action=save", data={
        "filename": "mm_slot_image.bin",
    })
    assert res.status_code != 200
    assert res.body["error"]["type"] == "not_supported_error"


def test_slot_erase_text_only_on_multimodal(mmproj_server):
    server = mmproj_server
    server.start()

    res = server.make_request("POST", "/completion", data={
        "prompt": "The quick brown fox jumps over the lazy dog.",
        "id_slot": 1,
        "cache_prompt": True,
    })
    assert res.status_code == 200
    prompt_n = res.body["timings"]["prompt_n"]
    assert prompt_n > 0  # all tokens are processed

    # Erasing a pure-text slot must succeed even though an mmproj is loaded.
    res = server.make_request("POST", "/slots/1?action=erase")
    assert res.status_code == 200

    # Re-running the same prompt should process all tokens again.
    res = server.make_request("POST", "/completion", data={
        "prompt": "The quick brown fox jumps over the lazy dog.",
        "id_slot": 1,
        "cache_prompt": True,
    })
    assert res.status_code == 200
    assert res.body["timings"]["prompt_n"] == prompt_n  # all tokens are processed again


#
# Context checkpoints across save/restore.
#
# SWA and hybrid/recurrent models need a context checkpoint to roll back to in
# order to partially reuse their cache on a divergent re-prompt. Checkpoints
# used to be discarded on restore (and were not part of the save file), forcing
# full prompt re-processing even when most of the prefix matched. They are now
# appended to the save file and restored with the slot: a restored slot must
# behave exactly like the live slot it was saved from.
#


@pytest.fixture
def swa_server():
    # tinygemma3 is a SWA model - its partial cache reuse relies on checkpoints.
    swa = ServerPreset.tinygemma3()
    swa.slot_save_path = "./tmp"
    swa.temperature = 0.0
    # the host prompt cache would transparently rescue the overwritten slot and
    # hide the save/restore path under test
    swa.cache_ram = 0
    # prompt-processing checkpoints are placed at (end - 4 - n_ubatch) and
    # (end - 4): keep n_ubatch small so that the first one lands before the
    # divergence point of the re-prompts below
    swa.n_ubatch = 32
    return swa


def test_slot_restore_preserves_context_checkpoints(swa_server):
    server = swa_server
    server.start()

    base = "The quick brown fox jumps over the lazy dog. " * 20

    # reference: on a live slot, a divergent re-prompt rolls back to a
    # mid-prompt checkpoint and only re-processes from there
    # the endings must span more tokens than the last checkpoint offset (4), so
    # that the checkpoint at (end - 4 - n_ubatch) sits before the divergence
    res = server.make_request("POST", "/completion", data={
        "prompt": base + "The first ending of this story is a happy one.",
        "id_slot": 1,
        "cache_prompt": True,
    })
    assert res.status_code == 200
    n_full = res.body["timings"]["prompt_n"]

    res = server.make_request("POST", "/completion", data={
        "prompt": base + "But the second ending was different and sad.",
        "id_slot": 1,
        "cache_prompt": True,
    })
    assert res.status_code == 200
    n_live = res.body["timings"]["prompt_n"]
    assert n_live < n_full  # partial reuse via checkpoint rollback

    # now the same divergent re-prompt, but after save -> overwrite -> restore
    res = server.make_request("POST", "/slots/1?action=erase")
    assert res.status_code == 200

    res = server.make_request("POST", "/completion", data={
        "prompt": base + "The first ending of this story is a happy one.",
        "id_slot": 1,
        "cache_prompt": True,
    })
    assert res.status_code == 200

    res = server.make_request("POST", "/slots/1?action=save", data={
        "filename": "ckpt_slot1.bin",
    })
    assert res.status_code == 200
    assert res.body["n_saved"] > 0

    # overwrite the slot with an unrelated prompt
    res = server.make_request("POST", "/completion", data={
        "prompt": "Unrelated text with no common prefix occupies the slot now.",
        "id_slot": 1,
        "cache_prompt": True,
    })
    assert res.status_code == 200

    res = server.make_request("POST", "/slots/1?action=restore", data={
        "filename": "ckpt_slot1.bin",
    })
    assert res.status_code == 200

    # the restored slot must roll back to the restored checkpoint, exactly like
    # the live slot did (before the fix: full re-processing, prompt_n == n_full)
    res = server.make_request("POST", "/completion", data={
        "prompt": base + "But the second ending was different and sad.",
        "id_slot": 1,
        "cache_prompt": True,
    })
    assert res.status_code == 200
    assert res.body["timings"]["prompt_n"] == n_live
