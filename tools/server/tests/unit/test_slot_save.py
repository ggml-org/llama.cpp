import pytest
from utils import *
import base64
import requests
import struct

# sequence state file: magic(4) version(4) payload_size(4), then payload_size llama_token words
STATE_FILE_HEADER_SIZE = 12

server = ServerPreset.tinyllama2()
server_noflag = ServerPreset.tinyllama2()

@pytest.fixture(autouse=True)
def create_server(tmp_path):
    global server, server_noflag
    server = ServerPreset.tinyllama2()
    server.slot_save_path = str(tmp_path)
    server.temperature = 0.0
    server.session_id_headers = "x-session-id,session_id"

    # same server without the --slot-save-sessions flag
    server_noflag = ServerPreset.tinyllama2()
    server_noflag.slot_save_path = str(tmp_path)
    server_noflag.temperature = 0.0


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


def test_slot_restore_legacy_token_list():
    global server
    server.start()

    res = server.make_request("POST", "/completion", data={
        "prompt": "What is the capital of France?",
        "id_slot": 1,
        "cache_prompt": True,
    })
    assert res.status_code == 200

    res = server.make_request("POST", "/slots/1?action=save", data={
        "filename": "slot_legacy.bin",
    })
    assert res.status_code == 200
    assert res.body["n_saved"] == 84

    # rewrite the token payload into a plain token list, as written by servers that predate the packed server_tokens format
    path = os.path.join(server.slot_save_path, "slot_legacy.bin")
    with open(path, "rb") as f:
        data = bytearray(f.read())

    # the payload written by this server starts with a packed header: LLAMA_TOKEN_NULL(4) version(4) n_tokens(4)
    packed_header_size = 12

    payload_size = struct.unpack_from("=I", data, STATE_FILE_HEADER_SIZE - 4)[0]
    payload_end = STATE_FILE_HEADER_SIZE + payload_size * 4
    n_tokens = struct.unpack_from("=I", data, STATE_FILE_HEADER_SIZE + 8)[0]
    assert n_tokens == 84

    tokens_start = STATE_FILE_HEADER_SIZE + packed_header_size
    data = data[:STATE_FILE_HEADER_SIZE] + data[tokens_start:tokens_start + n_tokens * 4] + data[payload_end:]
    struct.pack_into("=I", data, STATE_FILE_HEADER_SIZE - 4, n_tokens)

    with open(path, "wb") as f:
        f.write(data)

    # the plain token list must restore, and the restored KV must be reusable
    res = server.make_request("POST", "/slots/0?action=restore", data={
        "filename": "slot_legacy.bin",
    })
    assert res.status_code == 200
    assert res.body["n_restored"] == 84

    res = server.make_request("POST", "/completion", data={
        "prompt": "What is the capital of Germany?",
        "id_slot": 0,
        "cache_prompt": True,
    })
    assert res.status_code == 200
    assert res.body["timings"]["prompt_n"] == 6  # only the different part is processed



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
# A pure-text slot on a multimodal server and a slot containing images must both support save/restore.
# Erase remains gated on the slot's content.
#

IMG_URL_CAT = "https://huggingface.co/ggml-org/tinygemma3-GGUF/resolve/main/test/91_cat.png"
IMG_URL_TRUCK = "https://huggingface.co/ggml-org/tinygemma3-GGUF/resolve/main/test/11_truck.png"


def _get_img_base64(url: str) -> str:
    response = requests.get(url)
    response.raise_for_status()  # Raise an exception for bad status codes
    return base64.b64encode(response.content).decode("utf-8")


@pytest.fixture
def mmproj_server():
    # tinygemma3 is a small multimodal model: the mmproj is provided by the HF registry API and auto-downloaded on first run.
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

    # Prefix reuse is not checked with the default SWA cache.
    res = server.make_request("POST", "/completion", data={
        "prompt": "The quick brown fox jumps over the lazy dog.",
        "id_slot": 0,
        "cache_prompt": True,
    })
    assert res.status_code == 200


def test_slot_save_restore_with_image(mmproj_server):
    server = mmproj_server
    # Use the full SWA cache so the restored image prefix can be reused.
    server.swa_full = True
    server.start()

    prompt_cat = {
        "prompt_string": "What is this: <__media__>\n",
        "multimodal_data": [_get_img_base64(IMG_URL_CAT)],
    }
    res = server.make_request("POST", "/completions", data={
        "temperature": 0.0,
        "top_k": 1,
        "id_slot": 1,
        "cache_prompt": True,
        "prompt": prompt_cat,
    })
    assert res.status_code == 200
    content_cat = res.body["content"]
    prompt_n_full = res.body["timings"]["prompt_n"]
    assert res.body["timings"]["cache_n"] == 0
    assert prompt_n_full > 32  # text plus image tokens are all processed

    res = server.make_request("POST", "/slots/1?action=save", data={
        "filename": "mm_slot_image.bin",
    })
    assert res.status_code == 200
    n_saved = res.body["n_saved"]
    n_written = res.body["n_written"]
    assert n_saved > 0
    assert n_written > 0

    res = server.make_request("POST", "/slots/1?action=erase")
    assert res.status_code == 200

    res = server.make_request("POST", "/slots/0?action=restore", data={
        "filename": "mm_slot_image.bin",
    })
    assert res.status_code == 200
    assert res.body["n_restored"] == n_saved
    assert res.body["n_read"] == n_written

    # a different image must not reuse the restored image tokens; only the text prefix before the image is common
    res = server.make_request("POST", "/completions", data={
        "temperature": 0.0,
        "top_k": 1,
        "id_slot": 0,
        "cache_prompt": True,
        "prompt": {
            "prompt_string": "What is this: <__media__>\n",
            "multimodal_data": [_get_img_base64(IMG_URL_TRUCK)],
        },
    })
    assert res.status_code == 200
    cache_n = res.body["timings"]["cache_n"]
    assert cache_n < 16
    assert res.body["timings"]["prompt_n"] == prompt_n_full - cache_n

    # restore again and resend the same image: the image tokens must be reused and greedy sampling must reproduce the original content
    res = server.make_request("POST", "/slots/0?action=restore", data={
        "filename": "mm_slot_image.bin",
    })
    assert res.status_code == 200
    assert res.body["n_restored"] == n_saved

    res = server.make_request("POST", "/completions", data={
        "temperature": 0.0,
        "top_k": 1,
        "id_slot": 0,
        "cache_prompt": True,
        "prompt": prompt_cat,
    })
    assert res.status_code == 200
    assert res.body["timings"]["cache_n"] == prompt_n_full - 1
    assert res.body["timings"]["prompt_n"] == 1
    assert res.body["content"] == content_cat


def test_slot_save_restore_with_two_images(mmproj_server):
    server = mmproj_server
    server.swa_full = True
    server.n_ctx = 2048  # two images need more than the default 512 per slot
    server.start()

    prompt = {
        "prompt_string": "A: <__media__> B: <__media__>\n",
        "multimodal_data": [_get_img_base64(IMG_URL_CAT), _get_img_base64(IMG_URL_TRUCK)],
    }
    res = server.make_request("POST", "/completions", data={
        "temperature": 0.0,
        "top_k": 1,
        "id_slot": 1,
        "cache_prompt": True,
        "prompt": prompt,
    })
    assert res.status_code == 200
    prompt_n_full = res.body["timings"]["prompt_n"]
    assert prompt_n_full > 64

    res = server.make_request("POST", "/slots/1?action=save", data={
        "filename": "mm_slot_two_images.bin",
    })
    assert res.status_code == 200
    n_saved = res.body["n_saved"]

    res = server.make_request("POST", "/slots/0?action=restore", data={
        "filename": "mm_slot_two_images.bin",
    })
    assert res.status_code == 200
    assert res.body["n_restored"] == n_saved

    res = server.make_request("POST", "/completions", data={
        "temperature": 0.0,
        "top_k": 1,
        "id_slot": 0,
        "cache_prompt": True,
        "prompt": prompt,
    })
    assert res.status_code == 200
    assert res.body["timings"]["cache_n"] == prompt_n_full - 1
    assert res.body["timings"]["prompt_n"] == 1
    content = res.body["content"]

    res = server.make_request("POST", "/slots/1?action=restore", data={
        "filename": "mm_slot_two_images.bin",
    })
    assert res.status_code == 200
    assert res.body["n_restored"] == n_saved

    res = server.make_request("POST", "/completions", data={
        "temperature": 0.0,
        "top_k": 1,
        "id_slot": 0,
        "cache_prompt": True,
        "prompt": prompt,
    })
    assert res.status_code == 200
    assert res.body["timings"]["cache_n"] == prompt_n_full - 1
    assert res.body["timings"]["prompt_n"] == 1
    content = res.body["content"]

    assert res.body["content"] == content


def test_slot_save_restore_with_image_across_restart(mmproj_server):
    server = mmproj_server
    server.swa_full = True
    server.start()

    prompt_cat = {
        "prompt_string": "What is this: <__media__>\n",
        "multimodal_data": [_get_img_base64(IMG_URL_CAT)],
    }
    res = server.make_request("POST", "/completions", data={
        "temperature": 0.0,
        "top_k": 1,
        "id_slot": 0,
        "cache_prompt": True,
        "prompt": prompt_cat,
    })
    assert res.status_code == 200
    content = res.body["content"]
    prompt_n_full = res.body["timings"]["prompt_n"]

    res = server.make_request("POST", "/slots/0?action=save", data={
        "filename": "mm_slot_restart.bin",
    })
    assert res.status_code == 200
    n_saved = res.body["n_saved"]

    # restart the server with the same model and mmproj: the saved file must restore in the new process and the image KV must be reused
    server.stop()
    server.start()

    res = server.make_request("POST", "/slots/0?action=restore", data={
        "filename": "mm_slot_restart.bin",
    })
    assert res.status_code == 200
    assert res.body["n_restored"] == n_saved

    res = server.make_request("POST", "/completions", data={
        "temperature": 0.0,
        "top_k": 1,
        "id_slot": 0,
        "cache_prompt": True,
        "prompt": prompt_cat,
    })
    assert res.status_code == 200
    assert res.body["timings"]["cache_n"] == prompt_n_full - 1
    assert res.body["timings"]["prompt_n"] == 1
    assert res.body["content"] == content


def test_slot_save_restore_image_payload_larger_than_context(mmproj_server):
    server = mmproj_server
    server.swa_full = True
    server.start()

    # the slot context, as the server computed it (n_ctx split across the slots)
    res = server.make_request("GET", "/props")
    assert res.status_code == 200
    n_ctx_slot = res.body["default_generation_settings"]["n_ctx"]

    # a filler token, used to grow the prompt up to the slot context
    res = server.make_request("POST", "/tokenize", data={"content": " hello" * 8})
    assert res.status_code == 200
    assert len(res.body["tokens"]) == 8

    res = server.make_request("POST", "/completions", data={
        "temperature": 0.0,
        "top_k": 1,
        "id_slot": 0,
        "cache_prompt": True,
        "prompt": {
            "prompt_string": "What is this: <__media__>\n",
            "multimodal_data": [_get_img_base64(IMG_URL_CAT)],
        },
    })
    assert res.status_code == 200

    prompt_cat = {
        "prompt_string": "What is this: <__media__>\n" + " hello" * (n_ctx_slot - res.body["timings"]["prompt_n"] - 8),
        "multimodal_data": [_get_img_base64(IMG_URL_CAT)],
    }
    res = server.make_request("POST", "/completions", data={
        "temperature": 0.0,
        "top_k": 1,
        "id_slot": 0,
        "cache_prompt": True,
        "prompt": prompt_cat,
    })
    assert res.status_code == 200
    prompt_n_full = res.body["timings"]["cache_n"] + res.body["timings"]["prompt_n"]

    res = server.make_request("POST", "/slots/0?action=save", data={
        "filename": "mm_slot_large_payload.bin",
    })
    assert res.status_code == 200

    path = os.path.join(server.slot_save_path, "mm_slot_large_payload.bin")
    with open(path, "rb") as f:
        data = bytearray(f.read())
    payload_size = struct.unpack_from("=I", data, STATE_FILE_HEADER_SIZE - 4)[0]
    assert payload_size > n_ctx_slot  # the scenario under test: the payload does not fit in n_ctx

    # drop the image from the slot, then restore it from the file
    res = server.make_request("POST", "/completion", data={
        "prompt": "The quick brown fox",
        "id_slot": 0,
        "cache_prompt": True,
    })
    assert res.status_code == 200

    res = server.make_request("POST", "/slots/0?action=restore", data={
        "filename": "mm_slot_large_payload.bin",
    })
    assert res.status_code == 200

    res = server.make_request("POST", "/completions", data={
        "temperature": 0.0,
        "top_k": 1,
        "id_slot": 0,
        "cache_prompt": True,
        "prompt": prompt_cat,
    })
    assert res.status_code == 200
    assert res.body["timings"]["cache_n"] == prompt_n_full - 1
    assert res.body["timings"]["prompt_n"] == 1


def test_slot_restore_media_file_without_mmproj(mmproj_server):
    server = mmproj_server
    server.start()

    res = server.make_request("POST", "/completions", data={
        "temperature": 0.0,
        "top_k": 1,
        "id_slot": 0,
        "cache_prompt": True,
        "prompt": {
            "prompt_string": "What is this: <__media__>\n",
            "multimodal_data": [_get_img_base64(IMG_URL_CAT)],
        },
    })
    assert res.status_code == 200

    res = server.make_request("POST", "/slots/0?action=save", data={
        "filename": "mm_slot_no_mmproj.bin",
    })
    assert res.status_code == 200

    # restart the same model without the mmproj: restoring the media file must fail gracefully and leave the slot usable
    server.stop()
    server.no_mmproj = True
    server.start()

    res = server.make_request("POST", "/slots/0?action=restore", data={
        "filename": "mm_slot_no_mmproj.bin",
    })
    assert res.status_code == 400
    assert "Cannot restore media tokens without an mmproj" in res.body["error"]["message"]

    # A failed restore must leave the slot empty and usable.
    res = server.make_request("POST", "/completions", data={
        "temperature": 0.0,
        "top_k": 1,
        "id_slot": 1,
        "cache_prompt": True,
        "prompt": "The quick brown fox",
    })
    assert res.status_code == 200
    content = res.body["content"]

    res = server.make_request("POST", "/completions", data={
        "temperature": 0.0,
        "top_k": 1,
        "id_slot": 0,
        "cache_prompt": True,
        "prompt": "The quick brown fox",
    })
    assert res.status_code == 200
    assert res.body["timings"]["cache_n"] == 0
    assert res.body["content"] == content


#
# Session-keyed automatic save/restore.
#
# A session id in the request (body field or affinity headers) makes the server
# persist the evicted session's KV state to slot_save_path and restore it on the
# next request for the same session, without explicit save/restore calls.
#


def test_session_autosave_restore():
    global server
    server.server_slots = True
    server.start()

    prompt_a = "What is the capital of France?"

    # session a, fresh slot
    res = server.make_request("POST", "/completion", data={
        "prompt": prompt_a,
        "session_id": "a",
        "cache_prompt": True,
    })
    assert res.status_code == 200
    assert res.body["session_id"] == "a"
    prompt_n_a = res.body["timings"]["prompt_n"]
    assert prompt_n_a > 0
    slot_id_a = res.body["id_slot"]

    # session c pinned to the same slot evicts session a to disk
    res = server.make_request("POST", "/completion", data={
        "prompt": "What is the capital of Germany?",
        "session_id": "c",
        "id_slot": slot_id_a,
        "cache_prompt": True,
    })
    assert res.status_code == 200

    # session a again: its state must be restored from disk, full reuse
    res = server.make_request("POST", "/completion", data={
        "prompt": prompt_a,
        "session_id": "a",
        "cache_prompt": True,
    })
    assert res.status_code == 200
    assert res.body["session_id"] == "a"
    assert res.body["timings"]["prompt_n"] == 1
    assert res.body["timings"]["cache_n"] == prompt_n_a - 1

    # the evicted session file is on disk, named from the session id
    files = [f for f in os.listdir(server.slot_save_path) if f.endswith(".bin")]
    assert files == ["llama-session-a.bin"]

    # the sessions are visible per slot
    res = server.make_request("GET", "/slots")
    assert res.status_code == 200
    sessions = {s["session_id"] for s in res.body}
    assert "a" in sessions and "c" in sessions


def test_session_autosave_restore_headers():
    global server
    server.start()

    prompt_a = "What is the capital of France?"

    # session id delivered via the affinity headers instead of the body
    res = server.make_request("POST", "/completion", data={
        "prompt": prompt_a,
        "cache_prompt": True,
    }, headers={"x-session-id": "a"})
    assert res.status_code == 200
    assert res.body["session_id"] == "a"
    prompt_n_a = res.body["timings"]["prompt_n"]
    slot_id_a = res.body["id_slot"]

    # another session evicts it to disk
    res = server.make_request("POST", "/completion", data={
        "prompt": "What is the capital of Germany?",
        "id_slot": slot_id_a,
        "cache_prompt": True,
    }, headers={"x-session-id": "c"})
    assert res.status_code == 200

    # the session is restored from disk on the next request
    res = server.make_request("POST", "/completion", data={
        "prompt": prompt_a,
        "cache_prompt": True,
    }, headers={"session_id": "a"})
    assert res.status_code == 200
    assert res.body["session_id"] == "a"
    assert res.body["timings"]["prompt_n"] == 1
    assert res.body["timings"]["cache_n"] == prompt_n_a - 1


def test_session_id_conflict():
    global server
    server.start()

    # body and header disagree: the request must be rejected
    res = server.make_request("POST", "/completion", data={
        "prompt": "What is the capital of France?",
        "session_id": "a",
    }, headers={"x-session-id": "b"})
    assert res.status_code == 400
    assert "do not match" in res.body["error"]["message"]


def test_session_id_invalid_chars():
    global server
    server.start()

    # a session id that is invalid as a file name must be rejected
    res = server.make_request("POST", "/completion", data={
        "prompt": "What is the capital of France?",
        "session_id": "a/b",
    })
    assert res.status_code == 400
    assert "session_id" in res.body["error"]["message"]


def test_session_autosave_across_restart():
    global server
    server.start()

    prompt_a = "What is the capital of France?"

    res = server.make_request("POST", "/completion", data={
        "prompt": prompt_a,
        "session_id": "a",
        "cache_prompt": True,
    })
    assert res.status_code == 200
    prompt_n_a = res.body["timings"]["prompt_n"]
    slot_id_a = res.body["id_slot"]

    # evict session a to disk
    res = server.make_request("POST", "/completion", data={
        "prompt": "What is the capital of Germany?",
        "session_id": "c",
        "id_slot": slot_id_a,
        "cache_prompt": True,
    })
    assert res.status_code == 200

    # the file must survive the restart and restore in the new process
    server.stop()
    server.start()

    res = server.make_request("POST", "/completion", data={
        "prompt": prompt_a,
        "session_id": "a",
        "cache_prompt": True,
    })
    assert res.status_code == 200
    assert res.body["timings"]["prompt_n"] == 1
    assert res.body["timings"]["cache_n"] == prompt_n_a - 1


@pytest.mark.skipif(sys.platform == "win32", reason="TerminateProcess does not run the shutdown path on Windows")
def test_session_autosave_on_shutdown():
    global server
    server.start()

    prompt_a = "What is the capital of France?"

    # session a is never evicted, so it has no file on disk yet
    res = server.make_request("POST", "/completion", data={
        "prompt": prompt_a,
        "session_id": "a",
        "cache_prompt": True,
    })
    assert res.status_code == 200
    prompt_n_a = res.body["timings"]["prompt_n"]
    assert not any(f.endswith(".bin") for f in os.listdir(server.slot_save_path))

    # on shutdown the in-memory session state must be flushed to disk
    server.stop()
    files = [f for f in os.listdir(server.slot_save_path) if f.endswith(".bin")]
    assert files == ["llama-session-a.bin"]

    # in the new process the session must restore from that file
    server.start()

    res = server.make_request("POST", "/completion", data={
        "prompt": prompt_a,
        "session_id": "a",
        "cache_prompt": True,
    })
    assert res.status_code == 200
    assert res.body["timings"]["prompt_n"] == 1
    assert res.body["timings"]["cache_n"] == prompt_n_a - 1


def test_session_autosave_on_sleep():
    global server
    server.sleep_idle_seconds = 1
    server.start()

    prompt_a = "What is the capital of France?"

    # session a is never evicted, so it has no file on disk yet
    res = server.make_request("POST", "/completion", data={
        "prompt": prompt_a,
        "session_id": "a",
        "cache_prompt": True,
    })
    assert res.status_code == 200
    prompt_n_a = res.body["timings"]["prompt_n"]
    assert not any(f.endswith(".bin") for f in os.listdir(server.slot_save_path))

    # after the idle timeout the server sleeps, which must flush the session to disk
    # do not send requests while waiting, they reset the idle timer
    deadline = time.time() + 15.0
    while time.time() < deadline:
        time.sleep(2.0)
        res = server.make_request("GET", "/props")
        if res.body.get("is_sleeping"):
            break
    else:
        pytest.fail("server did not go to sleep")

    files = [f for f in os.listdir(server.slot_save_path) if f.endswith(".bin")]
    assert files == ["llama-session-a.bin"]

    # in the new process the session must restore from that file
    server.stop()
    server.start()

    res = server.make_request("POST", "/completion", data={
        "prompt": prompt_a,
        "session_id": "a",
        "cache_prompt": True,
    })
    assert res.status_code == 200
    assert res.body["timings"]["prompt_n"] == 1
    assert res.body["timings"]["cache_n"] == prompt_n_a - 1


def test_session_without_save_path():
    global server

    # a server with sessions enabled but no slot_save_path: session ids still
    # pin slots in memory, but evicted sessions cannot be restored
    srv = ServerPreset.tinyllama2()
    srv.temperature = 0.0
    srv.session_id_headers = "x-session-id,session_id"
    srv.start()
    server = srv

    prompt_a = "What is the capital of France?"

    res = srv.make_request("POST", "/completion", data={
        "prompt": prompt_a,
        "session_id": "a",
        "cache_prompt": True,
    })
    assert res.status_code == 200
    prompt_n_a = res.body["timings"]["prompt_n"]
    slot_id_a = res.body["id_slot"]

    res = srv.make_request("POST", "/completion", data={
        "prompt": "What is the capital of Germany?",
        "session_id": "c",
        "id_slot": slot_id_a,
        "cache_prompt": True,
    })
    assert res.status_code == 200

    # no save path means no restore: full re-prefill
    res = srv.make_request("POST", "/completion", data={
        "prompt": prompt_a,
        "session_id": "a",
        "cache_prompt": True,
    })
    assert res.status_code == 200
    assert res.body["timings"]["cache_n"] == 0
    assert res.body["timings"]["prompt_n"] == prompt_n_a


def test_session_id_ignored_without_flag():
    global server_noflag
    server_noflag.start()

    prompt_a = "What is the capital of France?"

    # without --slot-save-sessions the session id is ignored, in the body and in the headers
    res = server_noflag.make_request("POST", "/completion", data={
        "prompt": prompt_a,
        "session_id": "a",
        "cache_prompt": True,
    }, headers={"x-session-id": "a"})
    assert res.status_code == 200
    assert res.body["session_id"] == ""
    prompt_n_a = res.body["timings"]["prompt_n"]
    slot_id_a = res.body["id_slot"]

    # force the slot to be taken by another session
    res = server_noflag.make_request("POST", "/completion", data={
        "prompt": "What is the capital of Germany?",
        "id_slot": slot_id_a,
        "cache_prompt": True,
    })
    assert res.status_code == 200

    # no file was written
    server_noflag.stop()
    files = [f for f in os.listdir(server_noflag.slot_save_path) if f.endswith(".bin")]
    assert files == []

MODEL_DRAFT_FILE_URL = "https://huggingface.co/ggml-org/tiny-llamas/resolve/main/stories15M-q4_0.gguf"


def test_session_max_sessions():
    global server
    server.n_slots = 1
    server.session_max_sessions = 2
    server.start()

    prompts = [
        "What is the capital of France?",
        "What is the capital of Germany?",
        "What is the capital of Italy?",
    ]
    for sid, prompt in zip(["a", "b", "c"], prompts):
        res = server.make_request("POST", "/completion", data={
            "prompt": prompt,
            "session_id": sid,
            "cache_prompt": True,
        })
        assert res.status_code == 200

    # c holds the slot, b is on disk, the oldest (a) was pruned by the cap
    files = set(os.listdir(server.slot_save_path))
    assert files == {"llama-session-b.bin"}

    # request a new session: c is evicted to disk, b is pruned
    res = server.make_request("POST", "/completion", data={
        "prompt": "What is the capital of Spain?",
        "session_id": "d",
        "cache_prompt": True,
    })
    assert res.status_code == 200
    files = set(os.listdir(server.slot_save_path))
    assert files == {"llama-session-c.bin"}


def test_session_sweep_counts_disk_files():
    global server
    server.session_max_sessions = 1

    # a session file left by a previous run, older than anything created now
    fake = os.path.join(server.slot_save_path, "llama-session-z.bin")
    with open(fake, "wb") as f:
        f.write(b"")
    past = time.time() - 86400
    os.utime(fake, (past, past))

    server.start()

    # the file must be registered at startup and count against the cap
    res = server.make_request("POST", "/completion", data={
        "prompt": "What is the capital of France?",
        "session_id": "a",
        "cache_prompt": True,
    })
    assert res.status_code == 200

    # the pre-existing file was the oldest, so it was pruned
    assert not os.path.exists(fake)


def test_session_sweep_gated_by_flag():
    global server_noflag

    # without the flag, leftover session files are not indexed or pruned
    fake = os.path.join(server_noflag.slot_save_path, "llama-session-z.bin")
    with open(fake, "wb") as f:
        f.write(b"")

    server_noflag.start()
    assert os.path.exists(fake)
    server_noflag.stop()
    assert os.path.exists(fake)


def test_session_autosave_dft_pair(tmp_path):
    # speculative server: the draft context state is saved and restored alongside the main one
    server = ServerPreset.stories15m_moe()
    server.slot_save_path = str(tmp_path)
    server.temperature = 0.0
    server.session_id_headers = "x-session-id,session_id"
    server.model_draft = download_file(MODEL_DRAFT_FILE_URL)
    server.spec_type = "draft-simple"
    server.spec_draft_n_min = 1
    server.spec_draft_n_max = 2
    server.start()

    prompt_a = "What is the capital of France?"

    res = server.make_request("POST", "/completion", data={
        "prompt": prompt_a,
        "session_id": "a",
        "cache_prompt": True,
    })
    assert res.status_code == 200
    prompt_n_a = res.body["timings"]["prompt_n"]
    slot_id_a = res.body["id_slot"]

    # evict session a to disk
    res = server.make_request("POST", "/completion", data={
        "prompt": "What is the capital of Germany?",
        "session_id": "c",
        "id_slot": slot_id_a,
        "cache_prompt": True,
    })
    assert res.status_code == 200

    # both the target and the draft state files are written
    files = set(os.listdir(server.slot_save_path))
    assert files == {"llama-session-a.bin", "llama-session-a.dft"}

    # restart: the session restores with the draft state included
    server.stop()
    server.start()
    res = server.make_request("POST", "/completion", data={
        "prompt": prompt_a,
        "session_id": "a",
        "cache_prompt": True,
    })
    assert res.status_code == 200
    assert res.body["session_id"] == "a"
    assert res.body["timings"]["prompt_n"] == 1
    assert res.body["timings"]["cache_n"] == prompt_n_a - 1

    # with a cap of one session, exactly one pair of files survives the startup sweep
    server.session_max_sessions = 1
    server.stop()
    server.start()
    files = set(os.listdir(server.slot_save_path))
    assert len(files) == 2
    sid = next(f[len("llama-session-"):-4] for f in files if f.endswith(".bin"))
    assert files == {f"llama-session-{sid}.bin", f"llama-session-{sid}.dft"}

    # the surviving session restores from its pair
    prompt_map = {"a": prompt_a, "c": "What is the capital of Germany?"}
    res = server.make_request("POST", "/completion", data={
        "prompt": prompt_map[sid],
        "session_id": sid,
        "cache_prompt": True,
    })
    assert res.status_code == 200
    assert res.body["timings"]["prompt_n"] == 1

    # a new session evicts the survivor and prunes its pair
    res = server.make_request("POST", "/completion", data={
        "prompt": "What is the capital of Spain?",
        "session_id": "d",
        "cache_prompt": True,
    })
    assert res.status_code == 200
    files = set(os.listdir(server.slot_save_path))
    assert files == set()
