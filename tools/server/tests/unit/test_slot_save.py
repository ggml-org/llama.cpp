import pytest
from utils import *
import base64
import requests
import struct

# sequence state file: magic(4) version(4) payload_size(4), then payload_size llama_token words
STATE_FILE_HEADER_SIZE = 12
# the token payload holds a packed server_tokens object (see server_tokens::serialize()):
#   LLAMA_TOKEN_NULL(4) version(4) n_tokens(4) tokens, media list, zero padding to whole tokens
PACKED_HEADER_SIZE = 12  # LLAMA_TOKEN_NULL, version, n_tokens
LLAMA_TOKEN_NULL = 0xFFFFFFFF  # -1 read back as an unsigned word

# media list layout in the packed payload: n_media(4), then per image: start_idx(4) chunk_size(4) chunk blob
N_MEDIA_FIELD_SIZE = 4
START_IDX_FIELD_SIZE = 4
CHUNK_SIZE_FIELD_SIZE = 4

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
    path = os.path.join("tmp", "slot_legacy.bin")
    with open(path, "rb") as f:
        data = bytearray(f.read())

    payload_size = struct.unpack_from("=I", data, STATE_FILE_HEADER_SIZE - 4)[0]
    payload_end = STATE_FILE_HEADER_SIZE + payload_size * 4
    assert struct.unpack_from("=I", data, STATE_FILE_HEADER_SIZE)[0] == LLAMA_TOKEN_NULL
    n_tokens = struct.unpack_from("=I", data, STATE_FILE_HEADER_SIZE + 8)[0]
    assert n_tokens == 84

    tokens_start = STATE_FILE_HEADER_SIZE + PACKED_HEADER_SIZE
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


def test_slot_restore_unsupported_state_version():
    global server
    server.start()

    res = server.make_request("POST", "/completion", data={
        "prompt": "What is the capital of France?",
        "id_slot": 1,
        "cache_prompt": True,
    })
    assert res.status_code == 200

    res = server.make_request("POST", "/slots/1?action=save", data={
        "filename": "slot_bad_version.bin",
    })
    assert res.status_code == 200

    # a payload with an unknown format version must be rejected, not guessed at
    path = os.path.join("tmp", "slot_bad_version.bin")
    with open(path, "rb") as f:
        data = bytearray(f.read())

    assert struct.unpack_from("=I", data, STATE_FILE_HEADER_SIZE)[0] == LLAMA_TOKEN_NULL
    struct.pack_into("=I", data, STATE_FILE_HEADER_SIZE + 4, 99)  # version

    with open(path, "wb") as f:
        f.write(data)

    res = server.make_request("POST", "/slots/0?action=restore", data={
        "filename": "slot_bad_version.bin",
    })
    assert res.status_code == 400
    assert "Unsupported server tokens state version" in res.body["error"]["message"]

    res = server.make_request("POST", "/completion", data={
        "prompt": "The quick brown fox",
        "id_slot": 0,
        "cache_prompt": True,
    })
    assert res.status_code == 200


def test_slot_restore_truncated_file():
    global server
    server.start()

    res = server.make_request("POST", "/completion", data={
        "prompt": "What is the capital of France?",
        "id_slot": 1,
        "cache_prompt": True,
    })
    assert res.status_code == 200

    res = server.make_request("POST", "/slots/1?action=save", data={
        "filename": "slot_truncated.bin",
    })
    assert res.status_code == 200

    # cut the file inside the token payload: the payload size in the header can no longer be trusted, so the restore buffer falls back to n_ctx and loading reports the malformed file
    path = os.path.join("tmp", "slot_truncated.bin")
    with open(path, "r+b") as f:
        f.truncate(STATE_FILE_HEADER_SIZE + 4)

    res = server.make_request("POST", "/slots/0?action=restore", data={
        "filename": "slot_truncated.bin",
    })
    assert res.status_code == 400
    assert "No available space in KV cache or invalid slot save file" in res.body["error"]["message"]

    res = server.make_request("POST", "/completion", data={
        "prompt": "The quick brown fox",
        "id_slot": 0,
        "cache_prompt": True,
    })
    assert res.status_code == 200


def test_slot_restore_corrupt_payload_size():
    global server
    server.start()

    res = server.make_request("POST", "/completion", data={
        "prompt": "What is the capital of France?",
        "id_slot": 1,
        "cache_prompt": True,
    })
    assert res.status_code == 200

    res = server.make_request("POST", "/slots/1?action=save", data={
        "filename": "slot_bad_payload_size.bin",
    })
    assert res.status_code == 200

    # a payload size claiming ~4G tokens contradicts the actual file size, so it must not be trusted (or allocated): the restore buffer falls back to n_ctx and loading reports the malformed file
    path = os.path.join("tmp", "slot_bad_payload_size.bin")
    with open(path, "rb") as f:
        data = bytearray(f.read())

    struct.pack_into("=I", data, STATE_FILE_HEADER_SIZE - 4, 0xFFFFFFF0)

    with open(path, "wb") as f:
        f.write(data)

    res = server.make_request("POST", "/slots/0?action=restore", data={
        "filename": "slot_bad_payload_size.bin",
    })
    assert res.status_code == 400
    assert "No available space in KV cache or invalid slot save file" in res.body["error"]["message"]

    res = server.make_request("POST", "/completion", data={
        "prompt": "The quick brown fox",
        "id_slot": 0,
        "cache_prompt": True,
    })
    assert res.status_code == 200


def test_slot_restore_prompt_larger_than_slot_context():
    global server
    server.start()

    res = server.make_request("POST", "/completion", data={
        "prompt": "What is the capital of France?",
        "id_slot": 1,
        "cache_prompt": True,
    })
    assert res.status_code == 200

    res = server.make_request("POST", "/slots/1?action=save", data={
        "filename": "slot_large_prompt.bin",
    })
    assert res.status_code == 200

    # the slot context, as the server computed it (n_ctx split across the slots)
    res = server.make_request("GET", "/props")
    assert res.status_code == 200
    n_ctx_slot = res.body["default_generation_settings"]["n_ctx"]

    # grow the logical token list past the slot context while the KV section keeps its cells: only the restored prompt length check can reject such a file
    path = os.path.join("tmp", "slot_large_prompt.bin")
    with open(path, "rb") as f:
        data = bytearray(f.read())

    payload_size = struct.unpack_from("=I", data, STATE_FILE_HEADER_SIZE - 4)[0]
    assert struct.unpack_from("=I", data, STATE_FILE_HEADER_SIZE)[0] == LLAMA_TOKEN_NULL
    n_tokens = struct.unpack_from("=I", data, STATE_FILE_HEADER_SIZE + 8)[0]
    assert n_tokens == 84

    tokens_start = STATE_FILE_HEADER_SIZE + PACKED_HEADER_SIZE
    tokens = data[tokens_start:tokens_start + n_tokens * 4]
    filler = tokens * (n_ctx_slot // n_tokens + 1)
    filler = filler[:(n_ctx_slot + 1 - n_tokens) * 4]
    data = data[:tokens_start] + tokens + filler + data[tokens_start + n_tokens * 4:]
    struct.pack_into("=I", data, STATE_FILE_HEADER_SIZE - 4, payload_size + len(filler) // 4)
    struct.pack_into("=I", data, STATE_FILE_HEADER_SIZE + 8, n_ctx_slot + 1)

    with open(path, "wb") as f:
        f.write(data)

    res = server.make_request("POST", "/slots/0?action=restore", data={
        "filename": "slot_large_prompt.bin",
    })
    assert res.status_code == 400
    assert "Restored prompt does not fit in the slot context" in res.body["error"]["message"]

    res = server.make_request("POST", "/completion", data={
        "prompt": "The quick brown fox",
        "id_slot": 0,
        "cache_prompt": True,
    })
    assert res.status_code == 200


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


def _media_list_offset(data: bytearray) -> int:
    """Offset of the media list (the n_media field) inside the packed token payload."""
    assert struct.unpack_from("=I", data, STATE_FILE_HEADER_SIZE)[0] == LLAMA_TOKEN_NULL
    n_tokens = struct.unpack_from("=I", data, STATE_FILE_HEADER_SIZE + 8)[0]
    return STATE_FILE_HEADER_SIZE + PACKED_HEADER_SIZE + n_tokens * 4


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

    # The restored slot is usable for a follow-up completion.
    # We do NOT assert prefix reuse here: tinygemma3 is a SWA model, which forces full prompt re-processing after a restore (a model property, not the save/restore gate under test).
    res = server.make_request("POST", "/completion", data={
        "prompt": "The quick brown fox jumps over the lazy dog.",
        "id_slot": 0,
        "cache_prompt": True,
    })
    assert res.status_code == 200


def test_slot_save_restore_with_image(mmproj_server):
    server = mmproj_server
    # the SWA cache cannot be rolled back after a restore (checkpoints are not part of the save file), which would force full re-processing and hide the prefix reuse being verified below; use the full-size SWA cache instead
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

    # erase remains gated on media content
    res = server.make_request("POST", "/slots/1?action=erase")
    assert res.status_code == 501
    assert res.body["error"]["type"] == "not_supported_error"

    res = server.make_request("POST", "/slots/1?action=save", data={
        "filename": "mm_slot_image.bin",
    })
    assert res.status_code == 200
    n_saved = res.body["n_saved"]
    n_written = res.body["n_written"]
    assert n_saved > 0
    assert n_written > 0

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
    content = res.body["content"]
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
    # the token payload holds the packed server_tokens object (tokens plus media state), so it is longer than the number of tokens the slot holds: the restore buffer is sized from the file, not from n_ctx
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

    path = os.path.join("tmp", "mm_slot_large_payload.bin")
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
    assert "Cannot restore image tokens without an mmproj" in res.body["error"]["message"]

    res = server.make_request("POST", "/completion", data={
        "prompt": "The quick brown fox",
        "id_slot": 0,
        "cache_prompt": True,
    })
    assert res.status_code == 200


def test_slot_restore_corrupt_media_state(mmproj_server):
    server = mmproj_server
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
    content = res.body["content"]

    res = server.make_request("POST", "/slots/1?action=save", data={
        "filename": "mm_slot_corrupt.bin",
    })
    assert res.status_code == 200

    # corrupt the media state: grow the declared chunk size past the end of the payload
    path = os.path.join("tmp", "mm_slot_corrupt.bin")
    with open(path, "rb") as f:
        data = bytearray(f.read())

    media_offset = _media_list_offset(data)
    assert struct.unpack_from("=I", data, media_offset)[0] == 1  # n_media
    chunk_size_offset = media_offset + N_MEDIA_FIELD_SIZE + START_IDX_FIELD_SIZE
    struct.pack_into("=I", data, chunk_size_offset, 0x7FFFFFFF)

    with open(path, "wb") as f:
        f.write(data)

    res = server.make_request("POST", "/slots/0?action=restore", data={
        "filename": "mm_slot_corrupt.bin",
    })
    assert res.status_code == 400
    assert "Unexpected end of server tokens state" in res.body["error"]["message"]

    # the failed restore must leave slot 0 empty: no reusable prefix, and no stale KV cells that would corrupt greedy decoding
    res = server.make_request("POST", "/completions", data={
        "temperature": 0.0,
        "top_k": 1,
        "id_slot": 0,
        "cache_prompt": True,
        "prompt": prompt_cat,
    })
    assert res.status_code == 200
    assert res.body["timings"]["cache_n"] == 0
    assert res.body["content"] == content


def test_slot_restore_corrupt_state_padding(mmproj_server):
    server = mmproj_server
    server.start()

    res = server.make_request("POST", "/completions", data={
        "temperature": 0.0,
        "top_k": 1,
        "id_slot": 1,
        "cache_prompt": True,
        "prompt": {
            "prompt_string": "What is this: <__media__>\n",
            "multimodal_data": [_get_img_base64(IMG_URL_CAT)],
        },
    })
    assert res.status_code == 200

    res = server.make_request("POST", "/slots/1?action=save", data={
        "filename": "mm_slot_padding.bin",
    })
    assert res.status_code == 200

    # a non-zero byte in the zero padding at the end of the payload must be rejected
    path = os.path.join("tmp", "mm_slot_padding.bin")
    with open(path, "rb") as f:
        data = bytearray(f.read())

    payload_size = struct.unpack_from("=I", data, STATE_FILE_HEADER_SIZE - 4)[0]
    payload_end = STATE_FILE_HEADER_SIZE + payload_size * 4

    media_offset = _media_list_offset(data)
    assert struct.unpack_from("=I", data, media_offset)[0] == 1  # n_media
    chunk_size_offset = media_offset + N_MEDIA_FIELD_SIZE + START_IDX_FIELD_SIZE
    chunk_size = struct.unpack_from("=I", data, chunk_size_offset)[0]
    stream_end = chunk_size_offset + CHUNK_SIZE_FIELD_SIZE + chunk_size

    assert payload_end - stream_end > 0  # this image must leave padding bytes to corrupt
    data[stream_end] = 0xFF

    with open(path, "wb") as f:
        f.write(data)

    res = server.make_request("POST", "/slots/0?action=restore", data={
        "filename": "mm_slot_padding.bin",
    })
    assert res.status_code == 400
    assert "Invalid padding in server tokens state" in res.body["error"]["message"]

    res = server.make_request("POST", "/completion", data={
        "prompt": "The quick brown fox",
        "id_slot": 0,
        "cache_prompt": True,
    })
    assert res.status_code == 200


def test_slot_restore_media_state_without_image_id(mmproj_server):
    server = mmproj_server
    server.start()

    res = server.make_request("POST", "/completions", data={
        "temperature": 0.0,
        "top_k": 1,
        "id_slot": 1,
        "cache_prompt": True,
        "prompt": {
            "prompt_string": "What is this: <__media__>\n",
            "multimodal_data": [_get_img_base64(IMG_URL_CAT)],
        },
    })
    assert res.status_code == 200

    res = server.make_request("POST", "/slots/1?action=save", data={
        "filename": "mm_slot_empty_image_id.bin",
    })
    assert res.status_code == 200

    path = os.path.join("tmp", "mm_slot_empty_image_id.bin")
    with open(path, "rb") as f:
        data = bytearray(f.read())

    payload_size = struct.unpack_from("=I", data, STATE_FILE_HEADER_SIZE - 4)[0]
    payload_end = STATE_FILE_HEADER_SIZE + payload_size * 4

    media_offset = _media_list_offset(data)
    assert struct.unpack_from("=I", data, media_offset)[0] == 1  # n_media
    # chunk blob (mtmd serialization v1): version(8) type(4) n_text_tokens(8) has_image(1)
    #   nx(4) ny(4) pos(4) image_idx(4) n_temporal_merge(4) id_size(8) id bytes ...
    blob_header_size = 41  # version, type, n_text_tokens, has_image, nx, ny, pos, image_idx, n_temporal_merge
    id_size_field_size = 8

    chunk_size_offset = media_offset + N_MEDIA_FIELD_SIZE + START_IDX_FIELD_SIZE
    chunk_size = struct.unpack_from("=I", data, chunk_size_offset)[0]
    blob_start = chunk_size_offset + CHUNK_SIZE_FIELD_SIZE
    stream_end = blob_start + chunk_size  # end of the packed payload, before the zero padding

    id_size_offset = blob_start + blob_header_size
    id_size = struct.unpack_from("=Q", data, id_size_offset)[0]
    assert id_size > 0
    struct.pack_into("=Q", data, id_size_offset, 0)
    struct.pack_into("=I", data, chunk_size_offset, chunk_size - id_size)

    # drop the id bytes and re-pad the payload to whole tokens
    id_offset = id_size_offset + id_size_field_size
    payload = data[STATE_FILE_HEADER_SIZE:id_offset] + data[id_offset + id_size:stream_end]
    payload += bytes(-len(payload) % 4)
    data = data[:STATE_FILE_HEADER_SIZE] + payload + data[payload_end:]
    struct.pack_into("=I", data, STATE_FILE_HEADER_SIZE - 4, len(payload) // 4)

    with open(path, "wb") as f:
        f.write(data)

    res = server.make_request("POST", "/slots/0?action=restore", data={
        "filename": "mm_slot_empty_image_id.bin",
    })
    assert res.status_code == 400
    assert "Image ID is missing in server tokens state" in res.body["error"]["message"]

    res = server.make_request("POST", "/completion", data={
        "prompt": "The quick brown fox",
        "id_slot": 0,
        "cache_prompt": True,
    })
    assert res.status_code == 200


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
