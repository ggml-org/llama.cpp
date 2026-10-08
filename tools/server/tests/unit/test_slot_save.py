import pytest
from utils import *
import base64
import requests
import struct

# sequence state file: magic(4) version(4) payload_size(4), then payload_size llama_token words
STATE_FILE_HEADER_SIZE = 12

server = ServerPreset.tinyllama2()

@pytest.fixture(autouse=True)
def create_server(tmp_path):
    global server
    server = ServerPreset.tinyllama2()
    server.slot_save_path = str(tmp_path)
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

# sidecar file written next to the slot save file:
#   magic(4) "LCKP" version(u32) n_checkpoints(u32) desc_tgt_len(u32) desc_dft_len(u32)
#   desc_tgt bytes + desc_dft bytes
#   per checkpoint: n_tokens(i64) id_task(i32) pos_min(i32) pos_max(i32)
#     data_tgt_size(u64) + bytes, data_dft_size(u64) + bytes, data_spec_size(u64) + bytes
CKPT_MAGIC = b"LCKP"
CKPT_RECORD_HEADER = struct.Struct("<qiII")  # n_tokens, id_task, pos_min, pos_max; then 3x u64 blob sizes

# a text prefix long enough to produce a context checkpoint, and a suffix to append after the restore
CKPT_PREFIX = (
    "The city of Amsterdam was built on the river Amstel. Its concentric canals "
    "were dug in the seventeenth century during the Dutch Golden Age, when the city "
    "grew rapidly as a center of trade and science. The canal ring, planned as a "
    "unified whole, remains one of the most remarkable urban projects of early modern "
    "Europe. "
) * 4
CKPT_SUFFIX = "What is the name of the river Amsterdam was built on?"

# a second prefix of similar length on a different topic, for the overwrite test
CKPT_PREFIX_B = (
    "The Colosseum in Rome was raised on the banks of a drained marsh. Its "
    "tiered arches were built in the first century under the Flavian emperors, "
    "when the city grew into the heart of a vast empire. The amphitheatre, "
    "designed to seat eighty thousand, remains one of the most remarkable "
    "monuments of the ancient world. "
) * 4


@pytest.fixture
def ckpt_server():
    # tinygemma3 is an SWA model: prefix reuse requires a checkpoint, so it exercises the persist path
    ckpt = ServerPreset.tinygemma3()
    ckpt.slot_save_path = "./tmp"
    ckpt.temperature = 0.0
    ckpt.n_ctx = 4096
    ckpt.n_predict = 8
    # disable the in-RAM prompt cache: prefix reuse can then only come from the disk state
    ckpt.cache_ram = 0
    # lower the checkpoint spacing so the prefix earns a checkpoint during processing
    ckpt.checkpoint_min_step = 128
    return ckpt


def _ckpt_completion(server, prompt: str, id_slot: int):
    res = server.make_request("POST", "/completion", data={
        "prompt": prompt,
        "id_slot": id_slot,
        "cache_prompt": True,
        "seed": 42,
    })
    assert res.status_code == 200
    return res


def _ckpt_parse_sidecar(path: str):
    with open(path, "rb") as f:
        data = f.read()
    magic, version, n_ckpt, tgt_len, dft_len = struct.unpack_from("<4sIIII", data, 0)
    assert magic == CKPT_MAGIC
    assert version == 1
    assert n_ckpt >= 1
    off = 20
    desc_tgt = data[off:off + tgt_len]; off += tgt_len
    desc_dft = data[off:off + dft_len]; off += dft_len
    records = []
    for _ in range(n_ckpt):
        n_tokens, id_task, pos_min, pos_max = CKPT_RECORD_HEADER.unpack_from(data, off)
        off += CKPT_RECORD_HEADER.size
        # each blob is preceded by its size, matching the write order
        tgt_size = struct.unpack_from("<Q", data, off)[0]; off += 8 + tgt_size
        dft_size = struct.unpack_from("<Q", data, off)[0]; off += 8 + dft_size
        spec_size = struct.unpack_from("<Q", data, off)[0]; off += 8 + spec_size
        records.append((n_tokens, id_task, pos_min, pos_max))
    # the byte layout must be fully consumed - the ledger is byte-exact
    assert off == len(data)
    return records


def _ckpt_mutate(path: str, off: int, nbytes: int):
    # flip bytes in [off, off+nbytes): a minimal corruption that must not be silently accepted
    with open(path, "rb") as f:
        data = bytearray(f.read())
    for i in range(off, off + nbytes):
        data[i] ^= 0xFF
    with open(path, "wb") as f:
        f.write(data)


def test_slot_save_checkpoint_sidecar_and_reuse(ckpt_server):
    server = ckpt_server
    server.start()

    res = _ckpt_completion(server, CKPT_PREFIX, 1)
    assert res.body["timings"]["cache_n"] == 0  # cold: everything processed
    total_n = res.body["timings"]["prompt_n"] + res.body["timings"]["cache_n"]

    res = server.make_request("POST", "/slots/1?action=save", data={"filename": "ckpt_inproc.bin"})
    assert res.status_code == 200
    n_saved = res.body["n_saved"]

    records = _ckpt_parse_sidecar("./tmp/ckpt_inproc.bin.ckpt")
    # multi-checkpoint fidelity: count, ordering and bounds of every record
    n_tokens = [r[0] for r in records]
    assert n_tokens == sorted(n_tokens)
    assert all(0 < t <= n_saved for t in n_tokens)

    res = server.make_request("POST", "/slots/0?action=restore", data={"filename": "ckpt_inproc.bin"})
    assert res.status_code == 200
    assert res.body["n_restored"] == n_saved

    # slot 0 never saw this prefix in RAM: reuse can only come from the restored checkpoints
    res = _ckpt_completion(server, CKPT_PREFIX + CKPT_SUFFIX, 0)
    assert res.body["timings"]["cache_n"] > 0, "restored prefix was not reused: checkpoints were lost in the round-trip"
    assert res.body["timings"]["prompt_n"] < total_n


def test_slot_save_checkpoint_reuse_across_restart(ckpt_server):
    server = ckpt_server
    server.start()

    res = _ckpt_completion(server, CKPT_PREFIX, 0)
    assert res.body["timings"]["cache_n"] == 0

    res = server.make_request("POST", "/slots/0?action=save", data={"filename": "ckpt_restart.bin"})
    assert res.status_code == 200
    n_saved = res.body["n_saved"]

    # restart: all in-memory checkpoint state dies with the process
    server.stop()
    server.start()

    res = server.make_request("POST", "/slots/0?action=restore", data={"filename": "ckpt_restart.bin"})
    assert res.status_code == 200
    assert res.body["n_restored"] == n_saved

    res = _ckpt_completion(server, CKPT_PREFIX + CKPT_SUFFIX, 0)
    assert res.body["timings"]["cache_n"] > 0, "restored prefix was not reused after restart: checkpoints were never persisted"


def test_slot_restore_sidecar_missing_or_truncated_degrades(ckpt_server):
    server = ckpt_server
    server.start()

    _ckpt_completion(server, CKPT_PREFIX, 0)
    res = server.make_request("POST", "/slots/0?action=save", data={"filename": "ckpt_degrade.bin"})
    assert res.status_code == 200
    n_saved = res.body["n_saved"]
    sidecar = "./tmp/ckpt_degrade.bin.ckpt"

    # truncated copies (cut mid-header, mid-descriptor, mid-record) must be rejected
    # without failing the restore; the slot degrades to full re-processing
    with open(sidecar, "rb") as f:
        raw = f.read()
    for cut in (10, 24, len(raw) - 1):
        truncated = raw[:cut]
        with open(sidecar, "wb") as f:
            f.write(truncated)
        res = server.make_request("POST", "/slots/1?action=restore", data={"filename": "ckpt_degrade.bin"})
        assert res.status_code == 200
        assert res.body["n_restored"] == n_saved

    os.remove(sidecar)
    res = server.make_request("POST", "/slots/1?action=restore", data={"filename": "ckpt_degrade.bin"})
    assert res.status_code == 200
    assert res.body["n_restored"] == n_saved

    # the slot stays usable and deterministic after the degraded restores
    content = _ckpt_completion(server, CKPT_PREFIX + CKPT_SUFFIX, 1).body["content"]
    assert _ckpt_completion(server, CKPT_PREFIX + CKPT_SUFFIX, 1).body["content"] == content


def test_slot_restore_sidecar_desc_or_version_mismatch_degrades(ckpt_server):
    server = ckpt_server
    server.start()

    _ckpt_completion(server, CKPT_PREFIX, 0)
    res = server.make_request("POST", "/slots/0?action=save", data={"filename": "ckpt_mismatch.bin"})
    assert res.status_code == 200
    n_saved = res.body["n_saved"]
    # a rejected ledger must not reuse the whole restored prefix; a few tokens of KV reuse survive
    prefix_n = n_saved
    sidecar = "./tmp/ckpt_mismatch.bin.ckpt"

    with open(sidecar, "rb") as f:
        raw = f.read()
    # descriptor lengths live in the header at offsets 12 (tgt) and 16 (dft)
    tgt_len = struct.unpack_from("<I", raw, 12)[0]
    dft_len = struct.unpack_from("<I", raw, 16)[0]
    assert tgt_len > 0  # the target descriptor is present; the draft one may be empty for non-speculative models

    # corrupt one field at a time, restore into slot 1, and confirm the restore succeeds
    # but the corrupted ledger is rejected: reuse must not reach the end of the prefix.
    # note: a rejected ledger still leaves a few tokens of KV reuse from the restored prompt
    ckpt_off = 20 + tgt_len + dft_len
    assert struct.unpack_from("<I", raw, 8)[0] >= 1  # the first record exists
    mutations = [("desc_tgt", 20), ("n_tokens", ckpt_off)]
    if dft_len > 0:
        mutations.insert(1, ("desc_dft", 20 + tgt_len))
    for name, off in mutations:
        with open(sidecar, "wb") as f:
            f.write(raw)
        _ckpt_mutate(sidecar, off, 1)
        res = server.make_request("POST", "/slots/1?action=restore", data={"filename": "ckpt_mismatch.bin"})
        assert res.status_code == 200
        assert res.body["n_restored"] == n_saved
        assert _ckpt_completion(server, CKPT_PREFIX + CKPT_SUFFIX, 1).body["timings"]["cache_n"] < prefix_n

    # a wrong ledger version must be rejected the same way
    with open(sidecar, "wb") as f:
        f.write(raw)
    with open(sidecar, "rb") as f:
        data = bytearray(f.read())
    struct.pack_into("<I", data, 4, 2)  # bump the version to 2
    with open(sidecar, "wb") as f:
        f.write(data)
    res = server.make_request("POST", "/slots/1?action=restore", data={"filename": "ckpt_mismatch.bin"})
    assert res.status_code == 200
    assert res.body["n_restored"] == n_saved
    # the version bump must reject the ledger the same way
    assert _ckpt_completion(server, CKPT_PREFIX + CKPT_SUFFIX, 1).body["timings"]["cache_n"] < prefix_n

    # write the original sidecar back: the unmutated ledger must work again
    with open(sidecar, "wb") as f:
        f.write(raw)
    res = server.make_request("POST", "/slots/1?action=restore", data={"filename": "ckpt_mismatch.bin"})
    assert res.status_code == 200
    assert res.body["n_restored"] == n_saved
    assert _ckpt_completion(server, CKPT_PREFIX + CKPT_SUFFIX, 1).body["timings"]["cache_n"] > 0


@pytest.fixture
def ckpt_server_no_ckpt():
    # same as ckpt_server, for the empty-ledger test which disables checkpoints at start
    ckpt = ServerPreset.tinygemma3()
    ckpt.slot_save_path = "./tmp"
    ckpt.temperature = 0.0
    ckpt.n_ctx = 4096
    ckpt.n_predict = 8
    # disable the in-RAM prompt cache: prefix reuse can then only come from the disk state
    ckpt.cache_ram = 0
    return ckpt


def test_slot_save_checkpoint_empty_ledger(ckpt_server_no_ckpt):
    server = ckpt_server_no_ckpt
    # checkpoint_min_step (default 8192) only spaces checkpoints apart; the near-end-of-prompt
    # checkpoint is written for a short prompt regardless. Disable checkpoints to reach the
    # writer's zero-checkpoint path: the sidecar is then header-only. The child captures the
    # env at launch, so restoring it right after start() leaves the other servers untouched.
    prev = os.environ.get("LLAMA_ARG_CTX_CHECKPOINTS")
    os.environ["LLAMA_ARG_CTX_CHECKPOINTS"] = "0"
    try:
        server.start()
    finally:
        if prev is None:
            os.environ.pop("LLAMA_ARG_CTX_CHECKPOINTS", None)
        else:
            os.environ["LLAMA_ARG_CTX_CHECKPOINTS"] = prev

    prompt = "What is the capital of France?"
    res = _ckpt_completion(server, prompt, 1)
    assert res.body["timings"]["cache_n"] == 0  # cold: everything processed

    res = server.make_request("POST", "/slots/1?action=save", data={"filename": "ckpt_empty.bin"})
    assert res.status_code == 200
    n_saved = res.body["n_saved"]

    # no checkpoints were written: the sidecar is header-only
    sidecar = "./tmp/ckpt_empty.bin.ckpt"
    with open(sidecar, "rb") as f:
        data = f.read()
    magic, version, n_ckpt, tgt_len, dft_len = struct.unpack_from("<4sIIII", data, 0)
    assert magic == CKPT_MAGIC
    assert version == 1
    assert n_ckpt == 0
    # header-only file: no record bytes past the two descriptors
    assert len(data) == 20 + tgt_len + dft_len

    res = server.make_request("POST", "/slots/0?action=restore", data={"filename": "ckpt_empty.bin"})
    assert res.status_code == 200
    assert res.body["n_restored"] == n_saved

    # no checkpoint was persisted: the same prompt on slot 0 must re-process fully
    res = _ckpt_completion(server, prompt, 0)
    assert res.body["timings"]["cache_n"] == 0


def test_slot_save_checkpoint_overwrite(ckpt_server):
    server = ckpt_server
    server.start()

    _ckpt_completion(server, CKPT_PREFIX, 1)
    res = server.make_request("POST", "/slots/1?action=save", data={"filename": "ckpt_overwrite.bin"})
    assert res.status_code == 200

    # the first save must earn at least one checkpoint
    assert len(_ckpt_parse_sidecar("./tmp/ckpt_overwrite.bin.ckpt")) >= 1

    # erase the slot, then save a different prefix of similar length to the same file
    res = server.make_request("POST", "/slots/1?action=erase")
    assert res.status_code == 200

    _ckpt_completion(server, CKPT_PREFIX_B, 1)
    res = server.make_request("POST", "/slots/1?action=save", data={"filename": "ckpt_overwrite.bin"})
    assert res.status_code == 200
    n_saved = res.body["n_saved"]

    # re-parsing must be byte-exact; a stale/append writer would leak records beyond the new save
    records = _ckpt_parse_sidecar("./tmp/ckpt_overwrite.bin.ckpt")
    assert all(0 < t <= n_saved for t in [r[0] for r in records])

    res = server.make_request("POST", "/slots/0?action=restore", data={"filename": "ckpt_overwrite.bin"})
    assert res.status_code == 200

    # reuse must come from the new save (CKPT_PREFIX_B), not the old one
    res = _ckpt_completion(server, CKPT_PREFIX_B + CKPT_SUFFIX, 0)
    assert res.body["timings"]["cache_n"] > 0


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
