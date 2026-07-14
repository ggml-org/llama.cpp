import os
import tempfile

from utils import *


def test_disaggregated_prefill_matches_local():
    prompt = (
        "Once upon a time in a land far away, a traveler crossed the mountains "
        "to find a hidden library and asked the keeper for a story."
    )
    request = {
        "prompt": prompt,
        "n_predict": 16,
        "seed": 42,
        "temperature": 0.0,
        "cache_prompt": False,
        "return_tokens": True,
    }

    local = ServerPreset.tinyllama2()
    local.start()
    expected = local.make_request("POST", "/completion", data=request)
    local.stop()

    prefill = ServerPreset.tinyllama2()
    prefill.prefill_devices = ["none"]
    prefill.prefill_min_tokens = 1
    prefill.debug = True
    fd, prefill.log_path = tempfile.mkstemp(suffix=".log")
    os.close(fd)

    try:
        prefill.start()
        actual = prefill.make_request("POST", "/completion", data=request)
        prefill.stop()

        assert actual.status_code == 200
        assert actual.body["content"] == expected.body["content"]
        assert actual.body["tokens"] == expected.body["tokens"]
        assert actual.body["timings"]["prompt_n"] == expected.body["timings"]["prompt_n"]
        assert actual.body["timings"]["cache_n"] == 0

        with open(prefill.log_path) as log:
            assert "__TEST_TAG_PREFILL_RESTORED__" in log.read()
    finally:
        prefill.stop()
        os.remove(prefill.log_path)
