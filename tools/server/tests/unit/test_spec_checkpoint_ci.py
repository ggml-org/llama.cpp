"""
CI-compatible speculative checkpoint tests using stories15M models.

These tests exercise the checkpoint code path on a pure transformer model.
They validate infrastructure correctness but cannot trigger hybrid-model-specific
bugs (KV leak, sampler state) because stories15M is not a hybrid model and
the standard KV guard disables checkpoints for it.

Note: With the standard KV guard (Fix 4), checkpoints are silently disabled
for stories15M. These tests verify that the non-checkpoint fallback path
(memory_seq_rm + rewind) remains correct with checkpoint CLI flags present.
"""

import pytest
from utils import ServerPreset


@pytest.fixture(scope="module", autouse=True)
def do_something():
    """Override conftest's load_all — we download models on demand."""
    pass


def test_spec_decoding_with_checkpoint_flags():
    """Spec decoding produces deterministic output with checkpoint flags on stories15M.

    The standard KV guard disables checkpoints for this pure transformer model,
    so this tests the fallback path (memory_seq_rm + rewind) with checkpoint
    CLI flags present. 10 identical requests must produce identical output.
    """
    server = ServerPreset.stories15m_moe()
    server.offline = False
    server.draft_min = 4
    server.draft_max = 8
    server.extra_args = [
        "--spec-type", "ngram-mod",
        "--spec-use-checkpoints", "on",
        "--ctx-checkpoints", "4",
    ]
    server.start()

    outputs = []
    for i in range(10):
        res = server.make_request("POST", "/completion", data={
            "prompt": "Once upon a time there was a little girl",
            "temperature": 0.0,
            "top_k": 1,
            "n_predict": 128,
        })
        assert res.status_code == 200, f"Request {i+1} failed: {res.status_code}"
        outputs.append(res.body["content"])

    assert len(set(outputs)) == 1, (
        f"Output divergence: {len(set(outputs))} unique outputs across "
        f"10 identical requests (stories15M, checkpoint flags)."
    )
