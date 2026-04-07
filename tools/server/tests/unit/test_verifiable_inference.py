import json
import os
import subprocess
import sys
import tempfile

from utils import *


server: ServerProcess


def test_verifiable_inference_oai_chat_completion():
    global server
    server = ServerPreset.tinyllama2()
    server.start()

    res = server.make_request("POST", "/v1/chat/completions", data={
        "messages": [{"role": "user", "content": "hello"}],
        "max_tokens": 8,
        "stream": False,
        "verifiable_inference": {
            "enabled": True,
            "samples": 1,
            "seed": 123,
            "max_proof_bytes": 2_000_000,
        },
    })

    assert res.status_code == 200
    assert isinstance(res.body, dict)
    assert "verifiable_inference" in res.body
    vi = res.body["verifiable_inference"]
    assert "merkle_root_sha256" in vi
    assert "proof_id" in vi
    assert len(vi.get("openings", [])) == 1
    assert "opened_bytes_url" in vi["openings"][0]

    # run reference verifier script
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(res.body, f)
        fname = f.name

    server_base = f"http://{server.server_host}:{server.server_port}"
    cp = subprocess.run(
        [sys.executable, "../vi_verify.py", "--file", fname, "--server-base", server_base],
        cwd=os.path.dirname(__file__),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert cp.returncode == 0, cp.stderr

