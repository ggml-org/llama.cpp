import pytest
from utils import *

server: ServerProcess

# a 3-byte lead followed by 6000 4-byte emojis (24004 bytes total): the
# server's 4096-byte pipe reads inside run_subprocess then land mid-sequence
# and exercise the UTF-8 carry. \U escapes keep the command line pure ASCII.
LEAD_3B = "\U0001f604"   # smiling face
EMOJI_4B = "\U0001f406"  # leopard
N_EMOJI = 6000

@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.router()
    server.server_tools = "all"

def test_tool_stream_utf8_emoji_across_chunk_boundaries():
    """A tool stream carrying 24 KB of emoji must arrive with every chunk valid
    UTF-8 and the full sequence intact (no U+FFFD substitution). The 4096-byte
    pipe reads land mid-emoji; the carry in run_subprocess must rejoin every
    split sequence. A regression here used to surface as either silent U+FFFD
    corruption (safe_json_to_str masking) or a server crash (unhandled dump()
    throw on the stream path)."""
    global server
    server.start()

    expected_bytes = (3 + 4 * N_EMOJI)

    # pure-ASCII command line: the embedded python source uses \U escapes, so
    # no shell quoting of raw multi-byte characters is involved
    cmd = (
        "python3 -c \"import sys; sys.stdout.write("
        f"'\\U0001f604' + '\\U0001f406' * {N_EMOJI}"
        ")\""
    )

    events = list(server.make_stream_request("POST", "/tools", data={
        "tool": "exec_shell_command",
        "params": {"command": cmd},
        "stream": True,
    }))

    assert len(events) >= 2, "expected chunk events plus a done event"
    last = events[-1]
    assert last.get("done") is True, last
    assert not last.get("error"), last

    chunks = [e["chunk"] for e in events[:-1]]

    # every individual chunk must be self-contained valid UTF-8 with no
    # substitution: the carry is what guarantees this at the source
    for i, c in enumerate(chunks):
        assert "\ufffd" not in c, f"chunk {i} contains U+FFFD (masking fired)"
        c.encode("utf-8")  # raises if the chunk is not valid UTF-8

    joined = "".join(chunks)

    # the stream must carry the full payload (plus the "[exit code: 0]"
    # trailer the tool appends): nothing lost, nothing substituted
    total = len(joined.encode("utf-8"))
    assert total >= expected_bytes, (
        f"expected at least {expected_bytes} payload bytes, got {total} total"
    )
    assert joined.count(EMOJI_4B) == N_EMOJI, (
        f"expected {N_EMOJI} leopard emojis, counted {joined.count(EMOJI_4B)}"
    )
    assert joined.count(LEAD_3B) == 1
