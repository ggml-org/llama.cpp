import base64
import json
import os
import socket
import struct
import pytest
from utils import *

server: ServerProcess


def ws_connect(host: str, port: int, path: str):
    sock = socket.create_connection((host, port), timeout=60)
    key = base64.b64encode(os.urandom(16)).decode()
    sock.sendall((f"GET {path} HTTP/1.1\r\nHost: {host}:{port}\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n"
                  f"Sec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\n\r\n").encode())
    stream = sock.makefile("rb")
    status = stream.readline()
    assert b" 101 " in status, status
    while stream.readline().strip():
        pass
    return sock, stream


def ws_send(sock, event: dict):
    payload = json.dumps(event).encode()
    n = len(payload)
    header = bytes([0x81]) + (bytes([0x80 | n]) if n < 126 else bytes([0x80 | 126]) + struct.pack(">H", n))
    mask = os.urandom(4)
    sock.sendall(header + mask + bytes(b ^ mask[i % 4] for i, b in enumerate(payload)))


def ws_recv(stream) -> dict:
    header = stream.read(2)
    assert len(header) == 2, "connection closed by the server"
    b0, b1 = header
    assert b0 & 0x0F == 1, f"expected a text frame, got opcode {b0 & 0x0F}"
    n = b1 & 0x7F
    if n == 126:
        n = struct.unpack(">H", stream.read(2))[0]
    elif n == 127:
        n = struct.unpack(">Q", stream.read(8))[0]
    return json.loads(stream.read(n))


@pytest.mark.slow
def test_realtime_bad_tool_schema_is_an_error_event():
    global server
    server = ServerProcess()
    server.model_hf_repo = "LiquidAI/LFM3-Omni-2B-GGUF:Q4_0"
    server.model_hf_file = None
    server.n_ctx = 4096
    server.n_slots = 1
    server.jinja = True
    server.start(timeout_seconds=600)  # large model needs time to download

    sock, stream = ws_connect(server.server_host, server.server_port, "/v1/realtime")
    assert ws_recv(stream)["type"] == "session.created"

    def update_tools(parameters):
        tool = {"type": "function", "name": "get_weather", "parameters": parameters}
        ws_send(sock, {"type": "session.update", "session": {"type": "realtime", "tools": [tool]}})
        return ws_recv(stream)

    # a schema the chat template cannot turn into a grammar must not take the server down
    for parameters in ["not an object", [], {"type": "nonsense"}]:
        ev = update_tools(parameters)
        assert ev["type"] == "error"
        assert server.make_request("GET", "/health").status_code == 200

    # the session is still usable
    ev = update_tools({"type": "object", "properties": {"city": {"type": "string"}}})
    assert ev["type"] == "session.updated"
    sock.close()
