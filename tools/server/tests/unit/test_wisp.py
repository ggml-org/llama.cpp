import socket
import struct
import threading
import pytest
from websockets.exceptions import InvalidStatus
from websockets.sync.client import connect, ClientConnection
from websockets.sync.connection import Connection
from utils import *

server = ServerPreset.tinyllama2()

WISP_CONNECT = 0x01
WISP_DATA = 0x02
WISP_CONTINUE = 0x03
WISP_CLOSE = 0x04
WISP_CLOSE_HOST_BLOCKED = 0x48


@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.tinyllama2()


def wisp_connect(query: str = "") -> ClientConnection:
    ws = connect(f"ws://{server.server_host}:{server.server_port}/wisp/{query}")
    packet_type, stream_id, payload = recv(ws)
    assert packet_type == WISP_CONTINUE
    assert stream_id == 0
    assert struct.unpack("<I", payload)[0] > 0
    return ws


def send(ws: Connection, packet_type: int, stream_id: int, payload: bytes):
    ws.send(struct.pack("<BI", packet_type, stream_id) + payload)


def recv(ws: Connection) -> tuple[int, int, bytes]:
    msg = ws.recv(timeout=10)
    assert isinstance(msg, bytes)
    packet_type, stream_id = struct.unpack("<BI", msg[:5])
    return packet_type, stream_id, msg[5:]


def open_stream(ws: Connection, stream_id: int, host: str, port: int):
    send(ws, WISP_CONNECT, stream_id, struct.pack("<BH", 0x01, port) + host.encode())


def start_echo_server() -> int:
    listener = socket.create_server(("127.0.0.1", 0))

    def serve():
        conn, _ = listener.accept()
        with conn:
            while data := conn.recv(4096):
                conn.sendall(data)
        listener.close()

    threading.Thread(target=serve, daemon=True).start()
    return listener.getsockname()[1]


def test_wisp_disabled():
    global server
    server.start()
    with pytest.raises(InvalidStatus):
        connect(f"ws://{server.server_host}:{server.server_port}/wisp/")


def test_wisp_echo():
    global server
    server.ui_linux_net = True
    server.ui_linux_net_allow_private_urls = True
    server.start()
    port = start_echo_server()
    with wisp_connect() as ws:
        open_stream(ws, 1, "127.0.0.1", port)
        send(ws, WISP_DATA, 1, b"hello wisp")
        received = b""
        while len(received) < len(b"hello wisp"):
            packet_type, stream_id, payload = recv(ws)
            assert packet_type == WISP_DATA
            assert stream_id == 1
            received += payload
        assert received == b"hello wisp"
        send(ws, WISP_CLOSE, 1, bytes([0x02]))


def test_wisp_private_address_refused():
    global server
    server.ui_linux_net = True
    server.start()
    port = start_echo_server()
    with wisp_connect() as ws:
        open_stream(ws, 1, "127.0.0.1", port)
        packet_type, stream_id, payload = recv(ws)
        assert packet_type == WISP_CLOSE
        assert stream_id == 1
        assert payload[0] == WISP_CLOSE_HOST_BLOCKED


def test_wisp_api_key():
    global server
    server.ui_linux_net = True
    server.api_key = "llama-test-key"
    server.start()
    with pytest.raises(InvalidStatus):
        connect(f"ws://{server.server_host}:{server.server_port}/wisp/")
    with pytest.raises(InvalidStatus):
        connect(f"ws://{server.server_host}:{server.server_port}/wisp/?api_key=wrong")
    with wisp_connect("?api_key=llama-test-key"):
        pass
