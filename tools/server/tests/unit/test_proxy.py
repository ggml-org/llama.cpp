import pytest
from utils import *
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

server = ServerPreset.tinyllama2()


@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.tinyllama2()


def test_mcp_no_proxy():
    global server
    server.ui_mcp_proxy = False
    server.start()

    res = server.make_request("GET", "/cors-proxy")
    assert res.status_code == 403


def test_mcp_proxy():
    global server
    server.ui_mcp_proxy = True
    server.start()

    url = f"http://{server.server_host}:{server.server_port}/cors-proxy?url=http://example.com"
    res = requests.get(url)
    assert res.status_code == 200
    assert "Example Domain" in res.text


def test_mcp_proxy_custom_port():
    global server
    server.ui_mcp_proxy = True
    server.start()

    # try getting the server's models API via the proxy
    res = server.make_request("GET", f"/cors-proxy?url=http://{server.server_host}:{server.server_port}/models")
    assert res.status_code == 200
    assert "data" in res.body


def test_mcp_proxy_no_content():
    # note: see issue #26598
    class NoContentHandler(BaseHTTPRequestHandler):
        def do_POST(self):
            self.send_response(204)
            self.end_headers()

        def log_message(self, format, *args):
            pass

    target = ThreadingHTTPServer(("127.0.0.1", 0), NoContentHandler)
    target_thread = threading.Thread(target=target.serve_forever, daemon=True)
    target_thread.start()

    try:
        global server
        server.ui_mcp_proxy = True
        server.start()

        res = server.make_request("POST", f"/cors-proxy?url=http://127.0.0.1:{target.server_port}/", data={})
        assert res.status_code == 204
        assert res.body in (None, b"", "")
    finally:
        target.shutdown()
        target.server_close()

