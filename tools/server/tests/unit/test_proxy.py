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
    server.debug_proxy_allow_private = True
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
        server.debug_proxy_allow_private = True
        server.start()

        res = server.make_request("POST", f"/cors-proxy?url=http://127.0.0.1:{target.server_port}/", data={})
        assert res.status_code == 204
        assert res.body in (None, b"", "")
    finally:
        target.shutdown()
        target.server_close()



@pytest.mark.parametrize("target", [
    "http://127.0.0.1:{port}/models",
    "http://localhost:{port}/models",
    "http://[::1]:{port}/models",
    "http://[::ffff:127.0.0.1]:{port}/models",
    "http://10.0.0.1/",
    "http://172.16.0.1/",
    "http://192.168.1.1/",
    "http://100.64.0.1/",
    "http://169.254.169.254/latest/meta-data/", # cloud metadata
    "http://0.0.0.0:{port}/models",
])
def test_mcp_proxy_refuses_non_public_targets(target: str):
    global server
    server.ui_mcp_proxy = True
    server.start()
    target = target.format(port=server.server_port)
    res = server.make_request("GET", f"/cors-proxy?url={target}")
    assert res.status_code == 400


def test_mcp_proxy_redirect_goes_through_proxy():
    class RedirectHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/start":
                self.send_response(302)
                self.send_header("Location", "/next")
                self.end_headers()
            else:
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b"final")

        def log_message(self, format, *args):
            pass

    target = ThreadingHTTPServer(("127.0.0.1", 0), RedirectHandler)
    target_thread = threading.Thread(target=target.serve_forever, daemon=True)
    target_thread.start()

    try:
        global server
        server.ui_mcp_proxy = True
        server.debug_proxy_allow_private = True
        server.start()

        url = server.make_url(f"/cors-proxy?url=http://127.0.0.1:{target.server_port}/start")
        # redirect points back to the proxy
        res = requests.get(url, allow_redirects=False)
        assert res.status_code == 302
        assert res.headers["Location"] == f"?url=http%3A%2F%2F127.0.0.1%3A{target.server_port}%2Fnext"
        res = requests.get(url)
        assert res.status_code == 200
        assert res.text == "final"
    finally:
        target.shutdown()
        target.server_close()
