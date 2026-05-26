import pytest
from utils import *


@pytest.fixture
def server(server_factory):
    return server_factory("tinyllama2")


def test_mcp_no_proxy(server):
    server.webui_mcp_proxy = False
    server.start()

    res = server.make_request("GET", "/cors-proxy")
    assert res.status_code == 404


def test_mcp_proxy(server):
    server.webui_mcp_proxy = True
    server.start()

    url = f"http://{server.server_host}:{server.server_port}/cors-proxy?url=http://example.com"
    res = requests.get(url)
    assert res.status_code == 200
    assert "Example Domain" in res.text


def test_mcp_proxy_custom_port(server):
    server.webui_mcp_proxy = True
    server.start()

    # try getting the server's models API via the proxy
    res = server.make_request("GET", f"/cors-proxy?url=http://{server.server_host}:{server.server_port}/models")
    assert res.status_code == 200
    assert "data" in res.body
