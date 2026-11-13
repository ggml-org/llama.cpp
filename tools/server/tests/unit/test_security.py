import pytest
from openai import OpenAI
from utils import *
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

server = ServerPreset.tinyllama2()

TEST_API_KEY = "sk-this-is-the-secret-key"

@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.tinyllama2()
    server.api_key = TEST_API_KEY


@pytest.mark.parametrize("endpoint", ["/health"])
def test_access_public_endpoint(endpoint: str):
    global server
    server.start()
    res = server.make_request("GET", endpoint)
    assert res.status_code == 200
    assert "error" not in res.body


def test_access_static_assets_without_api_key():
    """Static web UI assets should not require API key authentication (issue #21229)"""
    global server
    server.start()
    for path in ["/", "/sw.js", "/manifest.webmanifest", "/_app/version.json"]:
        res = server.make_request("GET", path)
        assert res.status_code == 200, f"Expected 200 for {path}, got {res.status_code}"


@pytest.mark.parametrize("api_key", [None, "invalid-key"])
def test_incorrect_api_key(api_key: str):
    global server
    server.start()
    res = server.make_request("POST", "/completions", data={
        "prompt": "I believe the meaning of life is",
    }, headers={
        "Authorization": f"Bearer {api_key}" if api_key else None,
    })
    assert res.status_code == 401
    assert "error" in res.body
    assert res.body["error"]["type"] == "authentication_error"


def test_correct_api_key():
    global server
    server.start()
    res = server.make_request("POST", "/completions", data={
        "prompt": "I believe the meaning of life is",
    }, headers={
        "Authorization": f"Bearer {TEST_API_KEY}",
    })
    assert res.status_code == 200
    assert "error" not in res.body
    assert "content" in res.body


def test_correct_api_key_anthropic_header():
    global server
    server.start()
    res = server.make_request("POST", "/completions", data={
        "prompt": "I believe the meaning of life is",
    }, headers={
        "X-Api-Key": TEST_API_KEY,
    })
    assert res.status_code == 200
    assert "error" not in res.body
    assert "content" in res.body


def test_openai_library_correct_api_key():
    global server
    server.start()
    client = OpenAI(api_key=TEST_API_KEY, base_url=f"http://{server.server_host}:{server.server_port}")
    res = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a chatbot."},
            {"role": "user", "content": "What is the meaning of life?"},
        ],
    )
    assert len(res.choices) == 1


@pytest.mark.parametrize("origin,cors_header,cors_header_value", [
    ("localhost", "Access-Control-Allow-Origin", "localhost"),
    ("web.mydomain.fr", "Access-Control-Allow-Origin", "web.mydomain.fr"),
    ("origin", "Access-Control-Allow-Credentials", "true"),
    ("web.mydomain.fr", "Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS"),
    ("web.mydomain.fr", "Access-Control-Allow-Headers", "*"),
])
def test_cors_options(origin: str, cors_header: str, cors_header_value: str):
    global server
    server.start()
    res = server.make_request("OPTIONS", "/completions", headers={
        "Origin": origin,
        "Access-Control-Request-Method": "POST",
        "Access-Control-Request-Headers": "Authorization",
    })
    assert res.status_code == 200
    assert cors_header in res.headers
    assert res.headers[cors_header] == cors_header_value


@pytest.mark.parametrize("origin", [
    "http://localhost",
    "http://localhost:8080",
    "http://127.0.0.1:3000",
    "http://[::1]:3000",
    "http://evil.com",
    "http://localhost.evil.com",
])
def test_cors_origins_localhost_no_cross_origin_header(origin: str):
    global server
    server = ServerPreset.router()
    server.cors_origins = "localhost"
    server.start()
    res = server.make_request("OPTIONS", "/completions", headers={
        "Origin": origin,
        "Access-Control-Request-Method": "POST",
        "Access-Control-Request-Headers": "Authorization",
    })
    assert res.status_code == 200
    assert "Access-Control-Allow-Origin" not in res.headers


@pytest.mark.parametrize("feature", ["server_tools", "ui_mcp_proxy"])
def test_cors_origins_defaults_to_localhost(feature: str):
    global server
    server = ServerPreset.router()
    if feature == "server_tools":
        server.server_tools = "all"
    else:
        server.ui_mcp_proxy = True
    server.start()
    res = server.make_request("GET", "/props", headers={"Origin": f"http://127.0.0.1:{server.server_port}"})
    assert res.status_code == 200
    res = server.make_request("GET", "/props", headers={"Origin": "http://localhost:5173"})
    assert res.status_code == 403


def _raw_post(path: str, body: str, headers: dict) -> requests.Response:
    # text/plain: "simple" request, no CORS preflight
    return requests.post(server.make_url(path), data=body, headers={"Content-Type": "text/plain", **headers})


@pytest.mark.parametrize("headers,expected_status", [
    ({}, 200),                                                    # non-browser client
    ({"Origin": "http://127.0.0.1:{port}"}, 200),                 # same-origin
    ({"Origin": "http://localhost:3000"}, 403),                   # other localhost app
    ({"Origin": "http://evil.com"}, 403),                         # cross-origin simple request
    ({"Origin": "http://localhost.evil.com"}, 403),
    ({"Origin": "null"}, 403),                                    # sandboxed iframe / file://
    ({"Sec-Fetch-Site": "cross-site", "Sec-Fetch-Mode": "no-cors"}, 403),
])
def test_cors_origins_localhost_blocks_simple_requests(headers: dict, expected_status: int):
    global server
    server.api_key = None
    server.cors_origins = "localhost"
    server.start()
    headers = {k: v.format(port=server.server_port) for k, v in headers.items()}
    res = _raw_post("/tokenize", '{"content": "hello"}', headers)
    assert res.status_code == expected_status


def test_cors_origins_localhost_allows_same_origin():
    global server
    server.api_key = None
    server.cors_origins = "localhost"
    server.server_host = "0.0.0.0"
    server.start()
    lan_ip = get_lan_ip(server)
    if lan_ip is None:
        pytest.skip("no LAN address reachable from this host")  # ty: ignore[too-many-positional-arguments]
    # UI opened via LAN address: Origin matches Host
    url = server.make_url("/tokenize", host=lan_ip)
    res = requests.post(url, data='{"content": "hello"}', timeout=DEFAULT_REQUEST_TIMEOUT, headers={
        "Content-Type": "text/plain",
        "Origin": f"http://{lan_ip}:{server.server_port}",
    })
    assert res.status_code == 200


@pytest.mark.parametrize("host,headers,expected_status", [
    ("localhost",       {}, 200),
    ("127.0.0.1",       {}, 200),
    ("127.0.0.2",       {}, 200),
    ("[::1]",           {}, 200),
    ("0.0.0.0",         {}, 200),
    ("evil.com",        {"Origin": "http://evil.com:{port}"}, 403),     # rebinding, Origin matches Host
    ("evil.com",        {"Sec-Fetch-Site": "same-origin"}, 403),        # rebinding, same-origin GET has no Origin
    ("127.0.0.1.evil.com", {}, 403),
    ("localhost.evil.com", {}, 403),
])
def test_cors_origins_localhost_blocks_dns_rebinding(host: str, headers: dict, expected_status: int):
    global server
    server.api_key = None
    server.cors_origins = "localhost"
    server.start()
    port = server.server_port
    headers = {k: v.format(port=port) for k, v in headers.items()}
    res = _raw_post("/tokenize", '{"content": "hello"}', {"Host": f"{host}:{port}", **headers})
    assert res.status_code == expected_status


def test_cors_origins_explicit_list_allows_its_hosts():
    global server
    server.api_key = None
    server.cors_origins = "https://ia.example.com"
    server.start()
    # reverse proxy that keeps the Host of the public name
    for host, expected_status in [
        ("ia.example.com", 200),
        ("evil.com", 403),
    ]:
        res = _raw_post("/tokenize", '{"content": "hello"}', {"Host": host, "Origin": f"https://{host}"})
        assert res.status_code == expected_status, host


def test_cors_origins_explicit_list_blocks_simple_requests():
    global server
    server.api_key = None
    server.cors_origins = "https://a.example, https://b.example"
    server.start()
    for origin, expected_status in [
        ("https://a.example", 200),
        ("https://b.example", 200),
        ("https://c.example", 403),
    ]:
        res = _raw_post("/tokenize", '{"content": "hello"}', {"Origin": origin})
        assert res.status_code == expected_status, origin


def test_cors_origins_wildcard_keeps_allowing_all():
    global server
    server.api_key = None
    server.cors_origins = "*"
    server.start()
    res = _raw_post("/tokenize", '{"content": "hello"}', {"Origin": "http://evil.com"})
    assert res.status_code == 200


def test_cors_cross_site_navigation_to_ui_allowed():
    global server
    server.api_key = None
    server.cors_origins = "localhost"
    server.start()
    nav = {"Sec-Fetch-Site": "cross-site", "Sec-Fetch-Mode": "navigate"}
    # UI may not be built in tests, only check it is not refused
    assert requests.get(server.make_url("/"), headers=nav).status_code != 403
    assert requests.get(server.make_url("/props"), headers=nav).status_code == 403


def test_tools_drive_by_exec_is_blocked():
    global server
    server.api_key = None
    server.server_tools = "exec_shell_command" # implies --cors-origins localhost
    server.start()
    body = '{"tool": "exec_shell_command", "params": {"command": "echo csrf-guard-test"}}'
    res = _raw_post("/tools", body, {"Origin": "http://evil.com"})
    assert res.status_code == 403
    assert "csrf-guard-test" not in res.text

    res = _raw_post("/tools", body, {"Origin": "http://localhost:5173"})
    assert res.status_code == 403

    res = _raw_post("/tools", body, {"Origin": f"http://127.0.0.1:{server.server_port}"})
    assert res.status_code == 200
    assert "csrf-guard-test" in res.text


@pytest.mark.parametrize("server_host,api_key,allow_unauth,should_start", [
    ("127.0.0.1", None, False, True),
    ("0.0.0.0", None, False, False),
    ("0.0.0.0", TEST_API_KEY, False, True),
    ("0.0.0.0", None, True, True),
])
def test_tools_refuse_unauthenticated_network_bind(server_host: str, api_key: str | None, allow_unauth: bool, should_start: bool):
    global server
    server = ServerPreset.router()
    server.server_tools = "all"
    server.server_host = server_host
    server.api_key = api_key
    server.allow_unauthenticated_tools = allow_unauth
    if should_start:
        server.start()
    else:
        with pytest.raises(RuntimeError):
            server.start()


def test_cors_proxy_only_forwards_explicit_proxy_headers():
    class CaptureHeadersHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.server.captured_headers = dict(self.headers)
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"ok")

        def log_message(self, format, *args):
            pass

    target = ThreadingHTTPServer(("127.0.0.1", 0), CaptureHeadersHandler)
    target.captured_headers = {}
    target_thread = threading.Thread(target=target.serve_forever, daemon=True)
    target_thread.start()

    try:
        server = ServerPreset.tinyllama2()
        server.api_key = TEST_API_KEY
        server.ui_mcp_proxy = True
        server.debug_proxy_allow_private = True
        server.start()

        res = server.make_request("GET", f"/cors-proxy?url=http://127.0.0.1:{target.server_port}/capture", headers={
            "Authorization": f"Bearer {TEST_API_KEY}",
            "Proxy-Authorization": "Basic secret",
            "X-Api-Key": TEST_API_KEY,
            "Cookie": "session=secret",
            "x-llama-server-proxy-header-accept": "application/json",
            "x-llama-server-proxy-header-authorization": "Bearer explicit",
        })

        assert res.status_code == 200
        captured = {key.lower(): value for key, value in target.captured_headers.items()}
        assert captured["accept"] == "application/json"
        assert captured["authorization"] == "Bearer explicit"
        assert "proxy-authorization" not in captured
        assert "x-api-key" not in captured
        assert "cookie" not in captured
    finally:
        target.shutdown()
        target.server_close()


@pytest.mark.parametrize(
    "media_path, image_url, success",
    [
        (None,             "file://mtmd/test-1.jpeg",    False), # disabled media path, should fail
        ("../../../tools", "file://mtmd/test-1.jpeg",    True),
        ("../../../tools", "file:////mtmd//test-1.jpeg", True),  # should be the same file as above
        ("../../../tools", "file://mtmd/notfound.jpeg",  False), # non-existent file
        ("../../../tools", "file://../mtmd/test-1.jpeg", False), # no directory traversal
    ]
)
def test_local_media_file(media_path, image_url, success,):
    server = ServerPreset.tinygemma3()
    server.media_path = media_path
    server.start()
    res = server.make_request("POST", "/chat/completions", data={
        "max_tokens": 1,
        "messages": [
            {"role": "user", "content": [
                {"type": "text", "text": "test"},
                {"type": "image_url", "image_url": {
                    "url": image_url,
                }},
            ]},
        ],
    })
    if success:
        assert res.status_code == 200
    else:
        assert res.status_code == 400
