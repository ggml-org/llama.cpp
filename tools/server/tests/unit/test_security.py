import pytest
from openai import OpenAI
from utils import *
import threading
import tempfile
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
    "http://127.0.0.1",
    "http://127.0.0.1:3000",
    "http://[::1]",
    "http://[::1]:3000",
])
def test_cors_origins_localhost_reflects(origin: str):
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
    assert res.headers["Access-Control-Allow-Origin"] == origin


@pytest.mark.parametrize("origin", [
    "http://web.mydomain.fr",
    "http://evil.com",
    "http://notlocalhost",
    "http://localhost.evil.com",
])
def test_cors_origins_localhost_rejects(origin: str):
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


def test_cors_origins_defaults_to_localhost_with_tools_enabled():
    global server
    server = ServerPreset.router()
    server.server_tools = "all"
    server.start()
    res = server.make_request("OPTIONS", "/completions", headers={
        "Origin": "http://localhost:8080",
        "Access-Control-Request-Method": "POST",
        "Access-Control-Request-Headers": "Authorization",
    })
    assert res.status_code == 200
    assert res.headers["Access-Control-Allow-Origin"] == "http://localhost:8080"

    res = server.make_request("OPTIONS", "/completions", headers={
        "Origin": "http://evil.com",
        "Access-Control-Request-Method": "POST",
        "Access-Control-Request-Headers": "Authorization",
    })
    assert res.status_code == 200
    assert "Access-Control-Allow-Origin" not in res.headers


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


# --- remote preset key allowlist (ref: issue #27857) ---------------------------
# A preset.ini fetched from a remote repo is untrusted input: it is parsed and
# rendered into the argv of a child llama-server, so it must only be able to set
# keys on the allowlist. These tests drive the real -hf flow against a local stub
# of the HF API, so no network and no remote repository are involved.

REMOTE_PRESET_REPO = "test-user/preset-repo"
REMOTE_PRESET_COMMIT = "0" * 40


def _hf_stub(preset_ini: bytes):
    repo = REMOTE_PRESET_REPO
    commit = REMOTE_PRESET_COMMIT

    class HfStubHandler(BaseHTTPRequestHandler):
        def _body_for(self, path):
            if path == f"/api/models/{repo}/refs":
                return json.dumps({"branches": [{"name": "main", "targetCommit": commit}]}).encode()
            if path == f"/api/models/{repo}/tree/{commit}":
                return json.dumps([{"type": "file", "path": "preset.ini", "size": len(preset_ini)}]).encode()
            if path == f"/{repo}/resolve/{commit}/preset.ini":
                return preset_ini
            return None

        def _respond(self, body, with_body):
            if body is None:
                self.send_response(404)
                self.end_headers()
                return
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Accept-Ranges", "bytes")
            self.end_headers()
            if with_body:
                self.wfile.write(body)

        def do_HEAD(self):
            self._respond(self._body_for(self.path.split("?")[0]), False)

        def do_GET(self):
            self._respond(self._body_for(self.path.split("?")[0]), True)

        def log_message(self, format, *args):
            pass

    hf = ThreadingHTTPServer(("127.0.0.1", 0), HfStubHandler)
    threading.Thread(target=hf.serve_forever, daemon=True).start()
    return hf


def _run_server_with_remote_preset(preset_ini: bytes, port: int, wait_for_start: bool):
    """Start llama-server with -hf pointed at the stub. Returns (started, output, models_json)."""
    hf = _hf_stub(preset_ini)
    with tempfile.TemporaryDirectory() as cache_dir:
        env = {
            **os.environ,
            "MODEL_ENDPOINT": f"http://127.0.0.1:{hf.server_port}/",
            "LLAMA_CACHE": cache_dir,
        }
        cmd = [
            "../../../build/bin/llama-server",
            "-hf", REMOTE_PRESET_REPO,
            "--host", "127.0.0.1",
            "--port", str(port),
        ]
        if not wait_for_start:
            try:
                proc = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=30)
                return False, proc.stdout + proc.stderr, None
            except subprocess.TimeoutExpired as e:
                out = (e.stdout or b"").decode(errors="replace") + (e.stderr or b"").decode(errors="replace")
                return True, out, None
            finally:
                hf.shutdown()

        proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        models = None
        try:
            for _ in range(60):
                if proc.poll() is not None:
                    break
                try:
                    r = requests.get(f"http://127.0.0.1:{port}/v1/models", timeout=2)
                    if r.status_code == 200:
                        models = r.json()
                        break
                except Exception:
                    pass
                time.sleep(1)
        finally:
            proc.terminate()
            try:
                out = proc.communicate(timeout=15)[0]
            except Exception:
                proc.kill()
                out = proc.communicate()[0]
            hf.shutdown()
        return models is not None, out, models


# every key below is a real registered llama-server option that the UNFILTERED
# local preset path accepts; each is outside the remote allowlist, so on the
# remote path it must be rejected by the allowlist specifically
@pytest.mark.parametrize("key, value", [
    ("mcp-servers-json",   '{"mcpServers":{}}'),
    ("mcp-servers-config", "/dev/null"),
    ("webui-mcp-proxy",    "false"),
    ("host",               "127.0.0.1"),
    ("port",               "8099"),
])
def test_remote_preset_rejects_non_allowlisted_key(key, value):
    preset_ini = f"version = 1\n\n[*]\n{key} = {value}\n".encode()
    started, output, _ = _run_server_with_remote_preset(preset_ini, 8081, wait_for_start=False)
    assert not started, (
        f"server started with a remote preset that sets '{key}'; "
        "the remote-preset allowlist did not reject it"
    )
    assert f"option '{key}' is not allowed in remote presets" in output, output[-2000:]


def test_remote_preset_accepts_allowlisted_key():
    # batch-size is on the remote allowlist: the preset must load and take effect,
    # proving the allowlist filters keys rather than rejecting remote presets wholesale
    preset_ini = (
        b"version = 1\n\n[remote-model]\n"
        b"batch-size = 128\n"
    )
    started, output, models = _run_server_with_remote_preset(preset_ini, 8082, wait_for_start=True)
    assert started, f"server did not start with an allowlisted remote preset key: {output[-2000:]}"
    assert models is not None and models["data"], f"no models listed: {output[-2000:]}"
    args = models["data"][0]["status"]["args"]
    assert "--batch-size" in args, args
    assert args[args.index("--batch-size") + 1] == "128", args


def test_local_preset_is_not_filtered():
    # the local path is operator-authored and must stay unfiltered
    with tempfile.TemporaryDirectory() as d:
        ini = os.path.join(d, "preset.ini")
        with open(ini, "w") as f:
            f.write('version = 1\n\n[local-model]\nmcp-servers-json = {"mcpServers":{}}\n')
        server = ServerPreset.tinyllama2()
        server.api_key = None
        server.model_hf_repo = None
        server.model_hf_file = None
        server.model_file = None
        server.models_preset = ini
        server.start()
        try:
            res = server.make_request("GET", "/v1/models")
            assert res.status_code == 200
            args = res.body["data"][0]["status"]["args"]
            assert "--mcp-servers-json" in args, args
        finally:
            server.stop()
