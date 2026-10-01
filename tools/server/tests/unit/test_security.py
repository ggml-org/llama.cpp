from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import threading

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import threading

from openai import OpenAI
import pytest

from utils import ServerPreset

server = ServerPreset.tinyllama2()

TEST_API_KEY = "sk-this-is-the-secret-key"

@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.tinyllama2()
    server.api_key = TEST_API_KEY


@pytest.mark.parametrize("warn_unknown_env", [False, True])
@pytest.mark.parametrize("fallback_token", [None, ""])
def test_startup_warns_about_ignored_credential_environment_variables(monkeypatch, tmp_path, warn_unknown_env, fallback_token):
    global server
    ignored_api_key = "ignored-api-key-sentinel"
    ignored_hf_token = "ignored-hf-token-sentinel"
    monkeypatch.setenv("LLAMA_ARG_API_KEY", ignored_api_key)
    monkeypatch.setenv("HUGGINGFACE_HUB_TOKEN", ignored_hf_token)
    monkeypatch.delenv("LLAMA_API_KEY", raising=False)
    monkeypatch.delenv("LLAMA_ARG_API_KEY_FILE", raising=False)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("LLAMA_ARG_WARN_UNKNOWN_ENV", raising=False)
    if warn_unknown_env:
        monkeypatch.setenv("LLAMA_ARG_WARN_UNKNOWN_ENV", "1")
    if fallback_token is None:
        monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    else:
        monkeypatch.setenv("HUGGING_FACE_HUB_TOKEN", fallback_token)

    server = ServerPreset.router()
    log_path = tmp_path / "server.log"
    server.log_path = str(log_path)
    server.start()
    server.stop()
    startup_log = log_path.read_text()

    # Unknown LLAMA_ARG diagnostics are opt-in; the specialized message replaces the generic one.
    assert startup_log.count("LLAMA_ARG_API_KEY is ignored") == int(warn_unknown_env)
    assert ("LLAMA_API_KEY, LLAMA_ARG_API_KEY_FILE, --api-key, or --api-key-file" in startup_log) == warn_unknown_env
    assert "unknown environment variable: LLAMA_ARG_API_KEY" not in startup_log
    assert "HUGGINGFACE_HUB_TOKEN is ignored" in startup_log
    assert "HF_TOKEN or --hf-token" in startup_log
    assert ignored_api_key not in startup_log
    assert ignored_hf_token not in startup_log


@pytest.mark.parametrize("token_env", ["HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"])
def test_startup_does_not_warn_when_supported_credentials_are_configured(monkeypatch, tmp_path, token_env):
    global server
    credential_values = [
        "ignored-api-key-sentinel",
        "ignored-hf-token-sentinel",
        "supported-api-key-sentinel",
        "supported-hf-token-sentinel",
    ]
    monkeypatch.setenv("LLAMA_ARG_API_KEY", credential_values[0])
    monkeypatch.setenv("HUGGINGFACE_HUB_TOKEN", credential_values[1])
    monkeypatch.setenv("LLAMA_API_KEY", credential_values[2])
    monkeypatch.delenv("LLAMA_ARG_API_KEY_FILE", raising=False)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    monkeypatch.setenv(token_env, credential_values[3])
    monkeypatch.setenv("LLAMA_ARG_WARN_UNKNOWN_ENV", "1")

    server = ServerPreset.router()
    log_path = tmp_path / "server.log"
    server.log_path = str(log_path)
    server.start()
    server.stop()
    startup_log = log_path.read_text()

    assert "LLAMA_ARG_API_KEY is ignored" not in startup_log
    # A supported key suppresses credential advice, but the opt-in scan still reports the unknown name.
    assert startup_log.count("unknown environment variable: LLAMA_ARG_API_KEY") == 1
    assert "HUGGINGFACE_HUB_TOKEN is ignored" not in startup_log
    for credential_value in credential_values:
        assert credential_value not in startup_log


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


@pytest.mark.parametrize("api_key", [
    None,
    TEST_API_KEY[:-1] + "x",  # unequal, same length
    TEST_API_KEY[:-1],        # prefix
    TEST_API_KEY[1:],         # suffix
    "shorter",
    TEST_API_KEY + "-longer",
])
def test_incorrect_api_key(api_key: str | None):
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

def test_api_key_file_trims_entries_and_ignores_comments(tmp_path):
    global server
    api_key_file = tmp_path / "api-keys.txt"
    api_key_file.write_bytes(
        b"# comment\r\n"
        b"  \r\n"
        b"  file-secret-key  \r\n"
        b"\t# indented comment"
    )
    server.api_key = None
    server.api_key_file = str(api_key_file)
    server.start()

    valid = server.make_request("POST", "/completions", data={
        "prompt": "I believe the meaning of life is",
    }, headers={
        "Authorization": "Bearer file-secret-key",
    })
    assert valid.status_code == 200

    for invalid_key in ["  file-secret-key  ", "# comment", "\t# indented comment"]:
        invalid = server.make_request("POST", "/completions", data={
            "prompt": "I believe the meaning of life is",
        }, headers={
            "Authorization": f"Bearer {invalid_key}",
        })
        assert invalid.status_code == 401


def test_cli_api_key_overrides_environment_without_dropping_key_file(monkeypatch, tmp_path):
    global server
    api_key_file = tmp_path / "api-keys.txt"
    api_key_file.write_text("file-secret-key\n")
    monkeypatch.setenv("LLAMA_API_KEY", "stale-env-key")
    monkeypatch.setenv("LLAMA_ARG_API_KEY_FILE", str(api_key_file))
    server.start()

    for valid_key in [TEST_API_KEY, "file-secret-key"]:
        valid = server.make_request("POST", "/completions", data={
            "prompt": "I believe the meaning of life is",
        }, headers={
            "Authorization": f"Bearer {valid_key}",
        })
        assert valid.status_code == 200

    stale = server.make_request("POST", "/completions", data={
        "prompt": "I believe the meaning of life is",
    }, headers={
        "Authorization": "Bearer stale-env-key",
    })
    assert stale.status_code == 401


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


def test_cors_default_wildcard_preflight_is_non_credentialed():
    global server
    server.start()
    res = server.make_request("OPTIONS", "/completions", headers={
        "Origin": "https://arbitrary.example",
        "Access-Control-Request-Method": "POST",
        "Access-Control-Request-Headers": "Authorization",
    })

    assert res.status_code == 200
    assert res.headers["Access-Control-Allow-Origin"] == "*"
    assert res.headers["Access-Control-Allow-Credentials"] == "false"
    assert res.headers["Access-Control-Allow-Methods"] == "GET, POST, DELETE, OPTIONS"
    assert res.headers["Access-Control-Allow-Headers"] == "*"


@pytest.mark.parametrize("origin", [
    "https://one.example",
    "https://two.example",
])
def test_cors_wildcard_with_credentials_is_normalized(origin: str):
    global server
    server.cors_origins = "*"
    server.cors_credentials = True
    server.start()
    res = server.make_request("OPTIONS", "/completions", headers={
        "Origin": origin,
        "Access-Control-Request-Method": "POST",
    })

    assert res.status_code == 200
    assert res.headers["Access-Control-Allow-Origin"] == "*"
    assert res.headers["Access-Control-Allow-Credentials"] == "false"


def test_cors_wildcard_credentials_logs_normalization_warning(tmp_path):
    global server
    log_path = tmp_path / "server.log"
    server.cors_origins = "*"
    server.cors_credentials = True
    server.log_path = str(log_path)
    server.start()
    server.stop()

    assert log_path.read_text().count("CORS credentials are disabled") == 1


def test_cors_wildcard_get_never_reflects_origin_or_allows_credentials():
    global server
    server.start()
    res = server.make_request("GET", "/health", headers={
        "Origin": "null",
    })

    assert res.status_code == 200
    assert res.headers["Access-Control-Allow-Origin"] == "*"
    assert "Access-Control-Allow-Credentials" not in res.headers


def test_cors_explicit_origin_retains_credentials():
    global server
    server.cors_origins = "https://trusted.example"
    server.cors_credentials = True
    server.start()
    res = server.make_request("OPTIONS", "/completions", headers={
        "Origin": "https://trusted.example",
        "Access-Control-Request-Method": "POST",
    })

    assert res.status_code == 200
    assert res.headers["Access-Control-Allow-Origin"] == "https://trusted.example"
    assert res.headers["Access-Control-Allow-Credentials"] == "true"

    actual = server.make_request("GET", "/health", headers={
        "Origin": "https://trusted.example",
    })
    assert actual.status_code == 200
    assert actual.headers["Access-Control-Allow-Origin"] == "https://trusted.example"
    assert actual.headers["Access-Control-Allow-Credentials"] == "true"


@pytest.mark.parametrize("origin", ["https://trusted.example.evil", "https://evil.example"])
def test_cors_explicit_origin_rejects_near_matches(origin: str):
    global server
    server.cors_origins = "https://trusted.example"
    server.cors_credentials = True
    server.start()

    preflight = server.make_request("OPTIONS", "/completions", headers={
        "Origin": origin,
        "Access-Control-Request-Method": "POST",
    })
    assert preflight.status_code == 200
    assert "Access-Control-Allow-Origin" not in preflight.headers
    assert preflight.headers["Access-Control-Allow-Credentials"] == "false"

    actual = server.make_request("GET", "/health", headers={"Origin": origin})
    assert actual.status_code == 200
    assert "Access-Control-Allow-Origin" not in actual.headers
    assert "Access-Control-Allow-Credentials" not in actual.headers


def test_cors_explicit_origin_list_reflects_only_matching_origin():
    global server
    server.cors_origins = "https://one.example, https://two.example"
    server.cors_credentials = True
    server.start()

    actual = server.make_request("GET", "/health", headers={"Origin": "https://two.example"})
    assert actual.status_code == 200
    assert actual.headers["Access-Control-Allow-Origin"] == "https://two.example"
    assert actual.headers["Access-Control-Allow-Credentials"] == "true"


def test_cors_explicit_origin_omits_headers_without_origin():
    global server
    server.cors_origins = "https://trusted.example"
    server.cors_credentials = True
    server.start()

    preflight = server.make_request("OPTIONS", "/completions", headers={
        "Access-Control-Request-Method": "POST",
    })
    assert preflight.status_code == 200
    assert "Access-Control-Allow-Origin" not in preflight.headers
    assert preflight.headers["Access-Control-Allow-Credentials"] == "false"

    actual = server.make_request("GET", "/health")
    assert actual.status_code == 200
    assert "Access-Control-Allow-Origin" not in actual.headers
    assert "Access-Control-Allow-Credentials" not in actual.headers


def test_cors_explicitly_disabled_credentials_remain_disabled():
    global server
    server.cors_origins = "https://trusted.example"
    server.cors_credentials = False
    server.start()
    res = server.make_request("OPTIONS", "/completions", headers={
        "Origin": "https://trusted.example",
        "Access-Control-Request-Method": "POST",
    })

    assert res.status_code == 200
    assert res.headers["Access-Control-Allow-Origin"] == "https://trusted.example"
    assert res.headers["Access-Control-Allow-Credentials"] == "false"

    actual = server.make_request("GET", "/health", headers={
        "Origin": "https://trusted.example",
    })
    assert actual.status_code == 200
    assert actual.headers["Access-Control-Allow-Origin"] == "https://trusted.example"
    assert "Access-Control-Allow-Credentials" not in actual.headers


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
    server.cors_credentials = True
    server.start()
    res = server.make_request("OPTIONS", "/completions", headers={
        "Origin": origin,
        "Access-Control-Request-Method": "POST",
        "Access-Control-Request-Headers": "Authorization",
    })
    assert res.status_code == 200
    assert res.headers["Access-Control-Allow-Origin"] == origin
    assert res.headers["Access-Control-Allow-Credentials"] == "true"

    actual = server.make_request("GET", "/health", headers={
        "Origin": origin,
    })
    assert actual.status_code == 200
    assert actual.headers["Access-Control-Allow-Origin"] == origin
    assert actual.headers["Access-Control-Allow-Credentials"] == "true"

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
    server.api_key = TEST_API_KEY
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
        server.ui_mcp_proxy_allow = ["127.0.0.1"]
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
