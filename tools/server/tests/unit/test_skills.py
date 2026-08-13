import pytest
import time
from pathlib import Path

from utils import ServerPreset, ServerResponse


@pytest.fixture
def server():
    return ServerPreset.router()



# NOTE: the catalog/read contract is verified by the native direct-handler test
# `tests/test-server-skills.cpp`, which drives the constructed server_skills
# GET/POST handlers with real request headers and request JSON. The cache and
# token-measurement tests below exercise the same contract through the public
# HTTP surface; the route-facing tests cover the enabled registration, prefix,
# middleware, and envelope behavior.


def write_skill(root, provider, name, body="Use <safe> & sound.", description="Use <safe> & sound."):
    """Create a valid skill at <root>/.<provider>/skills/<name>/SKILL.md."""
    skill = Path(root) / f".{provider}" / "skills" / name
    (skill / "SKILL.md").parent.mkdir(parents=True, exist_ok=True)
    (skill / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: {description}\n---\n{body}", encoding="utf-8"
    )
    return skill
# Route-facing integration tests: enabled GET /skills and POST /skills/read,
# prefix behavior, shared error envelopes, middleware ordering, and the
# preflight that the shared CORS policy must keep allowing X-Skill-Cwd.

def _skill_server(tmp_path):
    """A router-mode skills server with an isolated HOME and one project skill."""
    server = ServerPreset.router()
    server.skills = True
    server.trust_project_skills = True
    home = tmp_path / "home"
    home.mkdir()
    server.skill_home = str(home)
    project = tmp_path / "proj"
    project.mkdir()
    write_skill(project, "agents", "demo", body="demo body", description="demo desc")
    return server, project


def test_skills_enabled_catalog_and_reads(tmp_path):
    """An enabled server serves the catalog and base/resource reads."""
    server, project = _skill_server(tmp_path)
    server.start()

    catalog = server.make_request("GET", "/skills", headers={"X-Skill-Cwd": str(project)})
    assert catalog.status_code == 200
    assert [s["name"] for s in catalog.body["skills"]] == ["demo"]
    assert catalog.body["catalog_instruction_xml"] != ""
    assert "diagnostics" in catalog.body

    skill_file = project / ".agents" / "skills" / "demo" / "SKILL.md"
    source = "---\nname: demo\ndescription: demo skill\n---\n# Demo\n\nUse **carefully**.\n"
    skill_file.write_text(source, encoding="utf-8")

    base = server.make_request("POST", "/skills/read",
                               data={"name": "demo"}, headers={"X-Skill-Cwd": str(project)})
    assert base.status_code == 200
    assert base.body["kind"] == "skill"
    assert base.body["skill"]["name"] == "demo"
    assert base.body["source"] == source
    assert base.body["body_markdown"] == "# Demo\n\nUse **carefully**.\n"
    assert base.body["content_xml"].startswith("<skill_content name=\"demo\">")
    assert str(project) not in base.body["source"]
    assert str(tmp_path) not in base.body["source"]
    assert str(project) not in base.body["body_markdown"]
    assert str(tmp_path) not in base.body["body_markdown"]

    resource = write_skill(project, "agents", "demo", body="demo body", description="demo desc")
    (resource / "references").mkdir(exist_ok=True)
    (resource / "references" / "DETAILS.md").write_text("# details", encoding="utf-8")

    res = server.make_request("POST", "/skills/read",
                              data={"name": "demo", "path": "references/DETAILS.md"},
                              headers={"X-Skill-Cwd": str(project)})
    assert res.status_code == 200
    assert res.body["kind"] == "resource"
    assert res.body["resource"]["path"] == "references/DETAILS.md"
    assert "# details" in res.body["content_xml"]
    assert "source" not in res.body
    assert "body_markdown" not in res.body


def test_skills_api_prefix_applies_to_both_routes(tmp_path):
    """A configured API prefix applies to GET /skills and POST /skills/read."""
    server, project = _skill_server(tmp_path)
    server.api_prefix = "/api/v1"
    server.start()

    missing = server.make_request("GET", "/skills", headers={"X-Skill-Cwd": str(project)})
    assert missing.status_code == 404

    prefixed = server.make_request("GET", "/api/v1/skills", headers={"X-Skill-Cwd": str(project)})
    assert prefixed.status_code == 200
    assert [s["name"] for s in prefixed.body["skills"]] == ["demo"]

    prefixed_read = server.make_request("POST", "/api/v1/skills/read",
                                        data={"name": "demo"},
                                        headers={"X-Skill-Cwd": str(project)})
    assert prefixed_read.status_code == 200
    assert prefixed_read.body["kind"] == "skill"

    missing_read = server.make_request("POST", "/skills/read",
                                       data={"name": "demo"},
                                       headers={"X-Skill-Cwd": str(project)})
    assert missing_read.status_code == 404


def test_skills_malformed_body_and_invalid_cwd_use_400_envelope(tmp_path):
    """Malformed bodies and invalid CWDs share the 400 error envelope."""
    import requests as _requests

    server, project = _skill_server(tmp_path)
    server.start()

    def raw_post(body):
        response = _requests.post(
            f"http://{server.server_host}:{server.server_port}/skills/read",
            data=body, headers={"X-Skill-Cwd": str(project)}, timeout=10)
        result = ServerResponse()
        result.headers = dict(response.headers)
        result.status_code = response.status_code
        try:
            result.body = response.json()
        except Exception:
            result.body = response.text
        return result

    def assert_400(res):
        assert res.status_code == 400
        error = res.body["error"]
        assert "message" in error
        assert error["type"] == "invalid_request_error"
        assert error["code"] == 400

    # empty body, malformed JSON, a JSON array, and a JSON string are all
    # invalid skill requests
    for body in ["", "{", "[]", '"string"']:
        res = raw_post(body)
        assert_400(res)

    res = server.make_request("POST", "/skills/read", data={"path": "x"},
                              headers={"X-Skill-Cwd": str(project)})
    assert_400(res)
    res = server.make_request("POST", "/skills/read", data={"name": 42},
                              headers={"X-Skill-Cwd": str(project)})
    assert_400(res)
    res = server.make_request("POST", "/skills/read", data={"name": "demo", "extra": 1},
                              headers={"X-Skill-Cwd": str(project)})
    assert_400(res)

    invalid_cwd = server.make_request("GET", "/skills",
                                      headers={"X-Skill-Cwd": str(tmp_path / "missing")})
    assert_400(invalid_cwd)
    invalid_cwd_read = server.make_request("POST", "/skills/read", data={"name": "demo"},
                                           headers={"X-Skill-Cwd": str(tmp_path / "missing")})
    assert_400(invalid_cwd_read)


def test_skills_missing_skill_and_resource_use_404_envelope(tmp_path):
    """Missing skills and resources share the 404 error envelope."""
    server, project = _skill_server(tmp_path)
    server.start()

    missing_skill = server.make_request("POST", "/skills/read", data={"name": "nope"},
                                        headers={"X-Skill-Cwd": str(project)})
    assert missing_skill.status_code == 404
    error = missing_skill.body["error"]
    assert error["type"] == "not_found_error"
    assert error["code"] == 404
    assert "message" in error

    missing_resource = server.make_request("POST", "/skills/read",
                                           data={"name": "demo", "path": "no/such/file.txt"},
                                           headers={"X-Skill-Cwd": str(project)})
    assert missing_resource.status_code == 404
    error = missing_resource.body["error"]
    assert error["type"] == "not_found_error"
    assert error["code"] == 404
    assert "message" in error


def test_skills_api_key_middleware_rejects_before_handler(tmp_path):
    """API-key middleware rejects /skills before the Skills handler runs."""
    server, project = _skill_server(tmp_path)
    server.api_key = "sk-skills-secret"
    server.start()

    no_key = server.make_request("GET", "/skills", headers={"X-Skill-Cwd": str(project)})
    assert no_key.status_code == 401
    assert no_key.body["error"]["type"] == "authentication_error"

    bad_key = server.make_request("GET", "/skills",
                                  headers={"Authorization": "Bearer wrong", "X-Skill-Cwd": str(project)})
    assert bad_key.status_code == 401

    authed = server.make_request("GET", "/skills",
                                 headers={"Authorization": f"Bearer {server.api_key}",
                                          "X-Skill-Cwd": str(project)})
    assert authed.status_code == 200
    assert [s["name"] for s in authed.body["skills"]] == ["demo"]


def test_skills_readiness_middleware_rejects_before_handler(tmp_path):
    """Readiness middleware returns 503 before the Skills handler during load."""
    server = ServerPreset.tinyllama2()
    server.skills = True
    server.trust_project_skills = True
    server.offline = True
    home = tmp_path / "home"
    home.mkdir()
    server.skill_home = str(home)
    project = tmp_path / "proj"
    project.mkdir()
    write_skill(project, "agents", "demo", body="demo body", description="demo desc")
    # a local model keeps the server in the loading (not-ready) window after the
    # HTTP listener is up; the cached tinygemma3 model gives a reliable window
    tests_dir = Path(__file__).resolve().parents[1]  # tools/server/tests
    model_files = sorted(tests_dir.glob("tmp/models--ggml-org--tinygemma3-GGUF/snapshots/*/tinygemma3-Q8_0.gguf"))
    assert model_files, "cached tinygemma3 model missing; run the suite once to populate the model cache"
    server.model_file = str(model_files[0])
    server.model_hf_repo = None
    server.model_hf_file = None
    server.model_alias = None
    server.spawn()  # do not wait for readiness: we must observe the 503 window

    def request_skills():
        try:
            return server.make_request("GET", "/skills",
                                       headers={"X-Skill-Cwd": str(project)}, timeout=10)
        except Exception:
            return None

    deadline = time.time() + 30
    saw_503 = False
    while time.time() < deadline:
        res = request_skills()
        if res is not None and res.status_code == 503:
            saw_503 = True
            assert res.body["error"]["type"] == "unavailable_error"
            break
        if res is not None and res.status_code == 200:
            break  # model loaded before the first poll; nothing to observe
        if server.process.poll() is not None:
            break
        time.sleep(0.005)
    assert saw_503, "skills request during the model-load window did not answer 503"

    deadline = time.time() + 60
    ready = False
    while time.time() < deadline:
        try:
            res = server.make_request("GET", "/skills",
                                      headers={"X-Skill-Cwd": str(project)}, timeout=10)
            if res.status_code == 200:
                ready = True
                break
        except Exception:
            pass
        if server.process.poll() is not None:
            break
        time.sleep(0.05)
    assert ready, "skills request never became 200 after the model load"
    assert [s["name"] for s in res.body["skills"]] == ["demo"]


def test_skills_preflight_allows_x_skill_cwd(tmp_path):
    """OPTIONS preflight permits X-Skill-Cwd through the shared CORS policy."""
    server, project = _skill_server(tmp_path)
    server.start()

    for path, method in (("/skills", "GET"), ("/skills/read", "POST")):
        res = server.make_request("OPTIONS", path, headers={
            "Origin": "http://localhost:8080",
            "Access-Control-Request-Method": method,
            "Access-Control-Request-Headers": "X-Skill-Cwd",
        })
        assert res.status_code == 200
        assert res.headers["Access-Control-Allow-Methods"] == "GET, POST, DELETE, OPTIONS"
        allowed_headers = res.headers["Access-Control-Allow-Headers"]
        assert allowed_headers == "*" or "x-skill-cwd" in allowed_headers.lower()


def test_skills_disabled_keeps_route_missing(server):
    server.start()

    res = server.make_request("GET", "/skills")
    assert res.status_code == 404


def test_skills_accepts_normalized_provider_list_with_routes(tmp_path):
    # duplicates in --skill-providers are normalized at startup, and with the
    # routes bound an enabled server serves the catalog (project skills are
    # untrusted, so no fixture writes are needed: the catalog is simply empty)
    server = ServerPreset.router()
    server.skills = True
    server.skill_providers = ".claude,gemini,.claude"
    home = tmp_path / "home"
    home.mkdir()
    server.skill_home = str(home)
    server.start()

    res = server.make_request("GET", "/skills")
    assert res.status_code == 200
    assert res.body["skills"] == []


@pytest.mark.parametrize("providers", ["../claude", "claude/foo", "", ","])
def test_skills_rejects_invalid_provider_list_at_startup(server, providers):
    server.skills = True
    server.skill_providers = providers

    with pytest.raises(RuntimeError, match="return code"):
        server.start(timeout_seconds=1)

    assert server.process is not None
    assert server.process.returncode is not None
    assert server.process.returncode != 0


# Fills the 32-entry catalog cache past its cap with 33 distinct effective
# CWDs. Every catalog must resolve to its own fixture's skill, a recently
# revisited catalog must remain correct after an eviction candidate is
# revisited, and in router mode every instruction must be estimated as
# ceil(bytes / 4) with tokens_estimated true.
def test_skills_catalog_cache_lru_eviction(tmp_path):
    server = ServerPreset.router()
    server.skills = True
    server.trust_project_skills = True
    home = tmp_path / "home"
    home.mkdir()
    server.skill_home = str(home)
    projects = []
    for index in range(33):
        project = tmp_path / f"proj-{index:02d}"
        project.mkdir()
        write_skill(project, "agents", f"skill-{index:02d}",
                    body=f"body-{index:02d}", description=f"desc-{index:02d}")
        projects.append(project)
    server.start()

    def check(index):
        res = server.make_request("GET", "/skills", headers={"X-Skill-Cwd": str(projects[index])})
        assert res.status_code == 200
        assert len(res.body["skills"]) == 1
        skill = res.body["skills"][0]
        assert skill["name"] == f"skill-{index:02d}"
        assert skill["description"] == f"desc-{index:02d}"
        body_text = f"body-{index:02d}"
        instruction = skill["instruction"]
        assert instruction["bytes"] == len(body_text)
        assert instruction["tokens_estimated"] is True
        assert instruction["tokens"] == (len(body_text) + 3) // 4

    # fill the cache past the 32-entry cap: the 33rd fill evicts proj-00
    for index in range(33):
        check(index)
    # revisit the eviction candidates and recently revisited catalogs
    for index in (0, 0, 1, 32, 4):
        check(index)
    # every CWD must resolve to its own catalog (no cross-contamination)
    for index in range(33):
        check(index)


# Mutates SKILL.md in place and asserts the catalog refreshes bytes/body
# values without an identity change.
def test_skills_catalog_mutation_freshness(tmp_path):
    server = ServerPreset.router()
    server.skills = True
    server.trust_project_skills = True
    home = tmp_path / "home"
    home.mkdir()
    server.skill_home = str(home)
    project = tmp_path / "proj"
    project.mkdir()
    skill = write_skill(project, "agents", "mutable", body="first body", description="first desc")
    server.start()

    res = server.make_request("GET", "/skills", headers={"X-Skill-Cwd": str(project)})
    assert res.status_code == 200
    first = res.body["skills"][0]
    assert first["name"] == "mutable"
    assert first["description"] == "first desc"
    assert first["instruction"]["bytes"] == len("first body")
    assert first["instruction"]["tokens_estimated"] is True
    assert first["instruction"]["tokens"] == (len("first body") + 3) // 4
    first_id = first["id"]

    # in-place mutation: values refresh, identity is unchanged
    (skill / "SKILL.md").write_text(
        "---\nname: mutable\ndescription: second desc\n---\nsecond body", encoding="utf-8"
    )
    res = server.make_request("GET", "/skills", headers={"X-Skill-Cwd": str(project)})
    assert res.status_code == 200
    second = res.body["skills"][0]
    assert second["id"] == first_id
    assert second["name"] == "mutable"
    assert second["description"] == "second desc"
    assert second["instruction"]["bytes"] == len("second body")
    assert second["instruction"]["tokens_estimated"] is True
    assert second["instruction"]["tokens"] == (len("second body") + 3) // 4


# Router mode has no direct tokenizer: the catalog must report estimated
# counts equal to ceil(bytes / 4) instead of routing to a child model.
def test_skills_catalog_tokens_estimated_in_router_mode(tmp_path):
    server = ServerPreset.router()
    server.skills = True
    server.trust_project_skills = True
    home = tmp_path / "home"
    home.mkdir()
    server.skill_home = str(home)
    project = tmp_path / "proj"
    project.mkdir()
    write_skill(project, "agents", "estimated", body="estimate me", description="desc")
    server.start()

    res = server.make_request("GET", "/skills", headers={"X-Skill-Cwd": str(project)})
    assert res.status_code == 200
    instruction = res.body["skills"][0]["instruction"]
    assert instruction["tokens_estimated"] is True
    assert instruction["tokens"] == (instruction["bytes"] + 3) // 4
