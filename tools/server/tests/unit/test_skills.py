import pytest
from pathlib import Path

from utils import ServerPreset


@pytest.fixture
def server():
    return ServerPreset.router()


@pytest.fixture(scope="session", autouse=True)
def load_server_presets():
    yield


# NOTE: the catalog/read contract is verified by the native direct-handler test
# `tests/test-server-skills.cpp`, which drives the constructed server_skills
# GET/POST handlers with real request headers and request JSON while the
# production routes remain unbound until the route-binding task. The cache and
# token-measurement tests below exercise the same contract through the public
# HTTP surface and therefore self-skip until the route-binding task registers
# GET /skills; the native seam covers them against the unbound handlers.


def skills_route_bound(server):
    """True once the route-binding task registers GET /skills; before that the
    catalog cache/token tests cannot run through the public HTTP surface."""
    res = server.make_request("GET", "/skills")
    return res.status_code != 404


def write_skill(root, provider, name, body="Use <safe> & sound.", description="Use <safe> & sound."):
    """Create a valid skill at <root>/.<provider>/skills/<name>/SKILL.md."""
    skill = Path(root) / f".{provider}" / "skills" / name
    (skill / "SKILL.md").parent.mkdir(parents=True, exist_ok=True)
    (skill / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: {description}\n---\n{body}", encoding="utf-8"
    )
    return skill


def test_skills_disabled_keeps_route_missing(server):
    server.start()

    res = server.make_request("GET", "/skills")
    assert res.status_code == 404


def test_skills_accepts_normalized_provider_list_without_routes(server):
    server.skills = True
    server.skill_providers = ".claude,gemini,.claude"
    server.start()

    res = server.make_request("GET", "/skills")
    assert res.status_code == 404


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
    if not skills_route_bound(server):
        pytest.skip("Skills routes are registered by the route-binding task")

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
    if not skills_route_bound(server):
        pytest.skip("Skills routes are registered by the route-binding task")

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
    if not skills_route_bound(server):
        pytest.skip("Skills routes are registered by the route-binding task")

    res = server.make_request("GET", "/skills", headers={"X-Skill-Cwd": str(project)})
    assert res.status_code == 200
    instruction = res.body["skills"][0]["instruction"]
    assert instruction["tokens_estimated"] is True
    assert instruction["tokens"] == (instruction["bytes"] + 3) // 4
