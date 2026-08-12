import pytest

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
# production routes remain unbound until the route-binding task.

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
