import pytest
from utils import *


# ref: https://stackoverflow.com/questions/22627659/run-code-before-and-after-each-test-in-py-test
@pytest.fixture(autouse=True)
def stop_server_after_each_test():
    # do nothing before each test
    yield
    # stop all servers after each test
    instances = set(
        server_instances
    )  # copy the set to prevent 'Set changed size during iteration'
    for server in instances:
        server.stop()


@pytest.fixture(scope="module", autouse=True)
def do_something():
    # this will be run once per test session, before any tests
    ServerPreset.load_all()


@pytest.fixture
def server_factory():
    """Factory: returns a fresh, configured (but not started) ServerProcess."""
    def _build(preset="tinyllama2", **overrides):
        s = getattr(ServerPreset, preset)()
        for k, v in overrides.items():
            setattr(s, k, v)
        return s
    return _build
