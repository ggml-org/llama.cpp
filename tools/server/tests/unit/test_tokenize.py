import pytest
from utils import *

server = ServerPreset.tinyllama2()


@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.tinyllama2()


def test_tokenize_detokenize():
    global server
    server.start()
    # tokenize
    content = "What is the capital of France ?"
    res_tok = server.make_request("POST", "/tokenize", data={
        "content": content
    })
    assert res_tok.status_code == 200
    assert len(res_tok.body["tokens"]) > 5
    # detokenize
    res_detok = server.make_request("POST", "/detokenize", data={
        "tokens": res_tok.body["tokens"],
    })
    assert res_detok.status_code == 200
    assert res_detok.body["content"].strip() == content


def test_tokenize_with_bos():
    global server
    server.start()
    # tokenize
    content = "What is the capital of France ?"
    bosId = 1
    res_tok = server.make_request("POST", "/tokenize", data={
        "content": content,
        "add_special": True,
    })
    assert res_tok.status_code == 200
    assert res_tok.body["tokens"][0] == bosId


def test_tokenize_with_pieces():
    global server
    server.start()
    # tokenize
    content = "This is a test string with unicode 媽 and emoji 🤗"
    res_tok = server.make_request("POST", "/tokenize", data={
        "content": content,
        "with_pieces": True,
    })
    assert res_tok.status_code == 200
    for token in res_tok.body["tokens"]:
        assert "id" in token
        assert token["id"] > 0
        assert "piece" in token
        assert len(token["piece"]) > 0


# In a direct tiny-model server, the catalog instruction token count must
# equal the existing audited tokenizer operation with no special tokens
# (POST /tokenize defaults: add_special=false, parse_special=true). Skips
# until the route-binding task registers GET /skills.
def test_skills_catalog_token_count_matches_tokenize(tmp_path):
    global server
    server.skills = True
    server.trust_project_skills = True
    home = tmp_path / "home"
    home.mkdir()
    server.skill_home = str(home)
    project = tmp_path / "proj"
    project.mkdir()
    body = "Use the read_skill tool when a task matches a skill description."
    skill = project / ".agents" / "skills" / "counted"
    (skill / "SKILL.md").parent.mkdir(parents=True, exist_ok=True)
    (skill / "SKILL.md").write_text(
        f"---\nname: counted\ndescription: counted skill\n---\n{body}", encoding="utf-8"
    )
    server.start()

    res = server.make_request("GET", "/skills", headers={"X-Skill-Cwd": str(project)})
    if res.status_code == 404:
        pytest.skip("Skills routes are registered by the route-binding task")
    assert res.status_code == 200
    assert len(res.body["skills"]) == 1
    instruction = res.body["skills"][0]["instruction"]
    assert instruction["tokens_estimated"] is False
    assert instruction["bytes"] == len(body)
    # compare against the existing audited tokenizer operation with no special tokens
    res_tok = server.make_request("POST", "/tokenize", data={
        "content": body,
    })
    assert res_tok.status_code == 200
    assert instruction["tokens"] == len(res_tok.body["tokens"])
