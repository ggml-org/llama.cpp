"""Tests for chat persistence (chat_store)."""

from __future__ import annotations

import json

import pytest

from desktop_solver.chat_store import Chat, ChatStore, Turn
from desktop_solver.solver import SolveResult


@pytest.fixture()
def store(tmp_path):
    return ChatStore(root=tmp_path / "chats")


def _make_result(answer: str = "5", verified: bool = True) -> SolveResult:
    return SolveResult(
        problem="Solve 3x + 7 = 22",
        answer_repr=answer,
        latex="x = 5",
        code="x = Symbol('x'); answer = solve(3*x + 7 - 22, x)[0]",
        steps="Subtract 7, divide by 3.",
        verified=verified,
        verify_reason="" if verified else "residual 1.0",
    )


def test_new_chat_has_uuid_and_default_title(store):
    chat = store.new_chat()
    assert chat.id and len(chat.id) >= 32  # uuid4 is 36 chars
    assert chat.title == "New chat"
    assert chat.turns == []


def test_append_user_sets_title_from_problem(store):
    chat = store.new_chat()
    chat.append_user("Solve x^2 + 5x + 6 = 0")
    assert chat.title.startswith("Solve x^2")
    assert len(chat.turns) == 1
    assert chat.turns[0].role == "user"
    assert chat.turns[0].problem == "Solve x^2 + 5x + 6 = 0"
    assert chat.turns[0].timestamp


def test_append_assistant_records_result(store):
    chat = store.new_chat()
    chat.append_user("Solve 3x + 7 = 22")
    chat.append_assistant(_make_result(answer="5", verified=True))
    assert len(chat.turns) == 2
    assistant = chat.turns[1]
    assert assistant.role == "assistant"
    assert assistant.answer == "5"
    assert assistant.verified is True
    assert assistant.latex == "x = 5"


def test_save_and_reload_roundtrip(store):
    chat = store.new_chat()
    chat.append_user("Solve 3x + 7 = 22")
    chat.append_assistant(_make_result())
    store.save(chat)

    loaded = store.load(chat.id)
    assert loaded is not None
    assert loaded.id == chat.id
    assert loaded.title == chat.title
    assert len(loaded.turns) == 2
    assert loaded.turns[0].role == "user"
    assert loaded.turns[1].role == "assistant"
    assert loaded.turns[1].answer == "5"
    assert loaded.turns[1].verified is True


def test_list_chats_sorts_newest_first(store):
    a = store.new_chat()
    a.append_user("first")
    store.save(a)

    b = store.new_chat()
    b.append_user("second")
    store.save(b)

    chats = store.list_chats()
    assert [c.id for c in chats][0] == b.id


def test_delete_removes_file(store, tmp_path):
    chat = store.new_chat()
    chat.append_user("ephemeral")
    store.save(chat)

    target = tmp_path / "chats" / f"{chat.id}.json"
    assert target.exists()
    store.delete(chat.id)
    assert not target.exists()


def test_delete_missing_is_silent(store):
    store.delete("does-not-exist-id")  # must not raise


def test_save_is_atomic_via_tempfile(store, tmp_path):
    chat = store.new_chat()
    chat.append_user("hello")
    store.save(chat)
    final = tmp_path / "chats" / f"{chat.id}.json"
    tmp_residual = tmp_path / "chats" / f"{chat.id}.json.tmp"
    assert final.exists()
    assert not tmp_residual.exists()


def test_corrupt_json_is_skipped(store, tmp_path):
    (tmp_path / "chats" / "garbage.json").write_text("not json {", encoding="utf-8")
    chat = store.new_chat()
    chat.append_user("ok")
    store.save(chat)
    chats = store.list_chats()
    assert any(c.id == chat.id for c in chats)


def test_to_dict_from_dict_roundtrip():
    chat = Chat.new()
    chat.append_user("ping")
    d = chat.to_dict()
    j = json.loads(json.dumps(d))
    restored = Chat.from_dict(j)
    assert restored.id == chat.id
    assert restored.turns[0].problem == "ping"
