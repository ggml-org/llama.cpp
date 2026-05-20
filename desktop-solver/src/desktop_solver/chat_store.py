"""Persistent chat history for the desktop solver.

Each chat is a JSON file under ``$XDG_CONFIG_HOME/desktop-solver/chats/``
(or ``~/.config/desktop-solver/chats/``). One file per chat; atomic writes
via a temp-file + rename.
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from .solver import SolveResult

log = logging.getLogger(__name__)


def default_config_dir() -> Path:
    xdg = os.environ.get("XDG_CONFIG_HOME")
    base = Path(xdg) if xdg else Path.home() / ".config"
    return base / "desktop-solver"


def _now_iso() -> str:
    # microsecond resolution so two saves in the same second still sort correctly
    return datetime.now(timezone.utc).isoformat()


@dataclass
class Turn:
    role: str                       # "user" | "assistant"
    timestamp: str = ""
    # Populated for "user" turns:
    problem: str = ""
    # Populated for "assistant" turns:
    answer: str = ""
    latex: str = ""
    code: str = ""
    steps: str = ""
    verified: bool = False
    verify_reason: str = ""
    error: str = ""

    @classmethod
    def user(cls, problem: str) -> "Turn":
        return cls(role="user", problem=problem, timestamp=_now_iso())

    @classmethod
    def assistant_from_result(cls, result: SolveResult) -> "Turn":
        return cls(
            role="assistant",
            answer=result.answer_repr,
            latex=result.latex,
            code=result.code,
            steps=result.steps,
            verified=result.verified,
            verify_reason=result.verify_reason,
            error=result.error,
            timestamp=_now_iso(),
        )

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Turn":
        return cls(**{k: d.get(k, "") for k in (
            "role", "timestamp", "problem", "answer", "latex", "code",
            "steps", "verify_reason", "error",
        )}, verified=bool(d.get("verified", False)))


@dataclass
class Chat:
    id: str
    title: str = "New chat"
    created: str = ""
    updated: str = ""
    turns: list[Turn] = field(default_factory=list)

    @classmethod
    def new(cls) -> "Chat":
        now = _now_iso()
        return cls(id=str(uuid.uuid4()), title="New chat", created=now, updated=now)

    def append_user(self, problem: str) -> Turn:
        t = Turn.user(problem)
        if self.title == "New chat":
            self.title = _summarise(problem)
        self.turns.append(t)
        self.updated = t.timestamp
        return t

    def append_assistant(self, result: SolveResult) -> Turn:
        t = Turn.assistant_from_result(result)
        self.turns.append(t)
        self.updated = t.timestamp
        return t

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "created": self.created,
            "updated": self.updated,
            "turns": [asdict(t) for t in self.turns],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Chat":
        turns = [Turn.from_dict(t) for t in d.get("turns", [])]
        return cls(
            id=d["id"],
            title=d.get("title", "Untitled"),
            created=d.get("created", ""),
            updated=d.get("updated", ""),
            turns=turns,
        )


def _summarise(text: str, limit: int = 60) -> str:
    text = " ".join(text.split())
    if not text:
        return "New chat"
    return text[:limit] + ("…" if len(text) > limit else "")


class ChatStore:
    """File-backed chat persistence. One JSON file per chat."""

    def __init__(self, root: Optional[Path] = None) -> None:
        self.root = root or (default_config_dir() / "chats")
        self.root.mkdir(parents=True, exist_ok=True)

    def new_chat(self) -> Chat:
        return Chat.new()

    def list_chats(self) -> list[Chat]:
        chats: list[Chat] = []
        for p in self.root.glob("*.json"):
            try:
                with p.open() as f:
                    chats.append(Chat.from_dict(json.load(f)))
            except (json.JSONDecodeError, KeyError, OSError) as e:
                log.warning("skipping malformed chat file %s: %s", p, e)
        chats.sort(key=lambda c: c.updated or c.created, reverse=True)
        return chats

    def load(self, chat_id: str) -> Optional[Chat]:
        p = self.root / f"{chat_id}.json"
        if not p.exists():
            return None
        try:
            with p.open() as f:
                return Chat.from_dict(json.load(f))
        except (json.JSONDecodeError, KeyError, OSError) as e:
            log.warning("failed to load %s: %s", p, e)
            return None

    def save(self, chat: Chat) -> None:
        path = self.root / f"{chat.id}.json"
        tmp = path.with_suffix(".json.tmp")
        with tmp.open("w") as f:
            json.dump(chat.to_dict(), f, indent=2, ensure_ascii=False)
        os.replace(tmp, path)

    def delete(self, chat_id: str) -> None:
        try:
            (self.root / f"{chat_id}.json").unlink()
        except FileNotFoundError:
            pass
