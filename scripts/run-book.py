#!/usr/bin/env python3
"""Read and update the tessera program run-book.

The run-book (.zcode/program/run-book.json) is the cross-session state for the
conductor: it tracks what has been done, what is next, what is blocked. The
model reads it at dispatch time and updates it after each step. Atomic writes
(temp + rename) so a partial update never corrupts state.

stdlib only - no polars/numpy/etc. so this always runs.

Usage:
  python3 scripts/run-book.py show
  python3 scripts/run-book.py next
  python3 scripts/run-book.py add --capability <c> --summary "..." [--artifact PATH]
  python3 scripts/run-book.py update <id> --status <s> [--summary "..."] [--artifact PATH]
  python3 scripts/run-book.py decide "decided to ..."
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RUN_BOOK = REPO_ROOT / ".zcode" / "program" / "run-book.json"

CAPABILITIES = {"alphaevolve", "tessera-analyst", "findings-curator", "code-reviewer"}
STATUSES = {"pending", "in_progress", "done", "blocked"}
VALID = CAPABILITIES | STATUSES  # for capability validation only


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _load() -> dict:
    if not RUN_BOOK.exists():
        sys.stderr.write(
            f"run-book not found at {RUN_BOOK}. Create it with a minimal "
            "skeleton (see AGENTS.md Program Routing).\n"
        )
        sys.exit(1)
    return json.loads(RUN_BOOK.read_text())


def _atomic_write(d: dict) -> None:
    RUN_BOOK.parent.mkdir(parents=True, exist_ok=True)
    d["last_updated"] = _now()
    fd, tmp = tempfile.mkstemp(dir=str(RUN_BOOK.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(d, f, indent=2)
            f.write("\n")
        os.replace(tmp, RUN_BOOK)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _new_id(phases: list) -> str:
    used = {p["id"] for p in phases}
    n = len(phases) + 1
    while f"p{n}" in used:
        n += 1
    return f"p{n}"


def cmd_show(_args) -> int:
    d = _load()
    print(f"objective: {d.get('objective', '(none)')}")
    print(f"last_updated: {d.get('last_updated', '?')}")
    print(f"\nphases ({len(d.get('phases', []))}):")
    for p in d.get("phases", []):
        marker = {"done": "[x]", "in_progress": "[>]", "pending": "[ ]",
                  "blocked": "[!]"}.get(p.get("status"), "[?]")
        line = f"  {marker} {p['id']:6} {p.get('capability', '?'):18} {p.get('status', '?'):10}"
        if p.get("artifact"):
            line += f"  -> {p['artifact']}"
        print(line)
        if p.get("summary"):
            print(f"           {p['summary']}")
        if p.get("status") == "blocked" and p.get("blocked_on"):
            print(f"           blocked_on: {p['blocked_on']}")
    if d.get("open_questions"):
        print(f"\nopen questions ({len(d['open_questions'])}):")
        for q in d["open_questions"]:
            print(f"  - {q}")
    if d.get("decision_log"):
        print(f"\ndecision log ({len(d['decision_log'])}):")
        for e in d["decision_log"][-5:]:  # last 5
            cap = e.get("capability", "?")
            print(f"  {e.get('ts', '?')[:16]}  [{cap}] {e.get('decision', '')}")
    return 0


def cmd_next(_args) -> int:
    d = _load()
    # blocked takes priority (someone should look at why), then in_progress, then pending
    for status in ("blocked", "in_progress", "pending"):
        for p in d.get("phases", []):
            if p.get("status") == status:
                print(f"next: {p['id']} [{p.get('capability')}] ({status})")
                print(f"      {p.get('summary', '')}")
                if p.get("blocked_on"):
                    print(f"      blocked_on: {p['blocked_on']}")
                return 0
    print("next: nothing pending, blocked, or in progress")
    return 0


def cmd_add(args) -> int:
    if args.capability not in CAPABILITIES:
        sys.stderr.write(f"capability must be one of {sorted(CAPABILITIES)}\n")
        return 2
    d = _load()
    phase = {
        "id": _new_id(d.get("phases", [])),
        "capability": args.capability,
        "status": args.status,
        "summary": args.summary,
        "last_updated": _now(),
    }
    if args.artifact:
        phase["artifact"] = args.artifact
    d.setdefault("phases", []).append(phase)
    _atomic_write(d)
    print(f"added {phase['id']} [{phase['capability']}] ({phase['status']})")
    return 0


def cmd_update(args) -> int:
    d = _load()
    for p in d.get("phases", []):
        if p["id"] == args.id:
            if args.status:
                if args.status not in STATUSES:
                    sys.stderr.write(f"status must be one of {sorted(STATUSES)}\n")
                    return 2
                p["status"] = args.status
            if args.summary:
                p["summary"] = args.summary
            if args.artifact:
                p["artifact"] = args.artifact
            p["last_updated"] = _now()
            _atomic_write(d)
            print(f"updated {p['id']} -> {p['status']}")
            return 0
    sys.stderr.write(f"no phase with id {args.id!r}\n")
    return 1


def cmd_decide(args) -> int:
    d = _load()
    d.setdefault("decision_log", []).append({
        "ts": _now(),
        "decision": args.decision,
        "capability": args.capability or "",
    })
    _atomic_write(d)
    print("recorded decision")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("show", help="pretty-print the run-book")
    sub.add_parser("next", help="print the next pending/blocked phase")

    a = sub.add_parser("add", help="append a phase")
    a.add_argument("--capability", required=True)
    a.add_argument("--summary", required=True)
    a.add_argument("--artifact", default=None)
    a.add_argument("--status", default="pending", choices=sorted(STATUSES))

    u = sub.add_parser("update", help="update a phase by id")
    u.add_argument("id")
    u.add_argument("--status", default=None, choices=sorted(STATUSES))
    u.add_argument("--summary", default=None)
    u.add_argument("--artifact", default=None)

    dec = sub.add_parser("decide", help="append to the decision log")
    dec.add_argument("decision")
    dec.add_argument("--capability", default=None)

    args = ap.parse_args()
    return {
        "show": cmd_show, "next": cmd_next, "add": cmd_add,
        "update": cmd_update, "decide": cmd_decide,
    }[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
