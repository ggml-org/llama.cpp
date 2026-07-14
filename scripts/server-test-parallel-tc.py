#!/usr/bin/env python3
"""
Test parallel tool-calling capability via chat completions endpoint.

Only run this against models that actually support parallel tool calls — this
script does not attempt to toggle that setting on the server. Each scenario is
explicitly worded so that a capable model SHOULD emit multiple tool calls in a
single assistant turn (either the same tool N times, or several different
tools at once).

Each test case contains:
  - tools: list of tool definitions (OpenAI-compatible)
  - messages: initial conversation messages
  - mock_tool_responses: dict mapping tool_name -> callable(arguments) -> str (JSON)
  - expected_parallel: dict describing what constitutes a successful parallel turn
        {"min_parallel": int,                # minimum tool_calls in one turn
         "require_same_tool": Optional[str], # all parallel calls must be this tool
         "require_distinct_tools": Optional[int], # >= N distinct tool names in one turn
         "min_distinct_args_key": Optional[str]}  # parallel calls must span this
                                                   # many distinct values of this arg key
  - validate: callable(turns, all_tool_calls, final_content) -> (passed, reason)
"""

import argparse
import json
import requests
import sys

# ---------------------------------------------------------------------------
# Color / formatting helpers
# ---------------------------------------------------------------------------

RESET = "\x1b[0m"
BOLD = "\x1b[1m"
DIM = "\x1b[2m"
CYAN = "\x1b[36m"
YELLOW = "\x1b[33m"
GREEN = "\x1b[32m"
RED = "\x1b[31m"
BLUE = "\x1b[34m"
WHITE = "\x1b[97m"
MAGENTA = "\x1b[35m"


def _print(text="", end="\n"):
    sys.stdout.write(text + end)
    sys.stdout.flush()


def print_header(title):
    bar = "─" * 60
    _print(f"\n{BOLD}{CYAN}┌{bar}┐{RESET}")
    _print(
        f"{BOLD}{CYAN}│  {WHITE}{title}{CYAN}{' ' * max(0, 58 - len(title))}│{RESET}"
    )
    _print(f"{BOLD}{CYAN}└{bar}┘{RESET}")


def print_turn_banner(turn_idx, n_calls):
    color = MAGENTA if n_calls >= 2 else DIM
    _print(f"\n  {BOLD}{color}▶ turn {turn_idx} — {n_calls} tool call(s){RESET}")


def print_tool_call(name, args):
    args_str = json.dumps(args)
    _print(
        f"    {BOLD}{YELLOW}⚙ {name}{RESET}{DIM}({args_str}){RESET}"
    )


def print_tool_result(result):
    preview = result[:140] + ("…" if len(result) > 140 else "")
    _print(f"      {DIM}{BLUE}↳ {preview}{RESET}")


def print_model_output(text):
    sys.stdout.write(text)
    sys.stdout.flush()


def print_pass(reason):
    _print(f"\n{BOLD}{GREEN}✔ PASS{RESET}  {reason}")


def print_fail(reason):
    _print(f"\n{BOLD}{RED}✘ FAIL{RESET}  {reason}")


def print_info(msg):
    _print(f"{DIM}{msg}{RESET}")


def print_warn(msg):
    _print(f"{BOLD}{YELLOW}⚠ {msg}{RESET}")


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------


def chat_completion(url, messages, tools=None, stream=False):
    payload = {
        "messages": messages,
        "stream": stream,
        "max_tokens": 4096,
    }
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"

    try:
        response = requests.post(url, json=payload, stream=stream)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        body = e.response.content if (e.response is not None) else b""
        print_fail(f"Request error: {e} | body: {body}")
        return None

    full_content = ""
    reasoning_content = ""
    tool_calls: list[dict] = []

    if stream:
        for line in response.iter_lines():
            if not line:
                continue
            decoded = line.decode("utf-8")
            if not decoded.startswith("data: "):
                continue
            data_str = decoded[6:]
            if data_str == "[DONE]":
                break
            try:
                data = json.loads(data_str)
            except json.JSONDecodeError:
                continue
            choices = data.get("choices", [])
            if not choices:
                continue
            delta = choices[0].get("delta", {})
            if delta.get("reasoning_content"):
                reasoning_content += delta["reasoning_content"]
            if delta.get("content"):
                full_content += delta["content"]
                print_model_output(delta["content"])
            for tc in delta.get("tool_calls", []):
                idx = tc.get("index", 0)
                while len(tool_calls) <= idx:
                    tool_calls.append(
                        {
                            "id": "",
                            "type": "function",
                            "function": {"name": "", "arguments": ""},
                        }
                    )
                if "id" in tc:
                    tool_calls[idx]["id"] += tc["id"]
                if "function" in tc:
                    if "name" in tc["function"]:
                        tool_calls[idx]["function"]["name"] += tc["function"]["name"]
                    if "arguments" in tc["function"]:
                        tool_calls[idx]["function"]["arguments"] += tc["function"][
                            "arguments"
                        ]
    else:
        data = response.json()
        choices = data.get("choices", [])
        if choices:
            msg = choices[0].get("message", {})
            full_content = msg.get("content") or ""
            reasoning_content = msg.get("reasoning_content") or ""
            tool_calls = msg.get("tool_calls") or []
            if full_content:
                print_model_output(full_content)

    result = {"content": full_content, "tool_calls": tool_calls}
    if reasoning_content:
        result["reasoning_content"] = reasoning_content
    return result


def run_agentic_loop(url, messages, tools, mock_tool_responses, stream, max_turns=6):
    """
    Drive the multi-turn tool-call loop, but record each turn's tool calls
    separately so parallelism can be validated.

    Returns (turns, all_tool_calls, final_content) where `turns` is a list
    of dicts: {"index": int, "tool_calls": [...], "content": str}.
    """
    msgs = list(messages)
    turns: list[dict] = []
    all_tool_calls: list[dict] = []

    for turn_idx in range(max_turns):
        result = chat_completion(url, msgs, tools=tools, stream=stream)
        if result is None:
            return turns, all_tool_calls, None

        tcs = result.get("tool_calls") or []
        content = result.get("content") or ""

        turns.append(
            {"index": turn_idx, "tool_calls": list(tcs), "content": content}
        )

        if not tcs:
            if content:
                _print(f"\n{DIM}{'·' * 60}{RESET}")
                _print(f"{DIM}  model response:{RESET}\n")
            return turns, all_tool_calls, content

        print_turn_banner(turn_idx, len(tcs))
        all_tool_calls.extend(tcs)

        assistant_msg: dict = {
            "role": "assistant",
            "content": content,
            "tool_calls": tcs,
        }
        reasoning = result.get("reasoning_content")
        if reasoning:
            assistant_msg["reasoning_content"] = reasoning
        msgs.append(assistant_msg)

        for tc in tcs:
            tool_name = tc["function"]["name"]
            try:
                args = json.loads(tc["function"]["arguments"])
            except json.JSONDecodeError:
                args = {}

            print_tool_call(tool_name, args)

            mock_fn = mock_tool_responses.get(tool_name)
            if mock_fn:
                tool_result = mock_fn(args)
            else:
                tool_result = json.dumps({"error": f"Unknown tool: {tool_name}"})

            print_tool_result(tool_result)

            msgs.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.get("id", ""),
                    "content": tool_result,
                }
            )

    return turns, all_tool_calls, None


# ---------------------------------------------------------------------------
# Parallelism helpers
# ---------------------------------------------------------------------------


def _best_parallel_turn(turns):
    """Return the turn (dict) with the most tool calls, or None if no tools."""
    tool_turns = [t for t in turns if t["tool_calls"]]
    if not tool_turns:
        return None
    return max(tool_turns, key=lambda t: len(t["tool_calls"]))


def _distinct_tool_names(turn):
    return {tc["function"]["name"] for tc in turn["tool_calls"]}


def _distinct_arg_values(turn, key):
    values = set()
    for tc in turn["tool_calls"]:
        try:
            args = json.loads(tc["function"]["arguments"])
        except json.JSONDecodeError:
            continue
        v = args.get(key)
        if v is not None:
            if isinstance(v, str):
                values.add(v.strip().lower())
            else:
                values.add(v)
    return values


def _check_parallel(turns, expected):
    """
    Check that at least one turn satisfies the parallel-call expectations.
    Returns (ok, reason).
    """
    best = _best_parallel_turn(turns)
    if best is None:
        return False, "No tool calls were made at all"

    min_parallel = expected.get("min_parallel", 2)
    if len(best["tool_calls"]) < min_parallel:
        by_turn = [len(t["tool_calls"]) for t in turns]
        return False, (
            f"No turn had >= {min_parallel} parallel tool calls "
            f"(per-turn counts: {by_turn})"
        )

    require_same = expected.get("require_same_tool")
    if require_same is not None:
        names = [tc["function"]["name"] for tc in best["tool_calls"]]
        if any(n != require_same for n in names):
            return False, (
                f"Parallel turn mixed tools; expected all {require_same!r}, got {names}"
            )

    require_distinct = expected.get("require_distinct_tools")
    if require_distinct is not None:
        distinct = _distinct_tool_names(best)
        if len(distinct) < require_distinct:
            return False, (
                f"Parallel turn had only {len(distinct)} distinct tool names "
                f"({distinct}); need >= {require_distinct}"
            )

    distinct_key = expected.get("min_distinct_args_key")
    distinct_count = expected.get("min_distinct_args_count", min_parallel)
    if distinct_key is not None:
        values = _distinct_arg_values(best, distinct_key)
        if len(values) < distinct_count:
            return False, (
                f"Parallel turn had only {len(values)} distinct {distinct_key!r} "
                f"values ({values}); need >= {distinct_count}"
            )

    return True, (
        f"Parallel turn had {len(best['tool_calls'])} calls across "
        f"{len(_distinct_tool_names(best))} distinct tool(s)"
    )


# ---------------------------------------------------------------------------
# Test case runner
# ---------------------------------------------------------------------------


def run_test(url, test_case, stream):
    name = test_case["name"]
    mode = f"{'stream' if stream else 'non-stream'}"
    print_header(f"{name}  [{mode}]")

    turns, all_tool_calls, final_content = run_agentic_loop(
        url,
        messages=test_case["messages"],
        tools=test_case["tools"],
        mock_tool_responses=test_case["mock_tool_responses"],
        stream=stream,
    )

    if not turns:
        print_fail("No response from server.")
        return False

    parallel_ok, parallel_reason = _check_parallel(turns, test_case["expected_parallel"])
    if not parallel_ok:
        print_fail(parallel_reason)
        return False

    passed, reason = test_case["validate"](turns, all_tool_calls, final_content)
    if passed:
        print_pass(f"{parallel_reason}; {reason}")
    else:
        print_fail(reason)
    return passed


# ---------------------------------------------------------------------------
# Test case definitions
# ---------------------------------------------------------------------------

# ---- Test 1: Multi-file read (same tool, multiple distinct paths) ----

_FILE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": (
                "Read the full contents of a file from the local filesystem. "
                "Call this tool in parallel when asked to read several files — "
                "each path needs its own call."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Absolute or repo-relative path to a file",
                    },
                },
                "required": ["path"],
            },
        },
    },
]

_FILE_CONTENTS = {
    "config/database.yml": "host: db.internal\nport: 5432\nuser: svc_app\n",
    "config/redis.yml":    "host: cache.internal\nport: 6379\ndb: 0\n",
    "config/queue.yml":    "broker: rabbitmq.internal\nport: 5672\nvhost: prod\n",
    "config/auth.yml":     "provider: oidc\nissuer: https://auth.internal\n",
}


def _read_file_mock(args):
    path = args.get("path", "")
    norm = path.lstrip("./").lstrip("/")
    content = _FILE_CONTENTS.get(norm)
    if content is None:
        for k, v in _FILE_CONTENTS.items():
            if path.endswith(k):
                content = v
                break
    if content is None:
        return json.dumps({"path": path, "error": "not found"})
    return json.dumps({"path": path, "content": content})


MULTIFILE_READ_TEST = {
    "name": "Parallel multi-file read (same tool, 4 distinct paths)",
    "tools": _FILE_TOOLS,
    "messages": [
        {
            "role": "user",
            "content": (
                "Please read all four of these config files so I can review them "
                "together: config/database.yml, config/redis.yml, config/queue.yml, "
                "and config/auth.yml. Call read_file for every path in parallel in "
                "a single batch — do NOT read them one by one sequentially across "
                "turns. After you have all four, give me a one-line summary of each."
            ),
        }
    ],
    "mock_tool_responses": {"read_file": _read_file_mock},
    "expected_parallel": {
        "min_parallel": 4,
        "require_same_tool": "read_file",
        "min_distinct_args_key": "path",
        "min_distinct_args_count": 4,
    },
    "validate": lambda turns, tcs, content: _validate_multifile(turns, tcs, content),
}


def _validate_multifile(turns, tcs, content):
    del turns
    if not content:
        return False, "No final summary produced"
    return True, f"{len(tcs)} total read_file calls; content length={len(content)}"


# ---- Test 2: Batch TODO marking (same tool, N calls in one turn) ----

_TODO_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "mark_todo_complete",
            "description": (
                "Mark a single TODO item as complete by ID. When the user wants "
                "several items marked at once, call this tool in parallel — "
                "one call per item — rather than sequentially across turns."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "todo_id": {
                        "type": "string",
                        "description": "Identifier of the TODO item",
                    },
                    "note": {
                        "type": "string",
                        "description": "Optional completion note",
                    },
                },
                "required": ["todo_id"],
            },
        },
    },
]

_TODO_DB = {
    "T-101": "Draft onboarding doc",
    "T-102": "Update dependency lockfile",
    "T-103": "Fix flaky login test",
    "T-104": "Rotate service credentials",
    "T-105": "Archive Q4 reports",
}


def _mark_todo_mock(args):
    tid = args.get("todo_id", "")
    if tid in _TODO_DB:
        return json.dumps({"todo_id": tid, "title": _TODO_DB[tid], "status": "done"})
    return json.dumps({"todo_id": tid, "error": "unknown id"})


TODO_BATCH_TEST = {
    "name": "Batch TODO completion (same tool, 5 IDs in one turn)",
    "tools": _TODO_TOOLS,
    "messages": [
        {
            "role": "user",
            "content": (
                "I finished every item on today's list. Please mark all of the "
                "following TODOs as complete, in one parallel batch: T-101, T-102, "
                "T-103, T-104, T-105. Don't mark them one at a time across separate "
                "turns — issue all five mark_todo_complete calls at once. Afterwards "
                "confirm which ones succeeded."
            ),
        }
    ],
    "mock_tool_responses": {"mark_todo_complete": _mark_todo_mock},
    "expected_parallel": {
        "min_parallel": 5,
        "require_same_tool": "mark_todo_complete",
        "min_distinct_args_key": "todo_id",
        "min_distinct_args_count": 5,
    },
    "validate": lambda turns, tcs, content: _validate_todo(turns, tcs, content),
}


def _validate_todo(turns, tcs, content):
    del turns
    if not content:
        return False, "No confirmation summary produced"
    return True, f"{len(tcs)} total mark_todo_complete calls"


# ---- Test 3: Multi-city weather (same tool, N parallel locations) ----

_WEATHER_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": (
                "Fetch current weather for ONE city. When the user asks about "
                "several cities, call this tool in parallel — one call per city — "
                "instead of sequentially."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "city":  {"type": "string", "description": "City name"},
                    "units": {
                        "type": "string",
                        "enum": ["metric", "imperial"],
                        "default": "metric",
                    },
                },
                "required": ["city"],
            },
        },
    },
]

_WEATHER_DB = {
    "tokyo":  {"city": "Tokyo",  "temp_c": 18.4, "condition": "partly cloudy", "humidity": 64},
    "london": {"city": "London", "temp_c":  9.1, "condition": "overcast",       "humidity": 81},
    "new york": {"city": "New York", "temp_c": 12.7, "condition": "clear",      "humidity": 55},
    "paris":  {"city": "Paris",  "temp_c": 11.3, "condition": "light rain",     "humidity": 78},
}


def _weather_mock(args):
    city = args.get("city", "").strip().lower()
    if city.startswith("new york"):
        city = "new york"
    if city in _WEATHER_DB:
        return json.dumps(_WEATHER_DB[city])
    return json.dumps({"city": args.get("city", ""), "error": "unknown city"})


MULTI_WEATHER_TEST = {
    "name": "Parallel multi-city weather (same tool, 4 cities)",
    "tools": _WEATHER_TOOLS,
    "messages": [
        {
            "role": "user",
            "content": (
                "I'm comparing today's weather across four cities for a travel "
                "decision: Tokyo, London, New York, and Paris. Please call "
                "get_weather for all four in parallel in a single turn — don't "
                "fetch them one at a time. Then rank them from warmest to coolest."
            ),
        }
    ],
    "mock_tool_responses": {"get_weather": _weather_mock},
    "expected_parallel": {
        "min_parallel": 4,
        "require_same_tool": "get_weather",
        "min_distinct_args_key": "city",
        "min_distinct_args_count": 4,
    },
    "validate": lambda turns, tcs, content: _validate_weather(turns, tcs, content),
}


def _validate_weather(turns, tcs, content):
    del turns
    if not content or not any(
        kw in content.lower() for kw in ("warmest", "rank", "hot", "cool")
    ):
        return False, f"Final content missing a ranking: {content!r}"
    return True, f"{len(tcs)} total get_weather calls; ranking produced"


# ---- Test 4: Trip planning (different tools, parallel in one turn) ----

_TRIP_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_flights",
            "description": "Search one-way flights between two airports on a given date.",
            "parameters": {
                "type": "object",
                "properties": {
                    "from_airport": {"type": "string", "description": "IATA code, e.g. SFO"},
                    "to_airport":   {"type": "string", "description": "IATA code, e.g. JFK"},
                    "date":         {"type": "string", "description": "YYYY-MM-DD"},
                },
                "required": ["from_airport", "to_airport", "date"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_hotels",
            "description": "Search hotels in a city for a date range.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city":       {"type": "string"},
                    "check_in":   {"type": "string", "description": "YYYY-MM-DD"},
                    "check_out":  {"type": "string", "description": "YYYY-MM-DD"},
                    "max_price":  {"type": "integer"},
                },
                "required": ["city", "check_in", "check_out"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_restaurants",
            "description": "Search restaurants in a city by cuisine.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city":    {"type": "string"},
                    "cuisine": {"type": "string"},
                },
                "required": ["city"],
            },
        },
    },
]

_FLIGHTS_RESULT = {
    "results": [
        {"flight": "UA 1552", "depart": "08:15", "arrive": "16:45", "price": 389},
        {"flight": "AA  20",  "depart": "10:00", "arrive": "18:35", "price": 412},
    ]
}
_HOTELS_RESULT = {
    "results": [
        {"name": "Midtown Grand",    "nightly_rate": 245, "rating": 4.3},
        {"name": "Harbour Boutique", "nightly_rate": 312, "rating": 4.6},
    ]
}
_RESTAURANTS_RESULT = {
    "results": [
        {"name": "Trattoria Nona", "cuisine": "italian", "rating": 4.5},
        {"name": "Osteria Blu",    "cuisine": "italian", "rating": 4.4},
    ]
}

TRIP_PLAN_TEST = {
    "name": "Trip planning (3 different tools in parallel)",
    "tools": _TRIP_TOOLS,
    "messages": [
        {
            "role": "user",
            "content": (
                "I'm flying from SFO to JFK on 2026-06-12 and staying four nights "
                "(check out 2026-06-16). I'd also like some Italian restaurant "
                "suggestions in New York. Please call search_flights, search_hotels, "
                "and search_restaurants in parallel — all three in a single turn, "
                "since they don't depend on each other. Then give me a concise "
                "travel summary."
            ),
        }
    ],
    "mock_tool_responses": {
        "search_flights": lambda _: json.dumps(_FLIGHTS_RESULT),
        "search_hotels": lambda _: json.dumps(_HOTELS_RESULT),
        "search_restaurants": lambda _: json.dumps(_RESTAURANTS_RESULT),
    },
    "expected_parallel": {
        "min_parallel": 3,
        "require_distinct_tools": 3,
    },
    "validate": lambda turns, tcs, content: _validate_trip(turns, tcs, content),
}


def _validate_trip(turns, tcs, content):
    del turns
    names = {tc["function"]["name"] for tc in tcs}
    required = {"search_flights", "search_hotels", "search_restaurants"}
    missing = required - names
    if missing:
        return False, f"Missing tool calls: {missing}"
    if not content:
        return False, "No travel summary produced"
    return True, f"All three tools called; summary length={len(content)}"


# ---- Test 5: Portfolio check (same tool, parallel tickers) ----

_STOCK_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_stock_quote",
            "description": (
                "Get the latest quote for ONE ticker. When the user asks about "
                "multiple tickers, call this tool in parallel — one per symbol — "
                "rather than sequentially."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "Ticker symbol"},
                },
                "required": ["symbol"],
            },
        },
    },
]

_STOCK_DB = {
    "AAPL": {"symbol": "AAPL", "price": 218.45, "change_pct": "+0.8%"},
    "MSFT": {"symbol": "MSFT", "price": 421.10, "change_pct": "+1.2%"},
    "GOOGL":{"symbol": "GOOGL","price": 175.22, "change_pct": "-0.3%"},
    "AMZN": {"symbol": "AMZN", "price": 189.76, "change_pct": "+0.5%"},
    "NVDA": {"symbol": "NVDA", "price": 140.88, "change_pct": "+2.4%"},
}


def _stock_mock(args):
    sym = args.get("symbol", "").strip().upper()
    if sym in _STOCK_DB:
        return json.dumps(_STOCK_DB[sym])
    return json.dumps({"symbol": sym, "error": "unknown ticker"})


PORTFOLIO_TEST = {
    "name": "Portfolio check (same tool, 5 tickers in parallel)",
    "tools": _STOCK_TOOLS,
    "messages": [
        {
            "role": "user",
            "content": (
                "Pull the latest quote for every ticker in my portfolio — AAPL, "
                "MSFT, GOOGL, AMZN, and NVDA — in a single parallel batch. These "
                "lookups are independent, so please don't chain them across turns. "
                "Once you have all five, tell me which ticker had the biggest "
                "percentage change today."
            ),
        }
    ],
    "mock_tool_responses": {"get_stock_quote": _stock_mock},
    "expected_parallel": {
        "min_parallel": 5,
        "require_same_tool": "get_stock_quote",
        "min_distinct_args_key": "symbol",
        "min_distinct_args_count": 5,
    },
    "validate": lambda turns, tcs, content: _validate_portfolio(turns, tcs, content),
}


def _validate_portfolio(turns, tcs, content):
    del turns
    if not content or ("nvda" not in content.lower() and "NVDA" not in content):
        return False, f"Expected NVDA to be identified as the biggest mover: {content!r}"
    return True, f"{len(tcs)} total quotes pulled"


# ---- Test 6: Mixed — translate + dictionary in parallel for the same word ----

_LANG_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "translate_text",
            "description": "Translate a short text into a target language.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text":            {"type": "string"},
                    "target_language": {"type": "string",
                                        "description": "ISO 639-1 language code, e.g. 'es'"},
                },
                "required": ["text", "target_language"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_definition",
            "description": "Get the English dictionary definition of a word.",
            "parameters": {
                "type": "object",
                "properties": {
                    "word": {"type": "string"},
                },
                "required": ["word"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_synonyms",
            "description": "Get English synonyms for a word.",
            "parameters": {
                "type": "object",
                "properties": {
                    "word": {"type": "string"},
                },
                "required": ["word"],
            },
        },
    },
]


def _translate_mock(args):
    t = args.get("text", "")
    lang = args.get("target_language", "")
    return json.dumps({"source": t, "target_language": lang, "translation": f"[{lang}] {t}"})


def _definition_mock(args):
    w = args.get("word", "")
    return json.dumps({
        "word": w,
        "definition": f"A standard dictionary definition of {w!r}.",
    })


def _synonyms_mock(args):
    w = args.get("word", "")
    return json.dumps({
        "word": w,
        "synonyms": ["synonym_a", "synonym_b", "synonym_c"],
    })


LANG_TOOLKIT_TEST = {
    "name": "Language toolkit (translate + definition + synonyms in parallel)",
    "tools": _LANG_TOOLS,
    "messages": [
        {
            "role": "user",
            "content": (
                "For the English word 'resilient', I need three independent "
                "look-ups at once: (a) translate it into Spanish, (b) fetch its "
                "dictionary definition, and (c) list its synonyms. These three "
                "calls don't depend on each other — please issue them in parallel "
                "in a single turn. Then present the combined results as a short "
                "language note."
            ),
        }
    ],
    "mock_tool_responses": {
        "translate_text":  _translate_mock,
        "get_definition":  _definition_mock,
        "get_synonyms":    _synonyms_mock,
    },
    "expected_parallel": {
        "min_parallel": 3,
        "require_distinct_tools": 3,
    },
    "validate": lambda turns, tcs, content: _validate_lang(turns, tcs, content),
}


def _validate_lang(turns, tcs, content):
    del turns
    names = {tc["function"]["name"] for tc in tcs}
    required = {"translate_text", "get_definition", "get_synonyms"}
    missing = required - names
    if missing:
        return False, f"Missing tool calls: {missing}"
    if not content:
        return False, "No language note produced"
    return True, f"All three lookup tools called; note length={len(content)}"


# ---------------------------------------------------------------------------
# All test cases
# ---------------------------------------------------------------------------

ALL_TEST_CASES = [
    MULTIFILE_READ_TEST,
    TODO_BATCH_TEST,
    MULTI_WEATHER_TEST,
    TRIP_PLAN_TEST,
    PORTFOLIO_TEST,
    LANG_TOOLKIT_TEST,
]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Test llama-server parallel tool-calling capability. Run this only "
            "against models configured for parallel tool calls — this script "
            "does not configure that itself."
        )
    )
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", default=8080, type=int)
    parser.add_argument(
        "--no-stream", action="store_true", help="Disable streaming mode tests"
    )
    parser.add_argument(
        "--stream-only", action="store_true", help="Only run streaming mode tests"
    )
    parser.add_argument(
        "--test",
        help="Run only the test whose name contains this substring (case-insensitive)",
    )
    args = parser.parse_args()

    url = f"http://{args.host}:{args.port}/v1/chat/completions"
    print_info(f"Testing server at {url}")
    print_warn(
        "This script expects the target model to emit multiple tool calls in a "
        "single assistant turn. Run it only against parallel-tool-capable models."
    )

    modes: list[bool] = []
    if not args.stream_only:
        modes.append(False)
    if not args.no_stream:
        modes.append(True)

    cases: list[dict] = ALL_TEST_CASES
    if args.test:
        name_filter = args.test.lower()
        cases = [c for c in cases if name_filter in str(c["name"]).lower()]
        if not cases:
            print_fail(f"No test cases matched '{args.test}'")
            sys.exit(1)

    total = 0
    passed = 0
    for stream in modes:
        for case in cases:
            total += 1
            if run_test(url, case, stream=stream):
                passed += 1

    color = GREEN if passed == total else RED
    _print(f"\n{BOLD}{color}{'─' * 60}{RESET}")
    _print(f"{BOLD}{color}  Results: {passed}/{total} passed{RESET}")
    _print(f"{BOLD}{color}{'─' * 60}{RESET}\n")
    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
