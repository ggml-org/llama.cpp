#!/usr/bin/env python3
"""
Test structured output capability via chat completions endpoint.

Each test case contains:
  - response_format: OpenAI-compatible response_format specification.
                     Both "json_schema" and "json_object" are accepted; with
                     "json_object" a schema can be supplied via extra_body.
  - extra_body (optional): dict of extra top-level request fields merged into
                     the request payload (mirrors the OpenAI SDK's extra_body
                     feature; llama.cpp reads a top-level "json_schema" here).
  - messages: initial conversation messages
  - tools (optional): tool definitions (for mixed tool + structured tests)
  - mock_tool_responses (optional): dict mapping tool_name -> callable(arguments) -> str (JSON)
  - apply_stage: "always" to apply response_format to every request,
                 "after_tools" to run the tool loop plain, then request a
                 structured summary in a follow-up user turn.
  - followup (optional, for after_tools): user message appended before the
                 final structured call.
  - validate: callable(parsed_json, tool_calls_history, raw_content) -> (passed: bool, reason: str)
"""

import argparse
import json
import requests
import sys
from typing import Any, cast

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


def print_tool_call(name, args):
    args_str = json.dumps(args)
    _print(
        f"\n  {BOLD}{YELLOW}⚙ tool call{RESET}  {CYAN}{name}{RESET}{DIM}({args_str}){RESET}"
    )


def print_tool_result(result):
    preview = result[:160] + ("…" if len(result) > 160 else "")
    _print(f"  {DIM}{BLUE}↳ result{RESET}    {DIM}{preview}{RESET}")


def print_model_output(text):
    sys.stdout.write(text)
    sys.stdout.flush()


def print_pass(reason):
    _print(f"\n{BOLD}{GREEN}✔ PASS{RESET}  {reason}")


def print_fail(reason):
    _print(f"\n{BOLD}{RED}✘ FAIL{RESET}  {reason}")


def print_info(msg):
    _print(f"{DIM}{msg}{RESET}")


def print_schema_note(label, rf, extra_body=None):
    kind = rf.get("type", "?")
    name = ""
    if kind == "json_schema":
        name = rf.get("json_schema", {}).get("name", "")
    elif kind == "json_object" and extra_body and "json_schema" in extra_body:
        extra_schema = extra_body["json_schema"] or {}
        name = extra_schema.get("title") or "extra_body.json_schema"
    _print(f"{DIM}{MAGENTA}  ⟐ response_format [{label}]: {kind}"
           f"{(' / ' + name) if name else ''}{RESET}")


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------


def chat_completion(url, messages, tools=None, response_format=None, stream=False,
                    extra_body=None):
    payload = {
        "messages": messages,
        "stream": stream,
        "max_tokens": 8192,
    }
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"
    if response_format is not None:
        payload["response_format"] = response_format
    if extra_body:
        payload.update(extra_body)

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


def run_tool_loop(
    url, messages, tools, mock_tool_responses, stream, response_format=None,
    extra_body=None, max_turns=6,
):
    """
    Drive the tool-call loop. If response_format is provided it is applied to
    every request. Returns (all_tool_calls, final_messages, final_content).
    """
    msgs = list(messages)
    all_tool_calls: list[dict] = []

    for _ in range(max_turns):
        result = chat_completion(
            url, msgs, tools=tools, response_format=response_format, stream=stream,
            extra_body=extra_body,
        )
        if result is None:
            return all_tool_calls, msgs, None

        tcs = result.get("tool_calls") or []
        content = result.get("content") or ""

        if not tcs:
            if content:
                _print(f"\n{DIM}{'·' * 60}{RESET}")
            return all_tool_calls, msgs, content

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

            mock_fn = mock_tool_responses.get(tool_name) if mock_tool_responses else None
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

    return all_tool_calls, msgs, None


# ---------------------------------------------------------------------------
# Test case runner
# ---------------------------------------------------------------------------


def _try_parse_json(text):
    """Attempt to parse text as JSON, trimming common markdown fences."""
    if text is None:
        return None
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        return None


def run_test(url, test_case, stream):
    name = test_case["name"]
    mode = f"{'stream' if stream else 'non-stream'}"
    apply_stage = test_case.get("apply_stage", "always")
    print_header(f"{name}  [{mode}] ({apply_stage})")

    response_format = test_case["response_format"]
    extra_body = test_case.get("extra_body")
    print_schema_note(apply_stage, response_format, extra_body)

    tools = test_case.get("tools")
    mocks = test_case.get("mock_tool_responses") or {}

    all_tcs: list[dict] = []
    final_content = None

    if apply_stage == "always":
        all_tcs, _msgs, final_content = run_tool_loop(
            url,
            messages=list(test_case["messages"]),
            tools=tools,
            mock_tool_responses=mocks,
            stream=stream,
            response_format=response_format,
            extra_body=extra_body,
        )
    elif apply_stage == "after_tools":
        # Phase 1: plain tool loop, no response_format applied yet.
        all_tcs, msgs, interim_content = run_tool_loop(
            url,
            messages=list(test_case["messages"]),
            tools=tools,
            mock_tool_responses=mocks,
            stream=stream,
            response_format=None,
        )
        if interim_content:
            msgs.append({"role": "assistant", "content": interim_content})
        followup = test_case.get(
            "followup",
            "Now output the answer strictly as JSON matching the provided schema. "
            "Do not include commentary.",
        )
        msgs.append({"role": "user", "content": followup})

        # Phase 2: request final structured output. Tools are not passed so the
        # model focuses on producing the schema-constrained answer.
        _print(f"\n{DIM}{MAGENTA}  ⟐ follow-up turn with response_format applied{RESET}")
        result = chat_completion(
            url, msgs, tools=None, response_format=response_format, stream=stream,
            extra_body=extra_body,
        )
        final_content = result["content"] if result else None
    else:
        print_fail(f"Unknown apply_stage: {apply_stage}")
        return False

    if final_content is None:
        print_fail("No final content from server.")
        return False

    parsed = _try_parse_json(final_content)
    if parsed is None:
        print_fail(f"Final content is not valid JSON: {final_content[:200]!r}")
        return False

    passed, reason = test_case["validate"](parsed, all_tcs, final_content)
    if passed:
        print_pass(reason)
    else:
        print_fail(reason)
    return passed


# ---------------------------------------------------------------------------
# Test case definitions
# ---------------------------------------------------------------------------

# ---- Test 1: Book metadata extraction (always / json_schema) ----

_BOOK_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "book_metadata",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "title": {"type": "string"},
                "author": {"type": "string"},
                "year": {"type": "integer"},
                "genre": {
                    "type": "string",
                    "enum": [
                        "fiction",
                        "non-fiction",
                        "fantasy",
                        "sci-fi",
                        "mystery",
                        "biography",
                        "history",
                        "other",
                    ],
                },
                "page_count": {"type": "integer"},
            },
            "required": ["title", "author", "year", "genre", "page_count"],
        },
    },
}

BOOK_TEST_CASE = {
    "name": "Book metadata extraction (json_schema, always)",
    "response_format": _BOOK_SCHEMA,
    "apply_stage": "always",
    "messages": [
        {
            "role": "user",
            "content": (
                "Extract book metadata from this description: "
                "'Dune is a 1965 science fiction epic by Frank Herbert, spanning roughly "
                "688 pages in its first edition, set on the desert planet Arrakis.' "
                "Return the data as JSON."
            ),
        }
    ],
    "validate": lambda parsed, tcs, raw: _validate_book(parsed),
}


def _validate_book(parsed):
    required = {"title", "author", "year", "genre", "page_count"}
    missing = required - parsed.keys()
    if missing:
        return False, f"Missing fields: {missing}"
    if not isinstance(parsed["title"], str) or not parsed["title"]:
        return False, "title must be a non-empty string"
    if not isinstance(parsed["author"], str) or "herbert" not in parsed["author"].lower():
        return False, f"author unexpected: {parsed['author']!r}"
    if not isinstance(parsed["year"], int) or parsed["year"] != 1965:
        return False, f"year should be 1965, got {parsed['year']!r}"
    if parsed["genre"] not in {
        "fiction", "non-fiction", "fantasy", "sci-fi", "mystery",
        "biography", "history", "other",
    }:
        return False, f"genre not in enum: {parsed['genre']!r}"
    if not isinstance(parsed["page_count"], int) or parsed["page_count"] <= 0:
        return False, f"page_count should be positive int: {parsed['page_count']!r}"
    return True, f"Book: {parsed['title']} ({parsed['year']}) / {parsed['genre']}"


# ---- Test 2: Sentiment classification (always / enum-constrained) ----

_SENTIMENT_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "sentiment_analysis",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "sentiment": {
                    "type": "string",
                    "enum": ["positive", "negative", "neutral"],
                },
                "confidence": {"type": "number"},
                "keywords": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 1,
                    "maxItems": 5,
                },
            },
            "required": ["sentiment", "confidence", "keywords"],
        },
    },
}

SENTIMENT_TEST_CASE = {
    "name": "Sentiment analysis with enum and array",
    "response_format": _SENTIMENT_SCHEMA,
    "apply_stage": "always",
    "messages": [
        {
            "role": "user",
            "content": (
                "Analyse the sentiment of this review and return JSON with the "
                "detected sentiment label, a confidence score between 0 and 1, "
                "and up to five keyword strings that drove the classification:\n\n"
                "'This product completely exceeded my expectations. The build "
                "quality is phenomenal, it arrived a day early, and customer "
                "support was delightful when I had a setup question.'"
            ),
        }
    ],
    "validate": lambda parsed, tcs, raw: _validate_sentiment(parsed),
}


def _validate_sentiment(parsed):
    if parsed.get("sentiment") not in {"positive", "negative", "neutral"}:
        return False, f"sentiment not in enum: {parsed.get('sentiment')!r}"
    if parsed["sentiment"] != "positive":
        return False, f"expected positive sentiment, got {parsed['sentiment']}"
    conf = parsed.get("confidence")
    if not isinstance(conf, (int, float)) or not (0.0 <= conf <= 1.0):
        return False, f"confidence not in [0,1]: {conf!r}"
    kws = parsed.get("keywords")
    if not isinstance(kws, list) or not (1 <= len(kws) <= 5):
        return False, f"keywords length out of range: {kws!r}"
    if not all(isinstance(k, str) and k for k in kws):
        return False, f"keywords must be non-empty strings: {kws!r}"
    return True, f"sentiment={parsed['sentiment']} conf={conf} kws={kws}"


# ---- Test: json_object + extra_body.json_schema (always) ----
#
# Exercises the llama.cpp-specific path where the OpenAI SDK would send
# response_format={"type": "json_object"} and tunnel the schema through
# extra_body.json_schema (which becomes a top-level "json_schema" field on
# the request body).

_PRODUCT_JSON_OBJECT_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": "https://example.com/product.schema.json",
    "title": "Product",
    "description": "A product in the catalog",
    "type": "object",
}

PRODUCT_JSON_OBJECT_TEST_CASE = {
    "name": "json_object response_format with extra_body json_schema",
    "response_format": {"type": "json_object"},
    "extra_body": {"json_schema": _PRODUCT_JSON_OBJECT_SCHEMA},
    "apply_stage": "always",
    "messages": [
        {
            "role": "system",
            "content": (
                "Extract structured data from the provided text according to the "
                "JSON schema. Return only valid JSON matching the schema exactly."
            ),
        },
        {
            "role": "user",
            "content": "Product: Wireless Headphones, ID: 101, In Stock: Yes",
        },
    ],
    "validate": lambda parsed, tcs, raw: _validate_product_json_object(parsed),
}


def _validate_product_json_object(parsed):
    if not isinstance(parsed, dict):
        return False, f"expected JSON object, got {type(parsed).__name__}: {parsed!r}"
    if not parsed:
        return False, f"expected non-empty object, got {parsed!r}"
    return True, f"product object with {len(parsed)} field(s): {sorted(parsed.keys())}"


# ---- Test 3: Nested recipe schema (always) ----

_RECIPE_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "recipe",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "name": {"type": "string"},
                "servings": {"type": "integer"},
                "ingredients": {
                    "type": "array",
                    "minItems": 2,
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "item": {"type": "string"},
                            "quantity": {"type": "string"},
                        },
                        "required": ["item", "quantity"],
                    },
                },
                "steps": {
                    "type": "array",
                    "minItems": 2,
                    "items": {"type": "string"},
                },
                "prep_time_minutes": {"type": "integer"},
            },
            "required": ["name", "servings", "ingredients", "steps", "prep_time_minutes"],
        },
    },
}

RECIPE_TEST_CASE = {
    "name": "Nested recipe with arrays of objects",
    "response_format": _RECIPE_SCHEMA,
    "apply_stage": "always",
    "messages": [
        {
            "role": "user",
            "content": (
                "Give me a simple 4-serving scrambled eggs recipe as structured JSON. "
                "Include the recipe name, servings, ingredients (each with item and "
                "quantity), preparation steps, and total prep time in minutes."
            ),
        }
    ],
    "validate": lambda parsed, tcs, raw: _validate_recipe(parsed),
}


def _validate_recipe(parsed):
    required = {"name", "servings", "ingredients", "steps", "prep_time_minutes"}
    missing = required - parsed.keys()
    if missing:
        return False, f"Missing fields: {missing}"
    if not isinstance(parsed["name"], str) or not parsed["name"]:
        return False, "name must be a non-empty string"
    if not isinstance(parsed["servings"], int) or parsed["servings"] <= 0:
        return False, f"servings must be positive int: {parsed['servings']!r}"
    ings = parsed["ingredients"]
    if not isinstance(ings, list) or len(ings) < 2:
        return False, f"ingredients must be array of >=2: got {ings!r}"
    for i, ing in enumerate(ings):
        if not isinstance(ing, dict):
            return False, f"ingredient[{i}] is not an object: {ing!r}"
        ing_d = cast(dict[str, Any], ing)
        item_val = ing_d.get("item")
        qty_val = ing_d.get("quantity")
        if item_val is None or qty_val is None:
            return False, f"ingredient[{i}] missing item/quantity: {ing!r}"
        if not isinstance(item_val, str) or not isinstance(qty_val, str):
            return False, f"ingredient[{i}] fields must be strings: {ing!r}"
    steps = parsed["steps"]
    if not isinstance(steps, list) or len(steps) < 2:
        return False, f"steps must be array of >=2 strings: got {steps!r}"
    if not all(isinstance(s, str) and s for s in steps):
        return False, "all steps must be non-empty strings"
    pt = parsed["prep_time_minutes"]
    if not isinstance(pt, int) or pt <= 0:
        return False, f"prep_time_minutes must be positive int: {pt!r}"
    return True, f"recipe '{parsed['name']}' with {len(ings)} ingredients, {len(steps)} steps"


# ---- Test 4: Tool call -> structured product comparison (after_tools) ----

_SHOP_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_products",
            "description": "Search a product catalogue by keyword.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_product_details",
            "description": "Get detailed specs for a product by ID.",
            "parameters": {
                "type": "object",
                "properties": {
                    "product_id": {"type": "string"},
                },
                "required": ["product_id"],
            },
        },
    },
]

_SHOP_SEARCH_RESULT = {
    "results": [
        {"product_id": "LAP-001", "title": "AeroBook 13 Pro",       "price": 1399.0, "rating": 4.7},
        {"product_id": "LAP-002", "title": "QuantumSlim 14",        "price": 1199.0, "rating": 4.4},
        {"product_id": "LAP-003", "title": "NimbusWork Ultra 15",   "price":  999.0, "rating": 4.2},
    ],
}
_SHOP_PRODUCT_DETAILS = {
    "LAP-001": {
        "product_id": "LAP-001",
        "title": "AeroBook 13 Pro",
        "cpu": "M-series 10-core",
        "ram_gb": 16,
        "storage_gb": 512,
        "battery_hours": 18,
        "weight_kg": 1.24,
        "price": 1399.0,
    },
    "LAP-002": {
        "product_id": "LAP-002",
        "title": "QuantumSlim 14",
        "cpu": "Core i7 12-core",
        "ram_gb": 16,
        "storage_gb": 512,
        "battery_hours": 12,
        "weight_kg": 1.35,
        "price": 1199.0,
    },
    "LAP-003": {
        "product_id": "LAP-003",
        "title": "NimbusWork Ultra 15",
        "cpu": "Ryzen 7 8-core",
        "ram_gb": 16,
        "storage_gb": 1024,
        "battery_hours": 10,
        "weight_kg": 1.70,
        "price": 999.0,
    },
}


def _shop_details_mock(args):
    pid = args.get("product_id", "")
    if pid in _SHOP_PRODUCT_DETAILS:
        return json.dumps(_SHOP_PRODUCT_DETAILS[pid])
    return json.dumps({"error": f"unknown product_id: {pid}"})


_SHOP_COMPARISON_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "laptop_comparison",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "recommendation": {"type": "string"},
                "ranked_candidates": {
                    "type": "array",
                    "minItems": 2,
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "product_id": {"type": "string"},
                            "title":      {"type": "string"},
                            "score":      {"type": "number"},
                            "reason":     {"type": "string"},
                        },
                        "required": ["product_id", "title", "score", "reason"],
                    },
                },
            },
            "required": ["recommendation", "ranked_candidates"],
        },
    },
}

SHOP_COMPARISON_TEST_CASE = {
    "name": "Tool calls then structured laptop comparison (after_tools)",
    "response_format": _SHOP_COMPARISON_SCHEMA,
    "apply_stage": "after_tools",
    "tools": _SHOP_TOOLS,
    "mock_tool_responses": {
        "search_products": lambda _: json.dumps(_SHOP_SEARCH_RESULT),
        "get_product_details": _shop_details_mock,
    },
    "messages": [
        {
            "role": "user",
            "content": (
                "I need a lightweight laptop for travel. Please search the catalogue "
                "for 'ultraportable laptop', then fetch detailed specs for at least two "
                "of the top candidates. Once you've gathered the data I'll ask you to "
                "produce a structured comparison."
            ),
        }
    ],
    "followup": (
        "Thanks. Now produce the final comparison strictly as JSON matching the "
        "laptop_comparison schema: your single best recommendation (the product_id), "
        "and a ranked_candidates array of at least two laptops, each with "
        "product_id, title, a numeric score, and a short reason."
    ),
    "validate": lambda parsed, tcs, raw: _validate_shop_comparison(parsed, tcs),
}


def _validate_shop_comparison(parsed, tcs):
    names = [tc["function"]["name"] for tc in tcs]
    if "search_products" not in names:
        return False, f"expected search_products tool call, got {names}"
    if "get_product_details" not in names:
        return False, f"expected get_product_details tool call, got {names}"
    if "recommendation" not in parsed or not isinstance(parsed["recommendation"], str):
        return False, f"recommendation missing or not a string: {parsed!r}"
    cands = parsed.get("ranked_candidates")
    if not isinstance(cands, list) or len(cands) < 2:
        return False, f"ranked_candidates must be >=2: {cands!r}"
    valid_ids = set(_SHOP_PRODUCT_DETAILS.keys())
    candidate_pids: list = []
    for i, c in enumerate(cands):
        if not isinstance(c, dict):
            return False, f"candidate[{i}] not an object: {c!r}"
        c_d = cast(dict[str, Any], c)
        pid = c_d.get("product_id")
        title = c_d.get("title")
        score = c_d.get("score")
        reason = c_d.get("reason")
        for k, v in (("product_id", pid), ("title", title),
                     ("score", score), ("reason", reason)):
            if v is None:
                return False, f"candidate[{i}] missing {k}: {c!r}"
        if pid not in valid_ids:
            return False, f"candidate[{i}].product_id not in catalogue: {pid!r}"
        if not isinstance(score, (int, float)):
            return False, f"candidate[{i}].score not numeric: {score!r}"
        candidate_pids.append(pid)
    recommendation = parsed["recommendation"]
    if recommendation not in valid_ids and recommendation not in candidate_pids:
        return False, f"recommendation {recommendation!r} not in candidates"
    return True, (
        f"tools={names}; recommended={parsed['recommendation']}; "
        f"{len(cands)} ranked candidates"
    )


# ---- Test 5: Multi-step research then structured report (after_tools) ----

_RESEARCH_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_country_stats",
            "description": "Fetch basic statistics for a country (population, GDP, capital).",
            "parameters": {
                "type": "object",
                "properties": {
                    "country": {"type": "string"},
                },
                "required": ["country"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_climate_info",
            "description": "Fetch climate information for a country.",
            "parameters": {
                "type": "object",
                "properties": {
                    "country": {"type": "string"},
                },
                "required": ["country"],
            },
        },
    },
]

_COUNTRY_STATS = {
    "norway": {
        "country": "Norway",
        "capital": "Oslo",
        "population": 5_480_000,
        "gdp_usd_trillion": 0.48,
        "currency": "NOK",
    }
}
_CLIMATE_INFO = {
    "norway": {
        "country": "Norway",
        "climate_zone": "subarctic / temperate coastal",
        "avg_winter_temp_c": -4.5,
        "avg_summer_temp_c": 16.0,
        "annual_precipitation_mm": 1400,
    }
}


def _country_stats_mock(args):
    c = args.get("country", "").strip().lower()
    if c in _COUNTRY_STATS:
        return json.dumps(_COUNTRY_STATS[c])
    return json.dumps({"error": f"unknown country: {c}"})


def _climate_info_mock(args):
    c = args.get("country", "").strip().lower()
    if c in _CLIMATE_INFO:
        return json.dumps(_CLIMATE_INFO[c])
    return json.dumps({"error": f"unknown country: {c}"})


_RESEARCH_REPORT_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "country_report",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "country": {"type": "string"},
                "capital": {"type": "string"},
                "population": {"type": "integer"},
                "climate_summary": {"type": "string"},
                "highlights": {
                    "type": "array",
                    "minItems": 2,
                    "maxItems": 5,
                    "items": {"type": "string"},
                },
                "suitable_for_tourism": {"type": "boolean"},
            },
            "required": [
                "country", "capital", "population",
                "climate_summary", "highlights", "suitable_for_tourism",
            ],
        },
    },
}

COUNTRY_REPORT_TEST_CASE = {
    "name": "Research pipeline then structured country report (after_tools)",
    "response_format": _RESEARCH_REPORT_SCHEMA,
    "apply_stage": "after_tools",
    "tools": _RESEARCH_TOOLS,
    "mock_tool_responses": {
        "get_country_stats": _country_stats_mock,
        "get_climate_info": _climate_info_mock,
    },
    "messages": [
        {
            "role": "user",
            "content": (
                "I'm preparing a short briefing on Norway. Please call the "
                "get_country_stats and get_climate_info tools to gather data "
                "first. Afterwards I'll ask for a structured summary."
            ),
        }
    ],
    "followup": (
        "Based on the tool results, produce the briefing as JSON matching the "
        "country_report schema. Populate every required field and provide between "
        "two and five highlights."
    ),
    "validate": lambda parsed, tcs, raw: _validate_country_report(parsed, tcs),
}


def _validate_country_report(parsed, tcs):
    names = [tc["function"]["name"] for tc in tcs]
    for required_tool in ("get_country_stats", "get_climate_info"):
        if required_tool not in names:
            return False, f"missing tool call {required_tool!r}: got {names}"
    required = {
        "country", "capital", "population",
        "climate_summary", "highlights", "suitable_for_tourism",
    }
    missing = required - parsed.keys()
    if missing:
        return False, f"missing report fields: {missing}"
    if "norway" not in parsed["country"].lower():
        return False, f"country should reference Norway: {parsed['country']!r}"
    if "oslo" not in parsed["capital"].lower():
        return False, f"capital should be Oslo: {parsed['capital']!r}"
    if not isinstance(parsed["population"], int) or parsed["population"] < 1_000_000:
        return False, f"population implausible: {parsed['population']!r}"
    if not isinstance(parsed["climate_summary"], str) or not parsed["climate_summary"]:
        return False, "climate_summary must be a non-empty string"
    hls = parsed["highlights"]
    if not isinstance(hls, list) or not (2 <= len(hls) <= 5):
        return False, f"highlights length out of range: {hls!r}"
    if not all(isinstance(h, str) and h for h in hls):
        return False, "each highlight must be a non-empty string"
    if not isinstance(parsed["suitable_for_tourism"], bool):
        return False, f"suitable_for_tourism must be bool: {parsed['suitable_for_tourism']!r}"
    return True, (
        f"tools={names}; report for {parsed['country']} "
        f"(pop {parsed['population']}, {len(hls)} highlights)"
    )


# ---------------------------------------------------------------------------
# All test cases
# ---------------------------------------------------------------------------

ALL_TEST_CASES = [
    BOOK_TEST_CASE,
    SENTIMENT_TEST_CASE,
    PRODUCT_JSON_OBJECT_TEST_CASE,
    RECIPE_TEST_CASE,
    SHOP_COMPARISON_TEST_CASE,
    COUNTRY_REPORT_TEST_CASE,
]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Test llama-server structured-output capability."
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
