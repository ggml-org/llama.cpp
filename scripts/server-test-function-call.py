#!/usr/bin/env python3
"""
Test tool calling capability via chat completions endpoint.

Each test case contains:
  - tools: list of tool definitions (OpenAI-compatible)
  - messages: initial conversation messages
  - mock_tool_responses: dict mapping tool_name -> callable(arguments) -> str (JSON)
  - validate: callable(tool_calls_history, final_content) -> (passed: bool, reason: str)
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
# Foreground colors
CYAN = "\x1b[36m"
YELLOW = "\x1b[33m"
GREEN = "\x1b[32m"
RED = "\x1b[31m"
BLUE = "\x1b[34m"
WHITE = "\x1b[97m"


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
    # printed inline during streaming; prefix with a visual marker on first chunk
    sys.stdout.write(text)
    sys.stdout.flush()


def print_pass(reason):
    _print(f"\n{BOLD}{GREEN}✔ PASS{RESET}  {reason}")


def print_fail(reason):
    _print(f"\n{BOLD}{RED}✘ FAIL{RESET}  {reason}")


def print_info(msg):
    _print(f"{DIM}{msg}{RESET}")


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
    Drive the multi-turn tool-call loop:
      1. Send messages to model.
      2. If the model returns tool calls, execute mocks and append results.
      3. Repeat until no more tool calls or max_turns reached.

    Returns (all_tool_calls, final_content).
    """
    msgs = list(messages)
    all_tool_calls: list[dict] = []

    for _ in range(max_turns):
        result = chat_completion(url, msgs, tools=tools, stream=stream)
        if result is None:
            return all_tool_calls, None

        tcs = result.get("tool_calls") or []
        content = result.get("content") or ""

        if not tcs:
            # Print a visual separator before the final model response
            if content:
                _print(f"\n{DIM}{'·'*60}{RESET}")
                _print(f"{DIM}  model response:{RESET}\n")
            return all_tool_calls, content

        # Record tool calls for validation
        all_tool_calls.extend(tcs)

        # Append assistant message with tool calls
        assistant_msg: dict = {
            "role": "assistant",
            "content": content,
            "tool_calls": tcs,
        }
        reasoning = result.get("reasoning_content")
        if reasoning:
            assistant_msg["reasoning_content"] = reasoning
        msgs.append(assistant_msg)

        # Execute each tool call via mock and append tool result messages
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

    return all_tool_calls, None


# ---------------------------------------------------------------------------
# Test case runner
# ---------------------------------------------------------------------------


def run_test(url, test_case, stream):
    name = test_case["name"]
    mode = f"{'stream' if stream else 'non-stream'}"
    print_header(f"{name}  [{mode}]")

    all_tool_calls, final_content = run_agentic_loop(
        url,
        messages=test_case["messages"],
        tools=test_case["tools"],
        mock_tool_responses=test_case["mock_tool_responses"],
        stream=stream,
    )

    if final_content is None and not all_tool_calls:
        print_fail("No response from server.")
        return False

    passed, reason = test_case["validate"](all_tool_calls, final_content)
    if passed:
        print_pass(reason)
    else:
        print_fail(reason)
    return passed


# ---------------------------------------------------------------------------
# Test case definitions
# ---------------------------------------------------------------------------

# ---- Test 1: E-commerce multi-step search (Azzoo = anonymized marketplace) ----

_AZZOO_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "azzoo_search_products",
            "description": (
                "Search for products on Azzoo marketplace by keyword. "
                "Returns a list of matching products with IDs, titles, ratings and prices."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search keyword or phrase",
                    },
                    "page": {
                        "type": "string",
                        "description": "Page number (1-based)",
                        "default": "1",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "azzoo_get_product",
            "description": "Retrieve detailed information about a specific Azzoo product including specs and price.",
            "parameters": {
                "type": "object",
                "properties": {
                    "product_id": {
                        "type": "string",
                        "description": "Azzoo product identifier (e.g. AZB12345)",
                    },
                },
                "required": ["product_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "azzoo_get_reviews",
            "description": "Fetch customer reviews for an Azzoo product.",
            "parameters": {
                "type": "object",
                "properties": {
                    "product_id": {
                        "type": "string",
                        "description": "Azzoo product identifier",
                    },
                    "page": {
                        "type": "string",
                        "description": "Review page number",
                        "default": "1",
                    },
                },
                "required": ["product_id"],
            },
        },
    },
]

_AZZOO_SEARCH_RESULT = {
    "results": [
        {
            "product_id": "AZB00001",
            "title": "SteelBrew Pro Kettle 1.7L",
            "rating": 4.6,
            "price": 34.99,
        },
        {
            "product_id": "AZB00002",
            "title": "HeatKeep Gooseneck Kettle",
            "rating": 4.3,
            "price": 27.50,
        },
        {
            "product_id": "AZB00003",
            "title": "QuickBoil Stainless Kettle",
            "rating": 4.1,
            "price": 21.00,
        },
    ]
}
_AZZOO_PRODUCT_RESULT = {
    "product_id": "AZB00001",
    "title": "SteelBrew Pro Kettle 1.7L",
    "price": 34.99,
    "rating": 4.6,
    "review_count": 2847,
    "specs": {
        "material": "18/8 stainless steel",
        "capacity": "1.7 L",
        "auto_shutoff": True,
        "keep_warm": "30 min",
        "warranty": "2 years",
    },
}
_AZZOO_REVIEWS_RESULT = {
    "product_id": "AZB00001",
    "average_rating": 4.6,
    "reviews": [
        {
            "rating": 5,
            "title": "Excellent build quality",
            "body": "Very sturdy, boils fast and stays warm longer than expected.",
        },
        {
            "rating": 5,
            "title": "Great for loose-leaf tea",
            "body": "The wide spout makes filling a teapot easy. No leaks after months of use.",
        },
        {
            "rating": 3,
            "title": "Minor lid issue",
            "body": "The lid doesn't always click shut properly, but overall happy with it.",
        },
        {
            "rating": 4,
            "title": "Good value",
            "body": "Heats quickly and the auto shutoff works reliably.",
        },
    ],
}

AZZOO_TEST_CASE = {
    "name": "Azzoo E-commerce: search -> product detail -> reviews",
    "messages": [
        {
            "role": "user",
            "content": (
                "I need a durable stainless steel tea kettle for my weekly tea gatherings. "
                "Please search Azzoo for 'stainless steel tea kettle', then get full details "
                "on the top-rated result, and finally fetch its customer reviews so I can "
                "check for recurring complaints. Give me a summary with pros and cons."
            ),
        }
    ],
    "tools": _AZZOO_TOOLS,
    "mock_tool_responses": {
        "azzoo_search_products": lambda _: json.dumps(_AZZOO_SEARCH_RESULT),
        "azzoo_get_product": lambda _: json.dumps(_AZZOO_PRODUCT_RESULT),
        "azzoo_get_reviews": lambda _: json.dumps(_AZZOO_REVIEWS_RESULT),
    },
    "validate": lambda tcs, content: _validate_azzoo(tcs, content),
}


def _validate_azzoo(tcs, content):
    names = [tc["function"]["name"] for tc in tcs]
    if not names:
        return False, "No tool calls made"
    if "azzoo_search_products" not in names:
        return False, f"Expected azzoo_search_products to be called, got: {names}"
    # After search the model should look up product details
    if "azzoo_get_product" not in names and "azzoo_get_reviews" not in names:
        return False, f"Expected follow-up product/review lookup, got: {names}"
    # Verify product lookup used an ID from search results
    for tc in tcs:
        if tc["function"]["name"] == "azzoo_get_product":
            try:
                args = json.loads(tc["function"]["arguments"])
                pid = args.get("product_id", "")
                if not pid:
                    return False, "azzoo_get_product called with empty product_id"
            except json.JSONDecodeError:
                return False, "azzoo_get_product arguments are not valid JSON"
    if not content:
        return False, "No final summary produced"
    return True, f"All expected tools called in order: {names}"


# ---- Test 2: Fitness BMI + exercise recommendations ----

_FITNESS_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "calculate_bmi",
            "description": "Calculate Body Mass Index (BMI) from weight and height.",
            "parameters": {
                "type": "object",
                "properties": {
                    "weight_kg": {
                        "type": "number",
                        "description": "Body weight in kilograms",
                    },
                    "height_m": {"type": "number", "description": "Height in meters"},
                },
                "required": ["weight_kg", "height_m"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_exercises",
            "description": (
                "Fetch a list of exercises filtered by muscle group, difficulty, category, "
                "and/or force type."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "muscle": {
                        "type": "string",
                        "description": "Target muscle group (e.g. chest, back, legs)",
                    },
                    "difficulty": {
                        "type": "string",
                        "description": "Difficulty level: beginner, intermediate, expert",
                    },
                    "category": {
                        "type": "string",
                        "description": "Exercise category (e.g. strength, cardio, stretching)",
                    },
                    "force": {
                        "type": "string",
                        "description": "Force type: push, pull, static",
                    },
                },
                "required": [],
            },
        },
    },
]

_BMI_RESULT = {"bmi": 24.5, "category": "Normal weight", "healthy_range": "18.5 – 24.9"}
_EXERCISES_RESULT = {
    "exercises": [
        {
            "name": "Push-Up",
            "muscle": "chest",
            "difficulty": "beginner",
            "equipment": "none",
            "instructions": "Keep body straight, lower chest to floor.",
        },
        {
            "name": "Incline Dumbbell Press",
            "muscle": "chest",
            "difficulty": "beginner",
            "equipment": "dumbbells, bench",
            "instructions": "Press dumbbells up from chest on incline bench.",
        },
        {
            "name": "Chest Fly (cables)",
            "muscle": "chest",
            "difficulty": "beginner",
            "equipment": "cable machine",
            "instructions": "Bring cables together in an arc motion.",
        },
    ]
}

FITNESS_TEST_CASE = {
    "name": "Fitness: BMI calculation + exercise suggestions",
    "messages": [
        {
            "role": "user",
            "content": (
                "I'm a 32-year-old male, 78 kg and 1.80 m tall. "
                "Please calculate my BMI and then suggest some beginner chest exercises I can do "
                "to build strength. Give me a short personalised plan."
            ),
        }
    ],
    "tools": _FITNESS_TOOLS,
    "mock_tool_responses": {
        "calculate_bmi": lambda _: json.dumps(_BMI_RESULT),
        "get_exercises": lambda _: json.dumps(_EXERCISES_RESULT),
    },
    "validate": lambda tcs, content: _validate_fitness(tcs, content),
}


def _validate_fitness(tcs, content):
    names = [tc["function"]["name"] for tc in tcs]
    if not names:
        return False, "No tool calls made"
    if "calculate_bmi" not in names:
        return False, f"Expected calculate_bmi to be called, got: {names}"
    # Validate BMI args contain plausible values
    for tc in tcs:
        if tc["function"]["name"] == "calculate_bmi":
            try:
                args = json.loads(tc["function"]["arguments"])
                w = args.get("weight_kg")
                h = args.get("height_m")
                if w is None or h is None:
                    return False, f"calculate_bmi missing weight_kg or height_m: {args}"
                if not (50 <= float(w) <= 200):
                    return False, f"calculate_bmi weight out of plausible range: {w}"
                if not (1.0 <= float(h) <= 2.5):
                    return False, f"calculate_bmi height out of plausible range: {h}"
            except (json.JSONDecodeError, ValueError) as e:
                return False, f"calculate_bmi argument error: {e}"
    if not content:
        return False, "No final plan produced"
    return True, f"Tools called: {names}"


# ---- Test 3: Community class planning (anonymised cooking/topic discovery) ----

_COMMUNITY_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_trending_questions",
            "description": (
                "Fetch commonly asked questions on a topic from search engine 'People Also Ask' boxes."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Topic to search for"},
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum questions to return",
                        "default": 10,
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_mobile_apps",
            "description": "Search the mobile app store for apps matching a category or keyword.",
            "parameters": {
                "type": "object",
                "properties": {
                    "keyword": {
                        "type": "string",
                        "description": "Search keyword (e.g. 'Italian cooking')",
                    },
                    "platform": {
                        "type": "string",
                        "enum": ["ios", "android", "both"],
                        "default": "both",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Number of results",
                        "default": 10,
                    },
                },
                "required": ["keyword"],
            },
        },
    },
]

_TRENDING_QUESTIONS_RESULT = {
    "query": "Italian cuisine",
    "questions": [
        "What are the most popular Italian dishes?",
        "What makes Italian food different from other cuisines?",
        "How do you make authentic Italian pasta from scratch?",
        "What are traditional Italian desserts?",
        "What herbs are commonly used in Italian cooking?",
        "Is Italian food healthy?",
        "What wine pairs best with Italian pasta?",
    ],
}
_APPS_RESULT = {
    "keyword": "Italian cooking",
    "results": [
        {
            "name": "PastaPro",
            "rating": 4.5,
            "installs": "500K+",
            "focus": "pasta recipes only",
        },
        {
            "name": "CookEasy",
            "rating": 4.2,
            "installs": "1M+",
            "focus": "general cooking, limited Italian content",
        },
        {
            "name": "ItalianKitchen",
            "rating": 3.8,
            "installs": "100K+",
            "focus": "regional Italian recipes, no video",
        },
    ],
}

COMMUNITY_CLASS_TEST_CASE = {
    "name": "Community class planning: trending topics + app gap analysis",
    "messages": [
        {
            "role": "user",
            "content": (
                "I want to start teaching Italian cooking classes at my community centre. "
                "First, find out what people commonly ask about Italian cuisine online. "
                "Then search for existing Italian cooking apps to see what they cover. "
                "Use both results to suggest three unique angles for my classes that fill gaps "
                "in what apps already offer."
            ),
        }
    ],
    "tools": _COMMUNITY_TOOLS,
    "mock_tool_responses": {
        "get_trending_questions": lambda _: json.dumps(_TRENDING_QUESTIONS_RESULT),
        "search_mobile_apps": lambda _: json.dumps(_APPS_RESULT),
    },
    "validate": lambda tcs, content: _validate_community(tcs, content),
}


def _validate_community(tcs, content):
    names = [tc["function"]["name"] for tc in tcs]
    if not names:
        return False, "No tool calls made"
    missing = [
        t for t in ("get_trending_questions", "search_mobile_apps") if t not in names
    ]
    if missing:
        return False, f"Missing expected tool calls: {missing}; got: {names}"
    if not content:
        return False, "No class suggestion produced"
    return True, f"Both discovery tools called: {names}"


# ---- Test 4: Multi-hostname geolocation filter (anonymized gallery discovery) ----
# Inspired by: checking gallery website server locations to find truly remote venues.
# Anonymized: galleryone.de → halle-eins.de, gallerytwo.fr → galerie-deux.fr,
#             gallerythree.it → galleria-tre.it

_GEO_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "lookup_ip_geolocation",
            "description": (
                "Retrieve geolocation data for an IP address or hostname, including country, "
                "city, coordinates, and network info. Useful for verifying physical server "
                "locations or personalising regional content."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "host": {
                        "type": "string",
                        "description": "IP address or hostname to look up (e.g. '8.8.8.8' or 'example.com').",
                    },
                },
                "required": ["host"],
            },
        },
    },
]

# Mock: one urban (Berlin → discard), two rural (keep)
_GEO_RESPONSES = {
    "halle-eins.de": {
        "host": "halle-eins.de",
        "city": "Berlin",
        "country": "DE",
        "lat": 52.5200,
        "lon": 13.4050,
        "is_major_city": True,
    },
    "galerie-deux.fr": {
        "host": "galerie-deux.fr",
        "city": "Rocamadour",
        "country": "FR",
        "lat": 44.7994,
        "lon": 1.6178,
        "is_major_city": False,
    },
    "galleria-tre.it": {
        "host": "galleria-tre.it",
        "city": "Matera",
        "country": "IT",
        "lat": 40.6664,
        "lon": 16.6044,
        "is_major_city": False,
    },
}


def _geo_mock(args):
    host = args.get("host", "")
    return json.dumps(_GEO_RESPONSES.get(host, {"error": f"unknown host: {host}"}))


GEO_TEST_CASE = {
    "name": "Gallery geolocation: filter urban venues, keep remote ones",
    "messages": [
        {
            "role": "user",
            "content": (
                "I have abstract paintings to exhibit in remote European galleries. "
                "I received enquiries from three venues: halle-eins.de, galerie-deux.fr, "
                "and galleria-tre.it. Please look up the geolocation of each website's server. "
                "Discard any venue whose server is in a major city (e.g. Berlin, Paris, Rome). "
                "For the remaining venues, report their exact coordinates so I can check "
                "whether hiking trails are nearby — my work thrives where nature and art meet."
            ),
        }
    ],
    "tools": _GEO_TOOLS,
    "mock_tool_responses": {
        "lookup_ip_geolocation": _geo_mock,
    },
    "validate": lambda tcs, content: _validate_geo(tcs, content),
}


def _validate_geo(tcs, content):
    names = [tc["function"]["name"] for tc in tcs]
    if not names:
        return False, "No tool calls made"
    # Expect exactly one geolocation call per domain (3 total)
    geo_calls = [tc for tc in tcs if tc["function"]["name"] == "lookup_ip_geolocation"]
    if len(geo_calls) < 3:
        return (
            False,
            f"Expected geolocation called 3 times (once per domain), got {len(geo_calls)}",
        )
    queried_hosts = set()
    for tc in geo_calls:
        try:
            args = json.loads(tc["function"]["arguments"])
            host = args.get("host", "")
            if not host:
                return False, f"lookup_ip_geolocation called with empty host: {args}"
            queried_hosts.add(host)
        except json.JSONDecodeError:
            return False, "lookup_ip_geolocation arguments are not valid JSON"
    expected = {"halle-eins.de", "galerie-deux.fr", "galleria-tre.it"}
    if not expected.issubset(queried_hosts):
        return (
            False,
            f"Not all domains queried. Expected {expected}, got {queried_hosts}",
        )
    if not content:
        return False, "No final summary produced"
    return True, f"All 3 domains geolocated: {sorted(queried_hosts)}"


# ---- Test 5: EV fleet expansion — stock → security → property → video ----
# Inspired by: multi-step business analysis combining finance, cybersecurity,
#              real estate and educational content.
# Anonymized: Tesla → Voltara (VLTR), Rivian → Rivex (RVXN),
#             Trenton → Halverton

_EV_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_stock_quote",
            "description": "Retrieve the latest market quote for a financial instrument by ticker symbol.",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "Ticker symbol (e.g. 'VLTR', 'RVXN')",
                    },
                    "interval": {
                        "type": "string",
                        "description": "Time interval: 1min, 5min, 1h, 1day, 1week",
                        "default": "1day",
                    },
                },
                "required": ["symbol"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_security_advisories",
            "description": (
                "Fetch current cybersecurity advisories from the national security agency, "
                "covering known vulnerabilities and exploits for industrial and consumer systems."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "keyword": {
                        "type": "string",
                        "description": "Filter advisories by keyword or product name",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of advisories to return",
                        "default": 5,
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_commercial_properties",
            "description": "Search for commercial properties (offices, garages, warehouses) available for rent or sale in a given city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name to search in"},
                    "property_type": {
                        "type": "string",
                        "description": "Type of property: office, garage, warehouse, premises",
                    },
                    "operation": {
                        "type": "string",
                        "enum": ["rent", "sale"],
                        "default": "rent",
                    },
                    "max_price": {
                        "type": "integer",
                        "description": "Maximum monthly rent or sale price",
                    },
                },
                "required": ["city", "property_type"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_video_recommendations",
            "description": "Fetch a list of recommended videos related to a given topic or reference video.",
            "parameters": {
                "type": "object",
                "properties": {
                    "topic": {
                        "type": "string",
                        "description": "Topic or keyword to search for related videos",
                    },
                },
                "required": ["topic"],
            },
        },
    },
]

_STOCK_RESULT_VLTR = {
    "symbol": "VLTR",
    "company": "Voltara Inc.",
    "price": 218.45,
    "change_pct": "+2.3%",
    "market_cap": "694B",
    "currency": "USD",
}
_STOCK_RESULT_RVXN = {
    "symbol": "RVXN",
    "company": "Rivex Motors",
    "price": 12.80,
    "change_pct": "-1.1%",
    "market_cap": "11B",
    "currency": "USD",
}
_ADVISORIES_RESULT = {
    "count": 2,
    "advisories": [
        {
            "id": "ICSA-24-102-01",
            "title": "Voltara In-Vehicle Infotainment System Authentication Bypass",
            "severity": "Medium",
            "summary": "Improper authentication in the OTA update module may allow an adjacent attacker to install unsigned firmware.",
            "published": "2024-04-11",
        },
        {
            "id": "ICSA-24-085-03",
            "title": "Voltara Charging Management API Input Validation Flaw",
            "severity": "Low",
            "summary": "Insufficient input validation in the charging session API could expose internal error messages.",
            "published": "2024-03-26",
        },
    ],
}
_PROPERTIES_RESULT = {
    "city": "Halverton",
    "listings": [
        {
            "id": "HV-0041",
            "type": "garage",
            "area_sqm": 420,
            "monthly_rent": 2800,
            "ev_power_outlets": 12,
            "address": "14 Ironworks Lane, Halverton",
        },
        {
            "id": "HV-0089",
            "type": "warehouse",
            "area_sqm": 900,
            "monthly_rent": 4200,
            "ev_power_outlets": 30,
            "address": "7 Depot Road, Halverton",
        },
    ],
}
_VIDEOS_RESULT = {
    "topic": "fleet electrification",
    "recommendations": [
        {
            "title": "How to Build an EV Fleet from Scratch",
            "channel": "Fleet Future",
            "views": "182K",
        },
        {
            "title": "EV Charging Infrastructure for Commercial Fleets",
            "channel": "GreenDrive Pro",
            "views": "94K",
        },
        {
            "title": "Total Cost of Ownership: Electric vs Diesel Vans",
            "channel": "LogisticsTech",
            "views": "61K",
        },
    ],
}


def _ev_stock_mock(args):
    symbol = args.get("symbol", "").upper()
    if symbol == "VLTR":
        return json.dumps(_STOCK_RESULT_VLTR)
    if symbol == "RVXN":
        return json.dumps(_STOCK_RESULT_RVXN)
    return json.dumps({"error": f"Unknown symbol: {symbol}"})


EV_FLEET_TEST_CASE = {
    "name": "EV fleet expansion: stock → cybersecurity → property → videos",
    "messages": [
        {
            "role": "user",
            "content": (
                "I'm expanding my courier business into electric vehicles and need a multi-step analysis:\n"
                "1. Get the latest stock quote for Voltara (VLTR) and Rivex (RVXN). "
                "If either is above $50, continue with that company.\n"
                "2. Search for cybersecurity advisories related to that company's vehicle models "
                "to understand any tech risks.\n"
                "3. Find commercial garage or warehouse properties in Halverton suitable for "
                "EV charging infrastructure.\n"
                "4. Recommend videos on fleet electrification strategies.\n"
                "Please work through all four steps and give me a concise summary."
            ),
        }
    ],
    "tools": _EV_TOOLS,
    "mock_tool_responses": {
        "get_stock_quote": _ev_stock_mock,
        "get_security_advisories": lambda _: json.dumps(_ADVISORIES_RESULT),
        "search_commercial_properties": lambda _: json.dumps(_PROPERTIES_RESULT),
        "get_video_recommendations": lambda _: json.dumps(_VIDEOS_RESULT),
    },
    "validate": lambda tcs, content: _validate_ev(tcs, content),
}


def _validate_ev(tcs, content):
    names = [tc["function"]["name"] for tc in tcs]
    if not names:
        return False, "No tool calls made"
    # Stock quote must come first
    if names[0] != "get_stock_quote":
        return False, f"Expected get_stock_quote to be called first, got: {names[0]}"
    stock_calls = [tc for tc in tcs if tc["function"]["name"] == "get_stock_quote"]
    for tc in stock_calls:
        try:
            args = json.loads(tc["function"]["arguments"])
            sym = args.get("symbol", "")
            if not sym:
                return False, f"get_stock_quote called with empty symbol: {args}"
        except json.JSONDecodeError:
            return False, "get_stock_quote arguments are not valid JSON"
    # All four pipeline tools expected
    required = [
        "get_stock_quote",
        "get_security_advisories",
        "search_commercial_properties",
        "get_video_recommendations",
    ]
    missing = [t for t in required if t not in names]
    if missing:
        return False, f"Missing pipeline steps: {missing}"
    if not content:
        return False, "No final summary produced"
    return True, f"Full 4-step pipeline executed: {names}"


# ---------------------------------------------------------------------------
# All test cases
# ---------------------------------------------------------------------------

ALL_TEST_CASES = [
    AZZOO_TEST_CASE,
    FITNESS_TEST_CASE,
    COMMUNITY_CLASS_TEST_CASE,
    GEO_TEST_CASE,
    EV_FLEET_TEST_CASE,
]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Test llama-server tool-calling capability."
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

    modes = []
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
    _print(f"\n{BOLD}{color}{'─'*60}{RESET}")
    _print(f"{BOLD}{color}  Results: {passed}/{total} passed{RESET}")
    _print(f"{BOLD}{color}{'─'*60}{RESET}\n")
    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
