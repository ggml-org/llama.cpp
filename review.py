import requests
import os
import json
import sys

API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    print("GEMINI_API_KEY not set.")
    sys.exit(1)

MODEL = "gemini-2.5-flash"

# ---------- Read diff ----------
try:
    with open("diff.txt", "r") as f:
        diff = f.read()
except FileNotFoundError:
    print("diff.txt not found.")
    sys.exit(1)

if not diff.strip():
    print("Empty diff.")
    sys.exit(0)

# ---------- Clean diff ----------
def clean_diff(diff):
    lines = diff.split("\n")
    filtered = []
    for line in lines:
        if line.startswith("diff --git"):
            continue
        if line.startswith("index "):
            continue
        if line.startswith("--- ") or line.startswith("+++ "):
            continue
        filtered.append(line)
    return "\n".join(filtered)

diff = clean_diff(diff)

# ---------- Truncate ----------
MAX_CHARS = 8000
if len(diff) > MAX_CHARS:
    diff = diff[:MAX_CHARS] + "\n... [TRUNCATED]"

# ---------- Prompt ----------
prompt = f"""
Return ONLY raw JSON. Do NOT wrap in markdown.
Do NOT use ```.

Format:
{{
  "summary": "",
  "critical_issues": [],
  "bugs": [],
  "security": [],
  "performance": [],
  "code_quality": [],
  "tech_debt": [],
  "suggestions": []
}}

Rules:
- No text outside JSON
- No praise
- Only real issues
- Be concise
- Tech debt = shortcuts, bad design, maintainability risks, missing abstractions, hacks

Review this git diff:
{diff}
"""

# ---------- API call ----------
url = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent?key={API_KEY}"

payload = {
    "contents": [
        {
            "parts": [{"text": prompt}]
        }
    ],
    "generationConfig": {
        "temperature": 0.1,
    }
}

try:
    response = requests.post(url, json=payload, timeout=60)
    response.raise_for_status()
except requests.exceptions.RequestException as e:
    print(f"API error: {e}")
    sys.exit(1)

data = response.json()

# ---------- Extract ----------
try:
    content = data["candidates"][0]["content"]["parts"][0]["text"]
except (KeyError, IndexError):
    print("Invalid response format.")
    print(json.dumps(data, indent=2))
    sys.exit(1)

# ---------- Parse JSON ----------
import re

def extract_json(text):
    text = text.strip()

    # Case 1: ```json ... ```
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence_match:
        return fence_match.group(1)

    # Case 2: any JSON object in text
    brace_match = re.search(r"\{.*\}", text, re.DOTALL)
    if brace_match:
        return brace_match.group(0)

    return None


json_str = extract_json(content)

if not json_str:
    parsed = {
        "summary": "No JSON found",
        "raw_output": content
    }
else:
    try:
        parsed = json.loads(json_str)
    except json.JSONDecodeError:
        parsed = {
            "summary": "Invalid JSON after extraction",
            "raw_output": json_str
        }

# ---------- Save JSON ----------
with open("review.json", "w") as f:
    json.dump(parsed, f, indent=2)

# ---------- Format ----------
def format_review(r):
    lines = []
    lines.append("## LLM Code Review\n")
    lines.append(f"**Summary:** {r.get('summary','')}\n")

    def section(title, items):
        if items:
            lines.append(f"### {title}")
            for i, item in enumerate(items, 1):
                lines.append(f"{i}. {item}")
            lines.append("")

    section("Critical Issues", r.get("critical_issues", []))
    section("Bugs", r.get("bugs", []))
    section("Security", r.get("security", []))
    section("Performance", r.get("performance", []))
    section("Code Quality", r.get("code_quality", []))
    section("Suggestions", r.get("suggestions", []))

    return "\n".join(lines)

formatted = format_review(parsed)

# ---------- Save ----------
with open("review.txt", "w") as f:
    f.write(formatted)

print(formatted)