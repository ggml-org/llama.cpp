#!/usr/bin/env python
import pytest
import json
from typing import Any, Dict, List, Optional, Set

# ensure grandparent path is in sys.path
from pathlib import Path
import sys
path = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(path))

from utils import *

server: ServerProcess

TIMEOUT_START_SLOW = 15 * 60
TIMEOUT_HTTP_REQUEST = 120


# -- Tool definitions --

LIST_FILES_TOOL = {
    "type": "function",
    "function": {
        "name": "list_files",
        "description": "List files in a directory",
        "parameters": {
            "type": "object",
            "properties": {
                "directory": {
                    "type": "string",
                    "description": "The directory path to list files from"
                }
            },
            "required": ["directory"]
        }
    }
}

CREATE_TODO_TOOL = {
    "type": "function",
    "function": {
        "name": "create_todo",
        "description": "Create a todo list with items for a given task",
        "parameters": {
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": "The title of the todo list"
                },
                "items": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of todo items"
                }
            },
            "required": ["title", "items"]
        }
    }
}

EDIT_FILE_TOOL = {
    "type": "function",
    "function": {
        "name": "edit_file",
        "description": "Edit a file by replacing its contents with new text",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "The file path to edit"
                },
                "content": {
                    "type": "string",
                    "description": "The new content to write to the file"
                }
            },
            "required": ["path", "content"]
        }
    }
}

ALL_TOOLS = [LIST_FILES_TOOL, CREATE_TODO_TOOL, EDIT_FILE_TOOL]
TOOL_NAMES = {tool["function"]["name"] for tool in ALL_TOOLS}

WORKFLOW_FILE_REVIEW = {
    "system": (
        "You are an assistant that uses tools to help with file management tasks. "
        "Always use the provided tools when asked to perform actions. "
        "Be concise in your responses."
    ),
    "prompts": [
        "List the files in the current directory using the list_files tool. Use '.' as the directory.",
        "Now create a todo list for reviewing the codebase. Title it 'Code Review' and include items for each source file you found.",
        "Edit the CHANGES.md file to add a line saying 'Reviewed codebase structure'.",
    ],
}


def mock_tool_call(name: str, arguments: Dict[str, Any]) -> str:
    """Simulate tool execution and return a canned response"""
    if name == "list_files":
        return json.dumps({
            "files": ["README.md", "CHANGES.md", "src/main.py", "src/utils.py", "tests/test_main.py"]
        })
    elif name == "create_todo":
        return json.dumps({
            "status": "created",
            "id": 1,
            "title": arguments.get("title", "untitled"),
            "items_count": len(arguments.get("items", []))
        })
    elif name == "edit_file":
        return json.dumps({
            "status": "success",
            "path": arguments.get("path", "unknown"),
            "bytes_written": len(arguments.get("content", ""))
        })
    else:
        return json.dumps({"error": f"Unknown tool: {name}"})


class AgenticResult:
    def __init__(self, model_id: str = "unknown"):
        self.model_id = model_id
        self.turns: List[Dict[str, Any]] = []
        self.errors: List[str] = []

    @property
    def tool_names_called(self) -> List[str]:
        return [turn["tool_name"] for turn in self.turns if turn.get("tool_name")]

    @property
    def content_turns(self) -> List[Dict[str, Any]]:
        return [turn for turn in self.turns if "content" in turn and "tool_name" not in turn]

    def format_summary(self) -> str:
        """Format a readable summary of the workflow result for test output."""
        lines = [f"\n{'=' * 72}", f"Model: {self.model_id}", f"{'=' * 72}"]

        for turn in self.turns:
            reasoning = turn.get("reasoning_content")
            reasoning_marker = " (with reasoning)" if reasoning else ""

            if "tool_name" in turn:
                args_str = json.dumps(turn["arguments"], indent=None)
                if len(args_str) > 80:
                    args_str = args_str[:75] + "[...]"
                lines.append(f"  Turn {turn['turn']}: TOOL CALL  {turn['tool_name']}({args_str}){reasoning_marker}")
            else:
                content = turn.get("content", "")
                if not isinstance(content, str):
                    lines.append(f"  Turn {turn['turn']}: CONTENT    <unexpected type: {type(content).__name__}>")
                    continue
                preview = content.replace("\n", "\\n")
                if len(preview) > 100:
                    preview = preview[:100] + "[...]"
                lines.append(f"  Turn {turn['turn']}: CONTENT    \"{preview}\"{reasoning_marker}")

        if self.errors:
            lines.append(f"\n  ERRORS ({len(self.errors)}):")
            for err in self.errors:
                lines.append(f"    - {err}")

        # Diagnosis hint
        tool_calls = self.tool_names_called
        content_with_tool_json = []
        for turn in self.content_turns:
            content = turn.get("content", "")
            if isinstance(content, str) and any(name in content for name in TOOL_NAMES):
                content_with_tool_json.append(turn)

        if not tool_calls and content_with_tool_json:
            lines.append(f"\n  DIAGNOSIS: Model appears to be generating tool calls as plain")
            lines.append(f"  text ({len(content_with_tool_json)} turns), but the parser is not")
            lines.append(f"  extracting them. This is likely a parser/template issue.")
        elif not tool_calls:
            lines.append(f"\n  DIAGNOSIS: No tool calls detected. The model may not be")
            lines.append(f"  following the tool calling format for this template.")

        lines.append("")
        return "\n".join(lines)

    def assert_ok(self, min_turns: int = 1, min_tool_calls: int = 1, valid_names: Optional[Set[str]] = None):
        summary = self.format_summary()
        failures = []

        if self.errors:
            failures.append(f"Workflow had {len(self.errors)} error(s)")
        if len(self.turns) < min_turns:
            failures.append(f"Only {len(self.turns)}/{min_turns} turns completed")
        if len(self.tool_names_called) < min_tool_calls:
            failures.append(f"Only {len(self.tool_names_called)}/{min_tool_calls} tool call(s) parsed")
        if valid_names:
            bad = [name for name in self.tool_names_called if name not in valid_names]
            if bad:
                failures.append(f"Unknown tool(s) called: {bad}")

        if failures:
            pytest.fail(f"{'; '.join(failures)}{summary}", pytrace=False)


def run_agentic_workflow(
    server: ServerProcess,
    system_prompt: str,
    user_prompts: List[str],
    tools: List[Dict[str, Any]],
    model_id: str = "unknown",
    max_turns: int = 10,
    valid_tool_names: Optional[Set[str]] = None,
    **kwargs: Any,
) -> AgenticResult:
    """Multi-turn agentic workflow"""
    if valid_tool_names is None:
        valid_tool_names = {tool["function"]["name"] for tool in tools}

    result = AgenticResult(model_id=model_id)
    messages = [{"role": "system", "content": system_prompt}]
    prompt_idx = 0
    turn_count = 0

    messages.append({"role": "user", "content": user_prompts[prompt_idx]})
    prompt_idx += 1

    while turn_count < max_turns:
        turn_count += 1

        try:
            body = server.make_any_request("POST", "/v1/chat/completions", data={
                "max_tokens": 1024,
                "messages": messages,
                "tools": tools,
                "tool_choice": "auto",
                "parallel_tool_calls": False,
                **kwargs,
            }, timeout=TIMEOUT_HTTP_REQUEST)
        except Exception as e:
            result.errors.append(f"Turn {turn_count}: request failed: {e}")
            break

        choice = body["choices"][0]
        assistant_msg = choice["message"]
        tool_calls = assistant_msg.get("tool_calls")

        reasoning_content = assistant_msg.get("reasoning_content")

        if tool_calls:
            assistant_history_msg: Dict[str, Any] = {
                "role": "assistant",
                "content": assistant_msg.get("content"),
                "tool_calls": tool_calls,
            }
            if reasoning_content is not None:
                assistant_history_msg["reasoning_content"] = reasoning_content
            messages.append(assistant_history_msg)

            for tool_call in tool_calls:
                tool_name = tool_call["function"]["name"]
                try:
                    args_str = tool_call["function"]["arguments"]
                    arguments = json.loads(args_str) if isinstance(args_str, str) else args_str
                except (json.JSONDecodeError, TypeError) as e:
                    result.errors.append(
                        f"Turn {turn_count}: invalid JSON arguments for {tool_name}: "
                        f"{tool_call['function']['arguments']!r} ({e})"
                    )
                    arguments = {}

                if tool_name not in valid_tool_names:
                    result.errors.append(
                        f"Turn {turn_count}: unknown tool called: {tool_name}"
                    )

                mock_response = mock_tool_call(tool_name, arguments)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.get("id", f"call_{turn_count}"),
                    "name": tool_name,
                    "content": mock_response,
                })

                result.turns.append({
                    "turn": turn_count,
                    "tool_name": tool_name,
                    "arguments": arguments,
                    "mock_response": mock_response,
                    "reasoning_content": reasoning_content,
                })
        else:
            content = assistant_msg.get("content", "")
            if content is not None and not isinstance(content, str):
                result.errors.append(
                    f"Turn {turn_count}: content is {type(content).__name__}, expected str: {content!r}"
                )
                content = str(content)
            result.turns.append({
                "turn": turn_count,
                "content": content,
                "reasoning_content": reasoning_content,
            })
            assistant_history_msg = {
                "role": "assistant",
                "content": content,
            }
            if reasoning_content is not None:
                assistant_history_msg["reasoning_content"] = reasoning_content
            messages.append(assistant_history_msg)

            if prompt_idx < len(user_prompts):
                messages.append({"role": "user", "content": user_prompts[prompt_idx]})
                prompt_idx += 1
            else:
                break

    return result


def do_test_agentic_workflow(server: ServerProcess, model_id: str, **kwargs: Any) -> None:
    result = run_agentic_workflow(
        server,
        system_prompt=WORKFLOW_FILE_REVIEW["system"],
        user_prompts=WORKFLOW_FILE_REVIEW["prompts"],
        tools=ALL_TOOLS,
        model_id=model_id,
        max_turns=10,
        **kwargs,
    )
    result.assert_ok(min_turns=3, min_tool_calls=1, valid_names=TOOL_NAMES)


def make_qwen3_5_08b() -> ServerProcess:
    server = ServerProcess()
    server.offline = False
    server.model_hf_repo = "bartowski/Qwen_Qwen3.5-0.8B-GGUF"
    server.model_hf_file = None
    server.model_alias = "qwen3.5-0.8b"
    server.n_ctx = 8192
    server.n_batch = 2048
    server.n_slots = 1
    server.n_predict = 1024
    server.temperature = 0.0
    server.seed = 42
    return server


@pytest.fixture(autouse=True)
def create_server():
    global server
    server = make_qwen3_5_08b()
    server.server_port = 8082


# -- Fast tests (Qwen3.5-0.8B with its own template) --

def test_agentic_single_tool_call():
    """Verify the parser handles a tool call with a small model for fast testing"""
    global server
    server.jinja = True
    server.start(timeout_seconds=TIMEOUT_START_SLOW)
    result = run_agentic_workflow(
        server,
        system_prompt=WORKFLOW_FILE_REVIEW["system"],
        user_prompts=["List the files in the current directory using the list_files tool. Use '.' as the directory."],
        tools=ALL_TOOLS,
        model_id="Qwen3.5-0.8B",
        max_turns=3,
        temperature=0.0,
        top_k=1,
        top_p=1.0,
        tool_choice="required",
    )
    result.assert_ok(min_turns=1, min_tool_calls=1, valid_names=TOOL_NAMES)


def test_agentic_multi_turn():
    """Verify multi-turn tool calling works: tool call -> mock result -> next tool call"""
    global server
    server.jinja = True
    server.start(timeout_seconds=TIMEOUT_START_SLOW)
    do_test_agentic_workflow(server, model_id="Qwen3.5-0.8B",
                             temperature=0.0, top_k=1, top_p=1.0)


# -- Slow tests (real models, full agentic workflow) --

def _apply_template_override(server: ServerProcess, template_override: Any) -> None:
    """
    Apply a template override to the server

    template_override can be:
      None                       - use model's built-in template
      str                        - use a named template (e.g. "chatml")
      Tuple[str, Optional[str]]  - (hf_repo, variant) to load a .jinja file
    """
    if isinstance(template_override, tuple):
        template_hf_repo, template_variant = template_override
        server.chat_template_file = (
            f"../../../models/templates/"
            f"{template_hf_repo.replace('/', '-')}"
            f"{'-' + template_variant if template_variant else ''}.jinja"
        )
        assert os.path.exists(server.chat_template_file), \
            f"Template file {server.chat_template_file} does not exist."
    elif isinstance(template_override, str):
        server.chat_template = template_override


@pytest.mark.slow
@pytest.mark.parametrize("hf_repo,template_override", [
    ("bartowski/Qwen2.5-7B-Instruct-GGUF:Q4_K_M",             None),
    ("bartowski/Meta-Llama-3.1-8B-Instruct-GGUF:Q4_K_M",      None),
    ("bartowski/Qwen2.5-Coder-3B-Instruct-GGUF:Q4_K_M",       None),
    ("mradermacher/Hermes-3-Llama-3.1-8B-GGUF:Q4_K_M",        ("NousResearch/Hermes-3-Llama-3.1-8B", "tool_use")),
    ("bartowski/DeepSeek-R1-Distill-Qwen-7B-GGUF:Q4_K_M",     None),
    ("bartowski/Llama-3.2-3B-Instruct-GGUF:Q4_K_M",           ("meta-llama/Llama-3.2-3B-Instruct", None)),
    ("bartowski/arcee-ai_Trinity-Mini-GGUF:Q4_K_M",            None),
], ids=[
    "Qwen2.5-7B",
    "Llama-3.1-8B",
    "Qwen2.5-Coder-3B",
    "Hermes-3-8B",
    "DeepSeek-R1-Distill-7B",
    "Llama-3.2-3B",
    "Trinity-Mini-26B",
])
def test_agentic_workflow_real_model(hf_repo: str, template_override: Any):
    """Full multi-turn agentic workflow with real models"""
    global server
    server.jinja = True
    server.offline = False
    server.n_ctx = 8192
    server.n_predict = 1024
    server.model_hf_repo = hf_repo
    server.model_hf_file = None
    _apply_template_override(server, template_override)
    server.start(timeout_seconds=TIMEOUT_START_SLOW)
    do_test_agentic_workflow(server, model_id=hf_repo,
                             temperature=0.0, top_k=1, top_p=1.0)


@pytest.mark.slow
@pytest.mark.parametrize("hf_repo,template_override", [
    ("bartowski/Qwen2.5-7B-Instruct-GGUF:Q4_K_M",        None),
    ("bartowski/Meta-Llama-3.1-8B-Instruct-GGUF:Q4_K_M",  None),
], ids=[
    "Qwen2.5-7B",
    "Llama-3.1-8B",
])
def test_agentic_workflow_streamed_real_model(hf_repo: str, template_override: Any):
    """Same workflow but with streaming, to test streamed tool call parsing across turns"""
    global server
    server.jinja = True
    server.offline = False
    server.n_ctx = 8192
    server.n_predict = 1024
    server.model_hf_repo = hf_repo
    server.model_hf_file = None
    _apply_template_override(server, template_override)
    server.start(timeout_seconds=TIMEOUT_START_SLOW)
    do_test_agentic_workflow(server, model_id=f"{hf_repo} (streamed)",
                             temperature=0.0, top_k=1, top_p=1.0, stream=True)
