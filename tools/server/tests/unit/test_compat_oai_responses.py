import pytest
from openai import OpenAI
from utils import *

server: ServerProcess

@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.tinyllama2()

def test_responses_with_openai_library():
    global server
    server.start()
    client = OpenAI(api_key="dummy", base_url=f"http://{server.server_host}:{server.server_port}/v1")
    res = client.responses.create(
        model="gpt-4.1",
        input=[
            {"role": "system", "content": "Book"},
            {"role": "user", "content": "What is the best book"},
        ],
        max_output_tokens=8,
        temperature=0.8,
    )
    assert res.id.startswith("resp_")
    assert res.output[0].id is not None
    assert res.output[0].id.startswith("msg_")
    assert match_regex("(Suddenly)+", res.output_text)

def test_responses_stream_with_openai_library():
    global server
    server.start()
    client = OpenAI(api_key="dummy", base_url=f"http://{server.server_host}:{server.server_port}/v1")
    stream = client.responses.create(
        model="gpt-4.1",
        input=[
            {"role": "system", "content": "Book"},
            {"role": "user", "content": "What is the best book"},
        ],
        max_output_tokens=8,
        temperature=0.8,
        stream=True,
    )

    gathered_text = ''
    resp_id = ''
    msg_id = ''
    for r in stream:
        if r.type == "response.created":
            assert r.response.id.startswith("resp_")
            resp_id = r.response.id
        if r.type == "response.in_progress":
            assert r.response.id == resp_id
        if r.type == "response.output_item.added":
            assert r.item.id is not None
            assert r.item.id.startswith("msg_")
            msg_id = r.item.id
        if (r.type == "response.content_part.added" or
            r.type == "response.output_text.delta" or
            r.type == "response.output_text.done" or
            r.type == "response.content_part.done"):
            assert r.item_id == msg_id
        if r.type == "response.output_item.done":
            assert r.item.id == msg_id

        if r.type == "response.output_text.delta":
            gathered_text += r.delta
        if r.type == "response.completed":
            assert r.response.id.startswith("resp_")
            assert r.response.output[0].id is not None
            assert r.response.output[0].id.startswith("msg_")
            assert gathered_text == r.response.output_text
            assert match_regex("(Suddenly)+", r.response.output_text)


def test_responses_schema_fields():
    """Verify the 24 Response object fields added by this PR are present
    with correct types and default values. These fields are required by
    the OpenAI Responses API spec but were missing before this change."""
    global server
    server.start()
    res = server.make_request("POST", "/v1/responses", data={
        "model": "gpt-4.1",
        "input": "Book",
        "max_output_tokens": 8,
        "temperature": 0.8,
    })
    assert res.status_code == 200
    body = res.body
    # Usage sub-fields added by this PR
    usage = body["usage"]
    assert isinstance(usage["input_tokens_details"]["cached_tokens"], int)
    assert isinstance(usage["output_tokens_details"]["reasoning_tokens"], int)
    # All 24 fields added by this PR must be present with correct defaults
    assert body["incomplete_details"] is None
    assert body["previous_response_id"] is None
    assert body["instructions"] is None
    assert body["error"] is None
    assert body["tools"] == []
    assert body["tool_choice"] == "auto"
    assert body["truncation"] == "disabled"
    assert body["parallel_tool_calls"] == False
    assert body["text"] == {"format": {"type": "text"}}
    assert body["top_p"] == 1.0
    assert body["temperature"] == 1.0
    assert body["presence_penalty"] == 0.0
    assert body["frequency_penalty"] == 0.0
    assert body["top_logprobs"] == 0
    assert body["reasoning"] is None
    assert body["max_output_tokens"] is None
    assert body["store"] == False
    assert body["service_tier"] == "default"
    assert body["metadata"] == {}
    assert body["background"] == False
    assert body["safety_identifier"] is None
    assert body["prompt_cache_key"] is None
    assert body["max_tool_calls"] is None


def test_responses_stream_schema_fields():
    """Verify streaming done-events have the sequence_number, output_index,
    and content_index fields added by this PR. Also verify the completed
    response includes the 24 new schema fields."""
    global server
    server.start()
    res = server.make_stream_request("POST", "/v1/responses", data={
        "model": "gpt-4.1",
        "input": "Book",
        "max_output_tokens": 8,
        "temperature": 0.8,
        "stream": True,
    })
    seen_seq_nums = []
    saw_output_text_done = False
    saw_content_part_done = False
    saw_output_item_done = False
    completed_response = None
    for data in res:
        assert "sequence_number" in data, f"missing sequence_number in {data.get('type')}"
        seen_seq_nums.append(data["sequence_number"])
        if data.get("type") == "response.output_text.done":
            saw_output_text_done = True
            assert "content_index" in data
            assert "output_index" in data
            assert "logprobs" in data
            assert isinstance(data["logprobs"], list)
        if data.get("type") == "response.content_part.done":
            saw_content_part_done = True
            assert "content_index" in data
            assert "output_index" in data
        if data.get("type") == "response.output_item.done":
            saw_output_item_done = True
            assert "output_index" in data
        if data.get("type") == "response.completed":
            completed_response = data["response"]
    # Must have seen all done-event types
    assert saw_output_text_done, "never received response.output_text.done"
    assert saw_content_part_done, "never received response.content_part.done"
    assert saw_output_item_done, "never received response.output_item.done"
    # sequence_number must be present on done events and monotonically increasing
    assert len(seen_seq_nums) >= 4, f"expected >= 4 sequenced events, got {len(seen_seq_nums)}"
    assert all(a < b for a, b in zip(seen_seq_nums, seen_seq_nums[1:])), "sequence_numbers not strictly increasing"
    # completed response must have the new schema fields with correct values
    assert completed_response is not None
    assert completed_response["metadata"] == {}
    assert completed_response["store"] == False
    assert completed_response["truncation"] == "disabled"
    assert completed_response["usage"]["output_tokens_details"]["reasoning_tokens"] == 0


def test_responses_non_function_tool_skipped():
    """Non-function tool types must be silently skipped, producing a valid
    completion with no tools field in the converted chat request. Upstream
    rejects non-function types with 400; our code must return 200 and
    generate output as if no tools were provided."""
    global server
    server.start()
    res = server.make_request("POST", "/v1/responses", data={
        "model": "gpt-4.1",
        "input": [
            {"role": "system", "content": "Book"},
            {"role": "user", "content": "What is the best book"},
        ],
        "max_output_tokens": 8,
        "temperature": 0.8,
        "tools": [
            {"type": "web_search"},
            {"type": "code_interpreter"},
        ],
    })
    assert res.status_code == 200
    assert res.body["status"] == "completed"
    # With all tools skipped, the model must still produce text output
    assert len(res.body["output"]) > 0
    assert len(res.body["output_text"]) > 0


def test_responses_only_non_function_tools_same_as_no_tools():
    """When ALL tools are non-function types, they should all be filtered out
    and the result should be identical to a request with no tools at all.
    Compare token counts to confirm the tools field was truly empty."""
    global server
    server.start()
    no_tools = server.make_request("POST", "/v1/responses", data={
        "model": "gpt-4.1",
        "input": [
            {"role": "system", "content": "Book"},
            {"role": "user", "content": "What is the best book"},
        ],
        "max_output_tokens": 8,
        "temperature": 0.8,
    })
    with_skipped_tools = server.make_request("POST", "/v1/responses", data={
        "model": "gpt-4.1",
        "input": [
            {"role": "system", "content": "Book"},
            {"role": "user", "content": "What is the best book"},
        ],
        "max_output_tokens": 8,
        "temperature": 0.8,
        "tools": [
            {"type": "web_search"},
            {"type": "code_interpreter"},
            {"type": "file_search"},
        ],
    })
    assert no_tools.status_code == 200
    assert with_skipped_tools.status_code == 200
    # If tools were truly stripped, prompt token count must be identical
    assert with_skipped_tools.body["usage"]["input_tokens"] == no_tools.body["usage"]["input_tokens"]


def test_responses_extra_keys_stripped():
    """Responses-only request keys (store, include, prompt_cache_key, etc.)
    must be stripped before forwarding to the chat completions handler.
    The completion must succeed and produce the same output as a request
    without those keys."""
    global server
    server.start()
    # Baseline without extra keys
    baseline = server.make_request("POST", "/v1/responses", data={
        "model": "gpt-4.1",
        "input": [
            {"role": "system", "content": "Book"},
            {"role": "user", "content": "What is the best book"},
        ],
        "max_output_tokens": 8,
        "temperature": 0.8,
    })
    assert baseline.status_code == 200
    # Same request with extra Responses-only keys
    res = server.make_request("POST", "/v1/responses", data={
        "model": "gpt-4.1",
        "input": [
            {"role": "system", "content": "Book"},
            {"role": "user", "content": "What is the best book"},
        ],
        "max_output_tokens": 8,
        "temperature": 0.8,
        "store": True,
        "include": ["usage"],
        "prompt_cache_key": "test_key",
        "web_search": {"enabled": True},
        "text": {"format": {"type": "text"}},
        "truncation": "auto",
        "metadata": {"key": "value"},
    })
    assert res.status_code == 200
    assert res.body["status"] == "completed"
    # Extra keys must not affect token consumption
    assert res.body["usage"]["input_tokens"] == baseline.body["usage"]["input_tokens"]


def test_responses_developer_role_merging():
    """Developer role messages must be merged into the first system message
    at position 0. This ensures templates that require a single system
    message don't see developer content as a separate turn.

    We verify by comparing token counts: system + developer merged should
    consume the same prompt tokens as a single system message with the
    combined content."""
    global server
    server.start()
    # Single combined system message
    combined = server.make_request("POST", "/v1/responses", data={
        "model": "gpt-4.1",
        "input": [
            {"role": "system", "content": [
                {"type": "input_text", "text": "Book"},
                {"type": "input_text", "text": "Keep it short"},
            ]},
            {"role": "user", "content": [{"type": "input_text", "text": "What is the best book"}]},
        ],
        "max_output_tokens": 8,
        "temperature": 0.8,
    })
    assert combined.status_code == 200
    # Split system + developer (should be merged to same prompt)
    split = server.make_request("POST", "/v1/responses", data={
        "model": "gpt-4.1",
        "input": [
            {"role": "system", "content": [{"type": "input_text", "text": "Book"}]},
            {"role": "user", "content": [{"type": "input_text", "text": "What is the best book"}]},
            {"role": "developer", "content": [{"type": "input_text", "text": "Keep it short"}]},
        ],
        "max_output_tokens": 8,
        "temperature": 0.8,
    })
    assert split.status_code == 200
    assert split.body["status"] == "completed"
    # Merged prompt should consume same number of input tokens
    assert split.body["usage"]["input_tokens"] == combined.body["usage"]["input_tokens"]


def test_responses_input_text_type_multi_turn():
    """input_text type must be accepted for assistant messages (EasyInputMessage).
    An assistant message without explicit type:'message' must also be accepted
    (AssistantMessageItemParam). Verify the multi-turn context is preserved
    by checking the model sees the full conversation."""
    global server
    server.start()
    res = server.make_request("POST", "/v1/responses", data={
        "model": "gpt-4.1",
        "input": [
            {"role": "user", "content": [{"type": "input_text", "text": "Hello"}]},
            {
                "role": "assistant",
                "content": [{"type": "input_text", "text": "Hi there"}],
            },
            {"role": "user", "content": [{"type": "input_text", "text": "How are you"}]},
        ],
        "max_output_tokens": 8,
        "temperature": 0.8,
    })
    assert res.status_code == 200
    assert res.body["status"] == "completed"
    # Multi-turn input should result in more prompt tokens than single-turn
    single = server.make_request("POST", "/v1/responses", data={
        "model": "gpt-4.1",
        "input": "How are you",
        "max_output_tokens": 8,
        "temperature": 0.8,
    })
    assert single.status_code == 200
    assert res.body["usage"]["input_tokens"] > single.body["usage"]["input_tokens"]


def test_responses_output_text_matches_content():
    """output_text must be the concatenation of all output_text content parts.
    Verify this for both streaming and non-streaming responses."""
    global server
    server.start()
    # Non-streaming
    res = server.make_request("POST", "/v1/responses", data={
        "model": "gpt-4.1",
        "input": [
            {"role": "system", "content": "Book"},
            {"role": "user", "content": "What is the best book"},
        ],
        "max_output_tokens": 8,
        "temperature": 0.8,
    })
    assert res.status_code == 200
    # Manually reconstruct output_text from content parts
    reconstructed = ""
    for item in res.body["output"]:
        if item.get("type") == "message":
            for part in item["content"]:
                if part.get("type") == "output_text":
                    reconstructed += part["text"]
    assert res.body["output_text"] == reconstructed
    assert len(reconstructed) > 0


def test_responses_stream_output_text_consistency():
    """Streaming gathered text must match the output_text in response.completed."""
    global server
    server.start()
    res = server.make_stream_request("POST", "/v1/responses", data={
        "model": "gpt-4.1",
        "input": [
            {"role": "system", "content": "Book"},
            {"role": "user", "content": "What is the best book"},
        ],
        "max_output_tokens": 8,
        "temperature": 0.8,
        "stream": True,
    })
    gathered_text = ""
    completed_output_text = None
    for data in res:
        if data.get("type") == "response.output_text.delta":
            gathered_text += data["delta"]
        if data.get("type") == "response.completed":
            completed_output_text = data["response"]["output_text"]
            # Also verify content parts match
            for item in data["response"]["output"]:
                if item.get("type") == "message":
                    for part in item["content"]:
                        if part.get("type") == "output_text":
                            assert part["text"] == gathered_text
    assert completed_output_text is not None
    assert gathered_text == completed_output_text
    assert len(gathered_text) > 0


def test_responses_stream_created_event_has_full_response():
    """response.created must contain the full response object with all required
    fields, not just {id, object, status}. This is needed by strict client
    libraries like async-openai."""
    global server
    server.start()
    res = server.make_stream_request("POST", "/v1/responses", data={
        "model": "gpt-4.1",
        "input": [
            {"role": "system", "content": "Book"},
            {"role": "user", "content": "What is the best book"},
        ],
        "max_output_tokens": 8,
        "temperature": 0.8,
        "stream": True,
    })
    created_resp = None
    in_progress_resp = None
    for data in res:
        if data.get("type") == "response.created":
            created_resp = data["response"]
        if data.get("type") == "response.in_progress":
            in_progress_resp = data["response"]
    assert created_resp is not None, "never received response.created"
    assert in_progress_resp is not None, "never received response.in_progress"
    # Both must have the full response object, not just minimal fields
    for resp in [created_resp, in_progress_resp]:
        assert resp["status"] == "in_progress"
        assert resp["id"].startswith("resp_")
        assert resp["object"] == "response"
        assert resp["model"] is not None
        assert resp["completed_at"] is None
        assert resp["metadata"] == {}
        assert resp["store"] == False
        assert resp["truncation"] == "disabled"
        assert resp["tools"] == []
        assert resp["usage"]["input_tokens"] == 0
        assert resp["usage"]["output_tokens"] == 0
        assert resp["output"] == []
        assert resp["output_text"] == ""


def test_responses_stream_all_events_have_sequence_number():
    """Every streaming event must have a sequence_number field and they must
    be strictly increasing across the entire stream."""
    global server
    server.start()
    res = server.make_stream_request("POST", "/v1/responses", data={
        "model": "gpt-4.1",
        "input": [
            {"role": "system", "content": "Book"},
            {"role": "user", "content": "What is the best book"},
        ],
        "max_output_tokens": 8,
        "temperature": 0.8,
        "stream": True,
    })
    all_seq_nums = []
    event_types = []
    for data in res:
        assert "sequence_number" in data, f"missing sequence_number in event type {data.get('type')}"
        all_seq_nums.append(data["sequence_number"])
        event_types.append(data.get("type", "unknown"))
    # Must have received multiple events
    assert len(all_seq_nums) >= 6, f"expected >= 6 events, got {len(all_seq_nums)}: {event_types}"
    # Must be strictly increasing
    for i in range(1, len(all_seq_nums)):
        assert all_seq_nums[i] > all_seq_nums[i-1], \
            f"sequence_number not strictly increasing at index {i}: {all_seq_nums[i-1]} -> {all_seq_nums[i]} (events: {event_types[i-1]} -> {event_types[i]})"


def test_responses_stream_delta_events_have_indices():
    """Delta and added events must have output_index. Content-related events
    must also have content_index."""
    global server
    server.start()
    res = server.make_stream_request("POST", "/v1/responses", data={
        "model": "gpt-4.1",
        "input": [
            {"role": "system", "content": "Book"},
            {"role": "user", "content": "What is the best book"},
        ],
        "max_output_tokens": 8,
        "temperature": 0.8,
        "stream": True,
    })
    saw_output_item_added = False
    saw_content_part_added = False
    saw_output_text_delta = False
    for data in res:
        evt = data.get("type", "")
        if evt == "response.output_item.added":
            saw_output_item_added = True
            assert "output_index" in data, "output_item.added missing output_index"
        if evt == "response.content_part.added":
            saw_content_part_added = True
            assert "output_index" in data, "content_part.added missing output_index"
            assert "content_index" in data, "content_part.added missing content_index"
        if evt == "response.output_text.delta":
            saw_output_text_delta = True
            assert "output_index" in data, "output_text.delta missing output_index"
            assert "content_index" in data, "output_text.delta missing content_index"
    assert saw_output_item_added, "never received response.output_item.added"
    assert saw_content_part_added, "never received response.content_part.added"
    assert saw_output_text_delta, "never received response.output_text.delta"


def test_responses_reasoning_content_array():
    """Reasoning items with content as array (spec format) must be accepted."""
    global server
    server.start()
    res = server.make_request("POST", "/v1/responses", data={
        "model": "gpt-4.1",
        "input": [
            {"role": "user", "content": [{"type": "input_text", "text": "Hi"}]},
            {"type": "reasoning", "summary": [],
             "content": [{"type": "reasoning_text", "text": "thinking"}]},
            {"role": "assistant", "type": "message",
             "content": [{"type": "output_text", "text": "Hello"}]},
            {"role": "user", "content": [{"type": "input_text", "text": "How are you"}]},
        ],
        "max_output_tokens": 8,
        "temperature": 0.8,
    })
    assert res.status_code == 200
    assert res.body["status"] == "completed"


def test_responses_reasoning_content_string():
    """Reasoning items with content as plain string (OpenCode format) must be accepted."""
    global server
    server.start()
    res = server.make_request("POST", "/v1/responses", data={
        "model": "gpt-4.1",
        "input": [
            {"role": "user", "content": [{"type": "input_text", "text": "Hi"}]},
            {"type": "reasoning", "summary": [], "content": "thinking about it"},
            {"role": "assistant", "type": "message",
             "content": [{"type": "output_text", "text": "Hello"}]},
            {"role": "user", "content": [{"type": "input_text", "text": "How are you"}]},
        ],
        "max_output_tokens": 8,
        "temperature": 0.8,
    })
    assert res.status_code == 200
    assert res.body["status"] == "completed"


def test_responses_reasoning_content_null():
    """Reasoning items with content:null (Codex format, issue openai/codex#11834)
    must be accepted — content may be null when encrypted_content is present."""
    global server
    server.start()
    res = server.make_request("POST", "/v1/responses", data={
        "model": "gpt-4.1",
        "input": [
            {"role": "user", "content": [{"type": "input_text", "text": "Hi"}]},
            {"type": "reasoning", "summary": [], "content": None,
             "encrypted_content": "opaque_data_here"},
            {"role": "assistant", "type": "message",
             "content": [{"type": "output_text", "text": "Hello"}]},
            {"role": "user", "content": [{"type": "input_text", "text": "How are you"}]},
        ],
        "max_output_tokens": 8,
        "temperature": 0.8,
    })
    assert res.status_code == 200
    assert res.body["status"] == "completed"


def test_responses_reasoning_content_omitted():
    """Reasoning items with content omitted entirely must be accepted."""
    global server
    server.start()
    res = server.make_request("POST", "/v1/responses", data={
        "model": "gpt-4.1",
        "input": [
            {"role": "user", "content": [{"type": "input_text", "text": "Hi"}]},
            {"type": "reasoning", "summary": []},
            {"role": "assistant", "type": "message",
             "content": [{"type": "output_text", "text": "Hello"}]},
            {"role": "user", "content": [{"type": "input_text", "text": "How are you"}]},
        ],
        "max_output_tokens": 8,
        "temperature": 0.8,
    })
    assert res.status_code == 200
    assert res.body["status"] == "completed"
