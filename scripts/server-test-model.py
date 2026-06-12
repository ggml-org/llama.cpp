import argparse
import json
import requests
import logging
import sys

handler = logging.StreamHandler(sys.stdout)
handler.terminator = ""   # ← no newline
logging.basicConfig(level=logging.INFO, format='%(message)s', handlers=[handler])
logger = logging.getLogger("server-test-model")


def run_query(url, messages, tools=None, stream=False, tool_choice=None):
    payload = {
        "messages": messages,
        "stream": stream,
        "max_tokens": 5000,
    }
    if tools:
        payload["tools"] = tools
    if tool_choice:
        payload["tool_choice"] = tool_choice

    try:
        response = requests.post(url, json=payload, stream=stream)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        if e.response is not None:
            logger.info(f"Response error: {e} for {e.response.content}\n")
        else:
            logger.info(f"Error connecting to server: {e}\n")
        return None

    full_content = ""
    reasoning_content = ""
    tool_calls = []

    if stream:
        logger.info(f"--- Streaming response (Tools: {bool(tools)}) ---\n")
        for line in response.iter_lines():
            if line:
                decoded_line = line.decode("utf-8")
                if decoded_line.startswith("data: "):
                    data_str = decoded_line[6:]
                    if data_str == "[DONE]":
                        break
                    try:
                        data = json.loads(data_str)
                        if "choices" in data and len(data["choices"]) > 0:
                            delta = data["choices"][0].get("delta", {})

                            # Content
                            content_chunk = delta.get("content", "")
                            if content_chunk:
                                full_content += content_chunk
                                logger.info(content_chunk)

                            # Reasoning
                            reasoning_chunk = delta.get("reasoning_content", "")
                            if reasoning_chunk:
                                reasoning_content += reasoning_chunk
                                logger.info(f"\x1B[3m{reasoning_chunk}\x1B[0m")

                            # Tool calls
                            if "tool_calls" in delta:
                                for tc in delta["tool_calls"]:
                                    index = tc.get("index")
                                    if index is not None:
                                        while len(tool_calls) <= index:
                                            # Using "function" as type default but could be flexible
                                            tool_calls.append(
                                                {
                                                    "id": "",
                                                    "type": "function",
                                                    "function": {
                                                        "name": "",
                                                        "arguments": "",
                                                    },
                                                }
                                            )

                                        if "id" in tc:
                                            tool_calls[index]["id"] += tc["id"]
                                        if "function" in tc:
                                            if "name" in tc["function"]:
                                                tool_calls[index]["function"][
                                                    "name"
                                                ] += tc["function"]["name"]
                                            if "arguments" in tc["function"]:
                                                tool_calls[index]["function"][
                                                    "arguments"
                                                ] += tc["function"]["arguments"]

                    except json.JSONDecodeError:
                        logger.info(f"Failed to decode JSON: {data_str}\n")
        logger.info("\n--- End of Stream ---\n")
    else:
        logger.info(f"--- Non-streaming response (Tools: {bool(tools)}) ---\n")
        data = response.json()
        if "choices" in data and len(data["choices"]) > 0:
            message = data["choices"][0].get("message", {})
            full_content = message.get("content", "")
            reasoning_content = message.get("reasoning_content", "")
            tool_calls = message.get("tool_calls", [])
            logger.info(full_content)
        logger.info("--- End of Response ---\n")

    return {
        "content": full_content,
        "reasoning_content": reasoning_content,
        "tool_calls": tool_calls,
    }


def test_chat(url, stream):
    logger.info(f"\n=== Testing Chat (Stream={stream}) ===\n")
    messages = [{"role": "user", "content": "What is the capital of France?"}]
    result = run_query(url, messages, stream=stream)

    if result:
        if result["content"]:
            logger.info("PASS: Output received.\n")
        else:
            logger.info("WARN: No content received (valid if strict tool call, but unexpected here).\n")

        if result.get("reasoning_content"):
            logger.info(f"INFO: Reasoning content detected ({len(result['reasoning_content'])} chars).\n")
        else:
            logger.info("INFO: No reasoning content detected (Standard model behavior).\n")
    else:
        logger.info("FAIL: No result.\n")


def test_tool_call(url, stream):
    logger.info(f"\n=== Testing Tool Call (Stream={stream}) ===\n")
    messages = [
        {
            "role": "user",
            "content": "What is the weather in London? Please use the get_weather tool.",
        }
    ]
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the current weather in a given location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA",
                        },
                        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                    },
                    "required": ["location"],
                },
            },
        }
    ]

    result = run_query(url, messages, tools=tools, tool_choice="auto", stream=stream)

    if result:
        tcs = result.get("tool_calls")
        if tcs and len(tcs) > 0:
            logger.info("PASS: Tool calls detected.")
            for tc in tcs:
                func = tc.get("function", {})
                logger.info(f"  Tool: {func.get('name')}, Args: {func.get('arguments')}\n")
        else:
            logger.info(f"FAIL: No tool calls. Content: {result['content']}\n")

        if result.get("reasoning_content"):
            logger.info(
                f"INFO: Reasoning content detected during tool call ({len(result['reasoning_content'])} chars).\n"
            )
    else:
        logger.info("FAIL: Query failed.\n")


def main():
    parser = argparse.ArgumentParser(description="Test llama-server functionality.")
    parser.add_argument("--host", default="localhost", help="Server host")
    parser.add_argument("--port", default=8080, type=int, help="Server port")
    args = parser.parse_args()

    base_url = f"http://{args.host}:{args.port}/v1/chat/completions"
    logger.info(f"Testing server at {base_url}\n")

    # Non-streaming tests
    test_chat(base_url, stream=False)
    test_tool_call(base_url, stream=False)

    # Streaming tests
    test_chat(base_url, stream=True)
    test_tool_call(base_url, stream=True)


if __name__ == "__main__":
    main()
