#!/usr/bin/env python3
"""
Function Calling Agent

This example demonstrates:
- Tool/function definitions
- Function calling with LLMs
- Multi-step reasoning
- Error handling
"""

from llama_cpp import Llama
from typing import List, Dict, Any, Callable
import json
import re
from datetime import datetime


class Tool:
    """Tool definition for function calling."""

    def __init__(
        self,
        name: str,
        description: str,
        parameters: Dict[str, Any],
        function: Callable
    ):
        self.name = name
        self.description = description
        self.parameters = parameters
        self.function = function

    def execute(self, **kwargs) -> Any:
        """Execute the tool function."""
        try:
            return self.function(**kwargs)
        except Exception as e:
            return {"error": str(e)}

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters
        }


# Example tool functions
def get_current_time() -> str:
    """Get the current time."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def calculate(expression: str) -> float:
    """
    Evaluate a mathematical expression.

    Args:
        expression: Math expression (e.g., "2 + 2", "10 * 5")

    Returns:
        Result of calculation
    """
    try:
        # Safe evaluation (limited scope)
        result = eval(expression, {"__builtins__": {}}, {})
        return float(result)
    except Exception as e:
        return f"Error: {e}"


def get_weather(location: str) -> Dict[str, Any]:
    """
    Get weather for a location (simulated).

    Args:
        location: City name

    Returns:
        Weather information
    """
    # Simulated weather data
    weather_data = {
        "paris": {"temp": 18, "condition": "Sunny", "humidity": 65},
        "london": {"temp": 12, "condition": "Cloudy", "humidity": 75},
        "tokyo": {"temp": 22, "condition": "Clear", "humidity": 60},
        "new york": {"temp": 15, "condition": "Rainy", "humidity": 80},
    }

    location_lower = location.lower()
    if location_lower in weather_data:
        data = weather_data[location_lower]
        return {
            "location": location,
            "temperature": data["temp"],
            "condition": data["condition"],
            "humidity": data["humidity"]
        }
    else:
        return {
            "location": location,
            "error": "Weather data not available for this location"
        }


def search_web(query: str, num_results: int = 5) -> List[Dict[str, str]]:
    """
    Search the web (simulated).

    Args:
        query: Search query
        num_results: Number of results to return

    Returns:
        List of search results
    """
    # Simulated search results
    return [
        {
            "title": f"Result {i+1} for '{query}'",
            "url": f"https://example.com/result{i+1}",
            "snippet": f"This is a snippet for result {i+1} about {query}."
        }
        for i in range(num_results)
    ]


# Define tools
TOOLS = [
    Tool(
        name="get_current_time",
        description="Get the current date and time",
        parameters={
            "type": "object",
            "properties": {},
            "required": []
        },
        function=get_current_time
    ),
    Tool(
        name="calculate",
        description="Perform mathematical calculations",
        parameters={
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Mathematical expression to evaluate (e.g., '2 + 2', '10 * 5')"
                }
            },
            "required": ["expression"]
        },
        function=calculate
    ),
    Tool(
        name="get_weather",
        description="Get current weather for a location",
        parameters={
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City name"
                }
            },
            "required": ["location"]
        },
        function=get_weather
    ),
    Tool(
        name="search_web",
        description="Search the web for information",
        parameters={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query"
                },
                "num_results": {
                    "type": "integer",
                    "description": "Number of results (default: 5)"
                }
            },
            "required": ["query"]
        },
        function=search_web
    )
]


class FunctionCallingAgent:
    """Agent with function calling capabilities."""

    def __init__(self, llm: Llama, tools: List[Tool]):
        self.llm = llm
        self.tools = {tool.name: tool for tool in tools}

    def _build_tools_prompt(self) -> str:
        """Build tools description for prompt."""
        tools_desc = []

        for tool in self.tools.values():
            params_str = json.dumps(tool.parameters, indent=2)
            tools_desc.append(
                f"- {tool.name}: {tool.description}\n"
                f"  Parameters: {params_str}"
            )

        return "\n".join(tools_desc)

    def _parse_function_call(self, response: str) -> Dict[str, Any]:
        """Parse function call from LLM response."""
        try:
            # Look for JSON in response
            json_match = re.search(r'\{[^}]+\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                if 'function' in data and 'arguments' in data:
                    return data
        except json.JSONDecodeError:
            pass

        return None

    def run(self, query: str, max_iterations: int = 5, verbose: bool = True) -> str:
        """
        Run agent with query.

        Args:
            query: User query
            max_iterations: Maximum number of iterations
            verbose: Whether to print execution steps

        Returns:
            Final answer
        """
        tools_prompt = self._build_tools_prompt()

        system_prompt = f"""You are a helpful assistant with access to tools.

Available tools:
{tools_prompt}

To use a tool, output a JSON object in this exact format:
{{"function": "tool_name", "arguments": {{"arg1": "value1"}}}}

After receiving tool results, provide a natural language answer to the user.
If you can answer without tools, do so directly."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ]

        if verbose:
            print(f"\n{'='*60}")
            print(f"Query: {query}")
            print(f"{'='*60}\n")

        for iteration in range(max_iterations):
            if verbose:
                print(f"[Iteration {iteration + 1}]")

            # Generate response
            response = self.llm.create_chat_completion(
                messages=messages,
                max_tokens=512,
                temperature=0.7
            )

            assistant_message = response['choices'][0]['message']['content']
            messages.append({
                "role": "assistant",
                "content": assistant_message
            })

            if verbose:
                print(f"Assistant: {assistant_message}\n")

            # Check for function call
            function_call = self._parse_function_call(assistant_message)

            if function_call:
                func_name = function_call['function']
                arguments = function_call.get('arguments', {})

                if func_name in self.tools:
                    # Execute tool
                    if verbose:
                        print(f"Executing: {func_name}({arguments})")

                    try:
                        result = self.tools[func_name].execute(**arguments)
                        result_str = json.dumps(result, indent=2)

                        if verbose:
                            print(f"Result: {result_str}\n")

                        # Add result to conversation
                        messages.append({
                            "role": "user",
                            "content": f"Tool '{func_name}' returned:\n{result_str}\n\nProvide a natural language answer to the original query."
                        })

                    except Exception as e:
                        error_msg = f"Error executing {func_name}: {str(e)}"
                        if verbose:
                            print(f"{error_msg}\n")

                        messages.append({
                            "role": "user",
                            "content": f"Tool execution failed: {error_msg}\n\nTry a different approach or answer without tools."
                        })
                else:
                    # Unknown function
                    error_msg = f"Unknown function: {func_name}"
                    if verbose:
                        print(f"{error_msg}\n")

                    messages.append({
                        "role": "user",
                        "content": f"Error: {error_msg}\n\nUse only the available tools or answer directly."
                    })
            else:
                # No function call, this is the final answer
                if verbose:
                    print(f"{'='*60}")
                    print("Final Answer:")
                    print(f"{'='*60}")
                    print(assistant_message)

                return assistant_message

        return "Maximum iterations reached without final answer."


def main():
    """Example usage."""
    import sys

    if len(sys.argv) < 2:
        print("Usage: python 03_function_calling_agent.py <model_path>")
        print("\nExample:")
        print("  python 03_function_calling_agent.py ./models/model.gguf")
        sys.exit(1)

    model_path = sys.argv[1]

    # Initialize LLM
    print("Loading model...")
    llm = Llama(
        model_path=model_path,
        n_ctx=4096,
        n_gpu_layers=35,
        verbose=False
    )

    # Create agent
    agent = FunctionCallingAgent(llm, TOOLS)

    # Example queries
    examples = [
        "What time is it?",
        "What's the weather in Paris?",
        "Calculate 25 * 4 + 10",
        "What's the weather in London and Tokyo? Compare them.",
    ]

    print("\n" + "="*60)
    print("Function Calling Agent - Example Queries")
    print("="*60)

    for query in examples:
        result = agent.run(query, verbose=True)
        print("\n" + "-"*60 + "\n")

    # Interactive mode
    print("\n" + "="*60)
    print("Interactive Mode - Type 'quit' to exit")
    print("="*60 + "\n")

    while True:
        try:
            query = input("\nYou: ").strip()

            if query.lower() in ['quit', 'exit', 'q']:
                break

            if not query:
                continue

            agent.run(query, verbose=True)

        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}")


if __name__ == "__main__":
    main()
