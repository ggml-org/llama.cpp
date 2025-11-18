# Function Calling and Tool Use

**Module 8, Lesson 4**
**Estimated Time**: 3-4 hours
**Difficulty**: Advanced

## Overview

Function calling enables LLMs to interact with external tools and APIs, transforming them from conversational agents into powerful action-taking systems. Learn to implement structured outputs and tool use with llama.cpp.

## Learning Objectives

- Understand function calling concepts
- Implement structured JSON outputs
- Build tool-using agents
- Handle function execution and error cases
- Create production-ready agent systems

## Prerequisites

- Module 8, Lessons 1-3
- Understanding of JSON schemas
- Python programming experience

---

## 1. Function Calling Fundamentals

### What is Function Calling?

Function calling allows LLMs to:
- Generate structured JSON outputs
- Request execution of predefined functions
- Interact with external APIs and databases
- Perform multi-step reasoning and actions

### Architecture

```
┌─────────────────────────────────────────┐
│         Function Calling Agent          │
│                                         │
│  User Query                             │
│      │                                  │
│      ▼                                  │
│  ┌────────────────────────────────┐    │
│  │ LLM: Analyze and Plan          │    │
│  │ Output: Function call request  │    │
│  └────────────┬───────────────────┘    │
│               │                         │
│               ▼                         │
│  ┌────────────────────────────────┐    │
│  │ Parse Function Call            │    │
│  │ Extract: name, arguments       │    │
│  └────────────┬───────────────────┘    │
│               │                         │
│               ▼                         │
│  ┌────────────────────────────────┐    │
│  │ Execute Function               │    │
│  │ Call: tool/API/database        │    │
│  └────────────┬───────────────────┘    │
│               │                         │
│               ▼                         │
│  ┌────────────────────────────────┐    │
│  │ LLM: Format Result             │    │
│  │ Output: Final answer           │    │
│  └────────────────────────────────┘    │
└─────────────────────────────────────────┘
```

---

## 2. JSON Mode and Structured Outputs

### Basic JSON Generation

```python
from llama_cpp import Llama
import json

llm = Llama(
    model_path="./models/model.gguf",
    n_ctx=2048,
    n_gpu_layers=35
)

# Prompt for JSON output
prompt = """Generate a JSON object for a person with name, age, and occupation.
Output only valid JSON, no explanations.

JSON:"""

response = llm(
    prompt,
    max_tokens=100,
    temperature=0.1,  # Low temperature for consistent structure
    stop=["}\n"]      # Stop after closing brace
)

output = response['choices'][0]['text']
# Parse JSON
try:
    data = json.loads(output + "}")
    print(data)
except json.JSONDecodeError as e:
    print(f"Invalid JSON: {e}")
```

### Grammar-Constrained JSON

```python
from llama_cpp import Llama, LlamaGrammar

# Define JSON schema grammar
json_schema_grammar = r'''
root ::= object
object ::= "{" pair ("," pair)* "}"
pair ::= string ":" value
value ::= "true" | "false" | "null" | number | string | array | object
string ::= "\"" [^"]* "\""
number ::= "-"? [0-9]+ ("." [0-9]+)? ([eE] [+-]? [0-9]+)?
array ::= "[" (value ("," value)*)? "]"
'''

grammar = LlamaGrammar.from_string(json_schema_grammar)

llm = Llama(model_path="./models/model.gguf")

response = llm(
    "Generate a JSON with fields: name (string), age (number), hobbies (array):",
    max_tokens=200,
    grammar=grammar,
    temperature=0.7
)

print(response['choices'][0]['text'])
```

### Pydantic Schema Validation

```python
from pydantic import BaseModel, Field
from typing import List
import json

class Person(BaseModel):
    """Person data model."""
    name: str = Field(description="Full name")
    age: int = Field(ge=0, le=150, description="Age in years")
    occupation: str = Field(description="Job title")
    hobbies: List[str] = Field(description="List of hobbies")

def generate_structured_output(
    llm: Llama,
    model: BaseModel,
    prompt: str
) -> BaseModel:
    """Generate and validate structured output."""
    # Get schema
    schema = model.schema_json(indent=2)

    # Build prompt with schema
    full_prompt = f"""Generate a JSON object matching this schema:

{schema}

Request: {prompt}

Output only valid JSON matching the schema above.

JSON:"""

    # Generate
    response = llm(
        full_prompt,
        max_tokens=300,
        temperature=0.1
    )

    # Parse and validate
    json_str = response['choices'][0]['text']
    data = json.loads(json_str)

    return model(**data)

# Usage
llm = Llama(model_path="./models/model.gguf")
person = generate_structured_output(
    llm,
    Person,
    "Create a profile for a software engineer"
)

print(person.model_dump_json(indent=2))
```

---

## 3. Function Definitions

### Defining Tools

```python
from typing import List, Dict, Any, Callable

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

    def to_dict(self) -> Dict[str, Any]:
        """Convert to OpenAI function format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters
            }
        }

    def execute(self, **kwargs) -> Any:
        """Execute the tool function."""
        return self.function(**kwargs)

# Example tools
def get_weather(location: str) -> Dict[str, Any]:
    """Get weather for a location."""
    # Simulate API call
    return {
        "location": location,
        "temperature": 72,
        "condition": "sunny",
        "humidity": 45
    }

def calculate(expression: str) -> float:
    """Evaluate a mathematical expression."""
    try:
        # Safe evaluation (in production, use a proper math parser)
        return eval(expression, {"__builtins__": {}}, {})
    except Exception as e:
        return f"Error: {e}"

def search_database(query: str, limit: int = 5) -> List[Dict]:
    """Search database."""
    # Simulate database search
    return [
        {"id": i, "title": f"Result {i}", "content": f"Content for {query}"}
        for i in range(limit)
    ]

# Define tools
tools = [
    Tool(
        name="get_weather",
        description="Get current weather for a location",
        parameters={
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City name or location"
                }
            },
            "required": ["location"]
        },
        function=get_weather
    ),
    Tool(
        name="calculate",
        description="Perform mathematical calculations",
        parameters={
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Mathematical expression to evaluate"
                }
            },
            "required": ["expression"]
        },
        function=calculate
    ),
    Tool(
        name="search_database",
        description="Search the knowledge database",
        parameters={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query"
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum results",
                    "default": 5
                }
            },
            "required": ["query"]
        },
        function=search_database
    )
]
```

---

## 4. Agent Implementation

### Basic Function-Calling Agent

```python
from llama_cpp import Llama
from typing import List, Dict, Any
import json
import re

class FunctionCallingAgent:
    """Agent with function calling capabilities."""

    def __init__(self, llm: Llama, tools: List[Tool]):
        self.llm = llm
        self.tools = {tool.name: tool for tool in tools}
        self.conversation_history = []

    def _build_tools_description(self) -> str:
        """Build tools description for prompt."""
        descriptions = []
        for tool in self.tools.values():
            params_desc = json.dumps(tool.parameters, indent=2)
            descriptions.append(
                f"- {tool.name}: {tool.description}\n"
                f"  Parameters: {params_desc}"
            )
        return "\n".join(descriptions)

    def _parse_function_call(self, response: str) -> Dict[str, Any]:
        """Parse function call from LLM response."""
        # Look for JSON function call
        # Format: {"function": "tool_name", "arguments": {...}}
        try:
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                return data
        except json.JSONDecodeError:
            pass

        return None

    def run(self, query: str, max_iterations: int = 5) -> str:
        """Run agent with query."""
        tools_desc = self._build_tools_description()

        system_prompt = f"""You are a helpful assistant with access to tools.

Available tools:
{tools_desc}

To use a tool, output a JSON object:
{{"function": "tool_name", "arguments": {{"arg1": "value1"}}}}

After receiving tool results, provide a final answer.
"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ]

        for iteration in range(max_iterations):
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

            # Check for function call
            function_call = self._parse_function_call(assistant_message)

            if function_call and 'function' in function_call:
                func_name = function_call['function']
                arguments = function_call.get('arguments', {})

                if func_name in self.tools:
                    # Execute tool
                    try:
                        result = self.tools[func_name].execute(**arguments)
                        result_str = json.dumps(result, indent=2)

                        # Add result to conversation
                        messages.append({
                            "role": "function",
                            "name": func_name,
                            "content": result_str
                        })

                    except Exception as e:
                        messages.append({
                            "role": "function",
                            "name": func_name,
                            "content": f"Error: {str(e)}"
                        })
                else:
                    messages.append({
                        "role": "function",
                        "name": func_name,
                        "content": f"Error: Unknown function '{func_name}'"
                    })
            else:
                # No function call, return final answer
                return assistant_message

        return "Max iterations reached without final answer."

# Usage
llm = Llama(model_path="./models/model.gguf", n_ctx=4096)
agent = FunctionCallingAgent(llm, tools)

result = agent.run("What's the weather in Paris and calculate 25 * 4?")
print(result)
```

### ReAct Agent Pattern

```python
class ReActAgent:
    """ReAct (Reasoning + Acting) agent pattern."""

    def __init__(self, llm: Llama, tools: List[Tool]):
        self.llm = llm
        self.tools = {tool.name: tool for tool in tools}

    def run(self, query: str, max_steps: int = 10) -> str:
        """Run ReAct loop."""
        tools_desc = "\n".join([
            f"- {name}: {tool.description}"
            for name, tool in self.tools.items()
        ])

        prompt = f"""Answer the following question using this format:

Question: [input question]
Thought: [reasoning about what to do]
Action: [tool to use]
Action Input: [input to tool]
Observation: [tool result]
... (repeat Thought/Action/Observation as needed)
Thought: [final reasoning]
Answer: [final answer]

Available tools:
{tools_desc}

Question: {query}
Thought:"""

        full_response = ""
        current_prompt = prompt

        for step in range(max_steps):
            response = self.llm(
                current_prompt,
                max_tokens=256,
                temperature=0.7,
                stop=["Observation:"]
            )

            text = response['choices'][0]['text']
            full_response += text

            # Parse action
            if "Action:" in text and "Action Input:" in text:
                action_match = re.search(r'Action:\s*(\w+)', text)
                input_match = re.search(r'Action Input:\s*(.+)', text)

                if action_match and input_match:
                    action = action_match.group(1)
                    action_input = input_match.group(1).strip()

                    # Execute tool
                    if action in self.tools:
                        try:
                            result = self.tools[action].execute(
                                **{"query": action_input}
                            )
                            observation = f"Observation: {result}\nThought:"
                            full_response += observation
                            current_prompt += text + observation
                        except Exception as e:
                            observation = f"Observation: Error - {e}\nThought:"
                            full_response += observation
                            current_prompt += text + observation
                    else:
                        observation = f"Observation: Unknown tool '{action}'\nThought:"
                        full_response += observation
                        current_prompt += text + observation

            # Check for final answer
            if "Answer:" in text:
                return re.search(r'Answer:\s*(.+)', text).group(1).strip()

        return full_response
```

---

## 5. Advanced Patterns

### Multi-Tool Orchestration

```python
class ToolOrchestrator:
    """Orchestrate complex multi-tool workflows."""

    def __init__(self, llm: Llama, tools: List[Tool]):
        self.llm = llm
        self.tools = {tool.name: tool for tool in tools}
        self.execution_log = []

    def plan_and_execute(self, goal: str) -> Dict[str, Any]:
        """Plan multi-step workflow and execute."""
        # Step 1: Planning
        plan_prompt = f"""Given this goal: "{goal}"

Break it down into steps using available tools:
{self._build_tools_description()}

Output a JSON plan:
{{
  "steps": [
    {{"tool": "tool_name", "args": {{}}, "reason": "why this step"}},
    ...
  ]
}}

Plan:"""

        plan_response = self.llm(
            plan_prompt,
            max_tokens=500,
            temperature=0.3
        )

        # Parse plan
        try:
            plan = json.loads(plan_response['choices'][0]['text'])
        except json.JSONDecodeError:
            return {"error": "Failed to generate valid plan"}

        # Step 2: Execute plan
        results = []
        for i, step in enumerate(plan['steps']):
            tool_name = step['tool']
            args = step['args']

            if tool_name in self.tools:
                result = self.tools[tool_name].execute(**args)
                results.append({
                    "step": i + 1,
                    "tool": tool_name,
                    "args": args,
                    "result": result
                })
                self.execution_log.append({
                    "step": i + 1,
                    "tool": tool_name,
                    "result": result
                })

        # Step 3: Synthesize results
        synthesis_prompt = f"""Given these results:
{json.dumps(results, indent=2)}

Provide a concise answer to: "{goal}"

Answer:"""

        final_response = self.llm(
            synthesis_prompt,
            max_tokens=300,
            temperature=0.7
        )

        return {
            "goal": goal,
            "plan": plan,
            "execution": results,
            "answer": final_response['choices'][0]['text'].strip()
        }

    def _build_tools_description(self) -> str:
        """Build tools description."""
        return "\n".join([
            f"- {name}: {tool.description}"
            for name, tool in self.tools.items()
        ])
```

### Error Handling and Retries

```python
class RobustAgent:
    """Agent with error handling and retries."""

    def __init__(self, llm: Llama, tools: List[Tool], max_retries: int = 3):
        self.llm = llm
        self.tools = {tool.name: tool for tool in tools}
        self.max_retries = max_retries

    def execute_tool_with_retry(
        self,
        tool_name: str,
        arguments: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute tool with retry logic."""
        for attempt in range(self.max_retries):
            try:
                result = self.tools[tool_name].execute(**arguments)
                return {
                    "success": True,
                    "result": result,
                    "attempts": attempt + 1
                }
            except Exception as e:
                if attempt == self.max_retries - 1:
                    return {
                        "success": False,
                        "error": str(e),
                        "attempts": attempt + 1
                    }
                # Wait before retry
                import time
                time.sleep(2 ** attempt)  # Exponential backoff

        return {"success": False, "error": "Max retries exceeded"}

    def run_with_fallback(self, query: str) -> str:
        """Run agent with fallback strategies."""
        try:
            # Try primary strategy
            return self.run_primary(query)
        except Exception as e:
            # Fallback: Direct LLM response
            return self.run_fallback(query, error=str(e))

    def run_primary(self, query: str) -> str:
        """Primary execution with tools."""
        # Implementation similar to FunctionCallingAgent
        pass

    def run_fallback(self, query: str, error: str) -> str:
        """Fallback to direct LLM."""
        prompt = f"""Tool execution failed: {error}

Please answer this question without tools: {query}

Answer:"""

        response = self.llm(prompt, max_tokens=512)
        return response['choices'][0]['text'].strip()
```

---

## 6. Production Examples

### REST API Tool

```python
import requests

def create_api_tool(
    name: str,
    base_url: str,
    endpoints: Dict[str, Dict]
) -> Tool:
    """Create tool for REST API."""

    def api_call(endpoint: str, **params) -> Dict:
        """Call API endpoint."""
        config = endpoints.get(endpoint)
        if not config:
            raise ValueError(f"Unknown endpoint: {endpoint}")

        url = f"{base_url}{config['path']}"
        method = config.get('method', 'GET')

        if method == 'GET':
            response = requests.get(url, params=params)
        elif method == 'POST':
            response = requests.post(url, json=params)

        response.raise_for_status()
        return response.json()

    return Tool(
        name=name,
        description=f"Call {name} API",
        parameters={
            "type": "object",
            "properties": {
                "endpoint": {
                    "type": "string",
                    "enum": list(endpoints.keys())
                }
            }
        },
        function=api_call
    )
```

### Database Query Tool

```python
import sqlite3

class DatabaseTool(Tool):
    """Tool for database queries."""

    def __init__(self, db_path: str):
        def query_db(sql: str) -> List[Dict]:
            """Execute SQL query."""
            conn = sqlite3.connect(db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            cursor.execute(sql)
            results = [dict(row) for row in cursor.fetchall()]

            conn.close()
            return results

        super().__init__(
            name="query_database",
            description="Execute SQL query on database",
            parameters={
                "type": "object",
                "properties": {
                    "sql": {
                        "type": "string",
                        "description": "SQL SELECT query"
                    }
                },
                "required": ["sql"]
            },
            function=query_db
        )
```

---

## Summary

In this lesson, you learned:
- ✅ Function calling fundamentals
- ✅ Structured JSON output generation
- ✅ Building tool-using agents
- ✅ ReAct and multi-step reasoning patterns
- ✅ Error handling and production patterns

## Next Steps

- **Lesson 5**: Mobile Deployment
- **Lab 8.4**: Build a function-calling agent
- **Project**: Multi-tool agent system

## Additional Resources

- [OpenAI Function Calling Guide](https://platform.openai.com/docs/guides/function-calling)
- [ReAct Paper](https://arxiv.org/abs/2210.03629)
- [LangChain Agents](https://python.langchain.com/docs/modules/agents/)

---

**Module**: 08 - Integration & Applications
**Lesson**: 04 - Function Calling and Tool Use
**Version**: 1.0
**Last Updated**: 2025-11-18
