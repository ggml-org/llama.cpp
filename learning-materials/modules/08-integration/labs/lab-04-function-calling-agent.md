# Lab 8.4: Function-Calling Agent

**Estimated Time**: 3-4 hours
**Difficulty**: Advanced
**Prerequisites**: Lessons 8.1, 8.4

## Objective

Build an intelligent agent that can use multiple tools to accomplish complex tasks through function calling and multi-step reasoning.

## Part 1: Tool Framework (60 min)

### Tool Definition System

```python
# tools.py
from typing import Callable, Dict, Any, List
from pydantic import BaseModel, Field
import inspect

class Tool(BaseModel):
    name: str
    description: str
    parameters: Dict[str, Any]
    function: Callable

    class Config:
        arbitrary_types_allowed = True

    @classmethod
    def from_function(cls, func: Callable, description: str = None):
        """Create tool from function with type hints."""
        # TODO: Extract parameters from function signature
        sig = inspect.signature(func)
        parameters = {}
        # Convert type hints to JSON schema
        # TODO: Implement
        pass

    def execute(self, **kwargs) -> Any:
        """Execute tool with validation."""
        # TODO: Validate arguments against schema
        # TODO: Execute function
        # TODO: Handle errors
        pass
```

**Exercise**: Implement automatic tool creation from Python functions using type hints.

### Example Tools

Implement these tools:

```python
# example_tools.py
from typing import List, Dict
from datetime import datetime
import requests

def get_weather(city: str) -> Dict[str, Any]:
    """
    Get weather for a city.

    Args:
        city: City name

    Returns:
        Weather data
    """
    # TODO: Implement (use API or mock data)
    pass

def calculate(expression: str) -> float:
    """Evaluate math expression safely."""
    # TODO: Implement safe evaluation
    pass

def search_database(
    query: str,
    table: str,
    limit: int = 10
) -> List[Dict]:
    """Search database."""
    # TODO: Implement database search
    pass

def send_email(
    to: str,
    subject: str,
    body: str
) -> bool:
    """Send email (simulated)."""
    # TODO: Implement
    pass

def create_calendar_event(
    title: str,
    date: str,
    time: str
) -> Dict:
    """Create calendar event."""
    # TODO: Implement
    pass
```

## Part 2: Agent Implementation (90 min)

### ReAct Agent Pattern

```python
# react_agent.py
from llama_cpp import Llama
from typing import List
import re

class ReActAgent:
    """ReAct pattern agent with reasoning and acting."""

    def __init__(self, llm: Llama, tools: List[Tool]):
        self.llm = llm
        self.tools = {tool.name: tool for tool in tools}
        self.execution_log = []

    def run(self, task: str, max_steps: int = 10) -> str:
        """
        Execute task using ReAct pattern.

        Format:
        Thought: [reasoning]
        Action: [tool_name]
        Action Input: [arguments]
        Observation: [result]
        ... (repeat)
        Thought: [final reasoning]
        Answer: [final answer]
        """
        # TODO: Implement ReAct loop
        pass

    def _parse_action(self, text: str) -> Tuple[str, str]:
        """Parse action and input from text."""
        # TODO: Implement parsing
        pass

    def _execute_tool(self, tool_name: str, tool_input: str) -> str:
        """Execute tool and return observation."""
        # TODO: Implement
        pass
```

### Planning Agent

```python
# planning_agent.py
class PlanningAgent:
    """Agent that plans before acting."""

    def plan(self, goal: str) -> List[Dict]:
        """
        Create plan to achieve goal.

        Returns:
            List of steps with tools and arguments
        """
        # TODO: Implement planning
        pass

    def execute_plan(self, plan: List[Dict]) -> Dict:
        """Execute planned steps."""
        # TODO: Implement execution
        pass

    def replan_on_failure(self, step: Dict, error: str) -> List[Dict]:
        """Replan when step fails."""
        # TODO: Implement replanning
        pass
```

## Part 3: Advanced Features (60 min)

### Task 3.1: Tool Chaining

Implement automatic tool chaining:

```python
class ToolChain:
    """Chain multiple tools together."""

    def __init__(self, tools: List[Tool]):
        self.tools = tools

    def build_chain(self, goal: str) -> List[str]:
        """Determine tool execution order."""
        # TODO: Implement dependency analysis
        pass

    def execute_chain(self, tool_sequence: List[str], inputs: Dict) -> Any:
        """Execute tools in sequence."""
        # TODO: Implement
        pass
```

### Task 3.2: Error Recovery

```python
class RobustAgent(ReActAgent):
    """Agent with error recovery."""

    def execute_with_retry(
        self,
        tool_name: str,
        arguments: Dict,
        max_retries: int = 3
    ) -> Any:
        """Execute with retry logic."""
        # TODO: Implement exponential backoff retry
        pass

    def fallback_strategy(self, failed_tool: str, error: str) -> str:
        """Find alternative approach when tool fails."""
        # TODO: Implement fallback logic
        pass
```

## Part 4: Multi-Tool Scenarios (30 min)

### Create Complex Task Scenarios

Test your agent with these scenarios:

```python
scenarios = [
    {
        "task": "Find the weather in Paris, convert the temperature to Fahrenheit, and send me an email with the result.",
        "required_tools": ["get_weather", "calculate", "send_email"],
        "expected_steps": 3
    },
    {
        "task": "Search for Python courses in the database, calculate the average price, and create a calendar event to review the top course.",
        "required_tools": ["search_database", "calculate", "create_calendar_event"],
        "expected_steps": 4
    }
]

def test_agent_on_scenarios(agent: ReActAgent):
    for scenario in scenarios:
        result = agent.run(scenario["task"])
        # Verify correct tools were used
        # Verify steps match expected
```

**Exercise**: Run your agent on these scenarios and analyze the execution traces.

## Challenges

### Challenge 1: Natural Language Tool Discovery
Allow agent to ask for tool descriptions dynamically.

### Challenge 2: Parallel Tool Execution
Execute independent tools in parallel.

### Challenge 3: Learning from Feedback
Agent learns which tools work best for which tasks.

### Challenge 4: Tool Synthesis
Agent can create new tools by combining existing ones.

## Evaluation

```python
# evaluation.py
class AgentEvaluator:
    def evaluate_task_success(
        self,
        agent: ReActAgent,
        task: str,
        expected_outcome: Any
    ) -> Dict:
        """Evaluate if agent completed task correctly."""
        result = agent.run(task)
        return {
            "success": self._check_success(result, expected_outcome),
            "steps_taken": len(agent.execution_log),
            "tools_used": self._count_tools(agent.execution_log),
            "efficiency_score": self._calculate_efficiency(agent.execution_log)
        }

    def benchmark_agent(
        self,
        agent: ReActAgent,
        test_suite: List[Dict]
    ) -> Dict:
        """Run comprehensive benchmark."""
        results = []
        for test in test_suite:
            result = self.evaluate_task_success(
                agent,
                test["task"],
                test["expected"]
            )
            results.append(result)

        return {
            "total_tests": len(test_suite),
            "passed": sum(r["success"] for r in results),
            "avg_steps": np.mean([r["steps_taken"] for r in results]),
            "tool_usage": self._analyze_tool_usage(results)
        }
```

## Success Criteria

- [X] Tool framework with validation
- [X] ReAct agent working
- [X] Planning agent implemented
- [X] Error recovery functional
- [X] Complex scenarios solved
- [X] Evaluation passing

## Submission

Submit:
1. Complete agent code
2. Execution traces for test scenarios
3. Benchmark results
4. Analysis of agent behavior

---

**Lab**: 8.4 - Function-Calling Agent
**Module**: 08 - Integration & Applications
**Version**: 1.0
