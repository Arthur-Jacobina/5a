# Generic Agent API Documentation

## Overview

The `Agent` class is a generic, plugin-based agent that can be configured with various components like Memory, Tools, Signatures, and Observability features. This flexible design allows users to create agents tailored to their specific needs.

## Constructor Parameters

```python
Agent(
    memory: Optional[Memory] = None,
    tools: Optional[List[callable]] = None,
    signature: Optional[Any] = None,
    reasoning_system: str = "react",
    observability: bool = False,
    max_iters: int = 6,
    user_id: str = "default_user",
    **kwargs
)
```

### Parameters

- **memory** (`Optional[Memory]`): Mem0 Memory instance for storing and retrieving memories
- **tools** (`Optional[List[callable]]`): List of custom tools/functions the agent can use
- **signature** (`Optional[Any]`): DSPy signature for the agent's input/output format (defaults to `MemoryQA`)
- **reasoning_system** (`str`): Type of reasoning system to use:
  - `"react"`: ReAct reasoning with tools (default)
  - `"cot"`: Chain of Thought reasoning
  - `"basic"`: Basic prediction
- **observability** (`bool`): Enable/disable observability features (MLflow logging)
- **max_iters** (`int`): Maximum iterations for reasoning systems (default: 6)
- **user_id** (`str`): Default user ID for memory operations (default: "default_user")
- **kwargs**: Additional configuration parameters

## Usage Examples

### 1. Basic Agent (No Plugins)

```python
from src.core.agent import Agent

# Simple agent with basic reasoning
agent = Agent(reasoning_system="basic")
response = agent("Hello, how are you?")
```

### 2. Memory-Enabled Agent

```python
from mem0 import Memory
from src.core.agent import Agent

# Configure memory
memory_config = {
    "graph_store": {"config": {"url": "bolt://localhost:7687"}},
    "version": "v1.1"
}
memory = Memory.from_config(memory_config)

# Create agent with memory
agent = Agent(
    memory=memory,
    reasoning_system="react",
    observability=True
)

# Use memory features
response = agent("Remember that I like pizza")
preferences = agent.get_preferences("food")
```

### 3. Custom Tools Agent

```python
def my_calculator(expression: str) -> str:
    """Simple calculator tool."""
    try:
        result = eval(expression)
        return f"Result: {result}"
    except:
        return "Invalid expression"

def web_search(query: str) -> str:
    """Mock web search tool."""
    return f"Search results for: {query}"

agent = Agent(
    tools=[my_calculator, web_search],
    reasoning_system="react",
    max_iters=8
)

response = agent("Calculate 2 + 2 and search for python tutorials")
```

### 4. Full-Featured Agent

```python
from mem0 import Memory
from src.core.agent import Agent
from src.core.signatures.memory_qa import MemoryQA

# Setup memory
memory = Memory.from_config(memory_config)

# Custom tools
def sentiment_analyzer(text: str) -> str:
    return f"Sentiment: positive" if "good" in text.lower() else "Sentiment: neutral"

# Create comprehensive agent
agent = Agent(
    memory=memory,
    tools=[sentiment_analyzer],
    signature=MemoryQA,
    reasoning_system="react",
    observability=True,
    max_iters=10,
    user_id="user_123"
)

# Use all features
response = agent("I had a good day today. Remember this for later.")
```

### 5. Chain of Thought Agent

```python
agent = Agent(
    memory=memory,
    reasoning_system="cot",
    observability=True
)

response = agent("Explain step by step how to solve 2x + 5 = 11")
```

## Available Methods

### Core Methods

- **`forward(user_input: str, **kwargs)`**: Process user input through the reasoning system
- **`__call__(user_input: str, **kwargs)`**: Convenience method (same as forward)

### Tool Management

- **`add_tool(tool: callable)`**: Add a new tool to the agent dynamically
- **`remove_tool(tool_name: str)`**: Remove a tool by name

### Memory Methods (requires memory plugin)

- **`set_reminder(reminder_text: str, date_time: str = None, user_id: str = None)`**: Set a reminder
- **`get_preferences(category: str = "general", user_id: str = None)`**: Get user preferences
- **`update_preferences(category: str, preference: str, user_id: str = None)`**: Update preferences

### Observability Methods

- **`log_to_mlflow(run_name: str = None, experiment_name: str = "default")`**: Log to MLflow (requires observability=True)

## Dynamic Tool Management

```python
# Add tool dynamically
def new_tool(input_data: str) -> str:
    return f"Processed: {input_data}"

agent.add_tool(new_tool)

# Remove tool
agent.remove_tool("new_tool")
```

## Error Handling

The agent includes built-in error handling:

- **Memory not configured**: Raises `RuntimeError` when trying to use memory methods without memory plugin
- **Observability not enabled**: Raises `RuntimeError` when trying to log to MLflow without observability
- **Invalid reasoning system**: Raises `ValueError` for unsupported reasoning systems

## Backward Compatibility

For existing code using `MemoryReActAgent`, you can create a compatibility class:

```python
class MemoryReActAgent(Agent):
    def __init__(self, memory: Memory):
        super().__init__(
            memory=memory,
            reasoning_system="react",
            max_iters=6
        )
```

## Configuration Best Practices

1. **Memory**: Use memory for agents that need to remember user interactions
2. **Tools**: Add custom tools for domain-specific functionality
3. **Reasoning System**: 
   - Use "react" for tool-heavy agents
   - Use "cot" for analytical tasks
   - Use "basic" for simple input/output scenarios
4. **Observability**: Enable for production deployments to track performance
5. **Max Iterations**: Adjust based on task complexity (higher for complex reasoning)

## Thread Safety

The Agent class is designed to be thread-safe for read operations, but tool management and memory operations should be handled carefully in multi-threaded environments.

## Performance Considerations

- **Memory**: Adds overhead for storage/retrieval operations
- **Tools**: More tools increase reasoning complexity
- **Max Iterations**: Higher values allow more complex reasoning but increase latency
- **Observability**: Minimal overhead for logging operations 