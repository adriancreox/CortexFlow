# CortexFlow Documentation

Welcome to the documentation for CortexFlow — the Cognitive Operating System for production-grade AI agents.

## ⌨️ Command Line Interface (CLI)

The `cortexflow` command is your primary tool for scaffolding and running agents.

### Scaffolding a project
```bash
cortexflow init <project_name> --provider <openai|anthropic|gemini|deepseek|groq|ollama>
```

### Checking System Health
```bash
cortexflow status
```
This command verifies if the Kernel, EventMesh, and Memory Vault are operational.

---

## 🚀 Getting Started

### 1. Installation
CortexFlow requires Python 3.11+.

```bash
pip install cortexflow
```

Install with your preferred provider support:
```bash
pip install "cortexflow[openai]"      # OpenAI, DeepSeek, Groq
pip install "cortexflow[anthropic]"   # Anthropic Claude
pip install "cortexflow[google]"      # Google Gemini
pip install "cortexflow[ollama]"      # Local models
```

### 2. Your First Agent
Define an agent as a pure blueprint using `defineAgent`.

```python
from cortexflow import defineAgent, CortexRuntime
from cortexflow.providers.openai import OpenAIProvider

# 1. Define the agent
agent_def = defineAgent(
    name="assistant",
    instructions="You are a helpful and concise assistant.",
    provider=OpenAIProvider(model="gpt-4o")
)

# 2. Run the runtime
async def main():
    async with CortexRuntime() as runtime:
        # Spawn a stateful instance
        agent_id = await runtime.spawn(agent_def)
        
        # Send a message
        await runtime.send(agent_id, "Explain quantum entanglement in one sentence.")
```

---

## 🧠 Core Concepts

### Agents are Processes
Every agent spawned in CortexFlow gets a unique ID and its own **Agent Control Block (ACB)**. The ACB stores:
- **L1 Working Memory**: The current reasoning context.
- **Mailbox**: A queue of incoming events.
- **Reasoning Trace**: A history of every thought and action taken.

### The Memory Hierarchy (The Vault)
CortexFlow manages agent state automatically across 4 tiers:
1. **L1 Registers**: Hot context in memory (<0.1ms).
2. **L2 Cache**: Session persistence in Redis (~1ms).
3. **L3 Main Memory**: Semantic episodic memory via Vector Stores (~20ms).
4. **L4 Archive**: Complete audit logs in JSONL/S3 (Async).

---

## 🛠️ Working with Tools

CortexFlow tools are type-safe and feature native circuit breaking.

```python
from cortexflow.providers.tools import ToolRegistry

tools = ToolRegistry()

@tools.register(
    name="get_weather",
    description="Get the current weather for a city",
    timeout=5.0
)
async def get_weather(city: str) -> str:
    # Your logic here
    return f"The weather in {city} is sunny."
```

---

## 🔌 Supported Providers

CortexFlow is provider-agnostic. You can swap providers without changing your agent logic.

### OpenAI
```python
from cortexflow.providers.openai import OpenAIProvider
provider = OpenAIProvider(model="gpt-4o", api_key="sk-...")
```

### DeepSeek (New!)
```python
from cortexflow.providers.deepseek import DeepSeekProvider
provider = DeepSeekProvider(model="deepseek-chat", api_key="sk-...")
```

### Groq (New!)
```python
from cortexflow.providers.groq import GroqProvider
provider = GroqProvider(model="llama-3.3-70b-versatile", api_key="gsk_...")
```

### Google Gemini
```python
from cortexflow.providers.gemini import GeminiProvider
provider = GeminiProvider(model="gemini-1.5-pro", api_key="...")
```

---

## 📈 Observability & Debugging

### Snapshots
You can inspect an agent's state at any time using `runtime.get_snapshot(agent_id)`.

```python
snapshot = runtime.get_snapshot(agent_id)
print(f"Status: {snapshot.status}")
print(f"Current Iteration: {snapshot.iteration}")
print(f"Last Thought: {snapshot.reasoning_trace[-1].thought}")
```

### Time-Travel Debugging
Since every tick produces an immutable snapshot, you can theoretically "replay" an agent's reasoning from any historical state to understand where it went wrong.

---

## 🛡️ Production Readiness Checklist

1. **Use Redis**: For production, configure the runtime to use Redis for L2 cache and distributed locks.
   ```python
   runtime.use_redis(my_redis_client)
   ```
2. **Set Token Budgets**: Always set a `token_budget` in `defineAgent` to prevent runaway costs.
3. **Monitor Circuit Breakers**: Use `runtime.stats()` to check if any tools have been suspended due to repeated failures.
4. **Audit Logs**: Ensure L4 archiving is enabled to capture reasoning traces for compliance.
