# CortexFlow Specification (v0.1.0)

This document defines the architectural standards, data structures, and execution protocols of the CortexFlow Cognitive Runtime. It serves as the single source of truth for developers and contributors.

---

## 1. Core Philosophy: The Agent as a Process

CortexFlow abandons the "scripting" paradigm of LangChain and CrewAI. In CortexFlow, an agent is a **managed process** within a Cognitive Operating System.

- **Immutability**: Every state change produces a new snapshot. History is never overwritten, only appended.
- **Determinism**: Given the same input event and the same state snapshot, the agent's reasoning process should be reproducible.
- **Serverless-First**: Agents do not consume compute while idle. They are "woken up" by events, process a single tick, commit their state, and go back to sleep.

---

## 2. The Agent Control Block (ACB)

Inspired by the Process Control Block (PCB) in traditional OS kernels, the ACB is the internal registry entry for a live agent.

```python
@dataclass
class AgentRegistration:
    agent_id: str          # Unique Process ID (PID)
    agent_name: str        # Executable name
    snapshot: StateSnapshot # Current "Registers" and "Stack"
    provider: LLMProvider # The "CPU" instruction set
    is_processing: bool    # Mutex flag
```

---

## 3. Memory Hierarchy (The Vault)

CortexFlow implements a 4-tier memory system to handle the "Context Window Problem" at production scale.

### L1: Registers (In-Process)
- **Backend**: Local LRU Cache.
- **Scope**: Current active reasoning trace + last N messages.
- **Latency**: <0.1ms.

### L2: Cache (Redis)
- **Backend**: Redis JSON / Protobuf.
- **Scope**: Session-level persistence. survives process restarts.
- **Latency**: ~1-2ms.

### L3: Main Memory (Vector)
- **Backend**: ChromaDB / pgvector / Pinecone.
- **Scope**: Long-term episodic memory. Semantic retrieval.
- **Latency**: ~10-50ms.

### L4: Archive (Cold Storage)
- **Backend**: JSONL on S3 / ClickHouse.
- **Scope**: Complete audit trail for compliance, fine-tuning, and debugging.
- **Latency**: Asynchronous.

---

## 4. The Cognitive Virtual Machine (CVM)

The CVM is the deterministic execution loop that transforms inputs into state updates.

### The Tick Cycle:
1. **Load**: Retrieve the current `StateSnapshot` (DNA) and the triggering `CortexEvent`.
2. **Reason**: The LLM processes the context and decides on an action (Text or Tool Call).
3. **Act**: If a tool call is requested, the CVM dispatches it to the Scheduler.
4. **Observe**: The result of the action is captured.
5. **Commit**: A new `StateSnapshot` is generated and persisted.

---

## 5. Event Mesh Protocol

All communication in CortexFlow happens via **CortexEvents**.

### Standard Event Schema:
- `event_id`: ULID (Sortable, Unique).
- `type`: `AGENT_TICK`, `MESSAGE_RECEIVED`, `TOOL_RESULT`, etc.
- `idempotency_key`: Prevents duplicate processing in distributed environments.
- `payload`: Data specific to the event type.

---

## 6. Provider Abstraction Layer (Synapse)

CortexFlow normalizes all LLM interactions into a single schema.

### Supported Providers:
- **OpenAI**: GPT-4o, GPT-3.5.
- **Anthropic**: Claude 3.5 Sonnet/Opus.
- **Google**: Gemini 1.5 Pro/Flash.
- **DeepSeek**: V3, R1.
- **Groq**: Ultra-low latency Llama 3 models.
- **Ollama**: Local execution.

---

## 7. Developer SDK Quick Start

### Defining an Agent
```python
from cortexflow import defineAgent
from cortexflow.providers.openai import OpenAIProvider

researcher = defineAgent(
    name="researcher",
    instructions="You are a meticulous researcher.",
    provider=OpenAIProvider(model="gpt-4o")
)
```

### Running the Runtime
```python
from cortexflow import CortexRuntime

async with CortexRuntime() as runtime:
    agent_id = await runtime.spawn(researcher)
    await runtime.send(agent_id, "Find latest AI news")
```

---

## 8. AI Security Layer (The Shield)

Every action in CortexFlow is gated by a multi-layered security protocol.

### 8.1 Permission Scopes
Agents do not have blanket access to tools. Every tool can define `required_scopes`, and every agent must be explicitly granted `allowed_scopes` in its blueprint.
- **Hierarchy**: Scopes follow a hierarchy (e.g., `filesystem:read` vs `filesystem:write`).
- **Enforcement**: The ToolRegistry validates scopes BEFORE execution.

### 8.2 Identity & Sandboxing
- **Agent Identity**: Every agent has a unique PID and Identity Graph.
- **Data Isolation**: Agents only access the vault tiers (L1-L4) assigned to their agent_id.

### 8.3 Cryptographic Audit Trail
Every tick result, tool call, and state transition is logged in the **L4 Archive**. In Phase 2, these logs will be signed cryptographically to ensure non-repudiation and compliance.

---

## 9. Reliability Guarantees

- **Infinite Loop Guard**: Agents have a `max_iterations` cap per event.
- **Distributed Locking**: Prevents race conditions when multiple workers try to wake the same agent.
- **Circuit Breaking**: Tools that fail repeatedly are automatically suspended to prevent "hallucination loops".
- **Time-Travel Debugging**: Because every tick is a snapshot, you can "rewind" an agent to any point in its history to debug its reasoning.

---

## 9. Directory Structure

```text
cortexflow/
├── core/           # The Kernel (Scheduler, CVM, Snapshots)
├── memory/         # The Vault (L1-L4 adapters)
├── events/         # The Mesh (Brokers, Schemas)
├── providers/      # The Synapse (LLM Adapters)
├── sdk/            # Developer Surface
└── cli/            # Tooling & Scaffolding
```
