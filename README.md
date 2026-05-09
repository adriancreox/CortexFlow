# 🧠 CortexFlow: The Cognitive Operating System for AI Agents

[![Version](https://img.shields.io/badge/version-1.0.0--gold-gold.svg)](https://github.com/adriancreox/cortexflow)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Engine](https://img.shields.io/badge/Engine-CVM--v1-red.svg)](docs/cvm.md)

**CortexFlow** is not another LLM framework. It is a production-grade **Cognitive Operating System** designed to run autonomous digital employees that are persistent, observable, and governed by business logic.

While others are building "prompt wrappers," we are building the **industrial nervous system** for the next generation of AI-native companies.

---

## 🏗️ The 4 Pillars of CortexFlow

### 1. The Kernel (Cognitive VM)
A deterministic execution loop that treats agent reasoning as a sequence of atomic state transitions.
- **Immutability**: Every reasoning step produces a new `StateSnapshot`.
- **Fault Tolerance**: Automatic recovery from provider errors and rate limits.
- **Safety**: Built-in infinite loop detection and budget enforcement.

### 2. The Vault (4-Layer Memory Hierarchy)
Inspired by CPU cache architectures, the Vault ensures agents never "forget" while keeping context windows lean.
- **L1 (Registers)**: Current reasoning trace.
- **L2 (Warm)**: Compressed semantic summaries.
- **L3 (Vector)**: Infinite semantic retrieval via NumPy-powered cosine similarity.
- **L4 (Archive)**: Immutable JSONL audit logs for industrial compliance.

### 3. The Shield (Governance Engine)
A constitutional layer that enforces business policies before any tool is executed.
- **Semantic Guardrails**: Block dangerous actions based on intent.
- **CostGuard**: Real-time token budget enforcement to protect your wallet.
- **Human-in-the-Loop**: Native support for approval workflows.

### 4. The Ghost Runtime (Simulation)
Predict the future before it happens.
- **Shadow Mode**: Stress-test agents with synthetic events.
- **Cost Prediction**: Estimate token usage and ROI before live deployment.

---

## 🚀 Quick Start (60 Seconds)

```bash
pip install cortexflow
```

```python
import asyncio
from cortexflow import defineAgent, CortexRuntime

async def main():
    # 1. Define your agent
    agent = defineAgent(
        name="Analyst",
        instructions="You are a data analyst. Be concise.",
        token_budget=1000
    )

    # 2. Run in a persistent runtime
    async with CortexRuntime() as runtime:
        aid = await runtime.spawn(agent)
        
        # 3. Send a task
        await runtime.send(aid, "Analyze the current market trends.")
        
        # Everything else (Persistence, Archival, Governance) 
        # is handled automatically by the Kernel.
        print(f"Agent {aid} is now thinking...")

if __name__ == "__main__":
    asyncio.run(main())
```

---

## 🛠️ Enterprise Ready
- **Persistence**: Built-in SQLite and Postgres adapters.
- **Distributed**: Powered by Redis Streams for multi-node event mesh.
- **Observable**: Structured JSON logging compatible with ELK/Datadog.
- **Scalable**: Async-first architecture capable of thousands of concurrent agents.

---

## ⚖️ License
Licensed under the **Apache License, Version 2.0**. Safe for commercial use and patent-protected.

---

**Built with ❤️ for the future of autonomous work.**  
*By [Adrian Creox](https://github.com/adriancreox) - CREOX Studios*
