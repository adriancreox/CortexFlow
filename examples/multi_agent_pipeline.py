"""
Multi-Agent Pipeline — Researcher → Analyst → Writer

This is the demo that goes viral.

Three specialized agents collaborate on a research task:
  1. Researcher — gathers raw information
  2. Analyst    — identifies patterns and insights
  3. Writer     — produces the final deliverable

Agents communicate via the CortexFlow Event Mesh.
No shared state, no globals — pure message passing.

Run:
    python examples/multi_agent_pipeline.py
"""

from __future__ import annotations

import asyncio

from cortexflow import CortexRuntime, defineAgent
from cortexflow.events.schema import CortexEvent
from cortexflow.sdk.testing import MockProvider
from cortexflow.sdk.workflow import defineWorkflow

# ── Agent Definitions ─────────────────────────────────────────────────────────

researcher = defineAgent(
    name="researcher",
    instructions="""
    You are a world-class research analyst with access to the internet.
    Your job: gather raw facts, data, and sources on any topic.
    Output a structured summary with key findings and citations.
    Be thorough but concise. Format: bullet points.
    """,
    provider=MockProvider(responses=[
        """Research findings on quantum computing (2025):
        • IBM Quantum reached 1,000+ qubit processors (Eagle, Heron series)
        • Google's Willow chip demonstrated quantum error correction below threshold
        • China's Jiuzhang 3 claimed 10^23x speedup over classical for specific tasks
        • Practical applications emerging in: drug discovery, logistics optimization
        • Key challenge: decoherence time still limiting real-world utility
        Sources: Nature, IBM Research Blog, Google Quantum AI"""
    ]),
    description="Gathers raw research data on any topic",
    tags=["research", "pipeline"],
)

analyst = defineAgent(
    name="analyst",
    instructions="""
    You are a strategic analyst. You receive research data and identify:
    1. The most significant trend
    2. Business implications
    3. Risks and opportunities
    4. A confidence score (0-100) for your analysis
    Output structured JSON.
    """,
    provider=MockProvider(responses=[
        """{
  "key_trend": "Quantum error correction breakthrough",
  "business_implication": "2-3 year window for enterprises to build quantum-ready infrastructure",
  "opportunities": ["Drug discovery acceleration", "Supply chain optimization", "Cryptography migration"],
  "risks": ["Post-quantum security vulnerabilities", "High capital requirements", "Talent scarcity"],
  "confidence_score": 78,
  "recommendation": "Begin quantum literacy programs and audit cryptographic dependencies now"
}"""
    ]),
    description="Identifies patterns and strategic insights",
    tags=["analysis", "pipeline"],
)

writer = defineAgent(
    name="writer",
    instructions="""
    You are an expert technical writer. Transform analysis into clear, compelling reports.
    Target audience: C-suite executives.
    Format: Executive Brief (300 words max).
    Tone: authoritative, actionable, no jargon.
    """,
    provider=MockProvider(responses=[
        """# Quantum Computing: Executive Brief Q2 2025

**Bottom Line:** The quantum computing sector crossed a critical threshold in 2025.
Google's error correction breakthrough and IBM's 1,000+ qubit processors signal
the transition from laboratory curiosity to enterprise consideration.

**What Changed:** For the first time, quantum systems have demonstrated the ability
to correct their own errors below the threshold required for practical computation.
This removes the primary technical barrier to useful quantum advantage.

**Your 18-Month Window:** Organizations that begin preparation now will be positioned
to capture early competitive advantages in drug discovery and logistics—estimated
$850B total addressable market by 2030.

**Immediate Actions Required:**
1. Audit all RSA/ECC cryptographic infrastructure for post-quantum vulnerability
2. Identify 2-3 high-value optimization problems suitable for quantum speedup
3. Establish quantum literacy program for technical leadership

**Risk of Inaction:** Companies that delay post-quantum migration face significant
security exposure as quantum decryption capabilities mature (est. 5-7 years).

*Confidence: 78/100 | Sources: Nature, IBM Research, Google Quantum AI*"""
    ]),
    description="Produces executive-ready reports",
    tags=["writing", "pipeline"],
)


# ── Pipeline Orchestration ────────────────────────────────────────────────────

async def main() -> None:
    print("⚡ CortexFlow — Multi-Agent Pipeline Demo")
    print("━" * 60)
    print("Pipeline: Researcher → Analyst → Writer\n")

    async with CortexRuntime() as runtime:
        # Spawn all agents
        researcher_id = await runtime.spawn(researcher)
        analyst_id = await runtime.spawn(analyst)
        writer_id = await runtime.spawn(writer)

        print(f"✅ Spawned 3 agents:")
        print(f"   🔬 Researcher: {researcher_id}")
        print(f"   📊 Analyst:    {analyst_id}")
        print(f"   ✍️  Writer:     {writer_id}\n")

        # Stage 1: Research
        print("🔬 Stage 1: Research")
        await runtime.send(researcher_id, "Research the latest developments in quantum computing")
        await asyncio.sleep(0.5)

        researcher_snapshot = runtime.get_snapshot(researcher_id)
        research_output = researcher_snapshot.working_memory.get("last_response") if researcher_snapshot else ""
        print(f"   ✅ Research complete ({len(research_output or '')} chars)\n")

        # Stage 2: Analysis
        print("📊 Stage 2: Analysis")
        await runtime.send(analyst_id, f"Analyze this research:\n\n{research_output}")
        await asyncio.sleep(0.5)

        analyst_snapshot = runtime.get_snapshot(analyst_id)
        analysis_output = analyst_snapshot.working_memory.get("last_response") if analyst_snapshot else ""
        print(f"   ✅ Analysis complete\n")

        # Stage 3: Writing
        print("✍️  Stage 3: Writing")
        await runtime.send(writer_id, f"Write an executive brief based on:\n\nRESEARCH:\n{research_output}\n\nANALYSIS:\n{analysis_output}")
        await asyncio.sleep(0.5)

        writer_snapshot = runtime.get_snapshot(writer_id)
        final_report = writer_snapshot.working_memory.get_sync("last_response") if writer_snapshot else ""

        print("\n" + "━" * 60)
        print("📄 FINAL REPORT:")
        print("━" * 60)
        print(final_report)

        # Pipeline stats
        print("\n" + "━" * 60)
        print("📊 Pipeline Statistics:")
        stats = runtime.stats()
        for aid, agent_stats in stats["scheduler"]["agents"].items():
            name = aid.split("-")[0]
            print(f"   {name:12} | ticks: {agent_stats['ticks']} | tokens: {agent_stats['tokens']} | status: {agent_stats['status']}")

    print("\n✅ Pipeline complete.")


if __name__ == "__main__":
    asyncio.run(main())
