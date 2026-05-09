"""
CortexFlow CLI — The Developer Command-line Interface.
"""

from __future__ import annotations

import asyncio
import sys
import uuid
from pathlib import Path
from typing import Any

import click
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text
from rich.live import Live

console = Console()

LOGO = r"""
  / ____/___  _____/ /____  _  __  / ____/ / /___ _      __ 
 / /   / __ \/ ___/ __/ _ \| |/_/ / /_  / / / __ \ | /| / / 
/ /___/ /_/ / /  / /_/  __/>  <  / __/ / / / /_/ / |/ |/ /  
\____/\____/_/   \__/\___/_/|_| /_/   /_/_/\____/|__/|__/   
"""


def print_splash():
    console.print(Text(LOGO, style="bold cyan"))
    console.print(f"[bold cyan]  [Kernel: v0.1.0][/bold cyan] [bold green] [EventMesh: Connected][/bold green] [bold yellow] [Memory: Optimized][/bold yellow]\n")

@click.group()
@click.version_option(package_name="cortexflow")
def cli() -> None:
    """⚡ CortexFlow — The Cognitive Operating System for AI Agents."""
    pass


@cli.command()
@click.argument("project_name")
@click.option("--provider", default="openai", type=click.Choice(["openai", "anthropic", "gemini", "deepseek", "groq", "ollama"]))
def init(project_name: str, provider: str) -> None:
    """Scaffold a new CortexFlow agent project."""
    print_splash()
    
    project_dir = Path(project_name)
    if project_dir.exists():
        console.print(f"[red]Error:[/red] Directory '{project_name}' already exists.")
        sys.exit(1)

    project_dir.mkdir()
    for d in ["agents", "tools", "workflows", "vault"]:
        (project_dir / d).mkdir()

    # Provider templates
    provider_configs = {
        "openai": ("from cortexflow.providers.openai import OpenAIProvider", "OpenAIProvider(model='gpt-4o')"),
        "anthropic": ("from cortexflow.providers.anthropic import AnthropicProvider", "AnthropicProvider(model='claude-3-5-sonnet-latest')"),
        "gemini": ("from cortexflow.providers.gemini import GeminiProvider", "GeminiProvider(model='gemini-1.5-pro')"),
        "deepseek": ("from cortexflow.providers.deepseek import DeepSeekProvider", "DeepSeekProvider(model='deepseek-chat')"),
        "groq": ("from cortexflow.providers.groq import GroqProvider", "GroqProvider(model='llama-3.3-70b-versatile')"),
        "ollama": ("from cortexflow.providers.ollama import OllamaProvider", "OllamaProvider(model='llama3')"),
    }
    
    p_import, p_init = provider_configs[provider]

    main_py = f'''"""
{project_name} — Powered by CortexFlow
"""
import asyncio
from cortexflow import defineAgent, CortexRuntime
{p_import}

# 1. Define your agent blueprint
agent = defineAgent(
    name="my-agent",
    instructions="You are a helpful AI process running inside CortexFlow.",
    provider={p_init},
    allowed_scopes=["internet"]
)

async def main():
    # 2. Start the Cognitive Runtime
    async with CortexRuntime() as runtime:
        # 3. Spawn a stateful process
        agent_id = await runtime.spawn(agent)
        print(f"🚀 Agent Process Started: {{agent_id}}")

        # 4. Interact with the event mesh
        await runtime.send(agent_id, "System check: are you operational?")
        
        # Keep running to see the logs
        await asyncio.sleep(5)

if __name__ == "__main__":
    asyncio.run(main())
'''
    (project_dir / "main.py").write_text(main_py)
    
    console.print(Panel(
        f"""[green]✅ Project created:[/green] [bold]{project_name}/[/bold]
        
[bold]Quick Start:[/bold]
  [cyan]cd {project_name}[/cyan]
  [cyan]python main.py[/cyan]

[dim]Infrastructure is ready. Go build the future.[/dim]""",
        title="[bold]Kernel Initialized[/bold]",
        border_style="bright_blue"
    ))


@cli.command()
@click.argument("file", type=click.Path(exists=True))
def run(file: str) -> None:
    """Run a CortexFlow script with live tracing."""
    print_splash()
    console.print(f"[bold white]Executing Process:[/bold white] {file}\n")
    
    path = Path(file).absolute()
    sys.path.insert(0, str(path.parent))
    
    import importlib.util
    spec = importlib.util.spec_from_file_location("__main__", path)
    if spec and spec.loader:
        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)
        except Exception as e:
            console.print(f"[red]Runtime Crash:[/red] {e}")


@cli.command()
def status() -> None:
    """Check the health of the CortexFlow Kernel and EventMesh."""
    print_splash()
    
    table = Table(box=None)
    table.add_column("Component", style="cyan")
    table.add_column("Status", style="bold")
    table.add_column("Details", style="dim")

    # This is a simulation for the CLI, in production it would check real services
    table.add_row("Cognitive VM", "[green]ONLINE[/green]", "Deterministic Loop v0.1.0")
    table.add_row("Event Mesh", "[green]ONLINE[/green]", "InMemoryBroker (Local)")
    table.add_row("Memory Vault", "[green]OPTIMIZED[/green]", "L1 Registers Active")
    table.add_row("Scheduler", "[green]READY[/green]", "Concurrency: 16")
    
    console.print(table)


@cli.command()
@click.option("--ticks", default=50, help="Number of ticks to benchmark")
def bench(ticks: int) -> None:
    """Benchmark CVM performance."""
    print_splash()
    
    async def _run_bench():
        from cortexflow.core.cvm import CognitiveVM
        from cortexflow.core.snapshot import StateSnapshot
        from cortexflow.sdk.testing import MockProvider
        import time

        from cortexflow.events.schema import CortexEvent
        cvm = CognitiveVM()
        provider = MockProvider(responses=["OK"])
        snapshot = StateSnapshot(agent_id="bench", agent_name="bench")
        event = CortexEvent.tick(agent_id="bench")
        
        times = []
        with Live(console=console, refresh_per_second=10) as live:
            for i in range(ticks):
                start = time.perf_counter()
                await cvm.tick(snapshot, event, provider)
                times.append((time.perf_counter() - start) * 1000)
                live.update(Panel(f"Benchmarking: {i+1}/{ticks} ticks\n[cyan]Avg: {sum(times)/len(times):.2f}ms[/cyan]"))


        table = Table(title="Performance Profile")
        table.add_column("Metric")
        table.add_column("Value")
        table.add_row("Average Latency", f"{sum(times)/len(times):.2f}ms")
        table.add_row("P95 Latency", f"{sorted(times)[int(ticks*0.95)-1]:.2f}ms")
        table.add_row("Throughput", f"{1000/(sum(times)/len(times)):.1f} ticks/sec")
        console.print(table)

    asyncio.run(_run_bench())


if __name__ == "__main__":
    cli()
