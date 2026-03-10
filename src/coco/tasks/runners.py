"""
Runner module for various tasks.

This module provides runner functions to execute the tasks like
Token Heist and Code Fix.
"""
import os
from typing import Any, Dict

from coco.core.agent import Agent, AgentTraits
from coco.core.database import DataManager
from coco.evolution.engine import EvolutionaryEngine
from coco.tasks.code_fix import CodeFixEnvironment
from coco.tasks.token_heist import TokenHeistEnvironment


async def run_token_heist_evolution(export_json: bool = False) -> None:
    """
    Run the default evolutionary Token Heist simulation.
    
    This function sets up the database, configures the evolutionary engine,
    and runs the simulation for a configured number of generations.
    
    Args:
        export_json: If True, exports the simulation logs to a JSON file.
    """
    import json
    from rich.console import Console
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
    from rich.table import Table

    console = Console()
    db_path = "token_heist_results.db"
    if os.path.exists(db_path):
        try:
            os.remove(db_path)
        except OSError as e:
            console.print(f"[red]Failed to remove existing database {db_path}: {e}[/red]")
            return

    db = DataManager(db_path)

    sim_config: Dict[str, Any] = {
        "task": "Token Heist",
        "population_size": 4,
        "generations": 3,
        "turns_per_gen": 5,
        "starting_tokens": 3,
        "consumption_rate": 1,
    }
    
    try:
        sim_id = db.create_simulation(sim_config)
    except Exception as e:
        console.print(f"[red]Failed to create simulation: {e}[/red]")
        return

    console.print(f"🚀 [bold green]Starting Evolution Simulation[/bold green] (ID: {sim_id})")

    try:
        engine = EvolutionaryEngine(
            environment_class=TokenHeistEnvironment,
            population_size=sim_config["population_size"],
            survival_rate=0.5,
            model="ollama/qwen2.5:1.5b",
            data_manager=db,
            simulation_id=sim_id,
        )
    except Exception as e:
        console.print(f"[red]Failed to initialize EvolutionaryEngine: {e}[/red]")
        return

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        gen_task = progress.add_task("[cyan]Generations", total=sim_config["generations"])
        
        for gen in range(sim_config["generations"]):
            turn_task = progress.add_task(f"  [yellow]Gen {gen} Turns", total=sim_config["turns_per_gen"])

            env = TokenHeistEnvironment(
                starting_tokens=sim_config["starting_tokens"],
                consumption_rate=sim_config["consumption_rate"],
            )
            env.data_manager = db
            env.simulation_id = sim_id
            setattr(env, "generation", gen)

            for agent in engine.population:
                env.register_agent(agent)

            for turn in range(sim_config["turns_per_gen"]):
                try:
                    await env.step()
                except Exception as e:
                    console.print(f"  [red]Turn {turn} failed: {e}[/red]")
                    break

                progress.update(turn_task, advance=1)
                
                dead_agents = env.state.get("dead_agents", [])
                if isinstance(dead_agents, list) and len(dead_agents) == len(engine.population):
                    break
            
            progress.remove_task(turn_task)

            # Calculate and log fitness
            summary_table = Table(title=f"Generation {gen} Results")
            summary_table.add_column("Agent ID", style="cyan")
            summary_table.add_column("Fitness", style="green")
            summary_table.add_column("Tokens", style="yellow")
            summary_table.add_column("Status", style="magenta")

            for agent in engine.population:
                survived_turns = sim_config["turns_per_gen"]
                is_dead = False
                
                dead_agents = env.state.get("dead_agents", [])
                if isinstance(dead_agents, list) and agent.agent_id in dead_agents:
                    is_dead = True
                    for entry in env.history:
                        if entry.get("type") == "death" and agent.agent_id in entry.get(
                            "message", ""
                        ):
                            survived_turns = 1
                            break

                agent.fitness = float(survived_turns + agent.resources.get("token", 0))
                summary_table.add_row(
                    agent.agent_id, 
                    f"{agent.fitness:.1f}", 
                    str(agent.resources.get('token', 0)),
                    "[red]DEAD[/red]" if is_dead else "[green]ALIVE[/green]"
                )

            console.print(summary_table)

            if gen < sim_config["generations"] - 1:
                try:
                    engine.evolve()
                except Exception as e:
                    console.print(f"[red]Evolution failed at generation {gen}: {e}[/red]")
                    break
            
            progress.update(gen_task, advance=1)

    console.print(f"\n✅ [bold green]Simulation Complete.[/bold green] Results saved to [blue]{db_path}[/blue]")

    if export_json:
        try:
            logs = db.get_simulation_logs(sim_id)
            export_path = f"simulation_{sim_id}_logs.json"
            with open(export_path, "w") as f:
                json.dump(logs, f, indent=2)
            console.print(f"📄 [bold cyan]Logs exported to {export_path}[/bold cyan]")
        except Exception as e:
            console.print(f"[red]Failed to export logs: {e}[/red]")


async def run_code_fix_example() -> None:
    """
    Run the Collaborative Code Fix example.
    
    Sets up two agents, 'The_Coder' and 'Script_Kiddie', with different traits,
    and runs them in a collaborative code fixing environment for a set number of turns.
    """
    from rich.console import Console
    from rich.panel import Panel
    from rich.tree import Tree

    console = Console()
    console.print(Panel("💻 [bold cyan]Initializing Collaborative Code Fixing[/bold cyan]"))
    env = CodeFixEnvironment()

    coder_traits = AgentTraits(
        collaboration_threshold=0.9, aggression_threshold=0.1, trust_level=0.8
    )
    coder = Agent(
        agent_id="The_Coder", traits=coder_traits, model="ollama/qwen2.5:1.5b"
    )

    sk_traits = AgentTraits(
        collaboration_threshold=0.1, aggression_threshold=0.9, trust_level=0.1
    )
    sk = Agent(agent_id="Script_Kiddie", traits=sk_traits, model="ollama/qwen2.5:1.5b")

    env.register_agent(coder)
    env.register_agent(sk)

    max_turns = 5

    for turn in range(1, max_turns + 1):
        console.print(f"\n[bold yellow]--- Turn {turn} ---[/bold yellow]")
        try:
            await env.step()
        except Exception as e:
            console.print(f"[red]Environment step failed on turn {turn}: {e}[/red]")
            break

        turn_actions = [h for h in env.history if h.get("action") and h.get("agent_id")]
        for action in turn_actions[-2:]:
            agent_id = action.get("agent_id")
            act = action.get("action", {})
            action_type = act.get("action_type")
            
            tree = Tree(f"[cyan]{agent_id}[/cyan] chose [bold]{action_type}[/bold]")
            if action_type == "propose_fix":
                tree.add(f"Method: [green]{act.get('method_id')}[/green]")
            elif action_type == "steal_snippet":
                tree.add(f"Target: [magenta]{act.get('target_id')}[/magenta] | Success: {'[green]YES[/green]' if action.get('success') else '[red]NO[/red]'}")
            
            console.print(tree)

    console.print("\n🏁 [bold green]Code Fix Concluded![/bold green]")
    passing_methods = env.state.get("passing_methods", {})
    if isinstance(passing_methods, dict):
        for agent_id, progress in passing_methods.items():
            console.print(f"  [cyan]{agent_id}[/cyan] fixed: [bold green]{progress}[/bold green]")
