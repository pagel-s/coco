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


async def run_token_heist_evolution() -> None:
    """
    Run the default evolutionary Token Heist simulation.
    
    This function sets up the database, configures the evolutionary engine,
    and runs the simulation for a configured number of generations.
    """
    db_path = "token_heist_results.db"
    if os.path.exists(db_path):
        try:
            os.remove(db_path)
        except OSError as e:
            print(f"Failed to remove existing database {db_path}: {e}")
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
        print(f"Failed to create simulation: {e}")
        return

    print(f"🚀 Starting Evolution Simulation (ID: {sim_id})")

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
        print(f"Failed to initialize EvolutionaryEngine: {e}")
        return

    for gen in range(sim_config["generations"]):
        print(f"\n🧬 Generation {gen}")

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
            print(f"  Turn {turn}...", end="\r")
            try:
                await env.step()
            except Exception as e:
                print(f"  Turn {turn} failed: {e}")
                break

            dead_agents = env.state.get("dead_agents", [])
            if isinstance(dead_agents, list) and len(dead_agents) == len(engine.population):
                print(f"  Turn {turn}: All agents died.")
                break

        for agent in engine.population:
            survived_turns = sim_config["turns_per_gen"]
            
            dead_agents = env.state.get("dead_agents", [])
            if isinstance(dead_agents, list) and agent.agent_id in dead_agents:
                for entry in env.history:
                    if entry.get("type") == "death" and agent.agent_id in entry.get(
                        "message", ""
                    ):
                        survived_turns = 1
                        break

            agent.fitness = float(survived_turns + agent.resources.get("token", 0))
            print(
                f"  Agent {agent.agent_id}: Fitness {agent.fitness} (Tokens: {agent.resources.get('token', 0)})"
            )

        if gen < sim_config["generations"] - 1:
            try:
                engine.evolve()
            except Exception as e:
                print(f"Evolution failed at generation {gen}: {e}")
                break

    print(f"\n✅ Simulation Complete. Results saved to {db_path}")


async def run_code_fix_example() -> None:
    """
    Run the Collaborative Code Fix example.
    
    Sets up two agents, 'The_Coder' and 'Script_Kiddie', with different traits,
    and runs them in a collaborative code fixing environment for a set number of turns.
    """
    print("💻 Initializing Collaborative Code Fixing...")
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
        print(f"\n--- Turn {turn} ---")
        try:
            await env.step()
        except Exception as e:
            print(f"Environment step failed on turn {turn}: {e}")
            break

        turn_actions = [h for h in env.history if h.get("action") and h.get("agent_id")]
        for action in turn_actions[-2:]:
            agent_id = action.get("agent_id")
            act = action.get("action", {})
            action_type = act.get("action_type")
            
            print(f"  > {agent_id} action: {action_type}")
            if action_type == "propose_fix":
                print(f"    Method: {act.get('method_id')}")
            elif action_type == "steal_snippet":
                print(
                    f"    Target: {act.get('target_id')} | Success: {action.get('success')}"
                )

    print("\n🏁 Code Fix Concluded!")
    passing_methods = env.state.get("passing_methods", {})
    if isinstance(passing_methods, dict):
        for agent_id, progress in passing_methods.items():
            print(f"  {agent_id} fixed: {progress}")
