import asyncio
import os
import sys
from typing import Dict, Any

# Ensure coco is in the python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from coco.core.database import DataManager
from coco.evolution.engine import EvolutionaryEngine
from coco.tasks.token_heist import TokenHeistEnvironment

async def run_simulation() -> None:
    db_path = "token_heist_results.db"
    if os.path.exists(db_path):
        os.remove(db_path)
        
    db = DataManager(db_path)
    
    sim_config: Dict[str, Any] = {
        "task": "Token Heist",
        "population_size": 4,
        "generations": 3,
        "turns_per_gen": 5,
        "starting_tokens": 3,
        "consumption_rate": 1
    }
    sim_id = db.create_simulation(sim_config)
    
    print(f"🚀 Starting Evolution Simulation (ID: {sim_id})")
    
    engine = EvolutionaryEngine(
        environment_class=TokenHeistEnvironment,
        population_size=sim_config["population_size"],
        survival_rate=0.5,
        model="ollama/qwen2.5:1.5b",
        data_manager=db,
        simulation_id=sim_id
    )
    
    for gen in range(sim_config["generations"]):
        print(f"\n🧬 Generation {gen}")
        
        # Initialize environment for this generation
        env = TokenHeistEnvironment(
            starting_tokens=sim_config["starting_tokens"], 
            consumption_rate=sim_config["consumption_rate"]
        )
        # Link environment to DB
        env.data_manager = db
        env.simulation_id = sim_id
        # Inject current generation number for logging
        setattr(env, "generation", gen)
        
        # Register the current population
        for agent in engine.population:
            env.register_agent(agent)
            
        # Run turns
        for turn in range(sim_config["turns_per_gen"]):
            print(f"  Turn {turn}...", end="\r")
            await env.step()
            
            # Check if everyone is dead
            if len(env.state["dead_agents"]) == len(engine.population):
                print(f"  Turn {turn}: All agents died.")
                break
        
        # Calculate fitness (turns survived)
        # This is a simple metric: tokens left + turn of death
        for agent in engine.population:
            survived_turns = sim_config["turns_per_gen"]
            if agent.agent_id in env.state["dead_agents"]:
                # Find when they died in history
                for entry in env.history:
                    if entry.get("type") == "death" and agent.agent_id in entry.get("message", ""):
                        # This is a bit hacky, but works for the demo
                        survived_turns = 1 # simplified
                        break
            
            agent.fitness = float(survived_turns + agent.resources.get("token", 0))
            print(f"  Agent {agent.agent_id}: Fitness {agent.fitness} (Tokens: {agent.resources.get('token', 0)})")

        # Evolve to next generation
        if gen < sim_config["generations"] - 1:
            engine.evolve()

    print(f"\n✅ Simulation Complete. Results saved to {db_path}")

def run_simulation_cli() -> None:
    asyncio.run(run_simulation())

if __name__ == "__main__":
    run_simulation_cli()
