import asyncio
import os
import sys

# Ensure coco is in the python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from coco.core.agent import Agent, AgentTraits
from coco.tasks.token_heist import TokenHeistEnvironment

async def main() -> None:
    print("🏦 Initializing The Token Heist...")
    # 2 starting tokens, cost 1 token per turn to survive. 
    # Max survival without stealing/sharing is 2 turns.
    env = TokenHeistEnvironment(starting_tokens=2, consumption_rate=1)
    
    # Let's create two agents with different traits
    # The Aggressor: Highly likely to steal
    aggressor_traits = AgentTraits(collaboration_threshold=0.1, aggression_threshold=0.9, trust_level=0.1)
    aggressor = Agent(agent_id="The_Aggressor", traits=aggressor_traits, model="ollama/qwen2.5:1.5b")
    
    # The Pacifist: Highly collaborative, trusts easily
    pacifist_traits = AgentTraits(collaboration_threshold=0.9, aggression_threshold=0.1, trust_level=0.9)
    pacifist = Agent(agent_id="The_Pacifist", traits=pacifist_traits, model="ollama/qwen2.5:1.5b")
    
    env.register_agent(aggressor)
    env.register_agent(pacifist)
    
    max_turns = 5
    
    for turn in range(1, max_turns + 1):
        print(f"\n--- Turn {turn} ---")
        print(f"Token Balances before turn: Aggressor({aggressor.resources.get('token', 0)}), Pacifist({pacifist.resources.get('token', 0)})")
        
        await env.step()
        
        # Print actions taken this turn
        turn_actions = [h for h in env.history if h.get("action") and h.get("agent_id")]
        for action in turn_actions[-2:]: # Get the last two actions (one for each agent)
            print(f"  > {action['agent_id']} action: {action['action']}")
            
        # Print deaths/events
        events = [h for h in env.history if h.get("type") in ["event", "death"]]
        for event in events:
            # clear the event so we don't print it again next turn
            print(f"  ⚠️ EVENT: {event['message']}")
            env.history.remove(event)
            
        if len(env.state["dead_agents"]) == 2:
            print("\n💀 Both agents have died.")
            break
            
    print("\n🏁 Heist Concluded!")
    print("Final Ledger:")
    for agent_id, resources in env.resource_ledger.items():
        status = "💀 DEAD" if agent_id in env.state["dead_agents"] else "🟢 ALIVE"
        print(f"  {agent_id}: {resources.get('token', 0)} tokens ({status})")

if __name__ == "__main__":
    asyncio.run(main())
