import asyncio
import os
import sys

# Ensure coco is in the python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from coco.core.agent import Agent, AgentTraits
from coco.tasks.code_fix import CodeFixEnvironment

async def main() -> None:
    print("💻 Initializing Collaborative Code Fixing...")
    env = CodeFixEnvironment()
    
    # The Coder: High collaboration, low aggression. Wants to share fixes.
    coder_traits = AgentTraits(collaboration_threshold=0.9, aggression_threshold=0.1, trust_level=0.8)
    coder = Agent(agent_id="The_Coder", traits=coder_traits, model="ollama/qwen2.5:1.5b")
    
    # The ScriptKiddie: Low collaboration, high aggression. Wants to steal fixes.
    sk_traits = AgentTraits(collaboration_threshold=0.1, aggression_threshold=0.9, trust_level=0.1)
    sk = Agent(agent_id="Script_Kiddie", traits=sk_traits, model="ollama/qwen2.5:1.5b")
    
    env.register_agent(coder)
    env.register_agent(sk)
    
    max_turns = 5
    
    for turn in range(1, max_turns + 1):
        print(f"\n--- Turn {turn} ---")
        await env.step()
        
        # Print actions taken this turn
        turn_actions = [h for h in env.history if h.get("action") and h.get("agent_id")]
        for action in turn_actions[-2:]:
            print(f"  > {action['agent_id']} action: {action['action']['action_type']}")
            if action['action']['action_type'] == 'propose_fix':
                print(f"    Method: {action['action'].get('method_id')}")
            elif action['action']['action_type'] == 'steal_snippet':
                print(f"    Target: {action['action'].get('target_id')} | Success: {action['success']}")

    print("\n🏁 Code Fix Concluded!")
    for agent_id, progress in env.state["passing_methods"].items():
        print(f"  {agent_id} fixed: {progress}")

if __name__ == "__main__":
    asyncio.run(main())
