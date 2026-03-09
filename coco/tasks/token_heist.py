from typing import Dict, Any, List
from coco.core.environment import Environment
from coco.core.agent import Agent

class TokenHeistEnvironment(Environment):
    """
    The Zero-Sum Benchmark (The Token Heist).
    
    Agents are placed in an environment with limited tokens. 
    Every turn an agent must 'consume' a token to survive.
    If an agent runs out of tokens, they die (are removed from the game).
    """
    
    def __init__(self, starting_tokens: int = 5, consumption_rate: int = 1, **kwargs: Any):
        super().__init__(**kwargs)
        self.starting_tokens = starting_tokens
        self.consumption_rate = consumption_rate
        self.state["turn_number"] = 0
        self.state["dead_agents"] = []

    def register_agent(self, agent: Agent) -> None:
        super().register_agent(agent)
        agent.resources["token"] = self.starting_tokens
        self.resource_ledger[agent.agent_id]["token"] = self.starting_tokens

    def get_agent_view(self, agent_id: str) -> Dict[str, Any]:
        view = super().get_agent_view(agent_id)
        view["global_state"]["dead_agents"] = self.state["dead_agents"]
        view["global_state"]["turn_number"] = self.state["turn_number"]
        
        if "other_agents" in view:
            view["other_agents"] = [
                a for a in view["other_agents"] 
                if a not in self.state["dead_agents"]
            ]
            
        view["available_actions"] = [
            {"action_type": "steal", "target_id": "<target_agent_id>", "resource_key": "token"},
            {"action_type": "share", "target_id": "<target_agent_id>", "resource_key": "token"},
            {"action_type": "pass"}
        ]
        return view

    async def attempt_theft(self, thief_id: str, victim_id: str, resource_key: str) -> bool:
        if victim_id in self.state["dead_agents"] or thief_id in self.state["dead_agents"]:
            return False

        if thief_id not in self.agents or victim_id not in self.agents:
            return False

        victim_agent = self.agents[victim_id]
        thief_agent = self.agents[thief_id]
        
        if resource_key not in victim_agent.resources or victim_agent.resources[resource_key] <= 0:
            return False

        import random
        base_chance = 0.5
        aggression = thief_agent.traits.aggression_threshold
        trust = victim_agent.traits.trust_level
        
        chance = base_chance + (aggression * 0.3) + (trust * 0.2)
        chance = max(0.1, min(0.9, chance))

        success = random.random() < chance
        
        if success:
            victim_agent.resources[resource_key] -= 1
            thief_agent.resources[resource_key] = thief_agent.resources.get(resource_key, 0) + 1
            
            self.resource_ledger[victim_id][resource_key] -= 1
            self.resource_ledger[thief_id][resource_key] = self.resource_ledger[thief_id].get(resource_key, 0) + 1
            
            self.history.append({
                "type": "event", 
                "message": f"{thief_id} successfully stole a token from {victim_id}!"
            })
            return True
            
        return False

    async def step(self) -> None:
        """Overridden step to handle token consumption and dead agents."""
        self.state["turn_number"] += 1
        self.turn_number = self.state["turn_number"] # For the logger hook
        
        self._pre_step()
        
        # Only alive agents can act
        active_ids = [
            k for k in self.agents.keys() 
            if k not in self.state["dead_agents"]
        ]
        
        await self._run_agent_actions(active_ids)
        
        # Token consumption
        for agent_id in active_ids:
            agent = self.agents[agent_id]
            current_tokens = agent.resources.get("token", 0)
            
            if current_tokens >= self.consumption_rate:
                agent.resources["token"] -= self.consumption_rate
                self.resource_ledger[agent_id]["token"] -= self.consumption_rate
            else:
                agent.resources["token"] = 0
                self.resource_ledger[agent_id]["token"] = 0
                self.state["dead_agents"].append(agent_id)
                self.history.append({
                    "type": "death", 
                    "message": f"{agent_id} has died of token starvation."
                })
        
        self._post_step()
