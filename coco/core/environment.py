from typing import Dict, Any, List

class Environment:
    def __init__(self):
        # The true state of the world
        self.state: Dict[str, Any] = {}
        
        # Registry of active agents
        self.agents: Dict[str, Any] = {}
        
        # Track what resources are owned by whom. 
        # Crucial for the steal mechanic.
        self.resource_ledger: Dict[str, Dict[str, Any]] = {}
        
    def register_agent(self, agent: Any):
        self.agents[agent.agent_id] = agent
        self.resource_ledger[agent.agent_id] = {}

    async def transfer_resource(self, source_id: str, target_id: str, resource_key: str) -> bool:
        """
        Handles voluntary sharing between agents.
        """
        if source_id not in self.agents or target_id not in self.agents:
            return False
            
        source_agent = self.agents[source_id]
        if resource_key in source_agent.resources:
            # Transfer logic
            resource = source_agent.resources.pop(resource_key)
            self.agents[target_id].resources[resource_key] = resource
            
            # Update ledger
            self.resource_ledger[target_id][resource_key] = resource
            if resource_key in self.resource_ledger[source_id]:
                del self.resource_ledger[source_id][resource_key]
                
            return True
        return False

    async def attempt_theft(self, thief_id: str, victim_id: str, resource_key: str) -> bool:
        """
        Handles the steal mechanic. 
        This is where we can implement complex rules: 
        e.g., probability of success based on thief's stealth vs victim's defense,
        or cost to the thief if caught.
        """
        if thief_id not in self.agents or victim_id not in self.agents:
            return False

        victim_agent = self.agents[victim_id]
        
        if resource_key not in victim_agent.resources:
            return False

        # TODO: Implement complex resolution logic (dice roll, stat check, etc.)
        # For now, a naive 50% chance if the resource exists
        import random
        success = random.random() > 0.5
        
        if success:
            resource = victim_agent.resources.pop(resource_key)
            self.agents[thief_id].resources[resource_key] = resource
            
            # Update ledger
            self.resource_ledger[thief_id][resource_key] = resource
            if resource_key in self.resource_ledger[victim_id]:
                del self.resource_ledger[victim_id][resource_key]
                
            # Note: We might want to alert the victim or update social ledger
            return True
            
        return False

    async def step(self):
        """
        Advances the simulation by one tick/turn.
        All agents will observe the environment and take an action.
        """
        # This will be orchestrated by asyncio.gather or similar in the main loop
        pass
