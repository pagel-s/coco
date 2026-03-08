from typing import Dict, Any, List
import asyncio

class Environment:
    def __init__(self) -> None:
        # The true state of the world
        self.state: Dict[str, Any] = {}
        
        # Registry of active agents
        self.agents: Dict[str, Any] = {}
        
        # Track what resources are owned by whom. 
        self.resource_ledger: Dict[str, Dict[str, Any]] = {}
        
        # Log of all actions taken in the environment
        self.history: List[Dict[str, Any]] = []

    def register_agent(self, agent: Any) -> None:
        self.agents[agent.agent_id] = agent
        self.resource_ledger[agent.agent_id] = {}

    def get_agent_view(self, agent_id: str) -> Dict[str, Any]:
        """
        Returns the subset of the environment state visible to a specific agent.
        For now, we return the full state, but this can be restricted later 
        (e.g. fog of war).
        """
        return {
            "global_state": self.state,
            "other_agents": [aid for aid in self.agents.keys() if aid != agent_id]
        }

    async def transfer_resource(self, source_id: str, target_id: str, resource_key: str) -> bool:
        if source_id not in self.agents or target_id not in self.agents:
            return False
            
        source_agent = self.agents[source_id]
        if resource_key in source_agent.resources:
            resource = source_agent.resources.pop(resource_key)
            self.agents[target_id].resources[resource_key] = resource
            
            self.resource_ledger[target_id][resource_key] = resource
            if resource_key in self.resource_ledger[source_id]:
                del self.resource_ledger[source_id][resource_key]
                
            return True
        return False

    async def attempt_theft(self, thief_id: str, victim_id: str, resource_key: str) -> bool:
        if thief_id not in self.agents or victim_id not in self.agents:
            return False

        victim_agent = self.agents[victim_id]
        
        if resource_key not in victim_agent.resources:
            return False

        import random
        # TODO: integrate traits (thief aggression vs victim trust/defense)
        success = random.random() > 0.5
        
        if success:
            resource = victim_agent.resources.pop(resource_key)
            self.agents[thief_id].resources[resource_key] = resource
            
            self.resource_ledger[thief_id][resource_key] = resource
            if resource_key in self.resource_ledger[victim_id]:
                del self.resource_ledger[victim_id][resource_key]
                
            return True
        return False

    async def execute_action(self, agent_id: str, action: Dict[str, Any]) -> None:
        """
        Parses and executes the JSON action returned by an agent.
        """
        action_type = action.get("action_type")
        success = False
        
        if action_type == "steal":
            success = await self.attempt_theft(
                thief_id=agent_id,
                victim_id=action.get("target_id", ""),
                resource_key=action.get("resource_key", "")
            )
        elif action_type == "share":
            success = await self.transfer_resource(
                source_id=agent_id,
                target_id=action.get("target_id", ""),
                resource_key=action.get("resource_key", "")
            )
        elif action_type == "pass":
            success = True
        else:
            # Delegate task-specific actions to a subclass or task handler
            success = await self.handle_custom_action(agent_id, action)

        self.history.append({
            "agent_id": agent_id,
            "action": action,
            "success": success
        })

    async def handle_custom_action(self, agent_id: str, action: Dict[str, Any]) -> bool:
        """
        To be overridden by specific Task environments (e.g. 'guess_number', 'write_code').
        """
        return False

    async def step(self) -> None:
        """
        Advances the simulation by one tick. 
        All agents observe the environment and take an action asynchronously.
        """
        tasks = []
        for agent_id, agent in self.agents.items():
            view = self.get_agent_view(agent_id)
            # Ask the agent to decide on an action
            tasks.append(agent.act(view))
        
        # Gather all actions (Agents "think" in parallel)
        actions = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Execute the actions sequentially to prevent race conditions on the ledger
        for agent_id, action in zip(self.agents.keys(), actions):
            if isinstance(action, Exception):
                print(f"Agent {agent_id} failed to act: {action}")
                continue
            
            # We know it's not an exception here, so cast it for mypy
            from typing import cast
            valid_action = cast(Dict[str, Any], action)
            await self.execute_action(agent_id, valid_action)
