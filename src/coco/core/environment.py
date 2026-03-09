import asyncio
from typing import Any, Dict, List, Optional


class Environment:
    def __init__(
        self, data_manager: Optional[Any] = None, simulation_id: Optional[int] = None
    ) -> None:
        # The true state of the world
        self.state: Dict[str, Any] = {}

        # Registry of active agents
        self.agents: Dict[str, Any] = {}

        # Track what resources are owned by whom.
        self.resource_ledger: Dict[str, Dict[str, Any]] = {}

        # Log of all actions taken in the environment
        self.history: List[Dict[str, Any]] = []

        # SQLite logging
        self.data_manager = data_manager
        self.simulation_id = simulation_id
        self.current_turn_id: Optional[int] = None

    def register_agent(self, agent: Any) -> None:
        self.agents[agent.agent_id] = agent
        self.resource_ledger[agent.agent_id] = {}

        # Log agent to database if manager is present
        if self.data_manager and self.simulation_id is not None:
            self.data_manager.log_agent(self.simulation_id, agent)

    def get_agent_view(self, agent_id: str) -> Dict[str, Any]:
        return {
            "global_state": self.state,
            "other_agents": [aid for aid in self.agents.keys() if aid != agent_id],
        }

    async def transfer_resource(
        self, source_id: str, target_id: str, resource_key: str
    ) -> bool:
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

    async def attempt_theft(
        self, thief_id: str, victim_id: str, resource_key: str
    ) -> bool:
        if thief_id not in self.agents or victim_id not in self.agents:
            return False

        victim_agent = self.agents[victim_id]

        if resource_key not in victim_agent.resources:
            return False

        import random

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
        action_type = action.get("action_type")
        success = False

        if action_type == "steal":
            success = await self.attempt_theft(
                thief_id=agent_id,
                victim_id=action.get("target_id", ""),
                resource_key=action.get("resource_key", ""),
            )
        elif action_type == "share":
            success = await self.transfer_resource(
                source_id=agent_id,
                target_id=action.get("target_id", ""),
                resource_key=action.get("resource_key", ""),
            )
        elif action_type == "pass":
            success = True
        else:
            success = await self.handle_custom_action(agent_id, action)

        self.history.append(
            {"agent_id": agent_id, "action": action, "success": success}
        )

        if self.data_manager and self.current_turn_id is not None:
            self.data_manager.log_interaction(
                self.current_turn_id, agent_id, action, success
            )
            self.data_manager.log_agent_snapshot(
                self.current_turn_id, self.agents[agent_id]
            )

    async def handle_custom_action(self, agent_id: str, action: Dict[str, Any]) -> bool:
        return False

    def _pre_step(self) -> None:
        """Internal hook to log the start of the turn."""
        if self.data_manager and self.simulation_id is not None:
            generation = getattr(self, "generation", 0)
            turn_number = getattr(self, "turn_number", 0)
            self.current_turn_id = self.data_manager.log_turn(
                self.simulation_id, generation, turn_number, self.state
            )

    async def _run_agent_actions(self, agent_ids: List[str]) -> None:
        """Prompts all active agents and executes their choices."""
        tasks = []
        for agent_id in agent_ids:
            agent = self.agents[agent_id]
            view = self.get_agent_view(agent_id)
            tasks.append(agent.act(view))

        actions = await asyncio.gather(*tasks, return_exceptions=True)

        from typing import cast

        for agent_id, action in zip(agent_ids, actions):
            if isinstance(action, Exception):
                print(f"Agent {agent_id} failed to act: {action}")
                continue

            valid_action = cast(Dict[str, Any], action)
            await self.execute_action(agent_id, valid_action)

    def _post_step(self) -> None:
        """Optional hook for environment-specific cleanup/rules."""
        pass

    async def step(self) -> None:
        """The standard step workflow."""
        self._pre_step()
        # By default, all registered agents are active
        await self._run_agent_actions(list(self.agents.keys()))
        self._post_step()
