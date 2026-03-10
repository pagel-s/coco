"""Module containing the core Environment implementation."""

import asyncio
import random
from typing import TYPE_CHECKING, Any, Dict, List, Optional, cast

if TYPE_CHECKING:
    from coco.core.agent import Agent


class Environment:
    """The central state manager for the simulation.

    Handles agent registration, resource management, and execution of agent actions.

    Attributes:
        state: A dictionary representing the true global state.
        agents: A dictionary of all registered agents by their IDs.
        resource_ledger: A dictionary mapping agent IDs to their resources.
        history: A list of executed actions in chronological order.
        data_manager: An optional database manager for logging events.
        simulation_id: An optional integer identifying the simulation.
        current_turn_id: The identifier of the currently active turn.
    """

    def __init__(
        self, data_manager: Optional[Any] = None, simulation_id: Optional[int] = None
    ) -> None:
        """Initializes an Environment instance.

        Args:
            data_manager: An optional object to manage SQLite logging.
            simulation_id: Optional numerical ID for tracking simulations.
        """
        self.state: Dict[str, Any] = {}
        self.agents: Dict[str, "Agent"] = {}
        self.resource_ledger: Dict[str, Dict[str, Any]] = {}
        self.history: List[Dict[str, Any]] = []

        self.data_manager = data_manager
        self.simulation_id = simulation_id
        self.current_turn_id: Optional[int] = None

    def register_agent(self, agent: "Agent") -> None:
        """Registers a new agent into the environment.

        Args:
            agent: The Agent instance to add.

        Raises:
            ValueError: If the agent is already registered.
        """
        if agent.agent_id in self.agents:
            raise ValueError(f"Agent with ID '{agent.agent_id}' is already registered.")

        self.agents[agent.agent_id] = agent
        self.resource_ledger[agent.agent_id] = {}

        if self.data_manager and self.simulation_id is not None:
            self.data_manager.log_agent(self.simulation_id, agent)

    def get_agent_view(self, agent_id: str) -> Dict[str, Any]:
        """Provides an agent-specific view of the environment.

        Args:
            agent_id: The identifier of the agent requesting the view.

        Returns:
            A dictionary containing the global state and visible agents.

        Raises:
            KeyError: If the agent is not registered.
        """
        if agent_id not in self.agents:
            raise KeyError(f"Agent with ID '{agent_id}' is not registered.")

        return {
            "global_state": self.state,
            "other_agents": [aid for aid in self.agents.keys() if aid != agent_id],
        }

    async def transfer_resource(
        self, source_id: str, target_id: str, resource_key: str
    ) -> bool:
        """Transfers a resource from one agent to another.

        Args:
            source_id: ID of the sender agent.
            target_id: ID of the receiver agent.
            resource_key: Identifier of the resource to be transferred.

        Returns:
            A boolean indicating if the transfer succeeded.
        """
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
        """Attempts a randomized theft of a resource between two agents.

        Args:
            thief_id: ID of the agent attempting to steal.
            victim_id: ID of the agent being stolen from.
            resource_key: Identifier of the target resource.

        Returns:
            A boolean indicating whether the theft succeeded.
        """
        if thief_id not in self.agents or victim_id not in self.agents:
            return False

        victim_agent = self.agents[victim_id]
        if resource_key not in victim_agent.resources:
            return False

        # 50% chance to succeed
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
        """Executes an action proposed by an agent.

        Args:
            agent_id: ID of the agent performing the action.
            action: A dictionary detailing the action type and parameters.

        Raises:
            ValueError: If the action is invalid or not actionable.
        """
        if not action or not isinstance(action, dict):
            raise ValueError("action must be a non-empty dictionary.")

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
        """Handles actions that are not standard (e.g. not steal/share/pass).

        Designed to be overridden by subclasses.

        Args:
            agent_id: The ID of the agent performing the action.
            action: The dictionary defining the action.

        Returns:
            A boolean indicating success. Defaults to False.
        """
        return False

    def _pre_step(self) -> None:
        """Hook called before the start of each simulation step."""
        if self.data_manager and self.simulation_id is not None:
            generation = getattr(self, "generation", 0)
            turn_number = getattr(self, "turn_number", 0)
            self.current_turn_id = self.data_manager.log_turn(
                self.simulation_id, generation, turn_number, self.state
            )

    async def _run_agent_actions(self, agent_ids: List[str]) -> None:
        """Gathers and executes the proposed actions for a list of agents.

        Args:
            agent_ids: List of agent identifiers participating in the current step.
        """
        tasks = []
        for agent_id in agent_ids:
            if agent_id not in self.agents:
                continue
            agent = self.agents[agent_id]
            view = self.get_agent_view(agent_id)
            tasks.append(agent.act(view))

        actions = await asyncio.gather(*tasks, return_exceptions=True)

        for agent_id, action in zip(agent_ids, actions):
            if isinstance(action, Exception):
                print(f"Agent {agent_id} failed to act: {action}")
                continue

            valid_action = cast(Dict[str, Any], action)
            try:
                await self.execute_action(agent_id, valid_action)
            except Exception as e:
                print(f"Failed to execute action for agent {agent_id}: {e}")

    def _post_step(self) -> None:
        """Hook called after the completion of each simulation step."""
        pass

    async def step(self) -> None:
        """Executes a single standard turn for all active agents.

        Invokes _pre_step, queries and applies agent actions, and then calls _post_step.
        """
        self._pre_step()
        await self._run_agent_actions(list(self.agents.keys()))
        self._post_step()
