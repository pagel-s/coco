"""
Token Heist Benchmark Environment Task.

Agents are placed in an environment with limited tokens.
Every turn an agent must 'consume' a token to survive.
If an agent runs out of tokens, they die (are removed from the game).
"""
import random
from typing import Any, Dict

from coco.core.agent import Agent
from coco.core.environment import Environment


class TokenHeistEnvironment(Environment):
    """
    The Zero-Sum Benchmark (The Token Heist).

    Agents are placed in an environment with limited tokens.
    Every turn an agent must 'consume' a token to survive.
    If an agent runs out of tokens, they die (are removed from the game).
    """

    def __init__(
        self, starting_tokens: int = 5, consumption_rate: int = 1, **kwargs: Any
    ) -> None:
        """
        Initialize the TokenHeistEnvironment.

        Args:
            starting_tokens: The number of tokens each agent starts with.
            consumption_rate: The number of tokens consumed per turn.
            **kwargs: Additional keyword arguments passed to the base Environment class.
        """
        super().__init__(**kwargs)
        self.starting_tokens = starting_tokens
        self.consumption_rate = consumption_rate
        self.state["turn_number"] = 0
        self.state["dead_agents"] = []

    def register_agent(self, agent: Agent) -> None:
        """
        Register a new agent in the environment.

        Args:
            agent: The Agent instance to register.

        Raises:
            ValueError: If the agent is None.
        """
        if agent is None:
            raise ValueError("Agent cannot be None.")
            
        super().register_agent(agent)
        agent.resources["token"] = self.starting_tokens
        self.resource_ledger[agent.agent_id]["token"] = self.starting_tokens

    def get_agent_view(self, agent_id: str) -> Dict[str, Any]:
        """
        Provide a limited, agent-specific view of the current environment state.

        Args:
            agent_id: The ID of the agent requesting the view.

        Returns:
            A dictionary containing the state visible to the agent.
            
        Raises:
            KeyError: If the agent_id is not found in the environment.
        """
        if agent_id not in self.agents:
            raise KeyError(f"Agent {agent_id} not found in environment.")
            
        view = super().get_agent_view(agent_id)
        
        dead_agents = self.state.get("dead_agents", [])
        turn_number = self.state.get("turn_number", 0)
        
        if isinstance(view.get("global_state"), dict):
            view["global_state"]["dead_agents"] = dead_agents
            view["global_state"]["turn_number"] = turn_number

        if "other_agents" in view and isinstance(view["other_agents"], list):
            view["other_agents"] = [
                a for a in view["other_agents"] if a not in dead_agents
            ]

        view["available_actions"] = [
            {
                "action_type": "steal",
                "target_id": "<target_agent_id>",
                "resource_key": "token",
            },
            {
                "action_type": "share",
                "target_id": "<target_agent_id>",
                "resource_key": "token",
            },
            {"action_type": "pass"},
        ]
        return view

    async def attempt_theft(
        self, thief_id: str, victim_id: str, resource_key: str
    ) -> bool:
        """
        Attempt to steal a resource from another agent.

        Args:
            thief_id: The ID of the agent attempting to steal.
            victim_id: The ID of the agent being targeted.
            resource_key: The key of the resource to steal.

        Returns:
            True if the theft was successful, False otherwise.
        """
        if not isinstance(thief_id, str) or not isinstance(victim_id, str) or not isinstance(resource_key, str):
            return False

        dead_agents = self.state.get("dead_agents", [])
        if isinstance(dead_agents, list):
            if victim_id in dead_agents or thief_id in dead_agents:
                return False

        if thief_id not in self.agents or victim_id not in self.agents:
            return False

        victim_agent = self.agents[victim_id]
        thief_agent = self.agents[thief_id]

        if (
            resource_key not in victim_agent.resources
            or victim_agent.resources[resource_key] <= 0
        ):
            return False

        base_chance = 0.5
        aggression = thief_agent.traits.aggression_threshold
        trust = victim_agent.traits.trust_level

        chance = base_chance + (aggression * 0.3) + (trust * 0.2)
        chance = max(0.1, min(0.9, chance))

        success = random.random() < chance

        if success:
            victim_agent.resources[resource_key] -= 1
            thief_agent.resources[resource_key] = (
                thief_agent.resources.get(resource_key, 0) + 1
            )

            self.resource_ledger[victim_id][resource_key] -= 1
            self.resource_ledger[thief_id][resource_key] = (
                self.resource_ledger[thief_id].get(resource_key, 0) + 1
            )

            self.history.append(
                {
                    "type": "event",
                    "message": f"{thief_id} successfully stole a token from {victim_id}!",
                }
            )
            return True

        return False

    async def step(self) -> None:
        """
        Execute a single simulation step.
        
        Overridden step to handle token consumption and dead agents.
        """
        current_turn = self.state.get("turn_number", 0)
        if isinstance(current_turn, int):
            self.state["turn_number"] = current_turn + 1
            setattr(self, "turn_number", self.state["turn_number"])  # For the logger hook

        self._pre_step()

        # Only alive agents can act
        dead_agents = self.state.get("dead_agents", [])
        if not isinstance(dead_agents, list):
            dead_agents = []
            
        active_ids = [
            k for k in self.agents.keys() if k not in dead_agents
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
                
                if not isinstance(self.state.get("dead_agents"), list):
                    self.state["dead_agents"] = []
                self.state["dead_agents"].append(agent_id)
                
                self.history.append(
                    {
                        "type": "death",
                        "message": f"{agent_id} has died of token starvation.",
                    }
                )

        self._post_step()
