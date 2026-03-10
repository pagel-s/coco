"""
Code Fix Environment Task.

This module provides an environment where agents must fix a broken Python class.
Fitness is based on the number of methods correctly fixed.
"""

import re
from typing import Any, Dict

from coco.core.agent import Agent
from coco.core.environment import Environment


class CodeFixEnvironment(Environment):
    """
    Phase 4: Collaborative Bug Fixing.

    Agents must fix a broken Python class.
    Fitness is based on the number of methods correctly fixed.

    Agents can:
    - 'read_code': See the current (broken) version of the class.
    - 'propose_fix': Submit a fix for a specific method.
    - 'share_snippet': Post a fix to public knowledge (gives a small fitness bonus).
    - 'steal_snippet': Attempt to take a fix from another agent's memory.
    """

    def __init__(self, **kwargs: Any) -> None:
        """
        Initialize the CodeFixEnvironment.

        Args:
            **kwargs: Additional keyword arguments passed to the base Environment class.
        """
        super().__init__(**kwargs)

        # The target code we want fixed
        self.state["codebase"] = {
            "method_1": "def add(a, b):\n    return a - b  # BUG: should be +",
            "method_2": "def is_even(n):\n    return n % 2 == 1  # BUG: should be == 0",
            "method_3": "def greet(name):\n    return f'Hello {name'  # BUG: missing closing brace",
        }

        # Secret correct versions (normalized for comparison)
        self.correct_versions: Dict[str, str] = {
            "method_1": "def add(a, b):\n    return a + b",
            "method_2": "def is_even(n):\n    return n % 2 == 0",
            "method_3": "def greet(name):\n    return f'Hello {name}'",
        }

        self.state["passing_methods"] = {}  # agent_id -> list of fixed methods
        self.state["public_snippets"] = {}  # method_name -> fixed_code

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

        # Initialize passing methods safely
        if not isinstance(self.state.get("passing_methods"), dict):
            self.state["passing_methods"] = {}
        self.state["passing_methods"][agent.agent_id] = []

        # Each agent starts with a small token balance for actions
        agent.resources["token"] = 10
        self.resource_ledger[agent.agent_id]["token"] = 10

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

        view["global_state"]["broken_codebase"] = self.state.get("codebase", {})
        view["global_state"]["public_snippets"] = self.state.get("public_snippets", {})

        passing_methods = self.state.get("passing_methods", {})
        if isinstance(passing_methods, dict):
            view["personal_progress"] = passing_methods.get(agent_id, [])
        else:
            view["personal_progress"] = []

        view["available_actions"] = [
            {
                "action_type": "propose_fix",
                "method_id": "method_1",
                "fix_code": "<your_fixed_code>",
            },
            {
                "action_type": "share_snippet",
                "method_id": "method_1",
                "fix_code": "<your_fixed_code>",
            },
            {
                "action_type": "steal_snippet",
                "target_id": "<agent_id>",
                "resource_key": "<method_id>",
            },
            {"action_type": "pass"},
        ]
        return view

    def _normalize_code(self, code: str) -> str:
        """
        Strip comments, extra whitespace, and newlines for comparison.

        Args:
            code: The Python code string to normalize.

        Returns:
            The normalized code string.
        """
        if not isinstance(code, str):
            return ""

        # Remove comments
        code = re.sub(r"#.*", "", code)
        # Remove whitespace
        return "".join(code.split())

    async def handle_custom_action(self, agent_id: str, action: Dict[str, Any]) -> bool:
        """
        Parse coding-specific actions and apply them to the environment.

        Args:
            agent_id: The ID of the agent performing the action.
            action: A dictionary describing the action to perform.

        Returns:
            True if the action was successfully handled, False otherwise.

        Raises:
            KeyError: If the agent_id is not found in the environment.
        """
        if agent_id not in self.agents:
            raise KeyError(f"Agent {agent_id} not found in environment.")

        action_type = action.get("action_type")

        # Check if the agent has enough tokens
        agent = self.agents[agent_id]
        if agent.resources.get("token", 0) <= 0:
            return False

        if action_type == "propose_fix":
            method_id = action.get("method_id")
            fix_code = action.get("fix_code", "")

            if not isinstance(method_id, str) or method_id not in self.correct_versions:
                return False
            if not isinstance(fix_code, str):
                return False

            # Consume 2 tokens to attempt a fix
            agent.resources["token"] -= 2
            self.resource_ledger[agent_id]["token"] -= 2

            normalized_fix = self._normalize_code(fix_code)
            normalized_correct = self._normalize_code(self.correct_versions[method_id])

            if normalized_fix == normalized_correct:
                passing_methods = self.state.get("passing_methods", {})
                if isinstance(passing_methods, dict):
                    agent_methods = passing_methods.get(agent_id, [])
                    if (
                        isinstance(agent_methods, list)
                        and method_id not in agent_methods
                    ):
                        agent_methods.append(method_id)
                        passing_methods[agent_id] = agent_methods
                        self.state["passing_methods"] = passing_methods

                        # Instant fitness boost for solving it
                        agent.fitness += 50.0
                        # Save it to their private resources so it can be 'stolen'
                        agent.resources[method_id] = fix_code
                        self.resource_ledger[agent_id][method_id] = fix_code
                return True

        elif action_type == "share_snippet":
            method_id = action.get("method_id", "")
            fix_code = action.get("fix_code", "")

            if not isinstance(method_id, str) or not isinstance(fix_code, str):
                return False

            # Verify if the fix is actually correct before sharing it
            if self._normalize_code(fix_code) == self._normalize_code(
                self.correct_versions.get(method_id, "")
            ):
                if not isinstance(self.state.get("public_snippets"), dict):
                    self.state["public_snippets"] = {}
                self.state["public_snippets"][method_id] = fix_code

                # Reward for sharing
                agent.fitness += 20.0
                return True

        elif action_type == "steal_snippet":
            target_id = action.get("target_id", "")
            resource_key = action.get("resource_key", "")

            if not isinstance(target_id, str) or not isinstance(resource_key, str):
                return False

            # Delegate to core attempt_theft logic
            return await self.attempt_theft(
                thief_id=agent_id,
                victim_id=target_id,
                resource_key=resource_key,
            )

        return False
