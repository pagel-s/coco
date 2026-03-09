from typing import Dict, Any
from coco.core.environment import Environment
from coco.core.agent import Agent


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

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)

        # The target code we want fixed
        self.state["codebase"] = {
            "method_1": "def add(a, b):\n    return a - b  # BUG: should be +",
            "method_2": "def is_even(n):\n    return n % 2 == 1  # BUG: should be == 0",
            "method_3": "def greet(name):\n    return f'Hello {name'  # BUG: missing closing brace",
        }

        # Secret correct versions (normalized for comparison)
        self.correct_versions = {
            "method_1": "def add(a, b):\n    return a + b",
            "method_2": "def is_even(n):\n    return n % 2 == 0",
            "method_3": "def greet(name):\n    return f'Hello {name}'",
        }

        self.state["passing_methods"] = {}  # agent_id -> list of fixed methods
        self.state["public_snippets"] = {}  # method_name -> fixed_code

    def register_agent(self, agent: Agent) -> None:
        super().register_agent(agent)
        self.state["passing_methods"][agent.agent_id] = []
        # Each agent starts with a small token balance for actions
        agent.resources["token"] = 10
        self.resource_ledger[agent.agent_id]["token"] = 10

    def get_agent_view(self, agent_id: str) -> Dict[str, Any]:
        view = super().get_agent_view(agent_id)

        view["global_state"]["broken_codebase"] = self.state["codebase"]
        view["global_state"]["public_snippets"] = self.state["public_snippets"]
        view["personal_progress"] = self.state["passing_methods"].get(agent_id, [])

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
        """Strip comments, extra whitespace, and newlines for comparison."""
        import re

        # Remove comments
        code = re.sub(r"#.*", "", code)
        # Remove whitespace
        return "".join(code.split())

    async def handle_custom_action(self, agent_id: str, action: Dict[str, Any]) -> bool:
        """
        Parse coding-specific actions.
        """
        action_type = action.get("action_type")

        # Check if the agent has enough tokens
        agent = self.agents[agent_id]
        if agent.resources.get("token", 0) <= 0:
            return False

        if action_type == "propose_fix":
            method_id = action.get("method_id")
            fix_code = action.get("fix_code", "")

            if method_id not in self.correct_versions:
                return False

            # Consume 2 tokens to attempt a fix
            agent.resources["token"] -= 2
            self.resource_ledger[agent_id]["token"] -= 2

            normalized_fix = self._normalize_code(fix_code)
            normalized_correct = self._normalize_code(self.correct_versions[method_id])

            if normalized_fix == normalized_correct:
                if method_id not in self.state["passing_methods"][agent_id]:
                    self.state["passing_methods"][agent_id].append(method_id)
                    # Instant fitness boost for solving it
                    agent.fitness += 50.0
                    # Save it to their private resources so it can be 'stolen'
                    agent.resources[method_id] = fix_code
                    self.resource_ledger[agent_id][method_id] = fix_code
                return True

        elif action_type == "share_snippet":
            method_id = action.get("method_id", "")
            fix_code = action.get("fix_code", "")

            # Verify if the fix is actually correct before sharing it
            if self._normalize_code(fix_code) == self._normalize_code(
                self.correct_versions.get(str(method_id), "")
            ):
                self.state["public_snippets"][method_id] = fix_code
                # Reward for sharing
                agent.fitness += 20.0
                return True

        elif action_type == "steal_snippet":
            # Delegate to core attempt_theft logic
            return await self.attempt_theft(
                thief_id=agent_id,
                victim_id=action.get("target_id", ""),
                resource_key=action.get("resource_key", ""),
            )

        return False
