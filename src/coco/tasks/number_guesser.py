"""
Number Guesser Task Environment.

This module provides a simple environment where agents must guess a target number.
It serves as a basic test of the LLM cognitive loop.
"""
from typing import Any, Dict

from coco.core.environment import Environment


class NumberGuesserEnvironment(Environment):
    """
    A simple task where the agent must guess a target number.
    This serves as the "Hello World" to verify the LLM cognitive loop.
    """

    def __init__(self, target_number: int = 42) -> None:
        """
        Initialize the NumberGuesserEnvironment.

        Args:
            target_number: The integer number the agents must guess.
        """
        super().__init__()
        self.state["target_number"] = target_number
        self.state["min_val"] = 1
        self.state["max_val"] = 100
        self.state["feedback"] = {}
        self.state["winners"] = []

    def get_agent_view(self, agent_id: str) -> Dict[str, Any]:
        """
        Hide the target number from the agents and provide task-specific actions.

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

        # Hide the target number so the LLM doesn't just read it
        if "target_number" in view.get("global_state", {}):
            hidden_state = dict(view["global_state"])
            del hidden_state["target_number"]
            view["global_state"] = hidden_state

        feedback_dict = self.state.get("feedback", {})
        if isinstance(feedback_dict, dict):
            view["personal_feedback"] = feedback_dict.get(
                agent_id, "No guesses made yet. Guess a number between 1 and 100."
            )
        else:
            view["personal_feedback"] = "No guesses made yet. Guess a number between 1 and 100."

        # Inject available actions so the agent knows what to do
        view["available_actions"] = [
            {"action_type": "guess", "value": "<insert integer here>"}
        ]
        return view

    async def handle_custom_action(self, agent_id: str, action: Dict[str, Any]) -> bool:
        """
        Parse the 'guess' action.

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
            
        if action.get("action_type") == "guess":
            guess = action.get("value")

            if not isinstance(self.state.get("feedback"), dict):
                self.state["feedback"] = {}
                
            feedback_dict = self.state["feedback"]

            # Type safety check, LLMs sometimes return strings
            if isinstance(guess, str) and guess.lstrip('-').isdigit():
                guess = int(guess)
            elif not isinstance(guess, int):
                feedback_dict[agent_id] = (
                    f"Invalid guess: '{guess}'. Must be an integer."
                )
                return False

            target = self.state.get("target_number")
            if not isinstance(target, int):
                # Fallback if target is somehow not an int
                return False

            if guess == target:
                feedback_dict[agent_id] = "Correct! You guessed the number."
                
                if not isinstance(self.state.get("winners"), list):
                    self.state["winners"] = []
                    
                winners_list = self.state["winners"]
                if agent_id not in winners_list:
                    winners_list.append(agent_id)
                return True
            elif guess < target:
                feedback_dict[agent_id] = f"Your guess {guess} is too low."
            else:
                feedback_dict[agent_id] = f"Your guess {guess} is too high."
            return True

        return False
