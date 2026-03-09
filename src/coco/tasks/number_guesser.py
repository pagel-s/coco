from typing import Any, Dict

from coco.core.environment import Environment


class NumberGuesserEnvironment(Environment):
    """
    A simple task where the agent must guess a target number.
    This serves as the "Hello World" to verify the LLM cognitive loop.
    """

    def __init__(self, target_number: int = 42):
        super().__init__()
        self.state["target_number"] = target_number
        self.state["min_val"] = 1
        self.state["max_val"] = 100
        self.state["feedback"] = {}
        self.state["winners"] = []

    def get_agent_view(self, agent_id: str) -> Dict[str, Any]:
        """
        Hide the target number from the agents and provide task-specific actions.
        """
        view = super().get_agent_view(agent_id)

        # Hide the target number so the LLM doesn't just read it
        if "target_number" in view["global_state"]:
            hidden_state = dict(view["global_state"])
            del hidden_state["target_number"]
            view["global_state"] = hidden_state

        view["personal_feedback"] = self.state["feedback"].get(
            agent_id, "No guesses made yet. Guess a number between 1 and 100."
        )

        # Inject available actions so the agent knows what to do
        view["available_actions"] = [
            {"action_type": "guess", "value": "<insert integer here>"}
        ]
        return view

    async def handle_custom_action(self, agent_id: str, action: Dict[str, Any]) -> bool:
        """
        Parse the 'guess' action.
        """
        if action.get("action_type") == "guess":
            guess = action.get("value")

            # Type safety check, LLMs sometimes return strings
            if isinstance(guess, str) and guess.isdigit():
                guess = int(guess)
            elif not isinstance(guess, int):
                self.state["feedback"][agent_id] = (
                    f"Invalid guess: '{guess}'. Must be an integer."
                )
                return False

            target = self.state["target_number"]
            if guess == target:
                self.state["feedback"][agent_id] = "Correct! You guessed the number."
                if agent_id not in self.state["winners"]:
                    self.state["winners"].append(agent_id)
                return True
            elif guess < target:
                self.state["feedback"][agent_id] = f"Your guess {guess} is too low."
            else:
                self.state["feedback"][agent_id] = f"Your guess {guess} is too high."
            return True

        return False
