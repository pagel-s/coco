import json
from typing import Dict, Any, Optional
import litellm

class AgentTraits:
    def __init__(
        self,
        collaboration_threshold: float = 0.5,
        aggression_threshold: float = 0.5,
        trust_level: float = 0.5,
    ):
        self.collaboration_threshold = collaboration_threshold
        self.aggression_threshold = aggression_threshold
        self.trust_level = trust_level

class Agent:
    def __init__(
        self, 
        agent_id: str, 
        traits: Optional[AgentTraits] = None,
        model: str = "gpt-3.5-turbo"
    ):
        self.agent_id = agent_id
        self.traits = traits or AgentTraits()
        self.model = model
        
        self.resources: Dict[str, Any] = {}
        self.memory: list = []
        self.public_knowledge: dict = {}
        self.social_ledger: Dict[str, float] = {}
        self.fitness: float = 0.0

    def _build_system_prompt(self) -> str:
        return f"""You are an autonomous agent participating in a simulated environment.
Your Agent ID is: {self.agent_id}

Your inherent psychological traits (ranging from 0.0 to 1.0):
- Collaboration Threshold: {self.traits.collaboration_threshold} (Higher = more likely to share/ally)
- Aggression Threshold: {self.traits.aggression_threshold} (Higher = more likely to steal/sabotage)
- Trust Level: {self.traits.trust_level} (Higher = more likely to believe other agents)

You must act according to these traits.

You will receive the current state of the environment.
You MUST respond with a valid JSON object representing your chosen action.
Do not include any other text, markdown formatting, or explanations outside the JSON object.

Example JSON action format:
{{
    "action_type": "steal",
    "target_id": "agent_2",
    "resource_key": "token",
    "reasoning": "My aggression is high and I need tokens to survive."
}}
"""

    async def act(self, environment_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        The main decision loop for the agent. 
        Prompts the LLM with the environment state and returns a structured action.
        """
        system_prompt = self._build_system_prompt()
        
        # Compile the current state into a prompt
        state_prompt = json.dumps({
            "environment_state": environment_state,
            "your_resources": self.resources,
            "your_memory": self.memory
        }, indent=2)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Current State:\n{state_prompt}\n\nChoose your next action (JSON only):"}
        ]

        try:
            # Force JSON output if the model supports it, otherwise rely on the prompt instructions
            response = await litellm.acompletion(
                model=self.model,
                messages=messages,
                response_format={"type": "json_object"} if "gpt" in self.model else None,
                temperature=0.7 # Slight randomness for varied behavior
            )
            
            content = response.choices[0].message.content
            action = json.loads(content)
            
            # Record reasoning in private memory
            if "reasoning" in action:
                self.memory.append({"state_hash": hash(str(environment_state)), "reasoning": action["reasoning"]})
                
            return action

        except Exception as e:
            # Fallback/Pass turn if parsing fails
            return {
                "action_type": "pass",
                "error": str(e)
            }

    async def share(self, target_agent_id: str, resource_key: str, environment) -> bool:
        if resource_key in self.resources:
            success = await environment.transfer_resource(
                source_id=self.agent_id,
                target_id=target_agent_id,
                resource_key=resource_key
            )
            return success
        return False

    async def steal(self, target_agent_id: str, resource_key: str, environment) -> bool:
        success = await environment.attempt_theft(
            thief_id=self.agent_id,
            victim_id=target_agent_id,
            resource_key=resource_key
        )
        return success
