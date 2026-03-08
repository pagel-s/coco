from typing import Dict, Any, Optional

class AgentTraits:
    def __init__(
        self,
        collaboration_threshold: float = 0.5,
        aggression_threshold: float = 0.5,
        trust_level: float = 0.5,
    ):
        # High value = more likely to collaborate
        self.collaboration_threshold = collaboration_threshold
        # High value = more likely to steal or sabotage
        self.aggression_threshold = aggression_threshold
        # High value = more likely to trust other agents' shared info
        self.trust_level = trust_level

class Agent:
    def __init__(self, agent_id: str, traits: Optional[AgentTraits] = None):
        self.agent_id = agent_id
        self.traits = traits or AgentTraits()
        
        # Resources this agent currently holds (e.g., compute tokens, pieces of information)
        self.resources: Dict[str, Any] = {}
        
        # Private memory (scratchpad)
        self.memory: list = []
        
        # Public knowledge the agent chooses to share
        self.public_knowledge: dict = {}
        
        # Track interactions with other agents to build trust/distrust
        self.social_ledger: Dict[str, float] = {}
        
        # Current fitness score
        self.fitness: float = 0.0

    async def act(self, environment_state: Any):
        """
        The main decision loop for the agent. Based on the environment,
        its traits, and its current goal, it chooses an action.
        """
        raise NotImplementedError("Subclasses must implement the act method.")

    async def share(self, target_agent_id: str, resource_key: str, environment) -> bool:
        """
        Attempt to share a resource or information with another agent.
        """
        if resource_key in self.resources:
            # The actual transfer logic should be handled by the Environment
            # to ensure consistency and prevent cheating.
            success = await environment.transfer_resource(
                source_id=self.agent_id,
                target_id=target_agent_id,
                resource_key=resource_key
            )
            return success
        return False

    async def steal(self, target_agent_id: str, resource_key: str, environment) -> bool:
        """
        Attempt to steal a resource or information from another agent.
        Success depends on the environment's rules and the target's defenses.
        """
        # The environment determines if the theft is successful
        success = await environment.attempt_theft(
            thief_id=self.agent_id,
            victim_id=target_agent_id,
            resource_key=resource_key
        )
        if success:
            # Note: This might cost "energy" or "reputation" even if successful
            pass
        return success
