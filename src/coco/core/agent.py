"""Module containing the core Agent implementation."""

import json
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import litellm

if TYPE_CHECKING:
    from coco.core.environment import Environment


class AgentTraits:
    """Psychological traits defining an agent's behavior.

    Attributes:
        collaboration_threshold: Float [0, 1]. Higher means more likely to share/ally.
        aggression_threshold: Float [0, 1]. Higher means more likely to steal/sabotage.
        trust_level: Float [0, 1]. Higher means more likely to believe other agents.
    """

    def __init__(
        self,
        collaboration_threshold: float = 0.5,
        aggression_threshold: float = 0.5,
        trust_level: float = 0.5,
    ) -> None:
        """Initializes AgentTraits with optional thresholds.

        Args:
            collaboration_threshold: The likelihood to collaborate (default: 0.5).
            aggression_threshold: The likelihood to be aggressive (default: 0.5).
            trust_level: The base trust in other agents (default: 0.5).

        Raises:
            ValueError: If any threshold is outside the [0.0, 1.0] range.
        """
        if not (0.0 <= collaboration_threshold <= 1.0):
            raise ValueError("collaboration_threshold must be between 0.0 and 1.0.")
        if not (0.0 <= aggression_threshold <= 1.0):
            raise ValueError("aggression_threshold must be between 0.0 and 1.0.")
        if not (0.0 <= trust_level <= 1.0):
            raise ValueError("trust_level must be between 0.0 and 1.0.")

        self.collaboration_threshold = collaboration_threshold
        self.aggression_threshold = aggression_threshold
        self.trust_level = trust_level


class MemoryConfig:
    """Configuration for an agent's memory system.

    Attributes:
        memory_type: Either "local" or "vector".
        embedding_model: The name of the embedding model to use.
        vector_db_path: The file path to the vector database.
        top_k: The number of relevant memories to retrieve.
    """

    def __init__(
        self,
        memory_type: str = "local",
        embedding_model: str = "text-embedding-3-small",
        vector_db_path: str = "./chroma_db",
        top_k: int = 3,
    ) -> None:
        """Initializes MemoryConfig.

        Args:
            memory_type: The type of memory storage ("local" or "vector").
            embedding_model: The model used for embeddings.
            vector_db_path: Path to the chroma DB.
            top_k: Number of memories to retrieve during queries.

        Raises:
            ValueError: If memory_type is not "local" or "vector", or if top_k < 1.
        """
        if memory_type not in ("local", "vector"):
            raise ValueError('memory_type must be "local" or "vector".')
        if top_k < 1:
            raise ValueError("top_k must be at least 1.")

        self.memory_type = memory_type
        self.embedding_model = embedding_model
        self.vector_db_path = vector_db_path
        self.top_k = top_k


class Agent:
    """An autonomous agent participating in the environment.

    Attributes:
        agent_id: A unique string identifying the agent.
        traits: The psychological traits of the agent.
        model: The LLM model used by the agent for decision making.
        generation: The generational level of the agent.
        parent_id: The ID of the agent's parent, if any.
        memory_config: The memory configuration of the agent.
        resources: A dictionary of resources currently held by the agent.
        memory: A list of recent memories (working memory).
        public_knowledge: A dictionary of publicly known facts.
        social_ledger: A dictionary mapping agent IDs to social scores.
        fitness: A float representing the agent's overall fitness score.
    """

    def __init__(
        self,
        agent_id: str,
        traits: Optional[AgentTraits] = None,
        model: str = "gpt-3.5-turbo",
        generation: int = 0,
        parent_id: Optional[str] = None,
        memory_config: Optional[MemoryConfig] = None,
    ) -> None:
        """Initializes an Agent with given parameters.

        Args:
            agent_id: Unique string identifier for the agent.
            traits: Optional AgentTraits object.
            model: The LLM to use.
            generation: Integer representing the generation number.
            parent_id: Optional parent agent identifier.
            memory_config: Optional MemoryConfig object.

        Raises:
            ValueError: If agent_id is empty or generation is negative.
        """
        if not agent_id:
            raise ValueError("agent_id cannot be empty.")
        if generation < 0:
            raise ValueError("generation cannot be negative.")

        self.agent_id = agent_id
        self.traits = traits or AgentTraits()
        self.model = model
        self.generation = generation
        self.parent_id = parent_id
        self.memory_config = memory_config or MemoryConfig()

        self.resources: Dict[str, Any] = {}
        self.memory: List[Dict[str, Any]] = []
        self.public_knowledge: Dict[str, Any] = {}
        self.social_ledger: Dict[str, float] = {}
        self.fitness: float = 0.0

        self._vector_collection: Optional[Any] = None
        if self.memory_config.memory_type == "vector":
            self._init_vector_db()

    def _init_vector_db(self) -> None:
        """Initializes the vector database for storing memories.

        If chromadb is not installed, it falls back to local memory.
        """
        try:
            import chromadb

            client = chromadb.PersistentClient(path=self.memory_config.vector_db_path)
            collection_name = f"agent_{self.agent_id}_gen_{self.generation}".replace(
                "-", "_"
            ).replace(".", "_")
            self._vector_collection = client.get_or_create_collection(
                name=collection_name
            )
        except ImportError:
            print("⚠️ chromadb not installed. Falling back to local memory.")
            self.memory_config.memory_type = "local"

    async def _add_to_vector_memory(self, text: str, metadata: Dict[str, Any]) -> None:
        """Adds a text snippet with metadata to the vector memory.

        Args:
            text: The text content to embed and store.
            metadata: Associated metadata dictionary.
        """
        if self._vector_collection is None:
            return

        response = await litellm.aembedding(
            model=self.memory_config.embedding_model, input=[text]
        )
        embedding = response.data[0].embedding

        self._vector_collection.add(
            embeddings=[embedding],
            documents=[text],
            metadatas=[metadata],
            ids=[f"mem_{len(self.memory)}"],
        )

    async def _query_vector_memory(self, query_text: str) -> List[str]:
        """Queries the vector database for relevant memories.

        Args:
            query_text: The text to search for.

        Returns:
            A list of retrieved memory strings.
        """
        if self._vector_collection is None:
            return []

        response = await litellm.aembedding(
            model=self.memory_config.embedding_model, input=[query_text]
        )
        query_embedding = response.data[0].embedding

        results = self._vector_collection.query(
            query_embeddings=[query_embedding], n_results=self.memory_config.top_k
        )

        # Results from chromadb query usually return lists of lists.
        # We fetch the first list of documents.
        if "documents" not in results or not results["documents"]:
            return []

        docs = results["documents"][0]
        if docs is None:
            return []
        return [str(doc) for doc in docs]

    def _build_system_prompt(
        self, relevant_memories: Optional[List[str]] = None
    ) -> str:
        """Constructs the system prompt to guide the agent's LLM.

        Args:
            relevant_memories: Optional list of retrieved memories to include.

        Returns:
            The complete system prompt string.
        """
        prompt = f"""You are an autonomous agent participating in a simulated environment.
Your Agent ID is: {self.agent_id}

Your inherent psychological traits (ranging from 0.0 to 1.0):
- Collaboration Threshold: {self.traits.collaboration_threshold} (Higher = more likely to share/ally)
- Aggression Threshold: {self.traits.aggression_threshold} (Higher = more likely to steal/sabotage)
- Trust Level: {self.traits.trust_level} (Higher = more likely to believe other agents)

You must act according to these traits.

You will receive the current state of the environment. Look for 'available_actions' in the state to know what you can do.
You MUST respond with a valid JSON object representing your chosen action.
Do not include any other text, markdown formatting, or explanations outside the JSON object.
"""
        if relevant_memories:
            prompt += "\nRelevant memories from your past:\n"
            for mem in relevant_memories:
                prompt += f"- {mem}\n"

        prompt += """
Example JSON action format (for core actions):
{
    "action_type": "steal",
    "target_id": "agent_2",
    "resource_key": "token",
    "reasoning": "My aggression is high and I need tokens to survive."
}
"""
        return prompt

    async def act(self, environment_state: Dict[str, Any]) -> Dict[str, Any]:
        """Decides the next action based on the environment state.

        Prompts the LLM with the current environment state, historical memory,
        and agent traits, expecting a structured JSON response.

        Args:
            environment_state: A dictionary representing the current state of the environment.

        Returns:
            A dictionary representing the action chosen by the agent.
        """
        relevant_memories = None
        if self.memory_config.memory_type == "vector":
            query = f"Current state: {json.dumps(environment_state)}"
            relevant_memories = await self._query_vector_memory(query)

        system_prompt = self._build_system_prompt(relevant_memories)

        state_prompt = json.dumps(
            {
                "environment_state": environment_state,
                "your_resources": self.resources,
                "recent_memory": self.memory[-5:],
            },
            indent=2,
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"Current State:\n{state_prompt}\n\nChoose your next action (JSON only):",
            },
        ]

        try:
            response = await litellm.acompletion(
                model=self.model,
                messages=messages,
                response_format={"type": "json_object"}
                if "gpt" in self.model
                else None,
                temperature=0.7,
            )

            content = response.choices[0].message.content
            if not content:
                raise ValueError("LLM returned empty content.")

            action = json.loads(content)
            if not isinstance(action, dict):
                raise ValueError("LLM did not return a JSON object.")

            reasoning = action.get("reasoning", "No reasoning provided.")
            mem_entry = {
                "state_hash": hash(str(environment_state)),
                "reasoning": reasoning,
            }
            self.memory.append(mem_entry)

            if self.memory_config.memory_type == "vector":
                await self._add_to_vector_memory(
                    text=f"In state {hash(str(environment_state))}, I chose {action.get('action_type')} because: {reasoning}",
                    metadata={"action": action.get("action_type", "unknown")},
                )

            return action

        except Exception as e:
            return {"action_type": "pass", "error": str(e)}

    async def share(
        self, target_agent_id: str, resource_key: str, environment: "Environment"
    ) -> bool:
        """Shares a resource with another agent.

        Args:
            target_agent_id: The ID of the agent receiving the resource.
            resource_key: The identifier of the resource to share.
            environment: The active Environment instance.

        Returns:
            A boolean indicating whether the transfer was successful.
        """
        if resource_key in self.resources:
            success: bool = await environment.transfer_resource(
                source_id=self.agent_id,
                target_id=target_agent_id,
                resource_key=resource_key,
            )
            return success
        return False

    async def steal(
        self, target_agent_id: str, resource_key: str, environment: "Environment"
    ) -> bool:
        """Attempts to steal a resource from another agent.

        Args:
            target_agent_id: The ID of the agent from whom to steal.
            resource_key: The identifier of the resource to steal.
            environment: The active Environment instance.

        Returns:
            A boolean indicating whether the theft was successful.
        """
        success: bool = await environment.attempt_theft(
            thief_id=self.agent_id, victim_id=target_agent_id, resource_key=resource_key
        )
        return success
