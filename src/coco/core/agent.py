import json
from typing import Dict, Any, Optional, List
import litellm
import os


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


class MemoryConfig:
    def __init__(
        self,
        memory_type: str = "local",  # "local" (list) or "vector" (chromadb)
        embedding_model: str = "text-embedding-3-small",
        vector_db_path: str = "./chroma_db",
        top_k: int = 3,
    ):
        self.memory_type = memory_type
        self.embedding_model = embedding_model
        self.vector_db_path = vector_db_path
        self.top_k = top_k


class Agent:
    def __init__(
        self,
        agent_id: str,
        traits: Optional[AgentTraits] = None,
        model: str = "gpt-3.5-turbo",
        generation: int = 0,
        parent_id: Optional[str] = None,
        memory_config: Optional[MemoryConfig] = None,
    ):
        self.agent_id = agent_id
        self.traits = traits or AgentTraits()
        self.model = model
        self.generation = generation
        self.parent_id = parent_id
        self.memory_config = memory_config or MemoryConfig()

        self.resources: Dict[str, Any] = {}
        self.memory: list = []  # Acts as short-term/working memory
        self.public_knowledge: dict = {}
        self.social_ledger: Dict[str, float] = {}
        self.fitness: float = 0.0

        # Initialize Vector DB if needed
        self._vector_collection: Optional[Any] = None
        if self.memory_config.memory_type == "vector":
            self._init_vector_db()

    def _init_vector_db(self) -> None:
        try:
            import chromadb
            from chromadb.config import Settings

            client = chromadb.PersistentClient(path=self.memory_config.vector_db_path)
            # Collection unique to this agent and generation
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
        if self._vector_collection is None:
            return

        # Generate embedding using litellm
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
        if self._vector_collection is None:
            return []

        response = await litellm.aembedding(
            model=self.memory_config.embedding_model, input=[query_text]
        )
        query_embedding = response.data[0].embedding

        results = self._vector_collection.query(
            query_embeddings=[query_embedding], n_results=self.memory_config.top_k
        )
        return results["documents"][0] if results["documents"] else []

    def _build_system_prompt(
        self, relevant_memories: Optional[List[str]] = None
    ) -> str:
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
{{
    "action_type": "steal",
    "target_id": "agent_2",
    "resource_key": "token",
    "reasoning": "My aggression is high and I need tokens to survive."
}}
"""
        return prompt

    async def act(self, environment_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        The main decision loop for the agent.
        Prompts the LLM with the environment state and returns a structured action.
        """
        relevant_memories = None
        if self.memory_config.memory_type == "vector":
            # Query for memories relevant to current state
            query = f"Current state: {json.dumps(environment_state)}"
            relevant_memories = await self._query_vector_memory(query)

        system_prompt = self._build_system_prompt(relevant_memories)

        # Compile the current state into a prompt
        state_prompt = json.dumps(
            {
                "environment_state": environment_state,
                "your_resources": self.resources,
                "recent_memory": self.memory[
                    -5:
                ],  # Always show last 5 turns as working memory
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
            if content is None:
                raise ValueError("LLM returned empty content")

            action = json.loads(content)
            if not isinstance(action, dict):
                raise ValueError("LLM did not return a JSON object")

            # Record reasoning in private memory
            reasoning = action.get("reasoning", "No reasoning provided.")
            mem_entry = {
                "state_hash": hash(str(environment_state)),
                "reasoning": reasoning,
            }
            self.memory.append(mem_entry)

            # Persist to Vector DB if enabled
            if self.memory_config.memory_type == "vector":
                await self._add_to_vector_memory(
                    text=f"In state {hash(str(environment_state))}, I chose {action.get('action_type')} because: {reasoning}",
                    metadata={"action": action.get("action_type", "unknown")},
                )

            return action  # type: ignore[no-any-return]

        except Exception as e:
            return {"action_type": "pass", "error": str(e)}

    async def share(
        self, target_agent_id: str, resource_key: str, environment: Any
    ) -> bool:
        if resource_key in self.resources:
            success: bool = await environment.transfer_resource(
                source_id=self.agent_id,
                target_id=target_agent_id,
                resource_key=resource_key,
            )
            return success
        return False

    async def steal(
        self, target_agent_id: str, resource_key: str, environment: Any
    ) -> bool:
        success: bool = await environment.attempt_theft(
            thief_id=self.agent_id, victim_id=target_agent_id, resource_key=resource_key
        )
        return success
