import json
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from coco.core.agent import Agent, MemoryConfig


@pytest.mark.asyncio
@patch("coco.core.agent.litellm.aembedding")
@patch("coco.core.agent.litellm.acompletion")
async def test_agent_vector_memory(mock_completion, mock_embedding) -> None:
    """Test the vector memory integration of the Agent class."""
    # 1. Mock chromadb entirely to avoid environment-specific import errors
    mock_chromadb = MagicMock()
    mock_collection = MagicMock()
    mock_collection.query.return_value = {"documents": [["Relevant memory content"]]}
    mock_client = MagicMock()
    mock_client.get_or_create_collection.return_value = mock_collection
    mock_chromadb.PersistentClient.return_value = mock_client

    with patch.dict(sys.modules, {"chromadb": mock_chromadb}):
        # 2. Setup LiteLLM Mocks
        mock_embedding.return_value = AsyncMock(
            data=[AsyncMock(embedding=[0.1, 0.2, 0.3])]
        )
        
        mock_completion.return_value = AsyncMock(
            choices=[
                AsyncMock(
                    message=AsyncMock(
                        content=json.dumps(
                            {"action_type": "pass", "reasoning": "I am testing vector memory."}
                        )
                    )
                )
            ]
        )

        memory_config = MemoryConfig(
            memory_type="vector",
            embedding_model="test-embedding-model",
            vector_db_path="./test_chroma_db",
        )

        # 3. Initialize Agent with vector memory
        agent = Agent(agent_id="vector_test_agent", memory_config=memory_config)

        # 4. Perform an action that should trigger vector memory logic
        action = await agent.act({"state": "initial"})

        # 5. Assertions
        assert action["action_type"] == "pass"
        assert agent.memory_config.memory_type == "vector"
        assert mock_chromadb.PersistentClient.called
        assert mock_client.get_or_create_collection.called
        assert mock_embedding.called
        assert mock_collection.add.called
        assert mock_collection.query.called
