import json
import os
from unittest.mock import AsyncMock, patch

import litellm
import pytest

from coco.core.agent import Agent, MemoryConfig


@pytest.mark.asyncio
async def test_agent_vector_memory() -> None:
    # Mock litellm.aembedding
    mock_embedding_response = AsyncMock()
    mock_embedding_response.data = [AsyncMock(embedding=[0.1, 0.2, 0.3])]

    # Mock litellm.acompletion
    mock_completion_response = AsyncMock()
    mock_completion_response.choices = [
        AsyncMock(
            message=AsyncMock(
                content=json.dumps(
                    {"action_type": "pass", "reasoning": "I am testing vector memory."}
                )
            )
        )
    ]

    memory_config = MemoryConfig(
        memory_type="vector",
        embedding_model="test-embedding-model",
        vector_db_path="./test_chroma_db",
    )

    # Use a unique agent_id to avoid collection collisions in tests
    agent = Agent(agent_id="vector_test_agent", memory_config=memory_config)

    with (
        patch("litellm.aembedding", return_value=mock_embedding_response),
        patch("litellm.acompletion", return_value=mock_completion_response),
    ):
        # 1. Act once to store something in vector memory
        await agent.act({"state": "initial"})

        assert agent.memory_config.memory_type == "vector"
        assert agent._vector_collection is not None

        # 2. Act again, which should trigger a query to vector memory
        await agent.act({"state": "next"})

        # Verify that aembedding was called for both adding and querying
        assert litellm.aembedding.call_count >= 2

    # Cleanup: remove the test database if possible or just rely on unique names
    import shutil

    if os.path.exists("./test_chroma_db"):
        shutil.rmtree("./test_chroma_db")
