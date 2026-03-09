from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from coco.tasks.runners import run_code_fix_example, run_token_heist_evolution


@pytest.mark.asyncio
@patch("coco.tasks.runners.DataManager")
@patch("coco.tasks.runners.EvolutionaryEngine")
@patch("coco.tasks.runners.TokenHeistEnvironment")
async def test_run_token_heist_evolution(mock_env, mock_engine, mock_db):
    # Setup mocks
    mock_env_instance = MagicMock()
    mock_env_instance.state = {"dead_agents": []}
    mock_env_instance.history = []
    # step is async
    mock_env_instance.step = AsyncMock()
    mock_env.return_value = mock_env_instance

    mock_engine_instance = MagicMock()
    # Create mock agents
    mock_agent_1 = MagicMock()
    mock_agent_1.agent_id = "agent_1"
    mock_agent_1.resources = {"token": 5}
    mock_agent_2 = MagicMock()
    mock_agent_2.agent_id = "agent_2"
    mock_agent_2.resources = {"token": 5}

    mock_engine_instance.population = [mock_agent_1, mock_agent_2]
    mock_engine.return_value = mock_engine_instance

    # Run the function
    with patch("os.path.exists", return_value=False):
        await run_token_heist_evolution()

    assert mock_db.called
    assert mock_engine.called
    assert mock_env.called
    assert mock_env_instance.step.called


@pytest.mark.asyncio
@patch("coco.tasks.runners.CodeFixEnvironment")
@patch("coco.tasks.runners.Agent")
async def test_run_code_fix_example(mock_agent, mock_env):
    # Setup mocks
    mock_env_instance = MagicMock()
    mock_env_instance.state = {"passing_methods": {}}
    mock_env_instance.history = []
    mock_env_instance.step = AsyncMock()
    mock_env.return_value = mock_env_instance

    await run_code_fix_example()

    assert mock_env.called
    assert mock_agent.called
    assert mock_env_instance.step.called
