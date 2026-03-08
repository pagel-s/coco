import pytest
from typing import Dict, Any

from coco.core.agent import Agent, AgentTraits
from coco.core.environment import Environment

class DummyEnvironment(Environment):
    """A dummy environment to test agent interactions without litellm calls."""
    async def transfer_resource(self, source_id: str, target_id: str, resource_key: str) -> bool:
        if resource_key == "success_transfer":
            return True
        return False

    async def attempt_theft(self, thief_id: str, victim_id: str, resource_key: str) -> bool:
        if resource_key == "success_steal":
            return True
        return False

@pytest.fixture
def agent() -> Agent:
    traits = AgentTraits(collaboration_threshold=0.8, aggression_threshold=0.2, trust_level=0.9)
    return Agent(agent_id="test_agent", traits=traits)

@pytest.fixture
def dummy_env() -> DummyEnvironment:
    return DummyEnvironment()

def test_agent_initialization(agent: Agent) -> None:
    assert agent.agent_id == "test_agent"
    assert agent.traits.collaboration_threshold == 0.8
    assert agent.traits.aggression_threshold == 0.2
    assert agent.traits.trust_level == 0.9
    assert agent.resources == {}

@pytest.mark.asyncio
async def test_agent_share_success(agent: Agent, dummy_env: DummyEnvironment) -> None:
    agent.resources["success_transfer"] = "some_value"
    result = await agent.share("target_agent", "success_transfer", dummy_env)
    assert result is True

@pytest.mark.asyncio
async def test_agent_share_failure_no_resource(agent: Agent, dummy_env: DummyEnvironment) -> None:
    # Resource not in agent.resources
    result = await agent.share("target_agent", "missing_resource", dummy_env)
    assert result is False

@pytest.mark.asyncio
async def test_agent_steal_success(agent: Agent, dummy_env: DummyEnvironment) -> None:
    result = await agent.steal("target_agent", "success_steal", dummy_env)
    assert result is True

@pytest.mark.asyncio
async def test_agent_steal_failure(agent: Agent, dummy_env: DummyEnvironment) -> None:
    result = await agent.steal("target_agent", "fail_steal", dummy_env)
    assert result is False

@pytest.mark.asyncio
async def test_agent_act_fallback() -> None:
    """Test the fallback mechanism when litellm fails/raises exception."""
    a = Agent(agent_id="err_agent", model="invalid_model")
    # This will trigger the broad Exception block in agent.act
    action = await a.act({"state": "test"})
    assert action["action_type"] == "pass"
    assert "error" in action
