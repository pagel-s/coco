import pytest

from coco.core.environment import Environment
from coco.core.agent import Agent

@pytest.fixture
def env() -> Environment:
    return Environment()

@pytest.fixture
def agent_a() -> Agent:
    a = Agent(agent_id="agent_a")
    a.resources = {"token": 10, "hint": "hello"}
    return a

@pytest.fixture
def agent_b() -> Agent:
    return Agent(agent_id="agent_b")

def test_register_agent(env: Environment, agent_a: Agent) -> None:
    env.register_agent(agent_a)
    assert "agent_a" in env.agents
    assert "agent_a" in env.resource_ledger
    assert env.resource_ledger["agent_a"] == {}

def test_get_agent_view(env: Environment, agent_a: Agent, agent_b: Agent) -> None:
    env.state = {"task_active": True}
    env.register_agent(agent_a)
    env.register_agent(agent_b)
    
    view = env.get_agent_view("agent_a")
    assert view["global_state"] == {"task_active": True}
    assert "agent_b" in view["other_agents"]
    assert "agent_a" not in view["other_agents"]

@pytest.mark.asyncio
async def test_transfer_resource_success(env: Environment, agent_a: Agent, agent_b: Agent) -> None:
    env.register_agent(agent_a)
    env.register_agent(agent_b)
    # Simulate agent A having a resource and the ledger knowing about it
    env.resource_ledger["agent_a"]["token"] = 10
    
    success = await env.transfer_resource("agent_a", "agent_b", "token")
    assert success is True
    assert "token" not in agent_a.resources
    assert agent_b.resources["token"] == 10
    assert env.resource_ledger["agent_b"]["token"] == 10
    assert "token" not in env.resource_ledger["agent_a"]

@pytest.mark.asyncio
async def test_transfer_resource_failure(env: Environment, agent_a: Agent, agent_b: Agent) -> None:
    env.register_agent(agent_a)
    env.register_agent(agent_b)
    
    # Try to transfer a non-existent resource
    success = await env.transfer_resource("agent_a", "agent_b", "missing_token")
    assert success is False

@pytest.mark.asyncio
async def test_attempt_theft_failure_no_resource(env: Environment, agent_a: Agent, agent_b: Agent) -> None:
    env.register_agent(agent_a)
    env.register_agent(agent_b)
    # Target has no resources
    success = await env.attempt_theft("agent_a", "agent_b", "token")
    assert success is False

@pytest.mark.asyncio
async def test_execute_action_share(env: Environment, agent_a: Agent, agent_b: Agent) -> None:
    env.register_agent(agent_a)
    env.register_agent(agent_b)
    env.resource_ledger["agent_a"]["token"] = 10
    
    action = {
        "action_type": "share",
        "target_id": "agent_b",
        "resource_key": "token"
    }
    await env.execute_action("agent_a", action)
    assert len(env.history) == 1
    assert env.history[0]["success"] is True
    assert env.history[0]["agent_id"] == "agent_a"
    assert env.history[0]["action"]["action_type"] == "share"
    
@pytest.mark.asyncio
async def test_execute_action_pass(env: Environment, agent_a: Agent) -> None:
    env.register_agent(agent_a)
    action = {"action_type": "pass"}
    await env.execute_action("agent_a", action)
    assert len(env.history) == 1
    assert env.history[0]["success"] is True
