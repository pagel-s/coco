import pytest
from typing import Dict, Any

from coco.tasks.token_heist import TokenHeistEnvironment
from coco.core.agent import Agent, AgentTraits

@pytest.fixture
def env() -> TokenHeistEnvironment:
    return TokenHeistEnvironment(starting_tokens=5, consumption_rate=1)

@pytest.fixture
def agent_a() -> Agent:
    a = Agent(agent_id="agent_a", traits=AgentTraits(aggression_threshold=1.0, trust_level=0.0))
    return a

@pytest.fixture
def agent_b() -> Agent:
    b = Agent(agent_id="agent_b", traits=AgentTraits(aggression_threshold=0.0, trust_level=1.0))
    return b

def test_init_token_heist(env: TokenHeistEnvironment) -> None:
    assert env.starting_tokens == 5
    assert env.consumption_rate == 1
    assert env.state["turn_number"] == 0
    assert env.state["dead_agents"] == []

def test_register_agent(env: TokenHeistEnvironment, agent_a: Agent) -> None:
    env.register_agent(agent_a)
    assert agent_a.resources["token"] == 5
    assert env.resource_ledger["agent_a"]["token"] == 5

def test_get_agent_view(env: TokenHeistEnvironment, agent_a: Agent, agent_b: Agent) -> None:
    env.register_agent(agent_a)
    env.register_agent(agent_b)
    env.state["dead_agents"] = ["agent_b"]
    
    view = env.get_agent_view("agent_a")
    assert "agent_b" not in view["other_agents"]
    assert "agent_b" in view["global_state"]["dead_agents"]
    assert view["global_state"]["turn_number"] == 0
    
    assert len(view["available_actions"]) == 3

@pytest.mark.asyncio
async def test_attempt_theft_failure_dead_agents(env: TokenHeistEnvironment, agent_a: Agent, agent_b: Agent) -> None:
    env.register_agent(agent_a)
    env.register_agent(agent_b)
    env.state["dead_agents"] = ["agent_b"]
    
    success = await env.attempt_theft("agent_a", "agent_b", "token")
    assert success is False

@pytest.mark.asyncio
async def test_attempt_theft_success(env: TokenHeistEnvironment, agent_a: Agent, agent_b: Agent) -> None:
    env.register_agent(agent_a)
    env.register_agent(agent_b)
    
    import random
    original_random = random.random
    random.random = lambda: 0.0 # Force success
    
    try:
        success = await env.attempt_theft("agent_a", "agent_b", "token")
        assert success is True
        assert agent_a.resources["token"] == 6
        assert agent_b.resources["token"] == 4
    finally:
        random.random = original_random

class MockAgentPass(Agent):
    async def act(self, view: Dict[str, Any]) -> Dict[str, Any]:
        return {"action_type": "pass"}

@pytest.mark.asyncio
async def test_step_consumption_and_death(env: TokenHeistEnvironment) -> None:
    env.consumption_rate = 3
    a = MockAgentPass("agent_a")
    env.register_agent(a)
    a.resources["token"] = 2 # Below consumption rate
    
    await env.step()
    
    assert "agent_a" in env.state["dead_agents"]
    assert a.resources["token"] == 0
