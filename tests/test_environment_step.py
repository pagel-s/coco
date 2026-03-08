import pytest
from typing import Dict, Any

from coco.core.environment import Environment
from coco.core.agent import Agent

class MockAgent(Agent):
    def __init__(self, agent_id: str, return_action: Dict[str, Any]):
        super().__init__(agent_id=agent_id)
        self.return_action = return_action
        
    async def act(self, environment_state: Dict[str, Any]) -> Dict[str, Any]:
        if self.return_action.get("action_type") == "error":
            raise ValueError("Intentional error for test")
        return self.return_action

@pytest.mark.asyncio
async def test_environment_step() -> None:
    env = Environment()
    
    # Agent 1 will pass
    agent_1 = MockAgent("agent_1", {"action_type": "pass"})
    
    # Agent 2 will fail/raise an error to test the exception handling
    agent_2 = MockAgent("agent_2", {"action_type": "error"})
    
    env.register_agent(agent_1)
    env.register_agent(agent_2)
    
    await env.step()
    
    # history should only contain agent_1's action since agent_2 raised an exception
    assert len(env.history) == 1
    assert env.history[0]["agent_id"] == "agent_1"
    assert env.history[0]["action"]["action_type"] == "pass"

@pytest.mark.asyncio
async def test_attempt_theft_success() -> None:
    # Use monkeypatch or just override the random behavior in environment for theft
    env = Environment()
    
    agent_1 = Agent("agent_1")
    agent_2 = Agent("agent_2")
    agent_2.resources["gold"] = 100
    
    env.register_agent(agent_1)
    env.register_agent(agent_2)
    env.resource_ledger["agent_2"]["gold"] = 100
    
    # Force the random probability to always succeed for testing
    import random
    original_random = random.random
    random.random = lambda: 1.0  # Always greater than 0.5
    
    try:
        success = await env.attempt_theft("agent_1", "agent_2", "gold")
        assert success is True
        assert "gold" in env.resource_ledger["agent_1"]
        assert "gold" not in env.resource_ledger["agent_2"]
    finally:
        random.random = original_random
