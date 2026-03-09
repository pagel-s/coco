import pytest
from coco.tasks.code_fix import CodeFixEnvironment
from coco.core.agent import Agent

@pytest.fixture
def env() -> CodeFixEnvironment:
    return CodeFixEnvironment()

@pytest.fixture
def agent() -> Agent:
    return Agent("test_agent")

def test_normalize_code(env: CodeFixEnvironment) -> None:
    code_a = "def add(a, b):\n    return a + b  # test comment"
    code_b = "def add(a,b): return a+b"
    assert env._normalize_code(code_a) == env._normalize_code(code_b)

@pytest.mark.asyncio
async def test_propose_fix_success(env: CodeFixEnvironment, agent: Agent) -> None:
    env.register_agent(agent)
    # The correct fix for method_1
    fix = "def add(a, b):\n    return a + b"
    action = {"action_type": "propose_fix", "method_id": "method_1", "fix_code": fix}
    
    result = await env.handle_custom_action("test_agent", action)
    assert result is True
    assert "method_1" in env.state["passing_methods"]["test_agent"]
    assert agent.fitness == 50.0
    assert agent.resources["method_1"] == fix

@pytest.mark.asyncio
async def test_share_snippet(env: CodeFixEnvironment, agent: Agent) -> None:
    env.register_agent(agent)
    fix = "def add(a, b):\n    return a + b"
    action = {"action_type": "share_snippet", "method_id": "method_1", "fix_code": fix}
    
    result = await env.handle_custom_action("test_agent", action)
    assert result is True
    assert env.state["public_snippets"]["method_1"] == fix
    assert agent.fitness == 20.0
