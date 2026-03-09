import pytest

from coco.core.agent import Agent
from coco.tasks.number_guesser import NumberGuesserEnvironment


@pytest.fixture
def env() -> NumberGuesserEnvironment:
    return NumberGuesserEnvironment(target_number=42)


@pytest.fixture
def agent() -> Agent:
    return Agent(agent_id="test_agent")


def test_init_number_guesser(env: NumberGuesserEnvironment) -> None:
    assert env.state["target_number"] == 42
    assert env.state["min_val"] == 1
    assert env.state["max_val"] == 100
    assert env.state["feedback"] == {}
    assert env.state["winners"] == []


def test_get_agent_view(env: NumberGuesserEnvironment, agent: Agent) -> None:
    env.register_agent(agent)
    view = env.get_agent_view("test_agent")

    # Check target number is hidden
    assert "target_number" not in view["global_state"]
    assert "min_val" in view["global_state"]

    # Check feedback
    assert (
        view["personal_feedback"]
        == "No guesses made yet. Guess a number between 1 and 100."
    )

    # Check available actions
    assert len(view["available_actions"]) == 1
    assert view["available_actions"][0]["action_type"] == "guess"


@pytest.mark.asyncio
async def test_handle_custom_action_correct_guess(
    env: NumberGuesserEnvironment, agent: Agent
) -> None:
    env.register_agent(agent)
    action = {"action_type": "guess", "value": 42}
    result = await env.handle_custom_action("test_agent", action)

    assert result is True
    assert "Correct!" in env.state["feedback"]["test_agent"]
    assert "test_agent" in env.state["winners"]


@pytest.mark.asyncio
async def test_handle_custom_action_low_guess(
    env: NumberGuesserEnvironment, agent: Agent
) -> None:
    env.register_agent(agent)
    action = {"action_type": "guess", "value": 10}
    result = await env.handle_custom_action("test_agent", action)

    assert result is True
    assert env.state["feedback"]["test_agent"] == "Your guess 10 is too low."


@pytest.mark.asyncio
async def test_handle_custom_action_high_guess(
    env: NumberGuesserEnvironment, agent: Agent
) -> None:
    env.register_agent(agent)
    action = {"action_type": "guess", "value": 90}
    result = await env.handle_custom_action("test_agent", action)

    assert result is True
    assert env.state["feedback"]["test_agent"] == "Your guess 90 is too high."


@pytest.mark.asyncio
async def test_handle_custom_action_invalid_guess_str(
    env: NumberGuesserEnvironment, agent: Agent
) -> None:
    env.register_agent(agent)
    action = {"action_type": "guess", "value": "42"}
    result = await env.handle_custom_action("test_agent", action)

    assert result is True
    assert "Correct!" in env.state["feedback"]["test_agent"]


@pytest.mark.asyncio
async def test_handle_custom_action_invalid_type(
    env: NumberGuesserEnvironment, agent: Agent
) -> None:
    env.register_agent(agent)
    action = {"action_type": "guess", "value": "not_a_number"}
    result = await env.handle_custom_action("test_agent", action)

    assert result is False
    assert "Invalid guess" in env.state["feedback"]["test_agent"]


@pytest.mark.asyncio
async def test_handle_custom_action_unknown_action(
    env: NumberGuesserEnvironment, agent: Agent
) -> None:
    env.register_agent(agent)
    action = {"action_type": "fly"}
    result = await env.handle_custom_action("test_agent", action)
    assert result is False
