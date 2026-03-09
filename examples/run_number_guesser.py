import asyncio

from coco.core.agent import Agent
from coco.tasks.number_guesser import NumberGuesserEnvironment


async def main() -> None:
    print("🎲 Initializing Number Guesser Environment...")
    env = NumberGuesserEnvironment(target_number=73)

    print("🤖 Registering Agent 'Alice' (using local Qwen 2.5 via Ollama)...")
    alice = Agent(agent_id="Alice", model="ollama/qwen2.5:1.5b")
    env.register_agent(alice)

    max_turns = 10

    for turn in range(1, max_turns + 1):
        print(f"\n--- Turn {turn} ---")
        await env.step()

        last_action = env.history[-1] if env.history else None
        if last_action and last_action["agent_id"] == "Alice":
            print(f"Alice's Action: {last_action['action']}")

        feedback = env.state["feedback"].get("Alice")
        print(f"Environment Feedback: {feedback}")

        if "Alice" in env.state["winners"]:
            print(f"\n🎉 Success! Alice guessed the number in {turn} turns!")
            break

    if "Alice" not in env.state["winners"]:
        print(
            f"\n❌ Failure! Alice could not guess the number ({env.state['target_number']}) in {max_turns} turns."
        )


if __name__ == "__main__":
    asyncio.run(main())
