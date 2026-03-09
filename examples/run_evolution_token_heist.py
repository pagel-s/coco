import asyncio

from coco.tasks.runners import run_token_heist_evolution


def run_simulation_cli() -> None:
    asyncio.run(run_token_heist_evolution())


if __name__ == "__main__":
    run_simulation_cli()
