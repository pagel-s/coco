import asyncio
import os
import subprocess

import typer

app = typer.Typer(help="CoCo - Collaborate || Compete CLI")


@app.command()
def sim(
    export: bool = typer.Option(
        False, "--export", help="Export simulation logs to JSON."
    ),
) -> None:
    """Run the default Token Heist evolutionary simulation."""
    from coco.tasks.runners import run_token_heist_evolution

    asyncio.run(run_token_heist_evolution(export_json=export))


@app.command()
def dashboard() -> None:
    """Launch the interactive Streamlit analysis dashboard."""
    run_dashboard()


@app.command()
def codefix() -> None:
    """Run the Collaborative Code Fix example."""
    from coco.tasks.runners import run_code_fix_example

    asyncio.run(run_code_fix_example())


def run_dashboard() -> None:
    """Helper to run streamlit from python."""
    # Find the path to app.py relative to this file
    base_path = os.path.dirname(__file__)
    app_path = os.path.join(base_path, "analysis", "app.py")

    print(f"🚀 Launching dashboard from {app_path}...")
    try:
        # Use sys.executable to ensure we use the same python environment
        import sys

        subprocess.run([sys.executable, "-m", "streamlit", "run", app_path], check=True)
    except KeyboardInterrupt:
        print("\n👋 Dashboard stopped.")
    except Exception as e:
        print(f"❌ Error launching dashboard: {e}")


if __name__ == "__main__":
    app()
