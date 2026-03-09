import typer
import subprocess
import sys
import os

app = typer.Typer(help="CoCo - Collaborate || Compete CLI")


@app.command()
def sim() -> None:
    """Run the default Token Heist evolutionary simulation."""
    from examples.run_evolution_token_heist import run_simulation
    import asyncio

    asyncio.run(run_simulation())


@app.command()
def dashboard() -> None:
    """Launch the interactive Streamlit analysis dashboard."""
    run_dashboard()


@app.command()
def codefix() -> None:
    """Run the Collaborative Code Fix example."""
    from examples.run_code_fix import main
    import asyncio

    asyncio.run(main())


def run_dashboard() -> None:
    """Helper to run streamlit from python."""
    # Find the path to app.py relative to this file
    base_path = os.path.dirname(__file__)
    app_path = os.path.join(base_path, "analysis", "app.py")

    print(f"🚀 Launching dashboard from {app_path}...")
    try:
        subprocess.run(["streamlit", "run", app_path], check=True)
    except KeyboardInterrupt:
        print("\n👋 Dashboard stopped.")
    except Exception as e:
        print(f"❌ Error launching dashboard: {e}")


if __name__ == "__main__":
    app()
