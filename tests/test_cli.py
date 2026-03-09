import pytest
from unittest.mock import patch, MagicMock
from typer.testing import CliRunner
from coco.cli import app

runner = CliRunner()

def test_cli_help():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "CoCo - Collaborate || Compete CLI" in result.output

@patch("examples.run_evolution_token_heist.run_simulation")
@patch("asyncio.run")
def test_cli_sim(mock_asyncio, mock_sim):
    result = runner.invoke(app, ["sim"])
    assert result.exit_code == 0
    mock_asyncio.assert_called_once()

@patch("examples.run_code_fix.main")
@patch("asyncio.run")
def test_cli_codefix(mock_asyncio, mock_main):
    result = runner.invoke(app, ["codefix"])
    assert result.exit_code == 0
    mock_asyncio.assert_called_once()

@patch("subprocess.run")
def test_cli_dashboard(mock_run):
    result = runner.invoke(app, ["dashboard"])
    assert result.exit_code == 0
    mock_run.assert_called_once()
    # Check if streamlit run was called
    args = mock_run.call_args[0][0]
    assert "streamlit" in args
    assert "run" in args
    assert "app.py" in args[2]
