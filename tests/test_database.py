import os
import sqlite3

import pytest

from coco.core.agent import Agent, AgentTraits
from coco.core.database import DataManager


@pytest.fixture
def db_path():
    path = "test_simulation.db"
    if os.path.exists(path):
        os.remove(path)
    yield path
    if os.path.exists(path):
        os.remove(path)


@pytest.fixture
def manager(db_path):
    return DataManager(db_path)


def test_database_initialization(manager, db_path):
    assert os.path.exists(db_path)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Check if tables exist
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [row[0] for row in cursor.fetchall()]
    assert "simulations" in tables
    assert "agents" in tables
    assert "turns" in tables
    assert "interactions" in tables
    assert "agent_snapshots" in tables
    conn.close()


def test_create_simulation(manager):
    config = {"task": "test", "pop": 10}
    sim_id = manager.create_simulation(config)
    assert isinstance(sim_id, int)

    conn = sqlite3.connect(manager.db_path)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT config_json FROM simulations WHERE simulation_id = ?", (sim_id,)
    )
    row = cursor.fetchone()
    assert row[0] == '{"task": "test", "pop": 10}'
    conn.close()


def test_log_agent(manager):
    sim_id = manager.create_simulation({})
    traits = AgentTraits(0.1, 0.2, 0.3)
    agent = Agent(agent_id="test_agent", traits=traits, model="test-model")

    manager.log_agent(sim_id, agent, parent_id="parent_1")

    conn = sqlite3.connect(manager.db_path)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT agent_id, collaboration_threshold, model, parent_id FROM agents WHERE simulation_id = ?",
        (sim_id,),
    )
    row = cursor.fetchone()
    assert row[0] == "test_agent"
    assert row[1] == 0.1
    assert row[2] == "test-model"
    assert row[3] == "parent_1"
    conn.close()


def test_log_turn(manager):
    sim_id = manager.create_simulation({})
    turn_id = manager.log_turn(sim_id, gen=1, turn_num=5, global_state={"key": "val"})
    assert isinstance(turn_id, int)

    conn = sqlite3.connect(manager.db_path)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT generation, turn_number, global_state_json FROM turns WHERE turn_id = ?",
        (turn_id,),
    )
    row = cursor.fetchone()
    assert row[0] == 1
    assert row[1] == 5
    assert row[2] == '{"key": "val"}'
    conn.close()


def test_log_interaction(manager):
    sim_id = manager.create_simulation({})
    turn_id = manager.log_turn(sim_id, 0, 1, {})

    action = {
        "action_type": "steal",
        "target_id": "victim",
        "resource_key": "token",
        "reasoning": "hungry",
    }
    manager.log_interaction(turn_id, "thief", action, success=True)

    conn = sqlite3.connect(manager.db_path)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT source_id, target_id, interaction_type, reasoning FROM interactions WHERE turn_id = ?",
        (turn_id,),
    )
    row = cursor.fetchone()
    assert row[0] == "thief"
    assert row[1] == "victim"
    assert row[2] == "steal"
    assert row[3] == "hungry"
    conn.close()


def test_get_simulation_logs(manager):
    sim_id = manager.create_simulation({"task": "test"})
    
    agent = Agent("test_agent")
    manager.log_agent(sim_id, agent)
    
    turn_id = manager.log_turn(sim_id, 0, 1, {"env": "state"})
    manager.log_interaction(turn_id, "test_agent", {"action_type": "move"}, True)
    manager.log_agent_snapshot(turn_id, agent)
    
    logs = manager.get_simulation_logs(sim_id)
    
    assert logs["simulation"]["simulation_id"] == sim_id
    assert len(logs["agents"]) == 1
    assert logs["agents"][0]["agent_id"] == "test_agent"
    assert len(logs["turns"]) == 1
    assert logs["turns"][0]["turn_number"] == 1
    assert len(logs["turns"][0]["interactions"]) == 1
    assert logs["turns"][0]["interactions"][0]["interaction_type"] == "move"
    assert len(logs["turns"][0]["snapshots"]) == 1
    assert logs["turns"][0]["snapshots"][0]["agent_id"] == "test_agent"
