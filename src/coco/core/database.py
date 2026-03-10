"""Module for managing local SQLite database logging.

This module provides the DataManager class, which handles tracking 
simulations, agents, turns, interactions, and agent snapshots over time.
"""
import json
import sqlite3
from contextlib import closing
from typing import Any, Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from coco.core.agent import Agent


class DataManager:
    """Manages a local SQLite database for logging simulation data.

    This includes tracking agent interactions, evolutionary lineages, 
    and cognitive traces over time.

    Attributes:
        db_path (str): The file path to the SQLite database.
    """

    def __init__(self, db_path: str = "coco_simulation.db") -> None:
        """Initializes the DataManager and sets up the database schema.

        Args:
            db_path (str): Path to the SQLite database file. Defaults to "coco_simulation.db".

        Raises:
            ValueError: If db_path is empty or invalid.
        """
        if not db_path:
            raise ValueError("db_path cannot be empty.")
        self.db_path = db_path
        self._initialize_db()

    def _initialize_db(self) -> None:
        """Sets up the SQLite schema for the simulation.

        Creates the necessary tables (simulations, agents, turns, interactions, 
        agent_snapshots) if they do not already exist.

        Raises:
            RuntimeError: If there is an issue executing the schema creation.
        """
        try:
            with closing(sqlite3.connect(self.db_path)) as conn:
                with conn:
                    cursor = conn.cursor()

                    # 1. Simulations: Metadata about each run
                    cursor.execute(
                        """
                        CREATE TABLE IF NOT EXISTS simulations (
                            simulation_id INTEGER PRIMARY KEY AUTOINCREMENT,
                            start_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            config_json TEXT
                        )
                        """
                    )

                    # 2. Agents: Tracks every agent across generations and their traits
                    cursor.execute(
                        """
                        CREATE TABLE IF NOT EXISTS agents (
                            agent_uid INTEGER PRIMARY KEY AUTOINCREMENT,
                            simulation_id INTEGER,
                            agent_id TEXT,
                            generation INTEGER,
                            parent_id TEXT,
                            collaboration_threshold REAL,
                            aggression_threshold REAL,
                            trust_level REAL,
                            model TEXT,
                            FOREIGN KEY (simulation_id) REFERENCES simulations (simulation_id)
                        )
                        """
                    )

                    # 3. Turns: Global environment state per turn
                    cursor.execute(
                        """
                        CREATE TABLE IF NOT EXISTS turns (
                            turn_id INTEGER PRIMARY KEY AUTOINCREMENT,
                            simulation_id INTEGER,
                            generation INTEGER,
                            turn_number INTEGER,
                            global_state_json TEXT,
                            FOREIGN KEY (simulation_id) REFERENCES simulations (simulation_id)
                        )
                        """
                    )

                    # 4. Interactions: The directed edges of the social graph
                    cursor.execute(
                        """
                        CREATE TABLE IF NOT EXISTS interactions (
                            interaction_id INTEGER PRIMARY KEY AUTOINCREMENT,
                            turn_id INTEGER,
                            source_id TEXT,
                            target_id TEXT,
                            interaction_type TEXT,
                            resource_key TEXT,
                            amount REAL,
                            success BOOLEAN,
                            reasoning TEXT,
                            FOREIGN KEY (turn_id) REFERENCES turns (turn_id)
                        )
                        """
                    )

                    # 5. Agent Snapshots: Private state of agents at each turn
                    cursor.execute(
                        """
                        CREATE TABLE IF NOT EXISTS agent_snapshots (
                            snapshot_id INTEGER PRIMARY KEY AUTOINCREMENT,
                            turn_id INTEGER,
                            agent_id TEXT,
                            resources_json TEXT,
                            memory_json TEXT,
                            social_ledger_json TEXT,
                            fitness REAL,
                            FOREIGN KEY (turn_id) REFERENCES turns (turn_id)
                        )
                        """
                    )
        except sqlite3.Error as e:
            raise RuntimeError(f"Database initialization failed: {e}") from e

    def create_simulation(self, config: Dict[str, Any]) -> int:
        """Creates a new simulation record in the database.

        Args:
            config (Dict[str, Any]): A dictionary containing simulation configuration.

        Returns:
            int: The primary key (simulation_id) of the newly created simulation.

        Raises:
            ValueError: If the config is None or not a dictionary.
            RuntimeError: If inserting the simulation fails.
        """
        if not isinstance(config, dict):
            raise ValueError("config must be a dictionary.")

        try:
            with closing(sqlite3.connect(self.db_path)) as conn:
                with conn:
                    cursor = conn.cursor()
                    cursor.execute(
                        "INSERT INTO simulations (config_json) VALUES (?)",
                        (json.dumps(config),),
                    )
                    sim_id = cursor.lastrowid
                    if sim_id is None:
                        raise RuntimeError("Failed to retrieve last inserted row id.")
                    return sim_id
        except sqlite3.Error as e:
            raise RuntimeError(f"Failed to create simulation: {e}") from e

    def log_agent(
        self, sim_id: int, agent: 'Agent', parent_id: Optional[str] = None
    ) -> None:
        """Logs an agent's existence and initial traits into the database.

        Args:
            sim_id (int): The ID of the simulation this agent belongs to.
            agent (Agent): The agent object to log.
            parent_id (Optional[str]): The ID of the parent agent, if any. Defaults to None.

        Raises:
            ValueError: If sim_id is invalid or agent is None.
            RuntimeError: If the database insertion fails.
        """
        if sim_id < 1:
            raise ValueError("sim_id must be a positive integer.")
        if agent is None:
            raise ValueError("agent cannot be None.")

        try:
            with closing(sqlite3.connect(self.db_path)) as conn:
                with conn:
                    cursor = conn.cursor()
                    cursor.execute(
                        """
                        INSERT INTO agents (
                            simulation_id, agent_id, generation, parent_id,
                            collaboration_threshold, aggression_threshold, trust_level, model
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            sim_id,
                            agent.agent_id,
                            getattr(agent, "generation", 0),
                            parent_id,
                            agent.traits.collaboration_threshold,
                            agent.traits.aggression_threshold,
                            agent.traits.trust_level,
                            agent.model,
                        ),
                    )
        except sqlite3.Error as e:
            raise RuntimeError(f"Failed to log agent: {e}") from e

    def log_turn(
        self, sim_id: int, gen: int, turn_num: int, global_state: Dict[str, Any]
    ) -> int:
        """Logs the global state of the environment for a specific turn.

        Args:
            sim_id (int): The ID of the active simulation.
            gen (int): The current generation number.
            turn_num (int): The turn number within the generation.
            global_state (Dict[str, Any]): A dictionary representing the environment state.

        Returns:
            int: The primary key (turn_id) of the newly logged turn.

        Raises:
            ValueError: If sim_id, gen, or turn_num are negative, or if global_state is None.
            RuntimeError: If inserting the turn fails.
        """
        if sim_id < 1:
            raise ValueError("sim_id must be a positive integer.")
        if gen < 0 or turn_num < 0:
            raise ValueError("Generation and turn numbers must be non-negative.")
        if not isinstance(global_state, dict):
            raise ValueError("global_state must be a dictionary.")

        try:
            with closing(sqlite3.connect(self.db_path)) as conn:
                with conn:
                    cursor = conn.cursor()
                    cursor.execute(
                        """
                        INSERT INTO turns (simulation_id, generation, turn_number, global_state_json)
                        VALUES (?, ?, ?, ?)
                        """,
                        (sim_id, gen, turn_num, json.dumps(global_state)),
                    )
                    turn_id = cursor.lastrowid
                    if turn_id is None:
                        raise RuntimeError("Failed to retrieve last inserted turn id.")
                    return turn_id
        except sqlite3.Error as e:
            raise RuntimeError(f"Failed to log turn: {e}") from e

    def log_interaction(
        self, turn_id: int, agent_id: str, action: Dict[str, Any], success: bool
    ) -> None:
        """Logs an interaction or action attempted by an agent.

        Args:
            turn_id (int): The ID of the turn during which the action occurred.
            agent_id (str): The ID of the agent performing the action.
            action (Dict[str, Any]): The action dictionary containing details.
            success (bool): Whether the action was successful or not.

        Raises:
            ValueError: If turn_id is invalid, agent_id is empty, or action is None.
            RuntimeError: If the database insertion fails.
        """
        if turn_id < 1:
            raise ValueError("turn_id must be a positive integer.")
        if not agent_id:
            raise ValueError("agent_id cannot be empty.")
        if not isinstance(action, dict):
            raise ValueError("action must be a dictionary.")

        try:
            with closing(sqlite3.connect(self.db_path)) as conn:
                with conn:
                    cursor = conn.cursor()
                    cursor.execute(
                        """
                        INSERT INTO interactions (
                            turn_id, source_id, target_id, interaction_type, 
                            resource_key, amount, success, reasoning
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            turn_id,
                            agent_id,
                            action.get("target_id"),
                            action.get("action_type"),
                            action.get("resource_key"),
                            action.get("amount", 1.0),
                            success,
                            action.get("reasoning"),
                        ),
                    )
        except sqlite3.Error as e:
            raise RuntimeError(f"Failed to log interaction: {e}") from e

    def log_agent_snapshot(self, turn_id: int, agent: 'Agent') -> None:
        """Logs a snapshot of an agent's private state for a given turn.

        Args:
            turn_id (int): The ID of the associated turn.
            agent (Agent): The agent whose snapshot is being recorded.

        Raises:
            ValueError: If turn_id is invalid or agent is None.
            RuntimeError: If the database insertion fails.
        """
        if turn_id < 1:
            raise ValueError("turn_id must be a positive integer.")
        if agent is None:
            raise ValueError("agent cannot be None.")

        try:
            with closing(sqlite3.connect(self.db_path)) as conn:
                with conn:
                    cursor = conn.cursor()
                    cursor.execute(
                        """
                        INSERT INTO agent_snapshots (
                            turn_id, agent_id, resources_json, memory_json, social_ledger_json, fitness
                        ) VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        (
                            turn_id,
                            agent.agent_id,
                            json.dumps(agent.resources),
                            json.dumps(agent.memory),
                            json.dumps(agent.social_ledger),
                            agent.fitness,
                        ),
                    )
        except sqlite3.Error as e:
            raise RuntimeError(f"Failed to log agent snapshot: {e}") from e

    def get_simulation_logs(self, sim_id: int) -> Dict[str, Any]:
        """Retrieves all logs for a specific simulation.

        Args:
            sim_id (int): The ID of the simulation to retrieve logs for.

        Returns:
            Dict[str, Any]: A dictionary containing all simulation data, 
                including agents, turns, and interactions.

        Raises:
            ValueError: If sim_id is invalid.
            RuntimeError: If the database query fails.
        """
        if sim_id < 1:
            raise ValueError("sim_id must be a positive integer.")

        try:
            with closing(sqlite3.connect(self.db_path)) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()

                # Get simulation metadata
                cursor.execute(
                    "SELECT * FROM simulations WHERE simulation_id = ?", (sim_id,)
                )
                sim_row = cursor.fetchone()
                if not sim_row:
                    return {}

                logs: Dict[str, Any] = {
                    "simulation": dict(sim_row),
                    "agents": [],
                    "turns": [],
                }

                # Get agents
                cursor.execute("SELECT * FROM agents WHERE simulation_id = ?", (sim_id,))
                logs["agents"] = [dict(row) for row in cursor.fetchall()]

                # Get turns and their interactions/snapshots
                cursor.execute("SELECT * FROM turns WHERE simulation_id = ?", (sim_id,))
                turn_rows = cursor.fetchall()

                for t_row in turn_rows:
                    turn_id = t_row["turn_id"]
                    turn_data = dict(t_row)

                    # Get interactions for this turn
                    cursor.execute(
                        "SELECT * FROM interactions WHERE turn_id = ?", (turn_id,)
                    )
                    turn_data["interactions"] = [dict(i_row) for i_row in cursor.fetchall()]

                    # Get snapshots for this turn
                    cursor.execute(
                        "SELECT * FROM agent_snapshots WHERE turn_id = ?", (turn_id,)
                    )
                    turn_data["snapshots"] = [dict(s_row) for s_row in cursor.fetchall()]

                    logs["turns"].append(turn_data)

                return logs
        except sqlite3.Error as e:
            raise RuntimeError(f"Failed to retrieve simulation logs: {e}") from e