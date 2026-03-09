import sqlite3
import json
from typing import Dict, Any, Optional, cast


class DataManager:
    """
    Manages a local SQLite database for logging agent interactions,
    evolutionary lineages, and cognitive traces over time.
    """

    def __init__(self, db_path: str = "coco_simulation.db"):
        self.db_path = db_path
        self._initialize_db()

    def _initialize_db(self) -> None:
        """Sets up the SQLite schema for the simulation."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # 1. Simulations: Metadata about each run
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS simulations (
                simulation_id INTEGER PRIMARY KEY AUTOINCREMENT,
                start_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                config_json TEXT
            )
        """)

        # 2. Agents: Tracks every agent across generations and their traits
        cursor.execute("""
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
        """)

        # 3. Turns: Global environment state per turn
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS turns (
                turn_id INTEGER PRIMARY KEY AUTOINCREMENT,
                simulation_id INTEGER,
                generation INTEGER,
                turn_number INTEGER,
                global_state_json TEXT,
                FOREIGN KEY (simulation_id) REFERENCES simulations (simulation_id)
            )
        """)

        # 4. Interactions: The directed edges of the social graph (Who did what to whom)
        cursor.execute("""
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
        """)

        # 5. Agent Snapshots: Private state of agents at each turn
        cursor.execute("""
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
        """)

        conn.commit()
        conn.close()

    def create_simulation(self, config: Dict[str, Any]) -> int:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO simulations (config_json) VALUES (?)", (json.dumps(config),)
        )
        sim_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return cast(int, sim_id)

    def log_agent(
        self, sim_id: int, agent: Any, parent_id: Optional[str] = None
    ) -> None:
        conn = sqlite3.connect(self.db_path)
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
        conn.commit()
        conn.close()

    def log_turn(
        self, sim_id: int, gen: int, turn_num: int, global_state: Dict[str, Any]
    ) -> int:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO turns (simulation_id, generation, turn_number, global_state_json)
            VALUES (?, ?, ?, ?)
        """,
            (sim_id, gen, turn_num, json.dumps(global_state)),
        )
        turn_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return cast(int, turn_id)

    def log_interaction(
        self, turn_id: int, agent_id: str, action: Dict[str, Any], success: bool
    ) -> None:
        conn = sqlite3.connect(self.db_path)
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
        conn.commit()
        conn.close()

    def log_agent_snapshot(self, turn_id: int, agent: Any) -> None:
        conn = sqlite3.connect(self.db_path)
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
        conn.commit()
        conn.close()
